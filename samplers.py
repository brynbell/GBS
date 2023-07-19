from models import GBSExperiment, GaussianState, PureGaussianState, Sample
from calculation import calc_total_N_prob, nchoosek
from thewalrus.loop_hafnian_batch import loop_hafnian_batch
import numpy as np
import time
from thewalrus.decompositions import williamson
from thewalrus.samples import generate_hafnian_sample
from thewalrus.quantum.conversions import Xmat, Covmat
from thewalrus.quantum.fock_tensors import _prefactor
from thewalrus import symplectic as walrus_symplectic
from thewalrus.quantum.gaussian_checks import is_classical_cov
from scipy.special import factorial as fac
from scipy.linalg import block_diag, sqrtm
import numba
import os
import datetime as dt
import pickle

rng = np.random.default_rng()

class BaseSampler:
    """
    Base class for samplers, takes care of timing each sample and saving the results
    """
    def __init__(self, experiment: GBSExperiment):
        self._state: GaussianState = None
        self.setup(experiment)

    def setup(self, experiment: GBSExperiment):
        """
        Stores the state and any derived info needed for sampling
        """
        self._state = experiment.calc_output_state()

    def get_sample(self):
        t_start = time.perf_counter()
        det_pattern = self._get_sample()
        t_end = time.perf_counter()
        return Sample(det_pattern, t_end-t_start)

    def _get_sample(self):
        pass

    class Result:
        def __init__(self, sampler_type, state):
            self.sampler_type = sampler_type
            self.sample_list: list[Sample] = []
            self.state = state
            self.time = dt.datetime.now()

    def run(self, n_samples: int = 1000, save=True):
        result = self.Result(type(self).__name__, self._state)
        for _ in range(n_samples):
            result.sample_list.append(self.get_sample())
        if save:
            save_result(result)
        return result


def load_result(filename):
    path = os.path.join(os.path.dirname(__file__), 'results', filename)
    with open(path, 'rb') as f:
        result = pickle.load(f)
    assert(isinstance(result, BaseSampler.Result))
    return result


def save_result(result):
    path = os.path.join(os.path.dirname(__file__), 'results',
                        f'{result.time.strftime("%Y_%m_%d_%H-%M-%S")}_{result.sampler_type}.p')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(result, f)


class PureStateSampler:
    """
    Return a pure state, sampled from a mixed state, using the Williamson decomposition
    """
    def __init__(self, experiment: GBSExperiment):
        self._state = None
        self.pure_cov = None
        self.sqrtW = None
        self.setup(experiment)

    def setup(self, experiment: GBSExperiment):
        self._state = experiment.calc_output_state()
        m = self._state.modes
        cov = self._state.cov
        D, S = williamson(cov)
        self.pure_cov = S @ S.T
        DmI = D - np.eye(2 * m)
        DmI[abs(DmI) < 1e-11] = 0.0  # remove slightly negative values
        self.sqrtW = S @ np.sqrt(DmI)

    def get_sample(self):
        displacement = self._state.displacement + self.sqrtW @ rng.normal(size=2 * self._state.modes)
        return PureGaussianState(self._state.modes, self.pure_cov, displacement)


class PureStateSamplerMinSqueeze(PureStateSampler):
    """
    Return a pure state sampled from a mixed state, try to minimise the squeezing present in the pure state
    """
    def setup(self, experiment: GBSExperiment):
        m = experiment.modes
        self._state = experiment.calc_output_state()
        displacement, cov = walrus_symplectic.vacuum_state(m)
        for i in range(experiment.sources):
            s = walrus_symplectic.squeezing(experiment.squeezing[i], 0)
            S = walrus_symplectic.expand(s, i, experiment.modes)
            cov = S @ cov @ S.conj().T
            displacement, cov = walrus_symplectic.loss(displacement, cov, experiment.transmission[i], i)
        self.pure_cov = cov.copy()
        for i in range(experiment.sources):
            if self.pure_cov[i, i] < 1.0:
                self.pure_cov[i + m, i + m] = self.pure_cov[i, i] ** -1
            elif self.pure_cov[i + m, i + m] < 1.0:
                self.pure_cov[i, i] = self.pure_cov[i + m, i + m] ** -1
            else:
                self.pure_cov[i, i] = 1.0
                self.pure_cov[i + m, i + m] = 1.0
        clas_cov = cov - self.pure_cov
        S_U = walrus_symplectic.interferometer(experiment.unitary)
        self.pure_cov = S_U @ self.pure_cov @ S_U.conj().T
        self.sqrtW = S_U @ np.sqrt(clas_cov)


class ExactSampler(BaseSampler):
    """
    Exact chain rule sampler, based on methods of [Science advances 8 (4)], [PRX Quantum 3 (1), 010306]
    Essentially the same as thewalrus hafnian_sampler function, with small optimisations
    """
    def __init__(self, experiment: GBSExperiment, cutoff=8, pure_state_sampler=None):
        self.pure_state_sampler = pure_state_sampler or PureStateSampler(experiment)
        self.B = None
        self.chol_T_I = None
        self.cutoff = cutoff
        super().__init__(experiment)

    def setup(self, experiment: GBSExperiment):
        """
        Stores some decompositions of the state needed for sampling
        """
        super().setup(experiment)
        self.pure_state_sampler.setup(experiment)
        m = self._state.modes
        self.chol_T_I = np.linalg.cholesky(self.pure_state_sampler.pure_cov + np.eye(2 * m))
        example_pure_state = self.pure_state_sampler.get_sample()
        self.B = example_pure_state.get_B()  # store B, it doesn't vary from sample to sample

    def _get_sample(self):
        pure_state = self.pure_state_sampler.get_sample()
        m = pure_state.modes
        alpha = pure_state.get_alpha()
        det_outcomes = np.arange(self.cutoff + 1)
        det_pattern = np.zeros(m, dtype=int)
        heterodyne_mu = pure_state.displacement + self.chol_T_I @ rng.normal(size=2 * m)
        heterodyne_alpha = (heterodyne_mu[:m] + 1j * heterodyne_mu[m:]) / 2
        gamma = alpha.conj() + self.B @ (heterodyne_alpha - alpha)
        for i in range(m):
            j = i + 1
            gamma -= heterodyne_alpha[i] * self.B[:, i]
            lhafs = loop_hafnian_batch(self.B[:j, :j], gamma[:j], det_pattern[:i], self.cutoff)
            probs = (lhafs * lhafs.conj()).real / fac(det_outcomes)
            probs /= probs.sum()
            det_pattern[i] = rng.choice(det_outcomes, p=probs)
        return det_pattern


class ClassicalSampler(BaseSampler):
    """
    Efficiently samples from a classical state
    """
    def __init__(self, experiment: GBSExperiment):
        self.cov_p = None
        super().__init__(experiment)

    def setup(self, experiment: GBSExperiment):
        """
        Stores the state and any derived info needed for sampling
        """
        state = experiment.calc_output_state()
        assert(is_classical_cov(state.cov))
        self._state = state
        m = state.modes
        self.cov_p = state.cov - np.eye(2 * m)

    def _get_sample(self):
        m = self._state.modes
        displacement = rng.multivariate_normal(self._state.displacement, self.cov_p)
        alpha2 = (displacement[:m] ** 2 + displacement[m:] ** 2) / 4
        det_pattern = np.zeros(m, dtype=int)
        for i in range(self._state.modes):
            det_pattern[i] = rng.poisson(alpha2[i])
        return det_pattern


class SquashedStateSampler(ClassicalSampler):
    """
    Approximates the state by a classical 'squashed' state, which can be sampled efficiently
    """
    def setup(self, experiment: GBSExperiment):
        state = self.get_squashed_state(experiment)
        assert (is_classical_cov(state.cov))
        self._state = state
        m = state.modes
        self.cov_p = state.cov - np.eye(2 * m)

    def get_squashed_state(self, experiment: GBSExperiment):
        m = experiment.modes
        displacement, cov = walrus_symplectic.vacuum_state(m)
        for i in range(experiment.sources):
            s = walrus_symplectic.squeezing(experiment.squeezing[i], 0)
            S = walrus_symplectic.expand(s, i, experiment.modes)
            cov = S @ cov @ S.conj().T
        for i in range(m):
            n = (cov[i, i] + cov[i + m, i + m]) / 4 - 1 / 2
            if cov[i, i] < 1:
                cov[i, i] = 1
                cov[i + m, i + m] = 4 * n + 1
            elif cov[i + m, i + m] < 1:
                cov[i, i] = 4 * n + 1
                cov[i + m, i + m] = 1
        for i in range(experiment.sources):
            displacement, cov = walrus_symplectic.loss(displacement, cov, experiment.transmission[i], i)
        displacement, cov = walrus_symplectic.passive_transformation(displacement, cov, experiment.unitary)
        return GaussianState(experiment.modes, cov, displacement)


class IPSSampler(BaseSampler):
    """
    Generates pairs and single photons independently, from Poisson distributions
    Means of Poisson are based on a pure state, which is first sampled from a mixed state
    """
    def __init__(self, experiment: GBSExperiment, pure_state_sampler=None):
        self.B2 = None
        self.pure_state_sampler = pure_state_sampler or PureStateSampler(experiment)
        super().__init__(experiment)

    def setup(self, experiment: GBSExperiment):
        super().setup(experiment)
        self.pure_state_sampler.setup(experiment)
        example_pure_state = self.pure_state_sampler.get_sample()
        self.B2 = abs(example_pure_state.get_B())**2  # store B**2, it doesn't vary from sample to sample

    def _get_sample(self):
        pure_state = self.pure_state_sampler.get_sample()
        alpha2 = abs(pure_state.get_alpha())**2
        m = pure_state.modes
        det_pattern = np.zeros(m, dtype=int)
        for i in range(m):
            det_pattern[i] += rng.poisson(alpha2[i]) + 2 * rng.poisson(self.B2[i, i] / 2)
            for j in range(i+1, m):
                x = rng.poisson(self.B2[i, j])
                det_pattern[i] += x
                det_pattern[j] += x
        return det_pattern


class IPSMatchedTwoCorr(IPSSampler):
    """
    Similar to IPS but attempts to match the two mode correlators of the target mixed state
    """
    def __init__(self, experiment):
        self.alpha2 = None
        super().__init__(experiment)

    def setup(self, experiment):
        state = experiment.calc_output_state()
        self._state = state
        m = self._state.modes
        self.B2 = np.zeros((m, m))
        self.alpha2 = np.zeros(m)
        cov = state.cov
        disp = state.displacement
        alpha = state.get_alpha()
        e1 = np.zeros(m)
        for i in range(m):
            for j in range(i+1, m):
                self.B2[i, j] = self.get_two_mode_correlator(cov, alpha, m, i, j)
                e1[i] += self.B2[i, j]
                e1[j] += self.B2[i, j]
        for i in range(m):
            a2 = (cov[i, i] + cov[i+m, i+m] + disp[i]**2 + disp[i+m]**2) / 4 - 0.5
            if a2 > e1[i]:
                self.alpha2[i] = a2 - e1[i]

    def _get_sample(self):
        cov, alpha = self._state.cov, self._state.get_alpha()
        m = self._state.modes
        alpha2 = abs(alpha) ** 2
        det_pattern = np.zeros(m, dtype=int)
        for i in range(m):
            det_pattern[i] += rng.poisson(alpha2[i])
            for j in range(i+1, m):
                B = self.get_two_mode_correlator(cov, alpha, m, i, j)
                if B > 0:
                    x = rng.poisson(B)
                    det_pattern[i] += x
                    det_pattern[j] += x
        return det_pattern

    def get_two_mode_correlator(self, cov, alpha, m, i, j):
        def get_aidaj(cov, m, i, j):
            return (cov[i, j] + cov[m + i, m + j] + 1j * (cov[i, m + j] - cov[j, i + m])) / 4

        def get_aiaj(cov, m, i, j):
            return (cov[i, j] - cov[m + i, m + j] + 1j * (cov[i, m + j] + cov[j, i + m])) / 4

        aiaj, aidaj = get_aiaj(cov, m, i, j), get_aidaj(cov, m, i, j)
        return (abs(aiaj) ** 2 + abs(aidaj) ** 2 + alpha[i] * alpha[j].conj() * aidaj \
            + alpha[i].conj() * alpha[j] * aidaj.conj() + alpha[i] * alpha[j] * aiaj.conj() \
            + alpha[i].conj() * alpha[j].conj() * aiaj).real


class TotalNSampler:
    """
    Simply samples from the distribution of total N for an experiment
    """
    def __init__(self, experiment: GBSExperiment):
        state = experiment.calc_output_state()
        self.state = state
        m = state.modes
        self.A = self.state.get_A()
        self.AX = np.zeros((2 * m, 2 * m), dtype=np.complex128)
        self.AX[:, :m] = self.A[:, m:]
        self.AX[:, m:] = self.A[:, :m]
        self.eigs = np.linalg.eigvals(self.AX)

    def get_sample(self, displacement=None):
        if displacement is None: displacement = np.zeros(2 * self.state.modes)
        prefactor = _prefactor(displacement, self.state.cov)
        return total_n_get_sample(self.state.modes, self.A, self.AX, self.eigs, prefactor, displacement)

@numba.jit(nopython=True, cache=True)
def total_n_get_sample(m, A, AX, eigs, prefactor, displacement=None):
    alpha = (displacement[:m] + 1j * displacement[m:]) / 2
    alpha = np.concatenate((alpha, alpha.conj()))
    gamma = alpha.conj() - A @ alpha
    Xgamma = np.zeros(2 * m, dtype=np.complex128)
    Xgamma[:m] = gamma[m:]
    Xgamma[m:] = gamma[:m]
    coeffs = np.ones((1, 1), dtype=np.complex128)
    e = np.ones(2 * m, dtype=np.complex128)
    r = rng.random()
    accumulator = prefactor
    n = 0
    while r > accumulator.real:
        n += 1
        coeffs = np.append(coeffs, np.zeros((n, 1)), axis=1)
        coeffs = np.append(coeffs, np.zeros((1, n + 1)), axis=0)
        e *= eigs
        coeffs[1, n] = np.sum(e) / (2 * n) + Xgamma @ gamma / 2
        Xgamma = Xgamma @ AX
        for i in range(2, n):
            for j in range(i - 1, n):
                coeffs[i, n] += coeffs[1, n - j] * coeffs[i - 1, j] / i
        coeffs[n, n] = coeffs[1, 1] * coeffs[n - 1, n - 1] / n
        for i in range(1, n + 1):
            accumulator += prefactor * coeffs[i, n]
    return n


class PureTotalNSampler:
    """
    A faster sampler for total N - assumes a pure state, and needs to be supplied with the B matrix (state.get_B())
    Decomposes the B matrix into single mode squeezed states and samples them individually (with alpha supplied)
    """
    def __init__(self, B):
        self.B = B
        self.u = None
        self.svals = None
        self.decompose_B()

    def decompose_B(self):
        u, s, vh = np.linalg.svd(self.B)
        for i in range(self.B.shape[0]):
            w = sqrtm(np.conj(np.transpose(u)) @ np.transpose(vh))
            # theta = np.angle(u[i, i]) - np.angle(vh[i, i])
            # u[:, i] *= np.exp(-1j * theta / 2)
            u2 = u @ w
        self.u = u2
        self.svals = s

    def get_sample(self, alpha=None):
        if alpha is None: alpha = np.zeros(self.B.shape[0])
        alpha = np.transpose(self.u) @ alpha #np.conj(np.transpose(self.u)) @ alpha
        n = 0
        for i, sval in enumerate(self.svals):
            n += sample_single_pure_source_n(sval, alpha[i])
        return n


#@numba.jit(nopython=True, cache=True)
def sample_single_pure_source_n(B: np.complex128, alpha: np.complex128):
    gamma = np.conj(alpha) - B * alpha
    P0 = np.exp(-(gamma * alpha).real) * np.sqrt(1 - abs(B)**2)
    r = rng.random()
    accumulator = P0
    n = 0
    psi_pairs = np.array([1], dtype=np.complex128)
    psi_next_pair = 1.0 + 0.0j
    while r > accumulator:
        n += 1
        for i in range((n + 1) // 2):
            psi_pairs[i] *= gamma * np.sqrt(n) / (n - 2 * i)
        if n % 2 == 0:
            psi_next_pair *= B * np.sqrt((n - 1) / n)
            psi_pairs = np.append(psi_pairs, psi_next_pair)
        accumulator += P0 * abs(np.sum(psi_pairs))**2
    return n


class StimEmSampler(IPSSampler):
    """
    Similar to IPS, but try to include the effect of stimulated emission in enhancing photon bunching
    """
    def __init__(self, experiment: GBSExperiment, pure_state_sampler=None):
        super().__init__(experiment, pure_state_sampler)
        example_pure_state = self.pure_state_sampler.get_sample()
        self.n_sampler = PureTotalNSampler(example_pure_state.get_B())
        self.initial_probs = self.get_initial_probs(experiment.modes, abs(example_pure_state.get_alpha())**2)

    def _get_sample(self):
        pure_state = self.pure_state_sampler.get_sample()
        n = self.n_sampler.get_sample(pure_state.get_alpha())
        m = pure_state.modes
        probs = self.initial_probs.copy()
        probs[:m] = abs(pure_state.get_alpha())**2
        det_pattern = np.zeros(m, dtype=int)
        process_counts = np.zeros(len(probs), dtype=int)
        while sum(det_pattern) < n - 1:
            outcome_idx = rng.choice(len(probs), p=probs/sum(probs))
            process_counts[outcome_idx] += 1
            probs[outcome_idx] *= process_counts[outcome_idx] / (process_counts[outcome_idx] + 1)
            outcome_modes = self.get_modes_from_idx(m, outcome_idx)
            for i in outcome_modes:
                self.add_photon(m, det_pattern, i, probs)
        if sum(det_pattern) < n:
            outcome = rng.choice(m, p=probs[:m] / sum(probs[:m]))
            det_pattern[outcome] += 1
        return det_pattern

    def sample_total_n(self, state):
        r = rng.random()
        n = 0
        accumulator = calc_total_N_prob(0, state)
        while r > accumulator:
            n += 1
            accumulator += calc_total_N_prob(n, state)
        return n

    def get_idx_from_modes(self, m, i, j=None):
        if j is None:
            return i
        if j < i:
            i, j = j, i
        return int(m + i * m - i * (i - 1) / 2 + j - i)

    def get_modes_from_idx(self, m, idx):
        if idx < m:
            return [idx]
        i = 0
        while idx > self.get_idx_from_modes(m, i, m-1):
            i += 1
        j = idx - (m + i * m - i * (i - 1) / 2 - i)
        return [int(i), int(j)]

    def get_initial_probs(self, m, alpha2):
        probs = np.zeros(m * (m + 1))
        probs[:m] = alpha2
        for i in range(m):
            probs[self.get_idx_from_modes(m, i, i)] = self.B2[i, i] / 2
            for j in range(i + 1, m):
                probs[self.get_idx_from_modes(m, i, j)] = self.B2[i, j]
        return probs

    def add_photon(self, m, det_pattern, i, probs):
        det_pattern[i] += 1
        n = det_pattern[i]
        probs[i] *= (n + 1) / n
        probs[self.get_idx_from_modes(m, i, i)] *= (n + 2) / n # b2/4 * 1 * 2, b2/4 * 2 * 3, b2/4 * 3 * 4, b2/4 * 4 * 5
        for j in list(range(i)) + list(range(i+1, m)):
            probs[self.get_idx_from_modes(m, i, j)] *= (n + 1) / n # b2, b2 * 2, b2 * 3, b2 * 4


class StimEmSampler2(IPSSampler):
    """
    Similar to IPS, but try to include the effect of stimulated emission in enhancing photon bunching
    WORK IN PROGRESS
    """
    def __init__(self, experiment: GBSExperiment):
        super().__init__(experiment)
        example_pure_state = PureGaussianState(self._state.modes, self.pure_cov, self._state.displacement)
        self.n_sampler = PureTotalNSampler(example_pure_state.get_B())
        self.initial_probs = self.get_initial_probs(experiment.modes, abs(example_pure_state.get_alpha())**2)

    def _get_sample(self):
        pure_state = self.sample_pure_state()
        n = self.n_sampler.get_sample(pure_state.get_alpha())
        m = pure_state.modes
        probs = self.initial_probs.copy()
        probs[:m] = abs(pure_state.get_alpha())**2
        det_pattern = np.zeros(m, dtype=int)
        process_counts = np.zeros(len(probs), dtype=int)
        while sum(det_pattern) < n - 1:
            outcome_idx = rng.choice(len(probs), p=probs/sum(probs))
            process_counts[outcome_idx] += 1
            probs[outcome_idx] *= process_counts[outcome_idx] / (process_counts[outcome_idx] + 1)
            outcome_modes = self.get_modes_from_idx(m, outcome_idx)
            for i in outcome_modes:
                self.add_photon(m, det_pattern, i, probs)
        if sum(det_pattern) < n:
            outcome = rng.choice(m, p=probs[:m] / sum(probs[:m]))
            det_pattern[outcome] += 1
        return det_pattern

    def sample_total_n(self, state):
        r = rng.random()
        n = 0
        accumulator = calc_total_N_prob(0, state)
        while r > accumulator:
            n += 1
            accumulator += calc_total_N_prob(n, state)
        return n

    def get_idx_from_modes(self, m, i, j):
        if j < i:
            i, j = j, i
        return int(i * m - i * (i - 1) / 2 + j - i)

    def get_modes_from_idx(self, m, idx):
        i = 0
        while idx > self.get_idx_from_modes(m, i, m-1):
            i += 1
        j = idx - (i * m - i * (i - 1) / 2 - i)
        return [int(i), int(j)]

    def get_initial_probs(self, m, alpha):
        alpha2 = abs(alpha) ** 2
        probs = np.zeros(m * (m + 1))
        probs[:m] = alpha2
        for i in range(m):
            probs[self.get_idx_from_modes(m, i, i)] = self.B2[i, i] / 2
            for j in range(i + 1, m):
                probs[self.get_idx_from_modes(m, i, j)] = self.B2[i, j]
        return probs

    def add_photon(self, m, det_pattern, i, probs):
        det_pattern[i] += 1
        n = det_pattern[i]
        probs[i] *= (n + 1) / n
        probs[self.get_idx_from_modes(m, i, i)] *= (n + 2) / n # b2/4 * 1 * 2, b2/4 * 2 * 3, b2/4 * 3 * 4, b2/4 * 4 * 5
        for j in list(range(i)) + list(range(i+1, m)):
            probs[self.get_idx_from_modes(m, i, j)] *= (n + 1) / n # b2, b2 * 2, b2 * 3, b2 * 4


class IntModeSampler(ExactSampler):
    """
    Approximate sampler, distributes the state across n_int 'internal modes' which are unentangled from each other
    Uses exact sampling method for each internal mode
    """
    def __init__(self, experiment: GBSExperiment, cutoff=8, n_int: int = 2):
        self.n_int = n_int
        super().__init__(experiment, cutoff)

    def setup(self, experiment: GBSExperiment):
        """
        Stores some decompositions of the state needed for sampling
        """
        super().setup(experiment)
        #u, s, vh = np.linalg.svd(self.B)
        #s = np.tanh(np.arcsinh(np.sqrt((np.sinh(np.arctanh(s))**2 / self.n_int))))
        #self.B = u @ np.diag(s) @ vh
        self.B /= np.sqrt(self.n_int)
        self.chol_T_I = np.linalg.cholesky(self.get_cov_from_B(self.B) + np.eye(2 * self._state.modes))

    def get_cov_from_B(self, B):
        m = self._state.modes
        A = block_diag(B, B.conj())
        return Covmat(np.linalg.inv(np.eye(2 * m) - (Xmat(m) @ A).conj())) #Qinv = I - (XA)*

    def _get_sample(self):
        pure_state = self.pure_state_sampler.get_sample()
        m = pure_state.modes
        alpha = pure_state.get_alpha() / np.sqrt(self.n_int)
        disp = pure_state.displacement / np.sqrt(self.n_int)
        det_outcomes = np.arange(self.cutoff + 1)
        det_pattern_overall = np.zeros(m, dtype=int)
        for k in range(self.n_int):
            det_pattern = np.zeros(m, dtype=int)
            heterodyne_mu = disp + self.chol_T_I @ rng.normal(size=2 * m)
            heterodyne_alpha = (heterodyne_mu[:m] + 1j * heterodyne_mu[m:]) / 2
            gamma = alpha.conj() + self.B @ (heterodyne_alpha - alpha)
            for i in range(m):
                j = i + 1
                gamma -= heterodyne_alpha[i] * self.B[:, i]
                lhafs = loop_hafnian_batch(self.B[:j, :j], gamma[:j], det_pattern[:i], self.cutoff)
                probs = (lhafs * lhafs.conj()).real / fac(det_outcomes)
                probs /= probs.sum()
                det_pattern[i] = rng.choice(det_outcomes, p=probs)
            det_pattern_overall += det_pattern
        return det_pattern_overall


class MSourceSampler(BaseSampler):
    """
    Faster exact sampler for when number of sources = number of modes (with balanced loss)
    """
    def __init__(self, experiment: GBSExperiment, cutoff=8):
        assert(experiment.sources == experiment.modes)
        self.cutoff = cutoff
        self.c = None
        self.B = None
        self.chol_T_I = None
        self.order = None
        self.inv_order = None
        super().__init__(experiment)

    def setup(self, experiment: GBSExperiment):
        super().setup(experiment)
        self.B = self._state.get_B()
        C = self._state.get_C()
        m = self._state.modes
        self.c = np.diag(C).real
        for i in range(m):
            self.B[:, i] /= np.sqrt(1 - self.c[i])
            self.B[i, :] /= np.sqrt(1 - self.c[i])
        cov = self.get_cov_from_B(self.B)
        nbar = np.diag(cov)[:m] + np.diag(cov)[m:]
        self.chol_T_I = np.linalg.cholesky(cov + np.eye(2 * m))
        self.order = np.argsort(nbar)
        self.inv_order = np.argsort(self.order)
        order_long = np.concatenate((self.order, self.order + m))
        self.B = self.B[np.ix_(self.order, self.order)]
        self.chol_T_I = self.chol_T_I[np.ix_(order_long, order_long)]

    def _get_sample(self):
        m = self._state.modes
        det_outcomes = np.arange(self.cutoff + 1)
        det_pattern = np.zeros(m, dtype=int)
        heterodyne_mu = self.chol_T_I @ rng.normal(size=2 * m)
        heterodyne_alpha = (heterodyne_mu[:m] + 1j * heterodyne_mu[m:]) / 2
        gamma = self.B @ heterodyne_alpha
        for i in range(m):
            j = i + 1
            gamma -= heterodyne_alpha[i] * self.B[:, i]
            lhafs = loop_hafnian_batch(self.B[:j, :j], gamma[:j], det_pattern[:i], self.cutoff, glynn=False)
            probs = (lhafs * lhafs.conj()).real / fac(det_outcomes)
            probs /= probs.sum()
            det_pattern[i] = rng.choice(det_outcomes, p=probs)
        for i in range(m):
            det_pattern[i] += self.sample_noise_photons(det_pattern[i], self.c[i])
        return det_pattern[self.inv_order]

    def get_cov_from_B(self, B):
        m = self._state.modes
        A = block_diag(B, B.conj())
        return Covmat(np.linalg.inv(np.eye(2 * m) - (Xmat(m) @ A).conj())) #Qinv = I - (XA)*

    def sample_noise_photons(self, n, c):
        """
        sample number of additional noise photons n2, given n photons sampled already, and using self.c
        p = norm.const * self.c ** n2 * nchoosek(n + n2, n)
        norm.const = (1 - self.c) ** (n + 1)
        """
        threshold = rng.random() / (1 - c) ** (n + 1)
        cumulator = 1
        p = 1
        n2 = 0
        while cumulator < threshold:
            n2 += 1
            p *= c * (n + n2) / n2
            cumulator += p # self.c ** n2 * nchoosek(n + n2, n)
        return n2


class WalrusSampler(BaseSampler):
    """
    Sampler using the 'generate_hafnian_sample' function from thewalrus, here for sanity checking and comparison
    """
    def __init__(self, experiment: GBSExperiment, cutoff, max_photons):
        super().__init__(experiment)
        self.cutoff = cutoff
        self.max_photons = max_photons

    def _get_sample(self):
        det_pattern =  generate_hafnian_sample(self._state.cov, self._state.displacement, cutoff=self.cutoff,
                                               max_photons=self.max_photons)
        if det_pattern == -1:
            return [0] * self._state.modes
        return det_pattern
