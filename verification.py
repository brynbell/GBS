from models import GaussianState, Sample
from calculation import calc_total_N_prob, calc_sample_prob, factorial
from thewalrus._hafnian import hafnian
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import itertools

rng = np.random.default_rng()

class Verifier:
    pass


def get_sample_count_dicts(sample_list: list[Sample], N_list: list[int]):
    count_dicts = {N: {} for N in N_list}
    for sample in sample_list:
        if (N:=sum(sample.det_pattern)) in N_list:
            count_dicts[N].setdefault(sample.string, 0)
            count_dicts[N][sample.string] += 1
    return count_dicts


def hist_total_distribution(sample_list: list[Sample], N_list: list[int]):
    count_dicts = get_sample_count_dicts(sample_list, N_list)
    for N in N_list:
        values = count_dicts[N].values()
        plt.bar(range(len(values)), count_dicts[N].values())
        plt.show()


class TotalNVerifier(Verifier):
    """
    Plots the total photon number distribution versus theory, also shows time per sample
    """
    def __call__(self, sample_list, state=None, log=False):
        n_list = [sum(sample.det_pattern) for sample in sample_list]
        max_n = max(n_list)
        t_list = [sample.time for sample in sample_list]
        n_counts = np.zeros(max_n+1)
        for n in n_list:
            n_counts[n] += 1
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.bar(range(max_n + 1), n_counts)
        if state:
            P = [calc_total_N_prob(N, state) for N in range(max_n+1)]
            ax1.plot(range(max_n+1), np.array(P) * len(n_list), c='orange')
        ax2.scatter(n_list, t_list)
        if log: ax2.set_yscale('log')
        ax1.set_ylabel('Occurrences')
        ax2.set_ylabel('Samples time (s)')
        plt.xlabel('Total photon number')
        plt.show()


class TotalDistributionVerifier(Verifier):
    """
    Checks the fidelity and tvd against ideal probabilities for the total photon numbers given in n_list
    """
    def __call__(self, sample_list, state, n_list):
        count_dicts = get_sample_count_dicts(sample_list, n_list)
        tvd = {}
        fidelity = {}
        for n in n_list:
            p_total_n = calc_total_N_prob(n, state)
            tvd[n] = 0
            fidelity[n] = 0
            count_dict = count_dicts[n]
            total_counts_n = sum(count_dict.values())
            p_calc_total = 0
            for sample_key, counts in count_dict.items():
                sample = Sample(sample_key)
                p_calc = calc_sample_prob(sample, state) / p_total_n
                p_calc_total += p_calc
                p_exp = counts / total_counts_n
                tvd[n] += 0.5 * np.abs(p_exp - p_calc)
                fidelity[n] += np.sqrt(p_calc * p_exp)
            fidelity[n] = abs(fidelity[n]) ** 2
            tvd[n] += 1 - p_calc_total
            print(f'{n} photon fidelity: {fidelity[n]}')
            print(f'{n} photon TVD: {tvd[n]}')


class TwoModeCorrelatorVerifier(Verifier):
    """
    Checks two mode correlators of samples versus theory
    """
    def __call__(self, sample_list: list[Sample], state: GaussianState):
        m = state.modes
        n_samples = len(sample_list)
        alpha = state.get_alpha()
        e1 = np.zeros(m)
        e2 = np.zeros((m, m))
        for sample in sample_list:
            for i in range(m):
                e1[i] += sample.det_pattern[i] / n_samples
                for j in range(i + 1, m):
                    e2[i, j] += sample.det_pattern[i] * sample.det_pattern[j] / n_samples
        C_exp = np.zeros((m, m))
        C_calc = np.zeros((m, m))
        C_calc_list = []
        C_exp_list = []
        for i in range(m):
            for j in range(i + 1, m):
                C_exp[i, j] = e2[i, j] - e1[i] * e1[j]
                aiaj, aidaj = self.get_aiaj(state, i, j), self.get_aidaj(state, i, j)
                C_calc[i, j] = abs(aiaj)**2 + abs(aidaj)**2 + (alpha[i] * alpha[j].conj() * aidaj \
                               + alpha[i].conj() * alpha[j] * aidaj.conj() + alpha[i] * alpha[j] * aiaj.conj() \
                               + alpha[i].conj() * alpha[j].conj() * aiaj).real
                C_exp_list.append(C_exp[i, j])
                C_calc_list.append(C_calc[i, j])
        plt.scatter(C_calc_list, C_exp_list)
        plt.xlabel('Calculated two-mode correlators')
        plt.ylabel('Sampled two-mode correlators')
        plt.plot([min(C_calc_list), max(C_calc_list)], [min(C_calc_list), max(C_calc_list)], c='black')
        plt.show()

    def get_aidaj(self, state, i, j):
        cov = state.cov
        m = state.modes
        return (cov[i, j] + cov[m + i, m + j] + 1j * (cov[i, m + j] - cov[j, i + m])) / 4

    def get_aiaj(self, state, i, j):
        cov = state.cov
        m = state.modes
        return (cov[i, j] - cov[m + i, m + j] + 1j * (cov[i, m + j] + cov[j, i + m])) / 4


def hist_total_n(sample_list: list[Sample]):
    """
    Display histogram of total photon number
    """
    n_list = [sum(sample.det_pattern) for sample in sample_list]
    max_n = max(n_list)
    plt.hist(n_list, bins=range(max_n))
    plt.show()


def plot_t_vs_n(sample_list: list[Sample], log=False):
    """
    scatter plot of sample time versus total photon number
    """
    n_list = [sum(sample.det_pattern) for sample in sample_list]
    t_list = [sample.time for sample in sample_list]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(n_list, t_list)
    if log: ax.set_yscale('log')
    plt.show()


class XEntropyVerifier(Verifier):
    """
    Plots cross-entropy versus total photon number
    """
    def __call__(self, sample_list: list[Sample], state: GaussianState, n_list: list[int]):
        count_dicts = get_sample_count_dicts(sample_list, n_list)
        xe = {n: 0 for n in n_list}
        var_xe = {n: 0 for n in n_list}
        std_err_xe = {n: 0 for n in n_list}
        for n in n_list:
            count_dict = count_dicts[n]
            n_samples = sum(count_dict.values())
            print(f'{n_samples} {n}-photon samples')
            prob_n = calc_total_N_prob(n, state)
            for sample_key, counts in count_dict.items():
                sample = Sample(sample_key)
                xe_sample = self.calc_xe_sample(sample, state, prob_n)
                xe[n] += xe_sample * counts
                var_xe[n] += xe_sample ** 2 * counts
            xe[n] /= n_samples
            var_xe[n] /= n_samples
            var_xe[n] -= xe[n] ** 2
            std_err_xe[n] = np.sqrt(var_xe[n] / n_samples)
        plt.errorbar(xe.keys(), xe.values(), yerr=list(std_err_xe.values()), ls='')
        plt.show()

    def calc_xe_sample(self, sample: Sample, state: GaussianState, prob_n):
        m = state.modes
        prob_sample = calc_sample_prob(sample, state)
        n = sum(sample.det_pattern)
        return np.log(prob_sample / prob_n * comb(n + m - 1, n)).real


class NCumulantVerifier(Verifier):
    """
    Compares Nth order cumulants of a sample list to theory
    """
    def __init__(self, state: GaussianState, sample_list: list[Sample], max_combinations=1000):
        self.cumulants_calc = {}
        self.cumulants_exp = {}
        self.expectation_ns_calc = {}
        self.expectation_ns_exp = {}
        self.sample_list = sample_list
        self.sample_array = np.array([sample.det_pattern for sample in sample_list])
        self.state = state
        self.max_combinations = max_combinations
        self.A = None
        self.set_matrix()

    def set_matrix(self):
        cov, alpha, m = self.state.cov, self.state.get_alpha(), self.state.modes
        A = np.zeros((m * 2, m * 2), dtype=np.complex128)
        for i in range(m):
            for j in range(m):
                # a = (x+ip)/2
                # <a_i a_j> = (x_i x_j + i x_i p_j + i p_i x_j - p_i p_j) / 4
                A[i, j] = (cov[i, j] + 1j * cov[i, m + j] + 1j * cov[i + m, j] - cov[i + m, j + m]) / 4 \
                          + alpha[i] * alpha[j]
                # <ad_i ad_j> = <a_i a_j>*
                A[i + m, j + m] = A[i, j].conj()
                # <ad_i a_j> = (x_i x_j + i x_i p_j - i p_i x_j + p_i p_j) / 4 - delta(i,j)/2
                A[i + m, j] = (cov[i, j] + 1j * cov[i, m + j] - 1j * cov[i + m, j] + cov[i + m, j + m]) / 4 \
                              + alpha[i].conj() * alpha[j] - 0.5 * (i == j)
                # <ad_j a_i> = (x_i x_j - i x_i p_j + i p_i x_j + p_i p_j) / 4 - delta(i,j)/2
                A[i, j + m] = (cov[i, j] - 1j * cov[i, m + j] + 1j * cov[i + m, j] + cov[i + m, j + m]) / 4 \
                              + alpha[i] * alpha[j].conj() - 0.5 * (i == j)
        self.A = A

    def __call__(self, n: int):
        if comb(self.state.modes, n) <= self.max_combinations:
            calc_list, exp_list = self.run_all_combinations(n)
        else:
            calc_list, exp_list = self.run_random_combinations(n)
        plt.scatter(calc_list, exp_list)
        plt.xlabel(f'Calculated {n}-mode correlators')
        plt.ylabel(f'Sampled {n}-mode correlators')
        plt.plot([min(calc_list), max(calc_list)], [min(calc_list), max(calc_list)], c='black')
        plt.show()

    def run_all_combinations(self, n: int):
        m = self.state.modes
        exp_list = []
        calc_list = []
        for combination in itertools.combinations(list(range(m)), n):
            modes = list(combination)
            key = str(np.array(modes))[1:-1]
            self.cumulants_calc.setdefault(key, self.calc_cumulant(modes))
            self.cumulants_exp.setdefault(key, self.exp_cumulant(modes))
            exp_list.append(self.cumulants_exp[key])
            calc_list.append(self.cumulants_calc[key])
        return calc_list, exp_list

    def run_random_combinations(self, n: int):
        m = self.state.modes
        exp_list = []
        calc_list = []
        keys = []
        while len(exp_list) < self.max_combinations:
            modes = list(np.sort(rng.choice(m, n, replace=False)))
            key = str(np.array(modes))[1:-1]
            if key not in keys:
                keys.append(key)
                self.cumulants_calc.setdefault(key, self.calc_cumulant(modes))
                self.cumulants_exp.setdefault(key, self.exp_cumulant(modes))
                exp_list.append(self.cumulants_exp[key])
                calc_list.append(self.cumulants_calc[key])
        return calc_list, exp_list

    def calc_cumulant(self, modes: list[int]):
        """
        Find the theory value of the n-mode cumulant across the given modes, where n is the number of modes
        """
        c = 0
        for partition in self.partitions(modes):
            n_partition = len(partition)
            d = factorial(n_partition - 1) * (-1) ** (n_partition - 1)
            for subpartition in partition:
                key = str(np.array(subpartition))[1:-1]
                d *= self.expectation_ns_calc.setdefault(key, self.calc_expectation_n(subpartition))
            c += d
        return c

    def exp_cumulant(self, modes: list[int]):
        """
        Find the experimental value of the n-mode cumulant across the given modes, where n is the number of modes
        """
        c = 0
        for partition in self.partitions(modes):
            n_partition = len(partition)
            d = factorial(n_partition - 1) * (-1) ** (n_partition - 1)
            for subpartition in partition:
                key = str(np.array(subpartition))[1:-1]
                d *= self.expectation_ns_exp.setdefault(key, self.exp_expectation_n(subpartition))
            c += d
        return c

    def calc_expectation_n(self, modes: list[int]):
        """
        Calculate expectation value <n_i n_j n_k ... > where i,j,k label the modes
        """
        m = self.state.modes
        kept_idx = np.array(modes)
        kept_idx = np.concatenate((kept_idx, kept_idx + m))
        A = self.A[np.ix_(kept_idx, kept_idx)]
        return hafnian(A, method="inclexcl").real

    def exp_expectation_n(self, modes: list[int]):
        """
        Find 'experimental' expectation value <n_i n_j n_k ... > from the sample list
        """
        n_samples = len(self.sample_list)
        sample_array = self.sample_array[:, modes]
        return np.sum(np.prod(sample_array, axis=1)) / n_samples

    def partitions(self, modes: list[int]):
        """
        Iterates for all partitions of the modes given
        """
        if len(modes) == 1:
            yield [modes]
            return
        first = modes[0]
        for smaller in self.partitions(modes[1:]):
            # insert `first` in each of the subpartition's subsets
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
            # put `first` in its own subset
            yield [[first]] + smaller
