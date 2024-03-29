from thewalrus.loop_hafnian_batch import loop_hafnian_batch
import numpy as np
import time
from thewalrus import symplectic as walrus_symplectic
from thewalrus.decompositions import williamson
from thewalrus.quantum import Amat
from scipy.stats import unitary_group
from scipy.special import factorial as fac
from os.path import join as path_join

class GBSExperiment:
    """
    Simple builder for a GBS experiment, identical sources, balanced loss, unitary transform
    """
    def __init__(self, modes=10, sources=5, squeezing=1.0, transmission=0.5, unitary=None):
        self.modes = modes
        self.sources = sources
        self.squeezing = squeezing if isinstance(squeezing, list) else [squeezing] * self.sources
        self.transmission = transmission if isinstance(transmission, list) else [transmission] * self.sources
        self.unitary = unitary if unitary is not None else unitary_group.rvs(self.modes)

    def calc_output_state(self):
        displacement, cov = walrus_symplectic.vacuum_state(self.modes)
        for i in range(self.sources):
            s = walrus_symplectic.squeezing(self.squeezing[i], 0)
            S = walrus_symplectic.expand(s, i, self.modes)
            cov = S @ cov @ S.conj().T
            displacement, cov = walrus_symplectic.loss(displacement, cov, self.transmission[i], i)
        displacement, cov = walrus_symplectic.passive_transformation(displacement, cov, self.unitary)
        return GaussianState(self.modes, cov, displacement)


class BorealisExperiment(GBSExperiment):
    def __init__(self, path):
        self.squeezing = np.load(path_join(path, 'r.npy'))
        self.T = np.load(path_join(path, 'T.npy'))
        self.modes = self.T.shape[0]
        self.sources = self.modes

    def calc_output_state(self):
        displacement, cov = walrus_symplectic.vacuum_state(self.modes)
        for i in range(self.modes):
            s = walrus_symplectic.squeezing(self.squeezing[i], 0)
            S = walrus_symplectic.expand(s, i, self.modes)
            cov = S @ cov @ S.conj().T
        displacement, cov = walrus_symplectic.passive_transformation(displacement, cov, self.T)
        return GaussianState(self.modes, cov, displacement)


class GaussianState:
    """
    A Gaussian state described by covariance matrix and displacement
    """
    def __init__(self, modes, cov, displacement):
        self.modes = modes
        self.cov = cov
        self.displacement = displacement

    def get_A(self):
        return Amat(self.cov)

    def get_B(self):
        return self.get_A()[:self.modes, :self.modes]

    def get_C(self):
        return self.get_A()[:self.modes, self.modes:]

    def get_alpha(self):
        return (self.displacement[:self.modes] + 1j * self.displacement[self.modes:]) / 2

    def get_alpha_long(self):
        alpha = self.get_alpha()
        return np.concatenate((alpha, alpha.conj()))

    def get_gamma(self):
        A, alpha = self.get_A(), self.get_alpha_long()
        return alpha.conj() - A @ alpha


class PureGaussianState(GaussianState):
    pass


class Sample:
    def __init__(self, det_pattern, time=None):
        if isinstance(det_pattern, str):
            self.string = det_pattern
            self.det_pattern = self.decode()
        else:
            self.det_pattern = det_pattern
            self.string = self.encode()
        self.time = time
        self.total_n = np.sum(self.det_pattern)
        self.modes = len(det_pattern)

    def encode(self):
        # convert detection pattern to string (space separated integers) for use as dictionary key, etc
        return str(self.det_pattern)[1:-1]

    def decode(self):
        # inverse of encode - get detection pattern as numpy array from a string
        return np.array([int(n) for n in self.string.split()], dtype=int)
