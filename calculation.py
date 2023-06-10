from models import GaussianState, Sample
from hafnian import WalrusLoopHafnianCalculator
from thewalrus.quantum.fock_tensors import _prefactor
import numpy as np
from thewalrus._hafnian import f_loop, get_submatrices
from scipy.special import comb, factorial as fac
import numba


def calc_sample_prob(sample: Sample, state: GaussianState):
    det_pattern_twice = list(sample.det_pattern) * 2
    A, gamma = state.get_A(), state.get_gamma()
    haf = WalrusLoopHafnianCalculator(glynn=False)(A, gamma, det_pattern_twice)
    return haf.real / np.prod(fac(sample.det_pattern)) * _prefactor(state.displacement, state.cov)


def calc_total_N_prob(N: int, state: GaussianState):
    A, gamma = state.get_A(), state.get_gamma()
    AX_S, XD_S, D_S, _ = get_submatrices(np.ones(A.shape[0]//2), A, gamma, None)
    AX = AX_S.copy()
    return (_prefactor(state.displacement, state.cov) * f_loop(AX, AX_S, XD_S, D_S, 2 * N)[N]).real


#@numba.jit(nopython=True, cache=True)
def calc_p_single_pure_source_n(B: np.complex128, alpha: np.complex128, n):
    # P(n) = P(0) * (1/n!) * abs(sum_k_(n-k)%2==0 nchoosek gamma**k B**((n-k)/2) (n-k-1)!! )**2
    # P(n) = P(0) * n! * abs(sum_k_(n-k)%2==0 gamma**k B**((n-k)/2) / (n-k) / k! / ((n-k-2)/2)! 2^-((n-k-2)/2) )**2
    # P(0) = np.exp(-0.5 * alpha @ Qinv @ alpha*) / np.sqrt(np.linalg.det(Q))
    # Qinv = I - (XA)* = I - ([[0, B], [B*,0]]) = [[1, -B], [-B*, 1]]
    # det(Qinv) = 1 - abs(B)**2
    # Q = 1/(1 - abs(B)**2) [[1, B], [B*, 1]]
    # gamma = alpha* - B.alpha
    # [alpha, alpha*] @ Qinv = [alpha - B* alpha*, alpha* - B alpha] = [gamma* gamma]
    # [gamma* gamma] @ [alpha* alpha] = (gamma alpha)* + gamma alpha
    # P(0) = np.exp(-(gamma alpha).real) * sqrt(1 - abs(B)**2)
    gamma = np.conj(alpha) - B * alpha
    P0 = np.exp(-(gamma * alpha).real) * np.sqrt(1 - abs(B)**2)
    psi = gamma ** n
    for pairs in range(1, n//2 + 1):
        k = n - 2 * pairs
        psi += gamma ** k * B ** pairs * nchoosek(n, k) * double_factorial(2 * pairs - 1)
    return abs(psi)**2 * P0 / factorial(n)


@numba.jit(nopython=True, cache=True)
def factorial(n):
    f = 1
    for i in range(1, n+1):
        f *= i
    return f


@numba.jit(nopython=True, cache=True)
def nchoosek(n, k):
    f = 1
    for i in range(k + 1, n + 1):
        f *= i
    for i in range(1, n - k + 1):
        f /= i
    return f


@numba.jit(nopython=True, cache=True)
def double_factorial(n):
    f = 1
    for i in range(n, 0, -2):
        f *= i
    return f