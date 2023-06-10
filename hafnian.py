from thewalrus._hafnian import loop_hafnian as walrus_loop_hafnian
from thewalrus.loop_hafnian_batch import loop_hafnian_batch
import numpy as np
import numba
from numba import complex128, int32


spec = [
    ('A', complex128[:,:]),               # a simple scalar field
    ('D', complex128[:]),
    ('reps', int32[:]),
    ('matched_edges', int32[:, :]),
    ('odd_node', int32)
]
@numba.experimental.jitclass(spec)
class LoopHafnianCalculator:
    """
    Functor for calculating loop hafnians
    """
    def __init__(self):
        self.A = None
        self.D = None
        self.reps = None
        self.matched_edges = None
        self.odd_node = None

    def __call__(self, A, D=None, reps=None):
        if D is None:
            D = np.zeros(A.shape[0], dtype=np.complex128)
        if reps is None:
            reps = np.ones(A.shape[0], dtype=int)
        else:
            reps = np.array(reps, dtype=int)
        if sum(reps) == 0:
            return np.complex128(1)
        self.A, self.D, self.reps = A, D, reps
        self.get_edge_matching()
        steps = 1
        for matched_edge in self.matched_edges:
            steps *= matched_edge[2] + 1
        H = 0
        for i in numba.prange(steps):
            H += 1
        return H

    def get_edge_matching(self):
        reps = self.reps.copy()
        nodes = np.arange(len(reps))
        sort_order = np.argsort(reps)[::-1]
        nodes, reps = nodes[sort_order], reps[sort_order]
        self.matched_edges = []
        while reps[0] > 1 or (reps[0] > 0 and reps[1] > 0):
            if len(reps) == 1 or reps[0] >= 2 * reps[1]:
                self.matched_edges.append([nodes[0], nodes[0], reps[0] // 2])
                reps[0] -= reps[0] // 2
            else:
                self.matched_edges.append([nodes[0], nodes[1], reps[1]])
                reps[0] -= reps[1]
                reps[1] = 0
            sort_order = np.argsort(reps)[::-1]
            nodes, reps = nodes[sort_order], reps[sort_order]
        if reps[0] == 1:
            self.odd_node = nodes[0]
        else:
            self.odd_node = -1


@numba.jit(nopython=True, cache=True)
def loop_hafnian(A, D=None, reps=None):
    if D is None:
        D = np.zeros(A.shape[0], dtype=np.complex128)
    if reps is None:
        reps = np.ones(A.shape[0])
    if sum(reps) == 0:
        return np.complex128(1)

    def get_edge_matching(reps):
        nodes = np.arange(len(reps))
        sort_order = np.argsort(reps)[::-1]
        nodes, reps = nodes[sort_order], reps[sort_order]
        matched_edges = []
        while reps[0] > 1 or (reps[0] > 0 and reps[1] > 0):
            if len(reps) == 1 or reps[0] >= 2 * reps[1]:
                matched_edges.append([nodes[0], nodes[0], reps[0] // 2])
                reps[0] -= reps[0] // 2
            else:
                matched_edges.append([nodes[0], nodes[1], reps[1]])
                reps[0] -= reps[1]
                reps[1] = 0
            sort_order = np.argsort(reps)[::-1]
            nodes, reps = nodes[sort_order], reps[sort_order]
        odd_node = nodes[0] if reps[0] == 1 else -1
        return matched_edges, odd_node

    def get_step_edge_reps(edge_reps, step):
        step_edge_reps = []
        for i in edge_reps:
            step_edge_reps.append(step % (edge_reps[i]+1))
            step //= edge_reps[i]+1
        return np.array(step_edge_reps[::-1], dtype=edge_reps.dtype)


    matched_edges, odd_node = get_edge_matching(reps)
    steps = 1
    for matched_edge in matched_edges:
        steps *= matched_edge[2] + 1
    H = 0
    for i in numba.prange(int(steps)):
        step_edge_reps = get_step_edge_reps(edge_reps, step)
    return H



class WalrusLoopHafnianCalculator:
    """
    Functor for calculating loop hafnians, using thewalrus
    """
    def __init__(self, glynn=True, use_diagonal_as_loops=False):
        self.glynn = glynn
        self.use_diagonal_as_loops = use_diagonal_as_loops

    def __call__(self, A, D=None, reps=None):
        if D is None:
            if self.use_diagonal_as_loops is False:
                D = np.zeros(A.shape[0], dtype=np.complex128)
            else:
                D = np.diag(A)
        return walrus_loop_hafnian(A, D, reps, glynn=self.glynn)


if __name__ == "__main__":
    A = np.ones((6, 6), dtype=np.complex128)
    #print(LoopHafnianCalculator()(A))
    print(loop_hafnian(A))