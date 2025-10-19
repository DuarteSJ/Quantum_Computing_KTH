"""Dense Cartesian quantum simulator implementation."""

import numpy as np
from src.circuits.base import QuantumSimulator


class DenseCartesianSim(QuantumSimulator):
    """Dense array with complex amplitudes (a + bi)."""

    def __init__(self, n: int):
        self.n = n
        self.state = np.zeros(2**n, dtype=complex)
        self.state[0] = 1.0

    def apply_single_qubit(self, gate: np.ndarray, target: int):
        new_state = np.zeros_like(self.state)
        for i in range(len(self.state)):
            bit = (i >> target) & 1
            i0 = i & ~(1 << target)
            i1 = i | (1 << target)
            new_state[i] += (
                gate[bit, 0] * self.state[i0] + gate[bit, 1] * self.state[i1]
            )
        self.state = new_state

    def apply_two_qubit(self, gate: np.ndarray, t1: int, t2: int):
        new_state = np.zeros_like(self.state)
        for i in range(len(self.state)):
            b0 = (i >> t1) & 1
            b1 = (i >> t2) & 1
            i00 = i & ~(1 << t1) & ~(1 << t2)
            i01 = i00 | (1 << t2)
            i10 = i00 | (1 << t1)
            i11 = i00 | (1 << t1) | (1 << t2)
            idx = b0 * 2 + b1
            new_state[i] += (
                gate[idx, 0] * self.state[i00]
                + gate[idx, 1] * self.state[i01]
                + gate[idx, 2] * self.state[i10]
                + gate[idx, 3] * self.state[i11]
            )
        self.state = new_state

    def get_statevector(self) -> np.ndarray:
        return self.state.copy()

    def measure_all(self) -> int:
        probs = np.abs(self.state) ** 2
        return np.random.choice(len(self.state), p=probs)
