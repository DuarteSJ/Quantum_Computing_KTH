"""Dense Polar quantum simulator implementation."""

import numpy as np
from src.circuits.base import QuantumSimulator

class DensePolarSim(QuantumSimulator):
    """Dense arrays storing magnitude and phase separately."""
    
    def __init__(self, n: int):
        self.n = n
        self.magnitude = np.zeros(2**n)
        self.phase = np.zeros(2**n)
        self.magnitude[0] = 1.0
    
    def _to_complex(self) -> np.ndarray:
        """Convert to complex representation."""
        return self.magnitude * np.exp(1j * self.phase)
    
    def _from_complex(self, state: np.ndarray):
        """Update from complex representation."""
        self.magnitude = np.abs(state)
        self.phase = np.angle(state)
    
    def apply_single_qubit(self, gate: np.ndarray, target: int):
        state = self._to_complex()
        new_state = np.zeros_like(state)
        for i in range(len(state)):
            bit = (i >> target) & 1
            i0 = i & ~(1 << target)
            i1 = i | (1 << target)
            new_state[i] += (
                gate[bit, 0] * state[i0] + 
                gate[bit, 1] * state[i1]
            )
        self._from_complex(new_state)
    
    def apply_two_qubit(self, gate: np.ndarray, t1: int, t2: int):
        state = self._to_complex()
        new_state = np.zeros_like(state)
        for i in range(len(state)):
            b0 = (i >> t1) & 1
            b1 = (i >> t2) & 1
            i00 = i & ~(1 << t1) & ~(1 << t2)
            i01 = i00 | (1 << t2)
            i10 = i00 | (1 << t1)
            i11 = i00 | (1 << t1) | (1 << t2)
            idx = b0 * 2 + b1
            new_state[i] += (
                gate[idx, 0] * state[i00] +
                gate[idx, 1] * state[i01] +
                gate[idx, 2] * state[i10] +
                gate[idx, 3] * state[i11]
            )
        self._from_complex(new_state)
    
    def get_statevector(self) -> np.ndarray:
        return self._to_complex()
    
    def measure_all(self) -> int:
        probs = self.magnitude**2
        return np.random.choice(len(self.magnitude), p=probs)
