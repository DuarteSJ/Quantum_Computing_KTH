"""Sparse Dictionary quantum simulator implementation."""

import numpy as np
from typing import Dict
from collections import defaultdict
from src.circuits.base import QuantumSimulator

class SparseDictSim(QuantumSimulator):
    """Sparse representation: only store non-zero amplitudes."""
    
    def __init__(self, n: int, threshold: float = 1e-10):
        self.n = n
        self.state: Dict[int, complex] = {0: 1.0+0j}
        self.threshold = threshold
    
    def _prune(self):
        """Remove negligible amplitudes."""
        to_remove = [k for k, v in self.state.items() 
                     if abs(v) < self.threshold]
        for k in to_remove:
            del self.state[k]
    
    def apply_single_qubit(self, gate: np.ndarray, target: int):
        new_state = defaultdict(complex)
        
        # Need to consider ALL indices that could be affected
        all_indices = set(self.state.keys())
        for idx in self.state.keys():
            # Add the related index (bit flip on target qubit)
            i0 = idx & ~(1 << target)
            i1 = idx | (1 << target)
            all_indices.add(i0)
            all_indices.add(i1)
        
        # Now compute new amplitudes for all relevant indices
        for i in all_indices:
            bit = (i >> target) & 1
            i0 = i & ~(1 << target)
            i1 = i | (1 << target)
            new_state[i] += gate[bit, 0] * self.state.get(i0, 0)
            new_state[i] += gate[bit, 1] * self.state.get(i1, 0)
        
        self.state = dict(new_state)
        self._prune()
        
    def apply_two_qubit(self, gate: np.ndarray, t1: int, t2: int):
        new_state = defaultdict(complex)
        # Need to iterate over all possible indices that could contribute
        all_indices = set(self.state.keys())
        for idx in self.state.keys():
            # Generate related indices
            for b1 in [0, 1]:
                for b2 in [0, 1]:
                    related = idx & ~(1 << t1) & ~(1 << t2)
                    if b1: related |= (1 << t1)
                    if b2: related |= (1 << t2)
                    all_indices.add(related)
        
        for i in all_indices:
            b0 = (i >> t1) & 1
            b1 = (i >> t2) & 1
            i00 = i & ~(1 << t1) & ~(1 << t2)
            i01 = i00 | (1 << t2)
            i10 = i00 | (1 << t1)
            i11 = i00 | (1 << t1) | (1 << t2)
            idx = b0 * 2 + b1
            new_state[i] += (
                gate[idx, 0] * self.state.get(i00, 0) +
                gate[idx, 1] * self.state.get(i01, 0) +
                gate[idx, 2] * self.state.get(i10, 0) +
                gate[idx, 3] * self.state.get(i11, 0)
            )
        self.state = dict(new_state)
        self._prune()
    
    def get_statevector(self) -> np.ndarray:
        vec = np.zeros(2**self.n, dtype=complex)
        for idx, amp in self.state.items():
            vec[idx] = amp
        return vec
    
    def measure_all(self) -> int:
        indices = list(self.state.keys())
        probs = np.array([abs(self.state[i])**2 for i in indices])
        probs /= probs.sum()
        return np.random.choice(indices, p=probs)
