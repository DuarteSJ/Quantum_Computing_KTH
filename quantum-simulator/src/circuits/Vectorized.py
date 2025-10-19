"""Vectorized quantum simulator implementation."""

import numpy as np
from src.circuits.base import QuantumSimulator


class VectorizedSim(QuantumSimulator):
    """Use numpy's vectorized operations for speed."""

    def __init__(self, n: int):
        self.n = n
        self.state = np.zeros(2**n, dtype=complex)
        self.state[0] = 1.0

    def apply_single_qubit(self, gate: np.ndarray, target: int):
        """Vectorized single-qubit gate using reshape."""
        d = 2**self.n
        # Reshape: axis i corresponds to qubit (n-1-i)
        shape_before = [2] * self.n
        state_tensor = self.state.reshape(shape_before)

        # Target qubit corresponds to axis (n-1-target)
        axis = self.n - 1 - target

        # Move that axis to position 0
        axes = list(range(self.n))
        axes[0], axes[axis] = axes[axis], axes[0]
        state_tensor = np.transpose(state_tensor, axes)

        # Apply gate
        original_shape = state_tensor.shape
        state_matrix = state_tensor.reshape(2, -1)
        state_matrix = gate @ state_matrix
        state_tensor = state_matrix.reshape(original_shape)

        # Transpose back
        state_tensor = np.transpose(state_tensor, axes)
        self.state = state_tensor.reshape(d)

    def apply_two_qubit(self, gate: np.ndarray, t1: int, t2: int):
        """Vectorized two-qubit gate."""
        d = 2**self.n
        shape_before = [2] * self.n
        state_tensor = self.state.reshape(shape_before)

        # Convert qubit indices to axis indices
        axis1 = self.n - 1 - t1
        axis2 = self.n - 1 - t2

        # Move both target axes to front (handle the order carefully)
        axes = list(range(self.n))

        # First move axis1 to position 0
        axes.remove(axis1)
        axes.insert(0, axis1)

        # Then move axis2 to position 1 (accounting for the shift)
        axes.remove(axis2)
        axes.insert(1, axis2)

        state_tensor = np.transpose(state_tensor, axes)

        # Apply gate
        original_shape = state_tensor.shape
        state_matrix = state_tensor.reshape(4, -1)
        state_matrix = gate @ state_matrix
        state_tensor = state_matrix.reshape(original_shape)

        # Transpose back (inverse permutation)
        inv_axes = [0] * self.n
        for i, ax in enumerate(axes):
            inv_axes[ax] = i
        state_tensor = np.transpose(state_tensor, inv_axes)
        self.state = state_tensor.reshape(d)

    def get_statevector(self) -> np.ndarray:
        return self.state.copy()

    def measure_all(self) -> int:
        probs = np.abs(self.state) ** 2
        return np.random.choice(len(self.state), p=probs)
