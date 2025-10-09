"""Quantum gate definitions and operations."""

import numpy as np


class QuantumGates:
    """Collection of quantum gates and their matrix representations."""

    @staticmethod
    def x() -> np.ndarray:
        """Pauli-X (NOT) gate."""
        return np.array([[0, 1], [1, 0]], dtype=complex)

    @staticmethod
    def y() -> np.ndarray:
        """Pauli-Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)

    @staticmethod
    def z() -> np.ndarray:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=complex)

    @staticmethod
    def h() -> np.ndarray:
        """Hadamard gate."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    @staticmethod
    def p(phi: float) -> np.ndarray:
        """Applies a phase shift of e^{iφ} to the |1⟩ component."""
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

    @staticmethod
    def rx(theta: float) -> np.ndarray:
        """Rotation around the X-axis."""
        return np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=complex,
        )

    @staticmethod
    def ry(theta: float) -> np.ndarray:
        """Rotation around the Y-axis."""
        return np.array(
            [
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=complex,
        )

    @staticmethod
    def rz(theta: float) -> np.ndarray:
        """Rotation around the Z-axis."""
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
        )

    @staticmethod
    def cx() -> np.ndarray:
        """CNOT gate (control = qubit 0, target = qubit 1 in 2-qubit space)."""
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )

    @staticmethod
    def cp(theta: float) -> np.ndarray:
        """Controlled-Phase gate (applies phase e^{i theta} to |11> state)."""
        return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)

    @staticmethod
    def swap() -> np.ndarray:
        """SWAP gate."""
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )

    @staticmethod
    def measure(state: np.ndarray) -> int:
        # TODO: Implement ts
        ...
