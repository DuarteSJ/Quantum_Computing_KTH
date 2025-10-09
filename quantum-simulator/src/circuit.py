"""Module for simulating n-qubit quantum states and applying quantum gates."""

import numpy as np
from typing import List, Optional, Tuple
from src.gates import QuantumGates
import matplotlib.pyplot as plt
import math


class QuantumCircuit:
    """Represents an n-qubit quantum statevector and allows gate application."""

    state: np.ndarray
    n: int

    def __init__(self, n: int) -> None:
        """Initialize an n-qubit quantum state in |0...0>."""
        self.n = n
        self.state = np.zeros(2**n, dtype=complex)
        self.state[0] = 1.0  # start in |0...0>

    def apply_single_qubit(self, gate_matrix: np.ndarray, target: int) -> None:
        """
        Apply a single-qubit gate to a target qubit.

        Args:
            gate_matrix (np.ndarray): 2x2 unitary matrix
            target (int): qubit index (0-indexed)
        """
        new_state = np.zeros_like(self.state)
        for i in range(len(self.state)):
            bit = (i >> target) & 1
            i0 = i & ~(1 << target)
            i1 = i | (1 << target)
            new_state[i] += (
                gate_matrix[bit, 0] * self.state[i0]
                + gate_matrix[bit, 1] * self.state[i1]
            )
        self.state = new_state

    def apply_two_qubit(
        self, gate_matrix: np.ndarray, target1: int, target2: int
    ) -> None:
        """
        Apply a two-qubit gate to two target qubits.

        Args:
            gate_matrix (np.ndarray): 4x4 unitary matrix
            targets (List[int]): two qubit indices [q0, q1]
        """
        new_state = np.zeros_like(self.state)
        for i in range(len(self.state)):
            b0 = (i >> target1) & 1
            b1 = (i >> target2) & 1
            i00 = i & ~(1 << target1) & ~(1 << target2)
            i01 = i00 | (1 << target2)
            i10 = i00 | (1 << target1)
            i11 = i00 | (1 << target1) | (1 << target2)
            idx = b0 * 2 + b1
            new_state[i] += (
                gate_matrix[idx, 0] * self.state[i00]
                + gate_matrix[idx, 1] * self.state[i01]
                + gate_matrix[idx, 2] * self.state[i10]
                + gate_matrix[idx, 3] * self.state[i11]
            )
        self.state = new_state

    def apply_measure(self):
        # TODO: Implement measurement
        ...

    # Convenience wrappers for all standard gates from QuantumGates
    def x(self, target: int) -> None:
        self.apply_single_qubit(QuantumGates.x(), target)

    def y(self, target: int) -> None:
        self.apply_single_qubit(QuantumGates.y(), target)

    def z(self, target: int) -> None:
        self.apply_single_qubit(QuantumGates.z(), target)

    def h(self, target: int) -> None:
        self.apply_single_qubit(QuantumGates.h(), target)

    def p(self, theta: float, target: int) -> None:
        self.apply_single_qubit(QuantumGates.p(theta), target)

    def s(self, target: int) -> None:
        self.apply_single_qubit(QuantumGates.p(np.pi / 2), target)

    def t(self, target: int) -> None:
        self.apply_single_qubit(QuantumGates.p(np.pi / 4), target)

    def rx(self, theta: float, target: int) -> None:
        self.apply_single_qubit(QuantumGates.rx(theta), target)

    def ry(self, theta: float, target: int) -> None:
        self.apply_single_qubit(QuantumGates.ry(theta), target)

    def rz(self, theta: float, target: int) -> None:
        self.apply_single_qubit(QuantumGates.rz(theta), target)

    def cx(self, control: int, target: int) -> None:
        self.apply_two_qubit(QuantumGates.cx(), control, target)

    def cp(self, theta: float, control: int, target: int) -> None:
        self.apply_two_qubit(QuantumGates.cp(theta), control, target)

    def cz(self, control: int, target: int) -> None:
        self.apply_two_qubit(QuantumGates.cp(np.pi), control, target)

    def swap(self, target1: int, target2: int) -> None:
        self.apply_two_qubit(QuantumGates.swap(), target1, target2)

    # Visualization method
    def viz_circle(
        self,
        max_cols: int = 8,
        figsize_scale: float = 2.3,
        label: str = "Quantum Circuit",
    ):
        """
        Visualize the quantum state in circle notation.

        Args:
            max_cols (int): Maximum number of columns in the grid
            figsize_scale (float): Scaling factor for figure size
            label (str): Title of the figure
        """

        # NOTE: This was adapted from the teaching material
        n_states = len(self.state)
        prob = np.abs(self.state) ** 2
        phase = np.angle(self.state)

        cols = max(1, min(max_cols, n_states))
        rows = int(math.ceil(n_states / cols))

        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * figsize_scale, rows * (figsize_scale + 0.2))
        )
        axes = np.atleast_2d(axes)

        def bitstr(i: int, n: int) -> str:
            return format(i, f"0{n}b")

        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            ax.set_aspect("equal")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            if idx >= n_states:
                ax.set_visible(False)
                continue

            # Outer reference circle
            ax.add_patch(plt.Circle((0.5, 0.5), 0.48, fill=False, lw=1.0, alpha=0.5))

            # Filled disk: radius ∝ sqrt(probability)
            radius = 0.48 * np.sqrt(prob[idx])
            ax.add_patch(plt.Circle((0.5, 0.5), radius, alpha=0.25, color="blue"))

            # Phase arrow
            angle = phase[idx]
            L = 0.45
            x2 = 0.5 + L * np.cos(angle)
            y2 = 0.5 + L * np.sin(angle)
            ax.arrow(
                0.5,
                0.5,
                x2 - 0.5,
                y2 - 0.5,
                head_width=0.03,
                head_length=0.05,
                length_includes_head=True,
                color="red",
            )

            ax.set_title(f"|{bitstr(idx, self.n)}⟩", fontsize=10)

        fig.suptitle(label, fontsize=12)
        plt.tight_layout()
        plt.show()
