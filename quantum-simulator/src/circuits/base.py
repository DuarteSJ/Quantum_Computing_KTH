"""Abstract base class for quantum simulators."""

from abc import ABC, abstractmethod
import numpy as np
from src.gates import QuantumGates

class QuantumSimulator(ABC):
    """Abstract base for quantum simulators."""

    @abstractmethod
    def __init__(self, n: int):
        """Initialize n-qubit system."""
        pass

    @abstractmethod
    def apply_single_qubit(self, gate: np.ndarray, target: int):
        """Apply single-qubit gate."""
        pass

    @abstractmethod
    def apply_two_qubit(self, gate: np.ndarray, t1: int, t2: int):
        """Apply two-qubit gate."""
        pass

    @abstractmethod
    def get_statevector(self) -> np.ndarray:
        """Return full statevector."""
        pass

    @abstractmethod
    def measure_all(self) -> int:
        """Measure all qubits, return basis state index."""
        pass

    # Convenience wrappers for all standard gates from QuantumGates
    def x(self, target: int) -> None:
        self.apply_single_qubit(QuantumGates.x(), target)

    def y(self, target: int) -> None:
        self.apply_single_qubit(QuantumGates.y(), target)

    def z(self, target: int) -> None:
        self.apply_single_qubit(QuantumGates.p(np.pi), target)

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

    def apply_measure(self):
        # TODO: Implement measurement
        ...


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
        import math
        import matplotlib.pyplot as plt

        state = self.get_statevector()  # Works for all variants!
        n_states = len(state)
        prob = np.abs(state) ** 2
        phase = np.angle(state)

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
