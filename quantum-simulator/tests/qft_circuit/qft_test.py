import numpy as np
import pytest
# from src.circuit import QuantumCircuit
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate

from qiskit_aer import Aer
from qiskit import transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import math

def sv_from_circ(circ: QiskitQuantumCircuit) -> np.ndarray:
    """Extract statevector from a quantum circuit."""
    sv = Statevector(circ)
    return np.asarray(sv, dtype=np.complex128)


class QubitSystem:
    def __init__(self, statevector: np.ndarray, label: str = "Qubit System"):
        self.label = label
        self.set_statevector(statevector)

    def set_statevector(self, statevector: np.ndarray):
        sv = np.asarray(statevector, dtype=np.complex128).flatten()
        if sv.ndim != 1:
            raise ValueError("Statevector must be 1D.")
        n_states = sv.size
        n_qubits = int(round(math.log2(n_states)))
        if 2**n_qubits != n_states:
            raise ValueError("Length must be a power of 2.")
        # Defensive normalization (harmless if already normalized)
        norm = np.linalg.norm(sv)
        if norm != 0 and not np.isclose(norm, 1.0):
            sv = sv / norm

        self.n_qubits = n_qubits
        self.n_states = n_states
        self.amps  = sv
        self.prob  = np.abs(sv)**2
        self.phase = np.angle(sv)

    def viz_circle(self, max_cols: int = 8, figsize_scale: float = 2.3, filename: str = "qft"):
        cols = max(1, min(max_cols, self.n_states))
        rows = int(math.ceil(self.n_states / cols))

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols*figsize_scale, rows*(figsize_scale+0.2))
        )
        axes = np.atleast_2d(axes)

        def bitstr(i: int, n: int) -> str:
            return format(i, f"0{n}b")

        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            ax.set_aspect("equal")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")

            if idx >= self.n_states:
                ax.set_visible(False)
                continue

            # Outer reference circle
            ax.add_patch(plt.Circle((0.5, 0.5), 0.48, fill=False, lw=1.0, alpha=0.5))

            # Filled disk: radius ∝ sqrt(probability) so area ∝ probability
            radius = 0.48 * np.sqrt(self.prob[idx])
            ax.add_patch(plt.Circle((0.5, 0.5), radius, alpha=0.25))

            # Phase arrow
            angle = self.phase[idx]
            L = 0.45
            x2 = 0.5 + L * np.cos(angle)
            y2 = 0.5 + L * np.sin(angle)
            ax.arrow(0.5, 0.5, x2 - 0.5, y2 - 0.5,
                     head_width=0.03, head_length=0.05, length_includes_head=True)

            ax.set_title(f"|{bitstr(idx, self.n_qubits)}⟩", fontsize=10)

        fig.suptitle(self.label, fontsize=12)
        plt.tight_layout()
        plt.savefig(filename + ".png")

# SHOTS = 1000
# sim = Aer.get_backend("qasm_simulator")

qc = QiskitQuantumCircuit(4)
for i in range(4):
    qc.h(i)
    qc.p(theta=np.pi/4 * (2**i), qubit=i)

qc.p(np.pi, 1)

sv = sv_from_circ(qc)
qs = QubitSystem(sv)
qs.viz_circle(filename="before")

# Default implementation
# qc.append(QFTGate(num_qubits=4).inverse(), qc.qubits)

def apply_our_qft(qc: QiskitQuantumCircuit) -> None:
    n = qc.num_qubits
    
    for i in reversed(range(n)):
        qc.h(i)
        for j in range(i):
            theta = -np.pi / (2**(j+1))
            qc.cp(theta, i-1-j, i)

    for i in range(n//2):
        qc.swap(i, n - 1 - i)

def apply_our_invqft(qc: QiskitQuantumCircuit) -> None:
    n = qc.num_qubits

    for i in range(n//2):
        qc.swap(i, n - 1 - i)
    
    for i in range(n):
        qc.h(i)
        theta = np.pi / 2
        for j in range(i+1,n):
            qc.cp(theta, i, j)
            theta = theta / 2

apply_our_qft(qc)
apply_our_invqft(qc)

# # invQFT implementation
# qc.h(3)
# qc.cp(-np.pi/2, 2, 3)
# qc.cp(-np.pi/4, 1, 3)
# qc.cp(-np.pi/8, 0, 3)
# qc.h(2)
# qc.cp(-np.pi/2, 1, 2)
# qc.cp(-np.pi/4, 0, 2)
# qc.h(1)
# qc.cp(-np.pi/2, 0, 1)
# qc.h(0)
# # Swap qubits to reverse order
# qc.swap(0, 3)
# qc.swap(1, 2)


print(qc)
sv = sv_from_circ(qc)
qs = QubitSystem(sv)
qs.viz_circle(filename="qft")

# print(qc)

# qc.measure_all()
# tqc = transpile(qc, sim)
# counts = sim.run(tqc, shots=SHOTS).result().get_counts()
# print(counts)