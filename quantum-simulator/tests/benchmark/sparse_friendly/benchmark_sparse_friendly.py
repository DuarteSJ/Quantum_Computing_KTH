"""
Sparse-Friendly Circuit benchmark.

Creates a sparse state vector with many zero amplitudes, which should
benefit sparse representations that don't store zeros.
"""

from qiskit import QuantumCircuit as QiskitQuantumCircuit
from tests.benchmark.benchmark_framework import run_benchmark
import numpy as np


def apply_sparse_friendly(qc) -> None:
    """
    Apply gates only to first 3 qubits.

    For n qubits, this creates a state with only 8 non-zero amplitudes
    out of 2^n total amplitudes. The sparse simulator should use
    O(1) memory instead of O(2^n).
    """
    n = qc.n

    # Only manipulate first 3 qubits (or fewer if n < 3)
    active_qubits = min(3, n)

    # Create superposition on active qubits
    for i in range(active_qubits):
        qc.h(i)

    # Add some entanglement between active qubits
    for i in range(active_qubits - 1):
        qc.cx(i, i + 1)

    # Add phase rotations between different qubits (skip i=0 to avoid cp(0,0))
    if active_qubits > 1:
        for i in range(1, active_qubits):
            theta = np.pi / (2 ** (i + 1))
            qc.cp(theta, 0, i)

    # All other qubits (n-3 to n-1) remain in |0⟩ state
    # This means only 2^3 = 8 non-zero amplitudes in a 2^n dimensional space


def apply_sparse_friendly_qiskit(qc: QiskitQuantumCircuit) -> None:
    """Apply sparse-friendly circuit to Qiskit."""
    n = qc.num_qubits

    active_qubits = min(3, n)

    for i in range(active_qubits):
        qc.h(i)

    for i in range(active_qubits - 1):
        qc.cx(i, i + 1)

    # Add phase rotations between different qubits (skip i=0 to avoid cp(0,0))
    if active_qubits > 1:
        for i in range(1, active_qubits):
            theta = np.pi / (2 ** (i + 1))
            qc.cp(theta, 0, i)


SPARSE_FRIENDLY_CIRCUIT = {
    "key": "sparse_friendly",
    "name": "SFC",  # for Sparse-Friendly Circuit
    "custom": apply_sparse_friendly,
    "qiskit": apply_sparse_friendly_qiskit,
}


def main():
    """Run Sparse-Friendly benchmark."""
    run_benchmark([SPARSE_FRIENDLY_CIRCUIT], max_qubits=15)


if __name__ == "__main__":
    main()
