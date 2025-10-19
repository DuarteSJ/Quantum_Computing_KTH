"""
Quantum Fourier Transform (QFT) benchmark.

This module defines the QFT circuit for benchmarking quantum simulators.
"""

import numpy as np
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit.library import QFTGate

from tests.benchmark.benchmark_framework import run_benchmark


def apply_qft(qc) -> None:
    """Apply QFT to custom quantum circuit."""
    n = qc.n
    for i in reversed(range(n)):
        qc.h(i)
        for j in range(i):
            theta = -np.pi / (2 ** (j + 1))
            qc.cp(theta, i - 1 - j, i)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)


def apply_qft_qiskit(qc: QiskitQuantumCircuit) -> None:
    """Apply QFT to Qiskit circuit."""
    qc.append(QFTGate(qc.num_qubits).inverse(), qc.qubits)


QFT_CIRCUIT = {
    "key": "qft",
    "name": "QFT",
    "custom": apply_qft,
    "qiskit": apply_qft_qiskit,
}


def main():
    """Run QFT benchmark."""
    run_benchmark([QFT_CIRCUIT], max_qubits=15)


if __name__ == "__main__":
    main()
