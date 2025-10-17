import numpy as np
import pytest
import random
from src.circuit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.quantum_info import Statevector

# ==================== QFT Circuit ====================

def apply_our_qft(qc: QuantumCircuit) -> None:
    n = qc.n
    
    for i in reversed(range(n)):
        qc.h(i)
        for j in range(i):
            theta = -np.pi / (2**(j+1))
            qc.cp(theta, i-1-j, i)

    for i in range(n//2):
        qc.swap(i, n - 1 - i)


def apply_our_invqft(qc: QuantumCircuit) -> None:
    n = qc.n

    for i in range(n//2):
        qc.swap(i, n - 1 - i)
    
    for i in range(n):
        qc.h(i)
        theta = np.pi / 2
        for j in range(i+1,n):
            qc.cp(theta, i, j)
            theta = theta / 2


def apply_qiskit_qft(qc: QiskitQuantumCircuit) -> None:
    qc.append(QFTGate(qc.num_qubits).inverse(), qc.qubits)


# ==================== Utilities ====================


def statevector_from_qiskit(circ: QiskitQuantumCircuit) -> np.ndarray:
    """Return a statevector as a NumPy array from a Qiskit circuit."""
    sv = Statevector.from_instruction(circ)
    return np.asarray(sv.data, dtype=np.complex128)


def compare_circuits(our_circuit: QuantumCircuit, qiskit_circuit: QiskitQuantumCircuit):
    """Compare statevectors from our circuit and Qiskit circuit."""
    sv_qiskit = statevector_from_qiskit(qiskit_circuit)
    sv_ours = our_circuit.state
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


def initialize_randomly(qs: QuantumCircuit, qc: QiskitQuantumCircuit, seed: int):
    random.seed(seed)
    n = qs.n

    for i in range(n):
        qs.h(i)
        qc.h(i)

    for i in range(n):
        theta = random.random() * 360
        qs.p(theta=theta, target=i)
        qc.p(theta=theta, qubit=i)
    return


DIMENSIONS = [4, 6, 8]

class TestQFT:

    def test_sanity(self):
        assert 1 == 1

    @pytest.mark.parametrize("dim", DIMENSIONS)
    def test_invqft(self, dim):
        qs1 = QuantumCircuit(dim)
        qs2 = QuantumCircuit(dim)

        # Initialize both in some arbitrary state with constant frequency
        for i in range(dim):
            qs1.h(i)
            qs2.h(i)
            qs1.p(theta=np.pi/4 * (2**i), target=i)
            qs2.p(theta=np.pi/4 * (2**i), target=i)

        apply_our_qft(qs2)
        apply_our_invqft(qs2)

        np.testing.assert_allclose(qs1.state, qs2.state, rtol=1e-12, atol=1e-12)


    @pytest.mark.parametrize("dim", DIMENSIONS)
    def test_zero_state_qft(self, dim):
        qc = QiskitQuantumCircuit(dim)
        apply_qiskit_qft(qc)

        qs = QuantumCircuit(dim)
        apply_our_qft(qs)

        compare_circuits(qs, qc)


    @pytest.mark.parametrize("dim", DIMENSIONS)
    def test_constant_frequency_qft(self, dim):
        qc = QiskitQuantumCircuit(dim)
        qs = QuantumCircuit(dim)

        for i in range(dim):
            qc.h(i)
            qs.h(i)
            qc.p(theta=np.pi/4 * (2**i), qubit=i)
            qs.p(theta=np.pi/4 * (2**i), target=i)

        apply_qiskit_qft(qc)
        apply_our_qft(qs)

        compare_circuits(qs, qc)


    @pytest.mark.parametrize("dim", DIMENSIONS)
    def test_random_state_qft(self, dim):
        qc = QiskitQuantumCircuit(dim)
        qs = QuantumCircuit(dim)

        SEED = 42
        initialize_randomly(qs, qc, SEED)
        apply_our_qft(qs)
        apply_qiskit_qft(qc)

        compare_circuits(qs, qc)