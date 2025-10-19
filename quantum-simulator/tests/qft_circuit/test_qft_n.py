import numpy as np
import pytest
import random
from qiskit.circuit.library import QFTGate
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.quantum_info import Statevector

from src.circuits.Dense_Cartesian import DenseCartesianSim
from src.circuits.Dense_Polar import DensePolarSim
from src.circuits.Sparse_Dict import SparseDictSim
from src.circuits.Vectorized import VectorizedSim


# ==================== QFT Circuit ====================


def apply_our_qft(qc) -> None:
    """Apply QFT to our quantum circuit."""
    n = qc.n
    
    for i in reversed(range(n)):
        qc.h(i)
        for j in range(i):
            theta = -np.pi / (2**(j+1))
            qc.cp(theta, i-1-j, i)
    
    for i in range(n//2):
        qc.swap(i, n - 1 - i)


def apply_our_invqft(qc) -> None:
    """Apply inverse QFT to our quantum circuit."""
    n = qc.n
    
    for i in range(n//2):
        qc.swap(i, n - 1 - i)
    
    for i in range(n):
        qc.h(i)
        theta = np.pi / 2
        for j in range(i+1, n):
            qc.cp(theta, i, j)
            theta = theta / 2


def apply_qiskit_qft(qc: QiskitQuantumCircuit) -> None:
    """Apply QFT to Qiskit circuit."""
    qc.append(QFTGate(qc.num_qubits).inverse(), qc.qubits)


# ==================== Utilities ====================


def statevector_from_qiskit(circ: QiskitQuantumCircuit) -> np.ndarray:
    """Return a statevector as a NumPy array from a Qiskit circuit."""
    sv = Statevector.from_instruction(circ)
    return np.asarray(sv.data, dtype=np.complex128)


def compare_circuits(our_circuit, qiskit_circuit: QiskitQuantumCircuit):
    """Compare statevectors from our circuit and Qiskit circuit."""
    sv_qiskit = statevector_from_qiskit(qiskit_circuit)
    sv_ours = our_circuit.get_statevector()  # Use abstract method
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


def initialize_randomly(qs, qc: QiskitQuantumCircuit, seed: int):
    """Initialize both circuits with the same random state."""
    random.seed(seed)
    n = qs.n
    
    for i in range(n):
        qs.h(i)
        qc.h(i)
    
    for i in range(n):
        theta = random.random() * 360
        qs.p(theta=theta, target=i)
        qc.p(theta=theta, qubit=i)


# All simulator implementations to test
ALL_SIMULATORS = [
    pytest.param(DenseCartesianSim, id="DenseCartesian"),
    pytest.param(DensePolarSim, id="DensePolar"),
    pytest.param(SparseDictSim, id="SparseDict"),
    pytest.param(VectorizedSim, id="Vectorized"),
]

DIMENSIONS = [4, 6, 8]


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestQFT:
    """QFT tests for all simulator implementations."""
    
    def test_sanity(self, SimClass):
        """Basic sanity check."""
        assert 1 == 1
    
    @pytest.mark.parametrize("dim", DIMENSIONS)
    def test_invqft(self, SimClass, dim):
        """Test that QFT followed by inverse QFT returns to original state."""
        qs1 = SimClass(dim)
        qs2 = SimClass(dim)
        
        # Initialize both in some arbitrary state with constant frequency
        for i in range(dim):
            qs1.h(i)
            qs2.h(i)
            qs1.p(theta=np.pi/4 * (2**i), target=i)
            qs2.p(theta=np.pi/4 * (2**i), target=i)
        
        apply_our_qft(qs2)
        apply_our_invqft(qs2)
        
        sv1 = qs1.get_statevector()
        sv2 = qs2.get_statevector()
        np.testing.assert_allclose(sv1, sv2, rtol=1e-12, atol=1e-12)
    
    @pytest.mark.parametrize("dim", DIMENSIONS)
    def test_zero_state_qft(self, SimClass, dim):
        """Test QFT on zero state."""
        qc = QiskitQuantumCircuit(dim)
        apply_qiskit_qft(qc)
        
        qs = SimClass(dim)
        apply_our_qft(qs)
        
        compare_circuits(qs, qc)
    
    @pytest.mark.parametrize("dim", DIMENSIONS)
    def test_constant_frequency_qft(self, SimClass, dim):
        """Test QFT on constant frequency state."""
        qc = QiskitQuantumCircuit(dim)
        qs = SimClass(dim)
        
        for i in range(dim):
            qc.h(i)
            qs.h(i)
            qc.p(theta=np.pi/4 * (2**i), qubit=i)
            qs.p(theta=np.pi/4 * (2**i), target=i)
        
        apply_qiskit_qft(qc)
        apply_our_qft(qs)
        
        compare_circuits(qs, qc)
    
    @pytest.mark.parametrize("dim", DIMENSIONS)
    def test_random_state_qft(self, SimClass, dim):
        """Test QFT on random state."""
        qc = QiskitQuantumCircuit(dim)
        qs = SimClass(dim)
        
        SEED = 42
        initialize_randomly(qs, qc, SEED)
        
        apply_our_qft(qs)
        apply_qiskit_qft(qc)
        
        compare_circuits(qs, qc)
