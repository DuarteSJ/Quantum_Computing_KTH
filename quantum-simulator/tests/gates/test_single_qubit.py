"""
Single-qubit gate tests for our quantum circuit simulator.

Tests verify correctness by comparing against Qiskit's implementations.
All tests use statevector comparison with high precision (rtol=1e-12, atol=1e-12).

Tests run against ALL simulator implementations automatically.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.quantum_info import Statevector

from src.circuits.Dense_Cartesian import DenseCartesianSim
from src.circuits.Dense_Polar import DensePolarSim
from src.circuits.Sparse_Dict import SparseDictSim
from src.circuits.Vectorized import VectorizedSim


# ==================== Utilities ====================


def statevector_from_qiskit(circ: QiskitQuantumCircuit) -> np.ndarray:
    """Return a statevector as a NumPy array from a Qiskit circuit."""
    sv = Statevector.from_instruction(circ)
    return np.asarray(sv.data, dtype=np.complex128)


def compare_circuits(our_circuit, qiskit_circuit: QiskitQuantumCircuit):
    """Compare statevectors from our circuit and Qiskit circuit."""
    sv_qiskit = statevector_from_qiskit(qiskit_circuit)
    sv_ours = our_circuit.get_statevector()
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


# Common test angles covering some edge cases and typical values
ROTATION_ANGLES = [
    0,
    np.pi / 8,
    np.pi / 4,
    np.pi / 2,
    np.pi,
    3 * np.pi / 2,
    2 * np.pi - np.pi / 4,
    -np.pi,
    -np.pi / 8,
    -2 * np.pi,
]


# All simulator implementations to test
ALL_SIMULATORS = [
    pytest.param(DenseCartesianSim, id="DenseCartesian"),
    pytest.param(DensePolarSim, id="DensePolar"),
    pytest.param(SparseDictSim, id="SparseDict"),
    pytest.param(VectorizedSim, id="Vectorized"),
]


# ==================== Hadamard (H) Gate ====================


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestHadamard:
    """Hadamard gate tests."""

    def test_single_qubit(self, SimClass):
        """Test H gate on a single qubit."""
        qc = QiskitQuantumCircuit(1)
        qc.h(0)

        qs = SimClass(1)
        qs.h(0)

        compare_circuits(qs, qc)

    def test_two_qubits(self, SimClass):
        """Test H gates on two qubits."""
        qc = QiskitQuantumCircuit(2)
        qc.h(0)
        qc.h(1)

        qs = SimClass(2)
        qs.h(0)
        qs.h(1)

        compare_circuits(qs, qc)

    def test_three_qubits_sparse(self, SimClass):
        """Test H gates on non-adjacent qubits."""
        qc = QiskitQuantumCircuit(3)
        qc.h(0)
        qc.h(2)

        qs = SimClass(3)
        qs.h(0)
        qs.h(2)

        compare_circuits(qs, qc)


# ==================== Pauli Gates ====================


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestPauliX:
    """Pauli-X (NOT) gate tests."""

    def test_single_qubit(self, SimClass):
        """Test X gate on a single qubit."""
        qc = QiskitQuantumCircuit(1)
        qc.x(0)

        qs = SimClass(1)
        qs.x(0)

        compare_circuits(qs, qc)

    def test_two_qubits_second_qubit(self, SimClass):
        """Test X gate on the second qubit of a two-qubit system."""
        qc = QiskitQuantumCircuit(2)
        qc.x(1)

        qs = SimClass(2)
        qs.x(1)

        compare_circuits(qs, qc)

    def test_three_qubits_sparse(self, SimClass):
        """Test X gates on non-adjacent qubits."""
        qc = QiskitQuantumCircuit(3)
        qc.x(0)
        qc.x(2)

        qs = SimClass(3)
        qs.x(0)
        qs.x(2)

        compare_circuits(qs, qc)


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestPauliY:
    """Pauli-Y gate tests."""

    def test_single_qubit(self, SimClass):
        """Test Y gate on a single qubit."""
        qc = QiskitQuantumCircuit(1)
        qc.y(0)

        qs = SimClass(1)
        qs.y(0)

        compare_circuits(qs, qc)


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestPauliZ:
    """Pauli-Z gate tests."""

    def test_single_qubit(self, SimClass):
        """Test Z gate on a single qubit."""
        qc = QiskitQuantumCircuit(1)
        qc.z(0)

        qs = SimClass(1)
        qs.z(0)

        compare_circuits(qs, qc)

    def test_with_superposition(self, SimClass):
        """Z gate on superposition state."""
        qc = QiskitQuantumCircuit(1)
        qc.h(0)
        qc.z(0)

        qs = SimClass(1)
        qs.h(0)
        qs.z(0)

        compare_circuits(qs, qc)


# ==================== Phase Gates ====================


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestTGate:
    """T gate (π/4 phase) tests."""

    def test_ground_state(self, SimClass):
        """T gate should have no observable effect."""
        qc = QiskitQuantumCircuit(1)
        qc.t(0)

        qs = SimClass(1)
        qs.t(0)

        compare_circuits(qs, qc)

    def test_with_superposition(self, SimClass):
        """T gate on superposition state."""
        qc = QiskitQuantumCircuit(1)
        qc.h(0)
        qc.t(0)

        qs = SimClass(1)
        qs.h(0)
        qs.t(0)

        compare_circuits(qs, qc)


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestSGate:
    """S gate (π/2 phase) tests."""

    def test_ground_state(self, SimClass):
        """S gate should have no observable effect."""
        qc = QiskitQuantumCircuit(1)
        qc.s(0)

        qs = SimClass(1)
        qs.s(0)

        compare_circuits(qs, qc)

    def test_with_superposition(self, SimClass):
        """S gate on superposition state."""
        qc = QiskitQuantumCircuit(1)
        qc.h(0)
        qc.s(0)

        qs = SimClass(1)
        qs.h(0)
        qs.s(0)

        compare_circuits(qs, qc)


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestPhaseGate:
    """Arbitrary phase gate P(θ) tests."""

    @pytest.mark.parametrize("theta", ROTATION_ANGLES)
    def test_various_angles(self, SimClass, theta):
        """Test P gate with various rotation angles."""
        qc = QiskitQuantumCircuit(1)
        qc.p(theta, 0)

        qs = SimClass(1)
        qs.p(theta, 0)

        compare_circuits(qs, qc)

    @pytest.mark.parametrize("theta", ROTATION_ANGLES)
    def test_with_superposition(self, SimClass, theta):
        """Test P gate on superposition state."""
        qc = QiskitQuantumCircuit(1)
        qc.h(0)
        qc.p(theta, 0)

        qs = SimClass(1)
        qs.h(0)
        qs.p(theta, 0)

        compare_circuits(qs, qc)


# ==================== Rotation Gates ====================


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestRXGate:
    """RX gate (rotation around X-axis) tests."""

    @pytest.mark.parametrize("theta", ROTATION_ANGLES)
    def test_various_angles(self, SimClass, theta):
        """Test RX rotation with various angles."""
        qc = QiskitQuantumCircuit(1)
        qc.rx(theta, 0)

        qs = SimClass(1)
        qs.rx(theta, 0)

        compare_circuits(qs, qc)


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestRYGate:
    """RY gate (rotation around Y-axis) tests."""

    @pytest.mark.parametrize("theta", ROTATION_ANGLES)
    def test_various_angles(self, SimClass, theta):
        """Test RY rotation with various angles."""
        qc = QiskitQuantumCircuit(1)
        qc.ry(theta, 0)

        qs = SimClass(1)
        qs.ry(theta, 0)

        compare_circuits(qs, qc)


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestRZGate:
    """RZ gate (rotation around Z-axis) tests."""

    @pytest.mark.parametrize("theta", ROTATION_ANGLES)
    def test_various_angles(self, SimClass, theta):
        """Test RZ rotation with various angles."""
        qc = QiskitQuantumCircuit(1)
        qc.rz(theta, 0)

        qs = SimClass(1)
        qs.rz(theta, 0)

        compare_circuits(qs, qc)
