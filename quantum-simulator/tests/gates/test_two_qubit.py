"""
Two-qubit gate tests for our quantum circuit simulator.

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
    sv_ours = our_circuit.get_statevector()  # Use abstract method
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


# Common test angles for parametric gates
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


# ==================== CNOT (CX) Gate ====================


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestCXGate:
    """CNOT (CX) gate tests."""

    def test_no_effect(self, SimClass):
        """CX gate with control=|0⟩ should have no effect."""
        qc = QiskitQuantumCircuit(2)
        qc.cx(0, 1)

        qs = SimClass(2)
        qs.cx(0, 1)

        compare_circuits(qs, qc)

    def test_control_one(self, SimClass):
        """CX gate with control=|1⟩ should flip target."""
        qc = QiskitQuantumCircuit(2)
        qc.x(0)
        qc.cx(0, 1)

        qs = SimClass(2)
        qs.x(0)
        qs.cx(0, 1)

        compare_circuits(qs, qc)

    def test_entanglement(self, SimClass):
        """CX gate creates Bell state from H|0⟩."""
        qc = QiskitQuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        qs = SimClass(2)
        qs.h(0)
        qs.cx(0, 1)

        compare_circuits(qs, qc)


# ==================== Controlled-Phase (CP) Gate ====================


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestCPGate:
    """Controlled-Phase gate tests."""

    @pytest.mark.parametrize("theta", ROTATION_ANGLES)
    def test_no_effect(self, SimClass, theta):
        """CP gate with control=|0⟩ should have no effect."""
        qc = QiskitQuantumCircuit(2)
        qc.x(1)  # Target in |1⟩ to make effect visible
        qc.cp(theta, 0, 1)

        qs = SimClass(2)
        qs.x(1)
        qs.cp(theta, 0, 1)

        compare_circuits(qs, qc)

    @pytest.mark.parametrize("theta", ROTATION_ANGLES)
    def test_control_one(self, SimClass, theta):
        """CP gate with control=|1⟩ applies phase to target."""
        qc = QiskitQuantumCircuit(2)
        qc.x(0)  # Control in |1⟩
        qc.x(1)  # Target in |1⟩ to make effect visible
        qc.cp(theta, 0, 1)

        qs = SimClass(2)
        qs.x(0)
        qs.x(1)
        qs.cp(theta, 0, 1)

        compare_circuits(qs, qc)


# ==================== Controlled-Z (CZ) Gate ====================


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestCZGate:
    """Controlled-Z gate tests."""

    def test_no_effect(self, SimClass):
        """CZ gate with control=|0⟩ should have no effect."""
        qc = QiskitQuantumCircuit(2)
        qc.x(1)  # Target in |1⟩ to make effect visible
        qc.cz(0, 1)

        qs = SimClass(2)
        qs.x(1)
        qs.cz(0, 1)

        compare_circuits(qs, qc)

    def test_control_one(self, SimClass):
        """CZ gate with both qubits=|1⟩ applies phase."""
        qc = QiskitQuantumCircuit(2)
        qc.x(0)  # Control in |1⟩
        qc.x(1)  # Target in |1⟩
        qc.cz(0, 1)

        qs = SimClass(2)
        qs.x(0)
        qs.x(1)
        qs.cz(0, 1)

        compare_circuits(qs, qc)


# ==================== SWAP Gate ====================


@pytest.mark.parametrize("SimClass", ALL_SIMULATORS)
class TestSWAPGate:
    """SWAP gate tests."""

    def test_ground_state(self, SimClass):
        """SWAP on |00⟩ should have no observable effect."""
        qc = QiskitQuantumCircuit(2)
        qc.swap(0, 1)

        qs = SimClass(2)
        qs.swap(0, 1)

        compare_circuits(qs, qc)

    def test_after_x_gate(self, SimClass):
        """SWAP should exchange qubit states."""
        qc = QiskitQuantumCircuit(2)
        qc.x(0)  # Put qubit 0 in |1⟩ to see the swap effect
        qc.swap(0, 1)

        qs = SimClass(2)
        qs.x(0)
        qs.swap(0, 1)

        compare_circuits(qs, qc)
