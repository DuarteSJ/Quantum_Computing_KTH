import numpy as np
import pytest
from src.circuit import QuantumCircuit

from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.quantum_info import Statevector


def statevector_from_qiskit(circ: QiskitQuantumCircuit) -> np.ndarray:
    """Return a statevector as a NumPy array from a Qiskit circuit."""
    sv = Statevector.from_instruction(circ)
    return np.asarray(sv.data, dtype=np.complex128)


# ----------------- Hadamard Tests -----------------


def test_h_single_qubit():
    # Qiskit
    qc = QiskitQuantumCircuit(1)
    qc.h(0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.h(0)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


def test_h_two_qubits():
    # Qiskit
    qc = QiskitQuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(2)
    qs.h(0)
    qs.h(1)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


def test_h_three_qubits():
    # Qiskit
    qc = QiskitQuantumCircuit(3)
    qc.h(0)
    qc.h(2)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(3)
    qs.h(0)
    qs.h(2)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


# TODO: Add more tests only have some simple ones for now

# ----------------- Pauli-X Tests -----------------


def test_x_gate_1_qubit():
    # Qiskit
    qc = QiskitQuantumCircuit(1)
    qc.x(0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.x(0)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


def test_x_gate_2_qubit():
    # Qiskit
    qc = QiskitQuantumCircuit(2)
    qc.x(1)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(2)
    qs.x(1)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


def test_x_gate_3_qubits():
    # Qiskit
    qc = QiskitQuantumCircuit(3)
    qc.x(0)
    qc.x(2)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(3)
    qs.x(0)
    qs.x(2)
    sv_ours = qs.state
    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


# TODO: add more idk what tho

# ----------------- Pauli-Y Tests -----------------


def test_y_gate():
    # Qiskit
    qc = QiskitQuantumCircuit(1)
    qc.y(0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.y(0)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


# TODO: Add more tests only have some simple ones for now


# ----------------- Pauli-Z Tests -----------------



def test_z_gate():
    # Qiskit
    qc = QiskitQuantumCircuit(1)
    qc.z(0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.z(0)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


# TODO: Add more tests only have some simple ones for now


# ----------------- T(Phase pi/4) Tests -----------------

def test_t_gate_no_superposition():
    # Qiskit
    qc = QiskitQuantumCircuit(1)
    qc.t(0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.t(0)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


def test_t_gate_with_superposition():
    # Qiskit
    qc = QiskitQuantumCircuit(1)
    qc.h(0)
    qc.t(0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.h(0)
    qs.t(0)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


# ----------------- S(Phase pi/2) Tests -----------------

def test_s_gate_no_superposition():
    # Qiskit
    qc = QiskitQuantumCircuit(1)
    qc.s(0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.s(0)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


def test_s_gate_with_superposition():
    # Qiskit
    qc = QiskitQuantumCircuit(1)
    qc.h(0)
    qc.s(0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.h(0)
    qs.s(0)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


# ------------------- Phase Tests -----------------

@pytest.mark.parametrize(
    "theta",
    [
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
    ],
)
def test_p_gate(theta):
    # Qiskit reference
    qc = QiskitQuantumCircuit(1)
    qc.p(theta, 0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.p(theta, 0)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "theta",
    [
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
    ],
)
def test_p_gate_superposition(theta):
    # Qiskit reference
    qc = QiskitQuantumCircuit(1)
    qc.h(0)
    qc.p(theta, 0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.h(0)
    qs.p(theta, 0)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


# ------------------- RX Tests -----------------

@pytest.mark.parametrize(
    "theta",
    [
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
    ],
)
def test_rx_gate(theta):
    # Qiskit reference
    qc = QiskitQuantumCircuit(1)
    qc.rx(theta, 0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.rx(theta, 0)
    sv_ours = qs.state
    print(sv_ours, sv_qiskit)

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


# ------------------- RY Tests -----------------

@pytest.mark.parametrize(
    "theta",
    [
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
    ],
)
def test_ry_gate(theta):
    # Qiskit reference
    qc = QiskitQuantumCircuit(1)
    qc.ry(theta, 0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.ry(theta, 0)
    sv_ours = qs.state
    print(sv_ours, sv_qiskit)

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


# ------------------- RZ Tests -----------------

@pytest.mark.parametrize(
    "theta",
    [
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
    ],
)
def test_rz_gate(theta):
    # Qiskit reference
    qc = QiskitQuantumCircuit(1)
    qc.rz(theta, 0)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(1)
    qs.rz(theta, 0)
    sv_ours = qs.state
    print(sv_ours, sv_qiskit)

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)
