import numpy as np
import pytest
from src.circuit import QuantumCircuit

from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.quantum_info import Statevector


def statevector_from_qiskit(circ: QiskitQuantumCircuit) -> np.ndarray:
    """Return a statevector as a NumPy array from a Qiskit circuit."""
    sv = Statevector.from_instruction(circ)
    return np.asarray(sv.data, dtype=np.complex128)


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
