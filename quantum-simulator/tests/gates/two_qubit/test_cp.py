import numpy as np
import pytest
from src.circuit import QuantumCircuit

from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.quantum_info import Statevector


def statevector_from_qiskit(circ: QiskitQuantumCircuit) -> np.ndarray:
    """Return a statevector as a NumPy array from a Qiskit circuit."""
    sv = Statevector.from_instruction(circ)
    return np.asarray(sv.data, dtype=np.complex128)


# ----------------- Tests -----------------


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
def test_cp_no_efect(theta):
    # Qiskit
    qc = QiskitQuantumCircuit(2)
    qc.x(1)  # so we can see the effect (in this case none)
    qc.cp(theta, 0, 1)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(2)
    qs.x(1)  # so we can see the effect (in this case none)
    qs.cp(theta, 0, 1)
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
def test_cp_control_one(theta):
    # Qiskit
    qc = QiskitQuantumCircuit(2)
    qc.x(0)
    qc.x(1)  # so we can see the effect
    qc.cp(theta, 0, 1)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(2)
    qs.x(0)
    qs.x(1)  # so we can see the effect
    qs.cp(theta, 0, 1)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)
