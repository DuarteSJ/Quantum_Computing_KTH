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


def test_cx_no_efect():
    # Qiskit
    qc = QiskitQuantumCircuit(2)
    qc.cx(0, 1)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(2)
    qs.cx(0, 1)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


def test_cx_control_one():
    # Qiskit
    qc = QiskitQuantumCircuit(2)
    qc.x(0)
    qc.cx(0, 1)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(2)
    qs.x(0)
    qs.cx(0, 1)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)


def test_cx_entanglement():
    # Qiskit
    qc = QiskitQuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    sv_qiskit = statevector_from_qiskit(qc)

    # Our simulator
    qs = QuantumCircuit(2)
    qs.h(0)
    qs.cx(0, 1)
    sv_ours = qs.state

    # Compare
    np.testing.assert_allclose(sv_ours, sv_qiskit, rtol=1e-12, atol=1e-12)
