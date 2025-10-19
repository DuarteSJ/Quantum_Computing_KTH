# Quantum Simulator Benchmark

Framework for benchmarking quantum circuit simulators.

## Running Benchmarks

```bash
python -m tests.benchmark.qft.benchmark_qft
```

## Adding a New Circuit

1. Create directory: `benchmark/my_circuit/`

2. Create `benchmark_my_circuit.py`:

```python
"""My Circuit benchmark."""

from qiskit import QuantumCircuit as QiskitQuantumCircuit
from tests.benchmark.benchmark_framework import run_benchmark


def apply_my_circuit(qc) -> None:
    """Apply circuit to custom simulator."""
    n = qc.n
    # Your implementation using: h(), x(), cx(), cp(), swap()
    qc.h(0)
    qc.cx(0, 1)


def apply_my_circuit_qiskit(qc: QiskitQuantumCircuit) -> None:
    """Apply circuit to Qiskit."""
    n = qc.num_qubits
    # Same implementation
    qc.h(0)
    qc.cx(0, 1)


MY_CIRCUIT = {
    "key": "my_circuit",
    "name": "My Circuit",
    "custom": apply_my_circuit,
    "qiskit": apply_my_circuit_qiskit,
}


def main():
    run_benchmark([MY_CIRCUIT], max_qubits=15)


if __name__ == "__main__":
    main()
```

3. Run: `python -m tests.benchmark.my_circuit.benchmark_my_circuit`

## Output

Generates `{circuit_key}_benchmark_results.png` with:
- Execution time vs qubits
- Memory usage vs qubits  
- Speedup vs Qiskit
