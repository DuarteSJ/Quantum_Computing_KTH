# Quantum Circuit Simulator

A quantum circuit simulator with multiple backend implementations for comparing performance and memory tradeoffs.

## Installation

### If you are using NixOs

```bash
# Enter development shell
nix develop

# Or run directly
nix develop --command python your_script.py
```

### Standard Python

```bash
pip install numpy qiskit matplotlib pytest
```

## Quick Start

```python
from src.circuits.Dense_Cartesian import DenseCartesianSim
import numpy as np

# Create a 3-qubit circuit
qc = DenseCartesianSim(3)

# Apply gates
qc.h(0)
qc.cx(0, 1)
qc.rx(np.pi/4, 2)

# Get statevector
state = qc.get_statevector()

# Measure
result = qc.measure_all()
```

## Backends

- **DenseCartesianSim**: Standard complex numbers. Good default choice.
- **DensePolarSim**: Stores magnitude and phase separately.
- **SparseDictSim**: Only stores non-zero amplitudes. Best for sparse states.
- **VectorizedSim**: Optimized NumPy operations. Fastest for most circuits.

All backends have identical APIs - just swap the class.

## Available Gates

**Single-qubit**: `x()`, `y()`, `z()`, `h()`, `s()`, `t()`, `p(theta)`, `rx(theta)`, `ry(theta)`, `rz(theta)`

**Two-qubit**: `cx(control, target)`, `cp(theta, control, target)`, `cz(control, target)`, `swap(q1, q2)`

## Testing

```bash
# Run all tests
pytest tests/

# Run specific backend
pytest tests/ -k "DenseCartesian"
```

## Benchmarking

```bash
# Run QFT benchmark
python -m tests.benchmark.qft.benchmark_qft

# Run sparse-friendly benchmark
python -m tests.benchmark.sparse_friendly.benchmark_sparse_friendly
```

See `tests/benchmark/README.md` for creating custom benchmarks.

## Visualization

```python
qc.viz_circle(label="My Circuit")
```

Shows quantum state with circle notation (radius = amplitude, arrow = phase).

## Project Structure

```
quantum-simulator/
├── src/
│   ├── circuits/
│   │   ├── base.py              # Abstract base class
│   │   ├── Dense_Cartesian.py   # Standard complex representation
│   │   ├── Dense_Polar.py       # Polar coordinate representation
│   │   ├── Sparse_Dict.py       # Sparse dictionary representation
│   │   └── Vectorized.py        # Optimized vectorized operations
│   └── gates.py                 # Quantum gate definitions
├── tests/
│   ├── gates/
│   │   ├── test_single_qubit.py # Single-qubit gate tests
│   │   └── test_two_qubit.py    # Two-qubit gate tests
│   └── benchmark/
│       ├── benchmark_framework.py
│       ├── README.md
│       ├── qft/
│       │   ├── benchmark_qft.py
│       │   └── test_qft_n.py
│       └── sparse_friendly/
│           └── benchmark_sparse_friendly.py
└── README.md
```
