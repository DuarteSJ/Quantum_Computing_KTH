# Quantum_Computing_KTH

Coursework and project material for a quantum computing course at KTH Royal
Institute of Technology.

- [`quantum-simulator/`](quantum-simulator/) - a quantum circuit simulator with
  multiple backend implementations, plus tests and benchmarks. See its
  [README](quantum-simulator/README.md).
- `programming-assignments/` - course assignment notebooks.

## Licence and reuse

The code in this repository written by us is released under the
[MIT License](LICENSE). Copyright (c) 2025 Duarte São José and Jens
Verherstraeten.

Reuse, adaptation and redistribution, including inclusion in research
datasets and use for machine learning training and evaluation, are permitted
under those terms, provided the copyright notice and licence text are retained.

Suggested attribution:

> Duarte São José and Jens Verherstraeten, *Quantum_Computing_KTH*,
> https://github.com/DuarteSJ/Quantum_Computing_KTH, MIT License.

When citing a specific file, please pin the commit hash rather than the `main`
branch so the reference stays reproducible.

### Scope and exclusions

The licence covers only material authored by us. It does not cover:

- **`programming-assignments/`** - the problem statements and any provided
  scaffolding in these notebooks are authored by the course staff at KTH and
  are not ours to license. Only our own solution code in those files falls
  under the licence above.
- **Third-party dependencies** - the simulator imports
  [Qiskit](https://github.com/Qiskit/qiskit) (Apache-2.0),
  [NumPy](https://numpy.org/) (BSD-3-Clause),
  [pytest](https://pytest.org/) (MIT) and
  [Matplotlib](https://matplotlib.org/) (PSF-based). No third-party source code
  is vendored here; each project's own licence applies to it.
