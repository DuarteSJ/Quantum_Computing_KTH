import numpy as np
import tracemalloc
import gc
import time
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit.library import QFTGate

from src.circuits.Dense_Cartesian import DenseCartesianSim
from src.circuits.Dense_Polar import DensePolarSim
from src.circuits.Sparse_Dict import SparseDictSim
from src.circuits.Vectorized import VectorizedSim


def apply_our_qft(qc) -> None:
    """Apply QFT to quantum circuit."""
    n = qc.n
    for i in reversed(range(n)):
        qc.h(i)
        for j in range(i):
            theta = -np.pi / (2 ** (j + 1))
            qc.cp(theta, i - 1 - j, i)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)


def apply_qiskit_qft(qc: QiskitQuantumCircuit) -> None:
    """Apply QFT to Qiskit circuit."""
    qc.append(QFTGate(qc.num_qubits).inverse(), qc.qubits)


def benchmark_simulator(SimClass, max_qubits=15):
    """Benchmark execution time and memory usage."""
    sim_name = "Qiskit" if SimClass == "qiskit" else SimClass.__name__
    print(f"Benchmarking {sim_name}...", end=" ", flush=True)

    time_results = {}
    memory_results = {}

    tracemalloc.start()

    for N in range(2, max_qubits):
        # Time measurement (averaged)
        repeats = max(1, 16 - N)
        times = []

        for _ in range(repeats):
            if SimClass == "qiskit":
                qc = QiskitQuantumCircuit(N)
                start = time.time()
                apply_qiskit_qft(qc)
                times.append(time.time() - start)
            else:
                qc = SimClass(N)
                start = time.time()
                apply_our_qft(qc)
                times.append(time.time() - start)
            del qc

        time_results[N] = np.mean(times)

        # Memory measurement
        gc.collect()
        snapshot_start = tracemalloc.take_snapshot()

        if SimClass == "qiskit":
            qc = QiskitQuantumCircuit(N)
            apply_qiskit_qft(qc)
        else:
            qc = SimClass(N)
            apply_our_qft(qc)

        snapshot_end = tracemalloc.take_snapshot()
        top_stats = snapshot_end.compare_to(snapshot_start, "lineno")
        memory_mib = sum(s.size_diff for s in top_stats if s.size_diff > 0) / (
            1024 * 1024
        )
        memory_results[N] = memory_mib

        del qc

    tracemalloc.stop()
    print("Done!")
    return time_results, memory_results


def plot_results(time_results, memory_results):
    """Create benchmark visualization plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        "Qiskit": "#000000",
        "DenseCartesianSim": "#2E86AB",
        "DensePolarSim": "#A23B72",
        "SparseDictSim": "#F18F01",
        "VectorizedSim": "#06A77D",
    }

    markers = {
        "Qiskit": "x",
        "DenseCartesianSim": "o",
        "DensePolarSim": "s",
        "SparseDictSim": "^",
        "VectorizedSim": "D",
    }

    # Plot 1: Execution Time
    for sim_name, times in time_results.items():
        qubits = sorted(times.keys())
        values = [times[q] for q in qubits]
        label = sim_name.replace("Sim", "") if sim_name != "Qiskit" else sim_name
        axes[0].plot(
            qubits,
            values,
            marker=markers[sim_name],
            color=colors[sim_name],
            label=label,
            linewidth=2,
            markersize=6,
            alpha=0.9,
        )

    axes[0].set_xlabel("Number of Qubits", fontsize=12)
    axes[0].set_ylabel("Execution Time (seconds)", fontsize=12)
    axes[0].set_title("QFT Execution Time", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # Plot 2: Memory Usage
    for sim_name, memory in memory_results.items():
        qubits = sorted(memory.keys())
        values = [memory[q] for q in qubits]
        label = sim_name.replace("Sim", "") if sim_name != "Qiskit" else sim_name
        axes[1].plot(
            qubits,
            values,
            marker=markers[sim_name],
            color=colors[sim_name],
            label=label,
            linewidth=2,
            markersize=6,
            alpha=0.9,
        )

    axes[1].set_xlabel("Number of Qubits", fontsize=12)
    axes[1].set_ylabel("Memory Usage (MiB)", fontsize=12)
    axes[1].set_title("QFT Memory Usage", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    # Plot 3: Speedup vs Qiskit
    baseline = time_results["Qiskit"]
    for sim_name, times in time_results.items():
        if sim_name == "Qiskit":
            continue
        qubits = sorted([q for q in times.keys() if q in baseline])
        speedups = [baseline[q] / times[q] for q in qubits]
        label = sim_name.replace("Sim", "")
        axes[2].plot(
            qubits,
            speedups,
            marker=markers[sim_name],
            color=colors[sim_name],
            label=label,
            linewidth=2,
            markersize=6,
        )

    axes[2].axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[2].set_xlabel("Number of Qubits", fontsize=12)
    axes[2].set_ylabel("Speedup (×)", fontsize=12)
    axes[2].set_title("Speedup vs Qiskit", fontsize=14, fontweight="bold")
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("qft_benchmark_results.png", dpi=300, bbox_inches="tight")
    print(f"\n✓ Results saved to 'qft_benchmark_results.png'")
    plt.show()


def main():
    """Run benchmarks and generate plots."""
    MAX_QUBITS = 15

    print("=" * 60)
    print("QUANTUM SIMULATOR BENCHMARK - QFT")
    print(f"Testing up to {MAX_QUBITS} qubits")
    print("=" * 60 + "\n")

    simulators = [
        "qiskit",
        DenseCartesianSim,
        DensePolarSim,
        SparseDictSim,
        VectorizedSim,
    ]

    all_time_results = {}
    all_memory_results = {}

    for sim in simulators:
        sim_name = "Qiskit" if sim == "qiskit" else sim.__name__
        times, memory = benchmark_simulator(sim, MAX_QUBITS)
        all_time_results[sim_name] = times
        all_memory_results[sim_name] = memory

    print("\n" + "=" * 60)
    print("Generating plots...")
    plot_results(all_time_results, all_memory_results)


if __name__ == "__main__":
    main()
