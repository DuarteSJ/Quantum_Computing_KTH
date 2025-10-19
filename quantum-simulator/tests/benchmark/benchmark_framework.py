"""
Core benchmark framework to evaluate quantum simulators.

This module provides the infrastructure for benchmarking different quantum
simulators on various quantum circuits in comparison to qiskit. It handles timing,
memory profiling, and visualization of results.

Usage:
    from benchmark_framework import run_benchmark
    from benchmark_CIRCUIT import CIRCUIT

    run_benchmark([CIRCUIT], max_qubits=15)
"""

import numpy as np
import tracemalloc
import gc
import time
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit as QiskitQuantumCircuit

from src.circuits.Dense_Cartesian import DenseCartesianSim
from src.circuits.Dense_Polar import DensePolarSim
from src.circuits.Sparse_Dict import SparseDictSim
from src.circuits.Vectorized import VectorizedSim


def benchmark_simulator(SimClass, circuit_def, max_qubits=15):
    """
    Benchmark execution time and memory usage for a specific circuit.

    Args:
        SimClass: Simulator class or "qiskit" string
        circuit_def: Dictionary with circuit definition (see CIRCUIT_TEMPLATE)
        max_qubits: Maximum number of qubits to test

    Returns:
        Tuple of (time_results, memory_results) dictionaries
    """
    sim_name = "Qiskit" if SimClass == "qiskit" else SimClass.__name__
    print(f"  {sim_name}...", end=" ", flush=True)

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
                circuit_def["qiskit"](qc)
                times.append(time.time() - start)
            else:
                qc = SimClass(N)
                start = time.time()
                circuit_def["custom"](qc)
                times.append(time.time() - start)
            del qc

        time_results[N] = np.mean(times)

        # Memory measurement
        gc.collect()
        snapshot_start = tracemalloc.take_snapshot()

        if SimClass == "qiskit":
            qc = QiskitQuantumCircuit(N)
            circuit_def["qiskit"](qc)
        else:
            qc = SimClass(N)
            circuit_def["custom"](qc)

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


def plot_results(time_results, memory_results, circuit_name, circuit_key):
    """
    Create benchmark visualization plots.

    Args:
        time_results: Dictionary mapping simulator names to timing data
        memory_results: Dictionary mapping simulator names to memory data
        circuit_name: Human-readable circuit name for titles
        circuit_key: Short key for filename
    """
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
            marker=markers.get(sim_name, "o"),
            color=colors.get(sim_name, "#999999"),
            label=label,
            linewidth=2,
            markersize=6,
            alpha=0.9,
        )

    axes[0].set_xlabel("Number of Qubits", fontsize=12)
    axes[0].set_ylabel("Execution Time (seconds)", fontsize=12)
    axes[0].set_title(
        f"{circuit_name} - Execution Time", fontsize=14, fontweight="bold"
    )
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
            marker=markers.get(sim_name, "o"),
            color=colors.get(sim_name, "#999999"),
            label=label,
            linewidth=2,
            markersize=6,
            alpha=0.9,
        )

    axes[1].set_xlabel("Number of Qubits", fontsize=12)
    axes[1].set_ylabel("Memory Usage (MiB)", fontsize=12)
    axes[1].set_title(f"{circuit_name} - Memory Usage", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    # Plot 3: Speedup vs Qiskit
    if "Qiskit" in time_results:
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
                marker=markers.get(sim_name, "o"),
                color=colors.get(sim_name, "#999999"),
                label=label,
                linewidth=2,
                markersize=6,
            )

        axes[2].axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        axes[2].set_xlabel("Number of Qubits", fontsize=12)
        axes[2].set_ylabel("Speedup (×)", fontsize=12)
        axes[2].set_title(
            f"{circuit_name} - Speedup vs Qiskit", fontsize=14, fontweight="bold"
        )
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{circuit_key.lower()}_benchmark_results.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n✓ Results saved to '{filename}'")
    plt.show()


def run_benchmark(circuits, max_qubits=15, simulators=None):
    """
    Run benchmarks for specified circuits and simulators.

    Args:
        circuits: List of circuit definitions (dictionaries)
        max_qubits: Maximum number of qubits to test
        simulators: List of simulator classes (default: all available)
    """
    if simulators is None:
        simulators = [
            "qiskit",
            DenseCartesianSim,
            DensePolarSim,
            SparseDictSim,
            VectorizedSim,
        ]

    for circuit_def in circuits:
        print("\n" + "=" * 60)
        print(f"BENCHMARKING: {circuit_def['name']}")
        print(f"Testing up to {max_qubits} qubits")
        print("=" * 60 + "\n")

        all_time_results = {}
        all_memory_results = {}

        for sim in simulators:
            sim_name = "Qiskit" if sim == "qiskit" else sim.__name__
            times, memory = benchmark_simulator(sim, circuit_def, max_qubits)
            all_time_results[sim_name] = times
            all_memory_results[sim_name] = memory

        print("\n" + "=" * 60)
        print("Generating plots...")
        plot_results(
            all_time_results,
            all_memory_results,
            circuit_def["name"],
            circuit_def["key"],
        )


# ============================================================================
# CIRCUIT DEFINITION TEMPLATE
# ============================================================================

CIRCUIT_TEMPLATE = {
    "key": "example",  # Short identifier for filenames
    "name": "Example Circuit",  # Human-readable name
    "custom": None,  # Function: custom_function(qc) -> None
    "qiskit": None,  # Function: qiskit_function(qc) -> None
}
