import numpy as np
import tracemalloc
import gc
import time
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit.library import QFTGate

# Import all simulator implementations
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
            theta = -np.pi / (2**(j+1))
            qc.cp(theta, i-1-j, i)
    
    for i in range(n//2):
        qc.swap(i, n - 1 - i)


def apply_qiskit_qft(qc: QiskitQuantumCircuit) -> None:
    """Apply QFT to Qiskit circuit."""
    qc.append(QFTGate(qc.num_qubits).inverse(), qc.qubits)


def benchmark_combined(SimClass, max_idx=15):
    """Benchmark both execution time and memory usage in one pass."""
    sim_name = SimClass.__name__ if hasattr(SimClass, '__name__') else 'Qiskit'
    
    print(f"\n{'='*70}")
    print(f"Benchmarking: {sim_name}")
    print(f"{'='*70}")
    
    # Warmup
    for i in range(2, 8):
        if SimClass == 'qiskit':
            qc = QiskitQuantumCircuit(i)
            apply_qiskit_qft(qc)
        else:
            qc = SimClass(i)
            apply_our_qft(qc)
    
    time_results = {}
    memory_results = {}
    
    print(f"{'Qubits':<8} {'Time (s)':<15} {'Memory (MiB)':<15}")
    print('-' * 70)
    
    tracemalloc.start()
    qc = None
    
    for N in range(2, max_idx):
        # Clean up before measurement
        if qc is not None:
            del qc
            gc.collect()
        
        # Measure execution time multiple times for each N to get average
        repeats = 16 - N # Fewer repeats for larger N or else this runs forever
        time_measurements = []
        
        for _ in range(repeats):
            if SimClass == 'qiskit':
                qc_temp = QiskitQuantumCircuit(N)
                start = time.time()
                apply_qiskit_qft(qc_temp)
                end = time.time()
            else:
                qc_temp = SimClass(N)
                start = time.time()
                apply_our_qft(qc_temp)
                end = time.time()
            
            time_measurements.append(end - start)
            del qc_temp
        
        avg_time = np.average(time_measurements)
        time_results[N] = avg_time
        
        # Measure memory usage for a single run
        gc.collect()
        snapshot_start = tracemalloc.take_snapshot()
        
        if SimClass == 'qiskit':
            qc = QiskitQuantumCircuit(N)
            apply_qiskit_qft(qc)
        else:
            qc = SimClass(N)
            apply_our_qft(qc)
        
        snapshot_end = tracemalloc.take_snapshot()
        
        top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        total_memory_change = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        total_mib = total_memory_change / (1024 * 1024)
        memory_results[N] = total_mib
        
        print(f"{N:<8} {avg_time:<15.6f} {total_mib:<15.2f}")
    
    tracemalloc.stop()
    return time_results, memory_results


def plot_results(time_results, memory_results, max_idx):
    """Create visualization plots for benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color scheme for each simulator
    colors = {
        'Qiskit': '#000000',
        'DenseCartesianSim': '#2E86AB',
        'DensePolarSim': '#A23B72',
        'SparseDictSim': '#F18F01',
        'VectorizedSim': '#06A77D'
    }
    
    markers = {
        'Qiskit': 'x',
        'DenseCartesianSim': 'o',
        'DensePolarSim': 's',
        'SparseDictSim': '^',
        'VectorizedSim': 'D'
    }
    
    # Plot execution time
    for sim_name, times in time_results.items():
        qubits = sorted(times.keys())
        values = [times[q] for q in qubits]
        display_name = sim_name.replace('Sim', '') if sim_name != 'Qiskit' else sim_name
        ax1.plot(qubits, values, 
                marker=markers[sim_name], 
                color=colors[sim_name],
                label=display_name,
                linewidth=2.5 if sim_name == 'Qiskit' else 2,
                markersize=8 if sim_name == 'Qiskit' else 6,
                alpha=1.0 if sim_name == 'Qiskit' else 0.8)
    
    ax1.set_xlabel('Number of Qubits', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('QFT Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot memory usage
    for sim_name, memory in memory_results.items():
        qubits = sorted([q for q in memory.keys() if q >= 2])
        values = [memory[q] for q in qubits]
        display_name = sim_name.replace('Sim', '') if sim_name != 'Qiskit' else sim_name
        ax2.plot(qubits, values,
                marker=markers[sim_name],
                color=colors[sim_name],
                label=display_name,
                linewidth=2.5 if sim_name == 'Qiskit' else 2,
                markersize=8 if sim_name == 'Qiskit' else 6,
                alpha=1.0 if sim_name == 'Qiskit' else 0.8)
    
    ax2.set_xlabel('Number of Qubits', fontsize=12)
    ax2.set_ylabel('Memory Usage (MiB)', fontsize=12)
    ax2.set_title('QFT Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('qft_benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plots saved to 'qft_benchmark_comparison.png'")
    plt.show()


def plot_speedup_comparison(time_results, max_idx):
    """Create speedup comparison plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {
        'Qiskit': '#000000',
        'DenseCartesianSim': '#2E86AB',
        'DensePolarSim': '#A23B72',
        'SparseDictSim': '#F18F01',
        'VectorizedSim': '#06A77D'
    }
    
    markers = {
        'Qiskit': 'x',
        'DenseCartesianSim': 'o',
        'DensePolarSim': 's',
        'SparseDictSim': '^',
        'VectorizedSim': 'D'
    }
    
    # Calculate speedup relative to Qiskit
    baseline_name = 'Qiskit'
    baseline = time_results[baseline_name]
    
    for sim_name, times in time_results.items():
        if sim_name == baseline_name:
            continue
        
        qubits = sorted([q for q in times.keys() if q in baseline])
        speedups = [baseline[q] / times[q] for q in qubits]
        display_name = sim_name.replace('Sim', '')
        
        ax.plot(qubits, speedups,
               marker=markers[sim_name],
               color=colors[sim_name],
               label=display_name,
               linewidth=2,
               markersize=6)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (Qiskit)')
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Speedup (relative to Qiskit)', fontsize=12)
    ax.set_title('Relative Performance: Speedup Factor vs Qiskit', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qft_speedup_comparison.png', dpi=300, bbox_inches='tight')
    print("Speedup plot saved to 'qft_speedup_comparison.png'")
    plt.show()


def compare_all_simulators(max_idx=15):
    """Run benchmarks for all simulator implementations."""
    simulators = [
        'qiskit',  # Add Qiskit as baseline
        DenseCartesianSim,
        DensePolarSim,
        SparseDictSim,
        VectorizedSim,
    ]
    
    time_results = {}
    memory_results = {}
    
    # Benchmark each simulator
    for SimClass in simulators:
        sim_name = 'Qiskit' if SimClass == 'qiskit' else SimClass.__name__
        times, memory = benchmark_combined(SimClass, max_idx)
        time_results[sim_name] = times
        memory_results[sim_name] = memory
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    
    print("\nExecution Time Comparison (seconds):")
    print(f"{'Qubits':<8}", end='')
    for sim_name in time_results.keys():
        print(f"{sim_name:<20}", end='')
    print()
    print('-' * 80)
    
    for N in range(2, max_idx):
        print(f"{N:<8}", end='')
        for sim_name in time_results.keys():
            if N in time_results[sim_name]:
                print(f"{time_results[sim_name][N]:<20.6f}", end='')
            else:
                print(f"{'N/A':<20}", end='')
        print()
    
    print("\nMemory Usage Comparison (MiB):")
    print(f"{'Qubits':<8}", end='')
    for sim_name in memory_results.keys():
        print(f"{sim_name:<20}", end='')
    print()
    print('-' * 80)
    
    for N in range(2, max_idx):
        print(f"{N:<8}", end='')
        for sim_name in memory_results.keys():
            if N in memory_results[sim_name]:
                print(f"{memory_results[sim_name][N]:<20.2f}", end='')
            else:
                print(f"{'N/A':<20}", end='')
        print()
    
    # Calculate speedup relative to slowest
    print("\n" + "="*80)
    print("SPEEDUP ANALYSIS (relative to slowest)")
    print("="*80)
    
    for N in [5, 10, 14]:  # Sample qubit counts
        if N >= max_idx:
            continue
        print(f"\n{N} Qubits:")
        times = [(name, time_results[name][N]) for name in time_results.keys() if N in time_results[name]]
        if not times:
            continue
        
        slowest = max(times, key=lambda x: x[1])
        print(f"  Slowest: {slowest[0]} ({slowest[1]:.6f}s)")
        
        for name, t in sorted(times, key=lambda x: x[1]):
            speedup = slowest[1] / t
            print(f"  {name:<20}: {speedup:>6.2f}x")
    
    # Generate plots
    plot_results(time_results, memory_results, max_idx)
    plot_speedup_comparison(time_results, max_idx)


def main():
    """Run all benchmarks and comparisons."""
    MAX_IDX = 15
    
    print("="*80)
    print("QUANTUM SIMULATOR BENCHMARK - QFT")
    print("="*80)
    print(f"Testing QFT implementation across all simulator variants")
    print(f"Maximum qubits: {MAX_IDX}")
    print("="*80)
    
    compare_all_simulators(MAX_IDX)


if __name__ == '__main__':
    main()
