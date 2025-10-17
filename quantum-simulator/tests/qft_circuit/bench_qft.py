import numpy as np
from src.circuit import QuantumCircuit
import tracemalloc
import gc
import time


def apply_our_qft(qc: QuantumCircuit) -> None:
    n = qc.n
    
    for i in reversed(range(n)):
        qc.h(i)
        for j in range(i):
            theta = -np.pi / (2**(j+1))
            qc.cp(theta, i-1-j, i)

    for i in range(n//2):
        qc.swap(i, n - 1 - i)


def main():

    MAX_IDX = 15

    # Warmup
    for i in range(8):
        qc = QuantumCircuit(i)
        apply_our_qft(qc)

    print(f"--- Total execution time for QFT ---")
    for N in range(2,MAX_IDX):
        repeats = 200 // N
        measurements = []
        for _ in range(repeats):
            qc = QuantumCircuit(N)

            start = time.time()
            apply_our_qft(qc)
            end = time.time()

            duration = end - start
            measurements.append(duration)

        avg = np.average(measurements)
        print(f"{N} Qubits: {avg:.4f} seconds")

    tracemalloc.start()
    print(f"--- Total memory allocated during runtime of QFT ---")

    for N in range(MAX_IDX):

        if N > 0:
            del qc
            gc.collect()

        snapshot_start = tracemalloc.take_snapshot()
        qc = QuantumCircuit(N)
        apply_our_qft(qc)
        snapshot_end = tracemalloc.take_snapshot()

        top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        total_memory_change = 0
        for stat in top_stats:
            if stat.size_diff > 0:
                total_memory_change += stat.size_diff
        total_mib = total_memory_change / (1024 * 1024)
        if N > 1:
            print(f"{N} Qubits: {total_mib:.2f} MiB ({total_memory_change} bytes)")

    tracemalloc.stop()

if __name__ == '__main__':
    main()

