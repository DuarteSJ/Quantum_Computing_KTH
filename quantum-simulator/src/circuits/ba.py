from src.circuits.Sparse_Dict import SparseDictSim
from src.circuits.Dense_Cartesian import DenseCartesianSim

qdense = DenseCartesianSim(1)
print("state vector for dense before had gate", qdense.get_statevector())
# qdense.viz_circle(label=" Dense Before H gate")

qdense.h(0)
print("state vector for dense after had gate", qdense.get_statevector())
# qdense.viz_circle(label=" Dense After H gate")


qsparse  = SparseDictSim(1)
print("state vector for sparse before had gate", qsparse.get_statevector())
# qsparse.viz_circle(label=" Sparse Before H gate")

qsparse.h(0)
print("state vector for sparse after had gate", qsparse.get_statevector())
# qsparse.viz_circle(label=" Sparse After H gate")
