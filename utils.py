import numpy as np
from math import log2
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, Operator
from qiskit.quantum_info.operators import Pauli

def circuit_from_qasm_str(qasm_text: str) -> QuantumCircuit:
    return QuantumCircuit.from_qasm_str(qasm_text)

def state_from_circuit(circ: QuantumCircuit, as_density=False):
    # Fast path: build statevector from instruction (no simulator required)
    sv = Statevector.from_instruction(circ)
    if as_density:
        return DensityMatrix(sv)
    return sv
def reduced_single_qubit_rhos(rho, num_qubits: int):
    from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector

    if isinstance(rho, Statevector):
        dm = DensityMatrix(rho)
    else:
        dm = rho

    rhos = []
    for i in range(num_qubits):
        # Trace out all other qubits
        traced = partial_trace(dm, [j for j in range(num_qubits) if j != i])
        mat = np.array(traced.data, dtype=complex).reshape(2,2)
        rhos.append(mat)
    return rhos


def pauli_expectations(rho_1q: np.ndarray):
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    def tr(a): return np.trace(a).real
    x = tr(rho_1q @ X)
    y = tr(rho_1q @ Y)
    z = tr(rho_1q @ Z)
    return float(x), float(y), float(z)

def purity(rho_1q: np.ndarray):
    return float(np.real(np.trace(rho_1q @ rho_1q)))

def von_neumann_entropy(rho_1q: np.ndarray, base=2):
    evals = np.linalg.eigvalsh(rho_1q)
    # Handle numerical issues
    evals = np.clip(evals.real, 1e-12, 1.0)
    ent = -np.sum(evals * (np.log(evals) / np.log(base)))
    return float(np.real(ent))

def example_circuit(name: str) -> QuantumCircuit:
    name = name.lower()
    if name == "bell":
        qc = QuantumCircuit(2)
        qc.h(0); qc.cx(0,1)
        return qc
    if name == "ghz3":
        qc = QuantumCircuit(3)
        qc.h(0); qc.cx(0,1); qc.cx(1,2)
        return qc
    if name == "qft3":
        qc = QuantumCircuit(3)
        # Simple 3-qubit QFT
        qc.h(0); qc.cp(np.pi/2,1,0); qc.cp(np.pi/4,2,0)
        qc.h(1); qc.cp(np.pi/2,2,1)
        qc.h(2)
        qc.swap(0,2)
        return qc
    # default: single-qubit bloch sweep
    qc = QuantumCircuit(1)
    qc.h(0); qc.p(np.pi/3, 0)
    return qc
