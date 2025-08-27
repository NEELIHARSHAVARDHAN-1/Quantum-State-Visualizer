import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from qiskit import QuantumCircuit
from utils import (circuit_from_qasm_str, state_from_circuit,
                   reduced_single_qubit_rhos, pauli_expectations,
                   purity, von_neumann_entropy, example_circuit)

def plot_bloch(ax, x, y, z, title):
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.1, linewidth=0)
    ax.quiver(0,0,0, 1,0,0, length=1, arrow_length_ratio=0.1)
    ax.quiver(0,0,0, 0,1,0, length=1, arrow_length_ratio=0.1)
    ax.quiver(0,0,0, 0,0,1, length=1, arrow_length_ratio=0.1)
    ax.quiver(0,0,0, x, y, z, length=1, arrow_length_ratio=0.1)
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_box_aspect([1,1,1])
    ax.set_title(title)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to .qasm file")
    parser.add_argument("--example", type=str, default=None, help="Example: bell, ghz3, qft3")
    parser.add_argument("--out", type=str, default="out", help="Output directory")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            qasm = f.read()
        qc = circuit_from_qasm_str(qasm)
    elif args.example:
        qc = example_circuit(args.example)
    else:
        raise SystemExit("Provide --file .qasm or --example")

    rho = state_from_circuit(qc, as_density=True)
    rhos = reduced_single_qubit_rhos(rho, qc.num_qubits)

    metrics = []
    for i, r in enumerate(rhos):
        x,y,z = pauli_expectations(r)
        P = purity(r)
        S = von_neumann_entropy(r)
        metrics.append(dict(qubit=i, x=float(x), y=float(y), z=float(z),
                            purity=float(P), entropy=float(S)))

    print(json.dumps(metrics, indent=2))

    os.makedirs(args.out, exist_ok=True)
    for m in metrics:
        i = m["qubit"]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_bloch(ax, m['x'], m['y'], m['z'], title=f"Qubit {i}")
        fig.savefig(os.path.join(args.out, f"bloch_qubit{i}.png"), dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()
