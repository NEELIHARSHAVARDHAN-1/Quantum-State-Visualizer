import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from qiskit import QuantumCircuit
from utils import (circuit_from_qasm_str, state_from_circuit,
                   reduced_single_qubit_rhos, pauli_expectations,
                   purity, von_neumann_entropy, example_circuit)

st.set_page_config(page_title="Quantum State Visualizer", layout="wide")

st.title("Quantum State Visualizer")
st.caption("Upload a multi-qubit circuit, compute single-qubit reduced density matrices via partial tracing, and visualize on Bloch spheres.")

with st.sidebar:
    st.header("Input")
    example = st.selectbox("Choose example circuit", ["(none)", "bell", "ghz3", "qft3"])
    uploaded = st.file_uploader("...or upload OpenQASM (.qasm)", type=["qasm"])
    as_density = st.checkbox("Use density-matrix internally", value=False)
    st.markdown("---")
    st.header("Export")
    export_prefix = st.text_input("Filename prefix", value="qsv_output")
    do_export = st.checkbox("Export figures & metrics", value=False)

# Load circuit
qc = None
if uploaded is not None:
    qasm_text = uploaded.read().decode("utf-8")
    qc = circuit_from_qasm_str(qasm_text)
elif example != "(none)":
    qc = example_circuit(example)

if qc is None:
    st.info("Select an example or upload a .qasm file to begin.")
    st.stop()

st.subheader("Circuit")
#st.code(qc.qasm(), language="qasm")
try:
    qasm_str = qc.qasm()
except Exception:
    # fallback for newer Qiskit versions
    from qiskit.qasm3 import dumps
    qasm_str = dumps(qc)

st.code(qasm_str, language="qasm")

# Compute state
rho = state_from_circuit(qc, as_density=as_density)
num_qubits = qc.num_qubits
rhos = reduced_single_qubit_rhos(rho, num_qubits)

# Bloch plotting helper
def plot_bloch(ax, x, y, z, title):
    import numpy as np

    # Sphere surface
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.15, color="lightblue", linewidth=0)

    # Axes
    ax.quiver(0,0,0, 1,0,0, color="black", linewidth=1, arrow_length_ratio=0.1)
    ax.quiver(0,0,0, 0,1,0, color="black", linewidth=1, arrow_length_ratio=0.1)
    ax.quiver(0,0,0, 0,0,1, color="black", linewidth=1, arrow_length_ratio=0.1)

    # State vector (only if not near zero)
    if abs(x) + abs(y) + abs(z) > 1e-6:
        ax.quiver(0,0,0, x, y, z, color="red", linewidth=2, arrow_length_ratio=0.15)
    else:
        ax.scatter([0],[0],[0], color="red", s=50)  # dot at center

    # Style
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_box_aspect([1,1,1])
    ax.set_title(title)
    ax.axis("off")

cols = st.columns(min(4, num_qubits) if num_qubits > 0 else 1)
rows = (num_qubits + len(cols)-1)//len(cols)

metrics = []
for i in range(num_qubits):
    x,y,z = pauli_expectations(rhos[i])
    P = purity(rhos[i])
    S = von_neumann_entropy(rhos[i])
    metrics.append(dict(qubit=i, x=x, y=y, z=z, purity=P, entropy=S))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_bloch(ax, x, y, z, title=f"Qubit {i}: (x={x:.2f}, y={y:.2f}, z={z:.2f})")

    if i < len(cols):
        cols[i].pyplot(fig)
    else:
        st.pyplot(fig)

st.subheader("Per-Qubit Metrics")
st.dataframe(metrics)

if do_export:
    import os, json
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    # Save metrics JSON
    with open(f"{out_dir}/{export_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    st.success(f"Saved metrics to {out_dir}/{export_prefix}_metrics.json")

    # Save figures
    for i,m in enumerate(metrics):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_bloch(ax, m['x'], m['y'], m['z'], title=f"Qubit {i}")
        fig.savefig(f"{out_dir}/{export_prefix}_qubit{i}.png", dpi=200, bbox_inches="tight")
    st.success(f"Saved {num_qubits} Bloch sphere images to {out_dir}/")
