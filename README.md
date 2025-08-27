# Quantum State Visualizer

A Streamlit app + CLI for:
- Loading multi-qubit quantum circuits (QASM/Qiskit).
- Computing reduced single-qubit density matrices via partial tracing.
- Plotting each qubit’s mixed state on a Bloch sphere.
- Reporting purity Tr(rho^2), von Neumann entropy, and Bloch vector.

## Quickstart (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a `.qasm` file or select an example circuit from the sidebar.

## CLI

```bash
python visualize.py --example bell
python visualize.py --file path/to/circuit.qasm
```

Outputs per-qubit Bloch vectors and purity/entropy in the terminal and saves figures to `out/`.
