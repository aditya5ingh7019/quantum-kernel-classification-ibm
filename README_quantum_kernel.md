# Quantum Kernel Classification on Real IBM Hardware

A quantum kernel method (Quantum SVM) built on features selected by a separate
[QUBO-based redundancy-aware feature selection project], validated on both a
local simulator and real IBM quantum hardware, including a real-hardware noise
study and an error mitigation comparison.

## Motivation

Quantum kernel methods use a parameterized quantum circuit to map classical
data into a quantum Hilbert space, then measure state overlaps to define a
kernel function for classical ML algorithms (here, an SVM). The theoretical
appeal is that some quantum feature maps encode similarity structures that are
classically hard to compute — though whether that translates into a practical
advantage on real (especially small, tabular) datasets is an open and actively
studied question. This project implements the method end-to-end and evaluates
it honestly against a classical baseline, on both simulated and real quantum
hardware.

## Dataset & Features

`sklearn.datasets.load_breast_cancer`, using 4 features (`mean perimeter`,
`mean symmetry`, `concavity error`, `worst area`) selected by a QUBO-based
redundancy-aware feature selection method in a companion project. Features
were scaled to `[0, π]` for angle encoding.

## Feature Map & Kernel

A 4-qubit feature map (one qubit per feature), inspired by the ZZFeatureMap
structure:
- **Layer 1**: each feature is encoded as a phase rotation (`RZ`) following a
  Hadamard, putting each qubit into a feature-dependent superposition.
- **Layer 2**: adjacent qubits are entangled via `CNOT`–`RZ`–`CNOT`, with the
  RZ angle depending on the *product* of the two adjacent features — this
  encodes pairwise feature interactions non-classically.

The kernel value between two points `x1, x2` is computed as the probability of
measuring the all-zeros state after applying the feature map for `x1` followed
by the inverse feature map for `x2` — this equals `|⟨φ(x2)|φ(x1)⟩|²`, the
squared state overlap.

## Methodology

1. Build the quantum kernel circuit in PennyLane (`default.qubit` simulator).
2. Precompute the full train/train and test/train kernel matrices, and train
   an `SVC(kernel='precomputed')` (Quantum SVM).
3. Train a classical `SVC(kernel='rbf')` on the same 4 features for a direct
   comparison.
4. Authenticate to IBM Quantum Platform (free Open Plan) via
   `qiskit-ibm-runtime`, using `qiskit.remote` (PennyLane-Qiskit) in job mode
   (Session mode is not available on the Open Plan).
5. Run a small sample of kernel-value pairs on real hardware and compare
   against simulator ground truth.
6. Repeat the same sample with dynamical decoupling enabled
   (`SamplerV2` option `dynamical_decoupling.enable = True`) to test whether
   basic error suppression narrows the sim-vs-hardware gap.

## Results

### Quantum SVM vs. Classical SVM (simulator)

| Method | Accuracy |
|---|---|
| Quantum Kernel SVM | 90.06% |
| Classical RBF SVM | 94.74% |

The classical RBF kernel outperformed the quantum kernel on this dataset. This
is a well-documented pattern in current QML research: generic small tabular
datasets typically don't have the kind of structure that gives quantum kernels
a computational edge over mature classical kernels — this result should be
read as an expected, informative finding rather than a failure of the method.

### Real Hardware Noise Study (backend: `ibm_marrakesh`, 200 shots/circuit)

| Pair | Simulator (ground truth) | Real HW | Real HW + Dynamical Decoupling |
|---|---|---|---|
| (163,57) | 0.0005 | 0.0100 | 0.0050 |
| (6,379) | 0.0115 | 0.0000 | 0.0150 |
| (70,125) | 0.0008 | 0.0000 | 0.0000 |
| (57,71) | 0.0046 | 0.0150 | 0.0250 |
| (26,346) | 0.3536 | 0.3550 | 0.2700 |

**Observations:**
- Real hardware introduces measurable deviation from the ideal simulator
  values, as expected. Deviations are larger in absolute terms for pairs with
  larger true kernel values, consistent with more coherent quantum information
  being exposed to decoherence in those cases.
- Dynamical decoupling **improved** results for 2 of 5 pairs, had no effect on
  1, and **worsened** 2 — notably degrading the largest-magnitude pair
  (0.3550 → 0.2700, versus a true value of 0.3536).
- This mixed result is expected and worth reporting plainly: dynamical
  decoupling targets idle-qubit dephasing specifically. It does not address
  gate errors or shot noise, and is not guaranteed to improve every circuit —
  a nuance often lost in simplified "error mitigation fixes noise" framing.
- With only 200 shots per circuit, part of the observed variation is
  statistical shot noise rather than hardware decoherence; this study does not
  attempt to separate the two contributions.

## Limitations

- Small sample size for the hardware comparison (5 pairs, 200 shots each) —
  chosen to stay within the free Open Plan's 10 free QPU minutes/month. A
  larger, statistically powered study would need significantly more QPU time.
- Dynamical decoupling was the only error suppression technique tested;
  readout error mitigation (available for `EstimatorV2`, not `SamplerV2`) was
  not evaluated here.
- Single dataset, single feature map design — results may not generalize to
  other quantum feature map architectures or datasets.

## Future Work

- Test readout error mitigation via `EstimatorV2` for a more complete error
  mitigation comparison.
- Increase shot count and sample size to statistically separate shot noise
  from hardware decoherence.
- Explore alternative feature map designs (different entanglement structures,
  depth) and their effect on both simulator accuracy and hardware noise
  sensitivity.

## Setup

```bash
pip install pennylane pennylane-qiskit qiskit-ibm-runtime scikit-learn numpy
```

Requires an IBM Quantum Platform account (free Open Plan) with a saved
account via `QiskitRuntimeService.save_account(...)`.

## Tech Stack

- `pennylane` / `pennylane-qiskit` — quantum feature map, kernel circuit, and
  hardware execution
- `qiskit-ibm-runtime` — real IBM QPU access via `SamplerV2`
- `scikit-learn` — classical baseline, SVM, evaluation
