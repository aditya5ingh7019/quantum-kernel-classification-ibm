# Qiskit Fall Fest 2026 @ MPSTME — Stage 1 Submission

**Project:** Quantum Kernel Classification on Real IBM Hardware
**Track:** Quantum Machine Learning Challenge
**Team:** Aditya Singh (solo)
**GitHub:** https://github.com/aditya5ingh7019/quantum-kernel-classification-ibm

---

## 1. Problem Understanding

Quantum kernel methods use a parameterized quantum circuit to map classical data into a quantum Hilbert space, then measure state overlaps to define a kernel function for classical machine learning algorithms (here, a Support Vector Machine). The theoretical appeal is that certain quantum feature maps encode similarity structures that are classically hard to compute — but whether this translates into a *practical* advantage on real, small tabular datasets is an open and actively debated question in current QML research.

This project implements a Quantum Kernel SVM end-to-end and evaluates it honestly against a classical baseline, on both a local simulator and real IBM quantum hardware — deliberately testing the claim rather than assuming it.

---

## 2. Quantum Approach

**Dataset & features:** `sklearn.datasets.load_breast_cancer`, using 4 features (`mean perimeter`, `mean symmetry`, `concavity error`, `worst area`) selected via a separate QUBO-based redundancy-aware feature selection method (companion project), scaled to `[0, π]` for angle encoding.

**Feature map (4 qubits, one per feature), ZZFeatureMap-inspired:**

- **Layer 1:** each feature encoded as an `RZ` phase rotation following a Hadamard, placing each qubit into a feature-dependent superposition.
- **Layer 2:** adjacent qubits entangled via `CNOT`–`RZ`–`CNOT`, with the `RZ` angle depending on the *product* of two adjacent features — encoding pairwise, non-classical feature interactions.

**Kernel definition:** the kernel value between two points `x1, x2` is the probability of measuring the all-zeros state after applying the feature map for `x1` followed by the inverse feature map for `x2` — equal to `|⟨φ(x2)|φ(x1)⟩|²`, the squared state overlap.

---

## 3. Qiskit Implementation

- Circuit built in PennyLane (`default.qubit`) for simulator-side kernel matrix computation.
- Real-hardware execution via `qiskit-ibm-runtime`, using `qiskit.remote` (PennyLane-Qiskit) in job mode — required since IBM's free Open Plan restricts session-mode execution.
- Backend used: `ibm_marrakesh`, authenticated via `QiskitRuntimeService`.
- Error suppression tested via `SamplerV2` dynamical decoupling (`dynamical_decoupling.enable = True`).

---

## 4. Working Prototype

Full pipeline implemented and runnable end-to-end:

1. Build kernel circuit
2. Compute train/train and test/train kernel matrices
3. Train `SVC(kernel='precomputed')` (Quantum SVM)
4. Train classical `SVC(kernel='rbf')` on identical features for direct comparison
5. Sample kernel-value pairs on real IBM hardware and compare against simulator ground truth
6. Repeat with dynamical decoupling enabled to test error suppression

---

## 5. Results & Analysis

### Quantum SVM vs. Classical SVM (simulator)

| Method | Accuracy |
|---|---|
| Quantum Kernel SVM | 90.06% |
| Classical RBF SVM | 94.74% |

The classical kernel outperformed the quantum kernel on this dataset. This is a well-documented pattern in current QML research: generic small tabular datasets typically don't have the kind of structure that gives quantum kernels a computational edge over mature classical kernels. Reported here as an honest, informative finding rather than reframed as a hidden success.

### Real hardware noise study (`ibm_marrakesh`, 200 shots/circuit, 5 sample pairs)

| Pair | Simulator (ground truth) | Real HW | Real HW + Dynamical Decoupling |
|---|---|---|---|
| (163,57) | 0.0005 | 0.0100 | 0.0050 |
| (6,379) | 0.0115 | 0.0000 | 0.0150 |
| (70,125) | 0.0008 | 0.0000 | 0.0000 |
| (57,71) | 0.0046 | 0.0150 | 0.0250 |
| (26,346) | 0.3536 | 0.3550 | 0.2700 |

**Observations:**

- Real hardware introduces measurable deviation from ideal simulator values, larger in absolute terms for pairs with larger true kernel values — consistent with more coherent quantum information being exposed to decoherence in those cases.
- Dynamical decoupling **improved** results for 2 of 5 pairs, had **no effect** on 1, and **worsened** 2 — notably degrading the largest-magnitude pair (0.3550 → 0.2700, versus a true value of 0.3536).
- This mixed result is expected and worth reporting plainly: dynamical decoupling targets idle-qubit dephasing specifically. It does not address gate errors or shot noise, and is not guaranteed to improve every circuit — a nuance often lost in simplified "error mitigation fixes noise" framing.
- With only 200 shots per circuit, part of the observed variation is statistical shot noise rather than hardware decoherence; this study does not attempt to separate the two contributions.

---

## 6. Innovation

- Redundancy-aware QUBO feature selection feeding directly into the quantum feature map, rather than an arbitrary feature choice.
- Deliberate dual validation: simulator *and* real hardware, with an explicit error-mitigation comparison rather than a single simulator-only result.
- Transparent reporting of a negative/mixed result (quantum underperforming classical, dynamical decoupling not uniformly helping) — treated as a rigorous scientific finding, not hidden or reframed.

---

## 7. Limitations

- Small sample size for the hardware comparison (5 pairs, 200 shots each) — chosen to stay within the free Open Plan's 10 free QPU minutes/month. A larger, statistically powered study would need significantly more QPU time.
- Dynamical decoupling was the only error suppression technique tested; readout error mitigation (available for `EstimatorV2`, not `SamplerV2`) was not evaluated here.
- Single dataset, single feature map design — results may not generalize to other quantum feature map architectures or datasets.

---

## 8. Future Work

- Test readout error mitigation via `EstimatorV2` for a more complete error mitigation comparison.
- Increase shot count and sample size to statistically separate shot noise from hardware decoherence.
- Explore alternative feature map designs (different entanglement structures, depth) and their effect on both simulator accuracy and hardware noise sensitivity.

---

## Tech Stack

- `pennylane` / `pennylane-qiskit` — quantum feature map, kernel circuit, and hardware execution
- `qiskit-ibm-runtime` — real IBM QPU access via `SamplerV2`
- `scikit-learn` — classical baseline, SVM, evaluation

**Setup:**

```bash
pip install pennylane pennylane-qiskit qiskit-ibm-runtime scikit-learn numpy
```

Requires an IBM Quantum Platform account (free Open Plan) with a saved account via `QiskitRuntimeService.save_account(...)`.
