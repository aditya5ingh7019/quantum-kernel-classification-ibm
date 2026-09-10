#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
print(f"Using backend: {backend.name}")

# 1. Build the simplest possible circuit: put a qubit in superposition and measure it
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

# 2. Transpile it for this specific backend's native gate set
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)

# 3. Submit the job
sampler = Sampler(mode=backend)
job = sampler.run([isa_circuit], shots=100)
print(f"Job ID: {job.job_id()}")
print("Status:", job.status())


# In[2]:


result = job.result()
counts = result[0].data.c.get_counts()
print(counts)


# In[4]:


from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names


# In[5]:


import pennylane as qml
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Prepare your QUBO-selected features
quantum_features = ['mean perimeter', 'mean symmetry', 'concavity error', 'worst area']
feature_indices = [list(feature_names).index(f) for f in quantum_features]
X_quantum = X[:, feature_indices]

# 2. Scale features to [0, pi] — angle encoding needs bounded inputs
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_quantum)

# 3. Define the feature map: angle encoding + entanglement (ZZFeatureMap-style)
n_qubits = len(quantum_features)  # 4 qubits, one per feature
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    # Layer 1: encode each feature as a rotation on its own qubit
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.RZ(2 * x[i], wires=i)
    # Layer 2: entangle adjacent qubits, encoding pairwise feature products
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])


# In[6]:


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    # Encode the first data point
    feature_map(x1)
    # Apply the inverse feature map for the second data point
    qml.adjoint(feature_map)(x2)
    # Measure the probability of returning to the all-zeros state
    return qml.probs(wires=range(n_qubits))

def quantum_kernel(x1, x2):
    # The |0...0> probability IS the kernel value (state overlap squared)
    probs = kernel_circuit(x1, x2)
    return probs[0]


# In[7]:


print(quantum_kernel(X_scaled[0], X_scaled[1]))
print(quantum_kernel(X_scaled[0], X_scaled[0]))  


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 2. Build kernel matrices (this computes quantum_kernel for every pair — can take a minute)
def compute_kernel_matrix(A, B):
    return np.array([[quantum_kernel(a, b) for b in B] for a in A])

K_train = compute_kernel_matrix(X_train, X_train)
K_test = compute_kernel_matrix(X_test, X_train)

# 3. Train an SVM using the precomputed quantum kernel
qsvm = SVC(kernel='precomputed')
qsvm.fit(K_train, y_train)

# 4. Evaluate
preds = qsvm.predict(K_test)
acc_qsvm = accuracy_score(y_test, preds)
print(f"Quantum Kernel SVM accuracy: {acc_qsvm:.4f}")


# In[9]:


# Classical SVM (RBF kernel) on the same 4 features, for a fair comparison
classical_svm = SVC(kernel='rbf')
classical_svm.fit(X_train, y_train)
classical_preds = classical_svm.predict(X_test)
acc_classical_svm = accuracy_score(y_test, classical_preds)
print(f"Classical RBF SVM accuracy: {acc_classical_svm:.4f}")


# In[10]:


from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
print(f"Using: {backend.name}")

# Real-hardware quantum device via PennyLane's qiskit.remote
# (runs in job mode by default — compatible with the free Open Plan, no Session needed)
dev_real = qml.device('qiskit.remote', wires=n_qubits, backend=backend, shots=200)

@qml.qnode(dev_real)
def kernel_circuit_real(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(n_qubits))

def quantum_kernel_real(x1, x2):
    probs = kernel_circuit_real(x1, x2)
    return probs[0]


# In[11]:


import random
random.seed(42)
sample_pairs = [(random.choice(range(len(X_test))), random.choice(range(len(X_train)))) for _ in range(5)]

print(f"{'Pair':<10}{'Simulator':<12}{'Real HW':<12}{'Diff'}")
for i, j in sample_pairs:
    sim_val = quantum_kernel(X_test[i], X_train[j])
    real_val = quantum_kernel_real(X_test[i], X_train[j])
    print(f"({i},{j})    {sim_val:.4f}      {real_val:.4f}      {abs(sim_val - real_val):.4f}")


# In[17]:


from qiskit import QuantumCircuit

def build_kernel_circuit_qiskit(x1, x2):
    qc = QuantumCircuit(n_qubits, n_qubits)

    # --- Forward feature map for x1 ---
    for i in range(n_qubits):
        qc.h(i)
        qc.rz(2 * x1[i], i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(2 * (np.pi - x1[i]) * (np.pi - x1[i + 1]), i + 1)
        qc.cx(i, i + 1)

    # --- Inverse feature map for x2 (reverse order, negated angles) ---
    for i in reversed(range(n_qubits - 1)):
        qc.cx(i, i + 1)
        qc.rz(-2 * (np.pi - x2[i]) * (np.pi - x2[i + 1]), i + 1)
        qc.cx(i, i + 1)
    for i in reversed(range(n_qubits)):
        qc.rz(-2 * x2[i], i)
        qc.h(i)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc

def quantum_kernel_mitigated_manual(x1, x2):
    qc = build_kernel_circuit_qiskit(x1, x2)
    isa_circuit = pm.run(qc)
    job = sampler_mitigated.run([isa_circuit], shots=200)
    result = job.result()[0]
    counts = result.data.c.get_counts()
    total = sum(counts.values())
    zero_state = '0' * n_qubits
    return counts.get(zero_state, 0) / total
print(f"{'Pair':<10}{'Simulator':<12}{'Real HW':<12}{'Real HW+EM':<12}")
for i, j in sample_pairs:
    sim_val = quantum_kernel(X_test[i], X_train[j])
    real_val = quantum_kernel_real(X_test[i], X_train[j])
    mitigated_val = quantum_kernel_mitigated_manual(X_test[i], X_train[j])
    print(f"({i},{j})    {sim_val:.4f}      {real_val:.4f}      {mitigated_val:.4f}")


# In[ ]:




