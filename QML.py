# ============================================================
# QISKIT + MACHINE LEARNING PROJECT
# Prediction of Teen Mental Health Indicators
# Targets:
#   1. stress_level
#   2. anxiety_level
#   3. addiction_level
#   4. depression_label
#
# Dataset Features:
#   age
#   gender
#   daily_social_media_hours
#   platform_usage
#   sleep_hours
#   screen_time_before_sleep
#   academic_performance
#   physical_activity
#   social_interaction_level
#
# ============================================================

# INSTALL REQUIRED LIBRARIES
# pip install pandas numpy scikit-learn qiskit matplotlib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier

# Qiskit Imports
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("Teen_Mental_Health_Dataset.csv")

print("\nDataset Preview:\n")
print(df.head())

# ============================================================
# HANDLE CATEGORICAL DATA
# ============================================================

label_encoders = {}

categorical_cols = [
    'gender',
    'platform_usage',
    'academic_performance',
    'physical_activity',
    'social_interaction_level',
    'stress_level',
    'anxiety_level',
    'addiction_level',
    'depression_label'
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ============================================================
# INPUT FEATURES
# ============================================================

X = df[
    [
        'age',
        'gender',
        'daily_social_media_hours',
        'platform_usage',
        'sleep_hours',
        'screen_time_before_sleep',
        'academic_performance',
        'physical_activity',
        'social_interaction_level'
    ]
]

# ============================================================
# TARGET VARIABLES
# ============================================================

targets = [
    'stress_level',
    'anxiety_level',
    'addiction_level',
    'depression_label'
]

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# QUANTUM FEATURE GENERATION USING QISKIT
# ============================================================

def quantum_feature_map(sample):

    qc = QuantumCircuit(4)

    # Encode first four features into quantum rotations
    for i in range(4):
        value = sample[i]
        qc.ry(value, i)

    # Add entanglement
    qc.cx(0,1)
    qc.cx(1,2)
    qc.cx(2,3)

    qc.measure_all()

    simulator = AerSimulator()

    result = simulator.run(qc, shots=256).result()

    counts = result.get_counts()

    # Quantum feature = number of unique states
    quantum_feature = len(counts)

    return quantum_feature

# Generate quantum features
quantum_features = []

for sample in X_scaled:
    qf = quantum_feature_map(sample)
    quantum_features.append(qf)

quantum_features = np.array(quantum_features).reshape(-1,1)

# Combine classical + quantum features
X_final = np.hstack((X_scaled, quantum_features))

print("\nQuantum Features Added Successfully")

# ============================================================
# TRAINING MODELS FOR EACH TARGET
# ============================================================

for target in targets:

    print("\n================================================")
    print(f"TARGET : {target}")
    print("================================================")

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=0.2,
        random_state=42
    )

    # Hybrid Classical + Quantum ML Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy for {target}: {accuracy:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

# ============================================================
# SAMPLE QUANTUM CIRCUIT VISUALIZATION
# ============================================================

sample = X_scaled[0]

qc = QuantumCircuit(4)

for i in range(4):
    qc.ry(sample[i], i)

qc.cx(0,1)
qc.cx(1,2)
qc.cx(2,3)

qc.measure_all()

print("\nQuantum Circuit:\n")
print(qc.draw())

# ============================================================
# RUN QUANTUM CIRCUIT
# ============================================================

simulator = AerSimulator()

result = simulator.run(qc, shots=1024).result()

counts = result.get_counts()

print("\nQuantum Measurement Counts:\n")
print(counts)

# ============================================================
# PLOT HISTOGRAM
# ============================================================

plot_histogram(counts)
plt.title("Quantum State Distribution")
plt.show()

# ============================================================
# PREDICT NEW STUDENT DATA
# ============================================================

new_student = pd.DataFrame({
    'age':[20],
    'gender':['Male'],
    'daily_social_media_hours':[6],
    'platform_usage':['Instagram'],
    'sleep_hours':[5],
    'screen_time_before_sleep':[3],
    'academic_performance':['Average'],
    'physical_activity':['Low'],
    'social_interaction_level':['Medium']
})

# Encode categorical features
for col in [
    'gender',
    'platform_usage',
    'academic_performance',
    'physical_activity',
    'social_interaction_level'
]:
    new_student[col] = label_encoders[col].transform(new_student[col])

# Scale data
new_student_scaled = scaler.transform(new_student)

# Quantum feature
new_qf = quantum_feature_map(new_student_scaled[0])

new_final = np.hstack((new_student_scaled, [[new_qf]]))

print("\n================================================")
print("NEW STUDENT PREDICTIONS")
print("================================================")

for target in targets:

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    prediction = model.predict(new_final)

    predicted_label = label_encoders[target].inverse_transform(prediction)

    print(f"{target} : {predicted_label[0]}")

# ============================================================
# END OF PROGRAM
# ============================================================