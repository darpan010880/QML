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

#   print("\nDataset Preview:\n")
#   print(df.shape)
#   print(df.head())


# ============================================================
# HANDLE CATEGORICAL DATA
# ============================================================

label_encoders = {}

categorical_cols = [
    'gender',
    'platform_usage',
    'social_interaction_level',
]
i=0;
j=0;

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
# ============================================================
# INPUT FEATURES
# ============================================================

Inputs = df[
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

for col in Inputs.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


input=df[['age', 'gender', 'daily_social_media_hours', 'platform_usage', 'sleep_hours', 'screen_time_before_sleep', 'academic_performance', 'physical_activity', 'social_interaction_level']]
labeled_output=df[['stress_level', 'anxiety_level', 'addiction_level', 'depression_label']]
print("Input Shape",input.shape)
scaler = StandardScaler()
Input_scaled = scaler.fit_transform(input)

def quantum_feature_map(sample):
    qc = QuantumCircuit(4)
    # Encode first four features into quantum rotations
    for i in range(9):
        value = sample[i]
        qc.ry(value, i)
    # Add entanglement
    qc.cx(0,1)
    qc.cx(1,2)
    qc.cx(2,3)
    qc.cx(3,4)
    qc.cx(4,5)
    qc.cx(5,6)  
    qc.cx(6,7)
    qc.cx(7,8)

    qc.measure_all()
    simulator = AerSimulator()
    result = simulator.run(qc, shots=256).result()
    counts = result.get_counts()
    # Quantum feature = number of unique states
    quantum_feature = len(counts)
    return quantum_feature

quantum_features = []

for sample in Input_scaled:
    qf = quantum_feature_map(sample)
    quantum_features.append(qf)

quantum_features = np.array(quantum_features).reshape(-1,1)
