import numpy as np
import pandas as pd

np.random.seed(42)

samples = 600

# Generate antenna geometry parameters
R1 = np.random.uniform(0.8,1.2,samples)
R2 = np.random.uniform(1.2,1.6,samples)
R3 = np.random.uniform(1.8,2.2,samples)
R4 = np.random.uniform(2.4,2.8,samples)
R5 = np.random.uniform(3.0,3.4,samples)
R6 = np.random.uniform(3.4,3.8,samples)
R7 = np.random.uniform(4.0,4.4,samples)
R8 = np.random.uniform(4.8,5.6,samples)

d = np.random.uniform(0.45,0.65,samples)
Wf = np.random.uniform(1.0,1.6,samples)

# Antenna electromagnetic relationships (synthetic but realistic)

F1 = 28 - 0.8*R1 -0.4*R2 +0.2*Wf + np.random.normal(0,0.001,samples)
F2 = 31 - 0.6*R3 -0.3*R4 +0.25*Wf + np.random.normal(0,0.001,samples)
F3 = 33 - 0.5*R5 -0.2*R6 +0.3*Wf + np.random.normal(0,0.001,samples)

BW1 = 0.8 +0.05*d +0.02*Wf + np.random.normal(0,0.001,samples)
BW2 = 1.0 +0.04*d +0.03*Wf + np.random.normal(0,0.001,samples)
BW3 = 0.7 +0.06*d +0.02*Wf + np.random.normal(0,0.001,samples)

dataset = pd.DataFrame({
"R1":R1,
"R2":R2,
"R3":R3,
"R4":R4,
"R5":R5,
"R6":R6,
"R7":R7,
"R8":R8,
"d":d,
"Wf":Wf,
"F1":F1,
"F2":F2,
"F3":F3,
"BW1":BW1,
"BW2":BW2,
"BW3":BW3
})

dataset.to_csv("antenna_dataset.csv",index=False)

print("\nDataset Generated Successfully!")
print("Total Samples:",len(dataset))
print("\nFirst 5 Rows:\n")
print(dataset.head())