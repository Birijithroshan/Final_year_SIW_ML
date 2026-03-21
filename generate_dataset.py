import numpy as np
import pandas as pd

np.random.seed(42)

samples = 600

# Generate antenna geometry parameters around validated HFSS design.
# Baseline (validated): R1..R8 = [0.8, 1.0, 1.75, 2.0, 2.7, 3.0, 3.65, 4.0],
# Dv (d) = 0.6, Wf = 1.4
R1 = np.random.uniform(0.6, 1.0, samples)
R2 = np.random.uniform(0.8, 1.2, samples)
R3 = np.random.uniform(1.5, 2.0, samples)
R4 = np.random.uniform(1.7, 2.3, samples)
R5 = np.random.uniform(2.4, 3.0, samples)
R6 = np.random.uniform(2.7, 3.3, samples)
R7 = np.random.uniform(3.3, 4.0, samples)
R8 = np.random.uniform(3.6, 4.4, samples)

d = np.random.uniform(0.5, 0.7, samples)
Wf = np.random.uniform(1.2, 1.6, samples)

# Antenna electromagnetic relationships calibrated for target bands:
# F1 ≈ 21-22 GHz, F2 ≈ 32-34 GHz, F3 ≈ 39-40 GHz
F1 = (
	23.2
	- 0.90 * R1
	- 0.80 * R2
	- 0.10 * d
	+ 0.20 * Wf
	+ np.random.normal(0, 0.004, samples)
)

F2 = (
	35.8
	- 0.95 * R3
	- 0.80 * R4
	- 0.08 * d
	+ 0.20 * Wf
	+ np.random.normal(0, 0.005, samples)
)

F3 = (
	43.7
	- 0.90 * R5
	- 0.70 * R6
	- 0.05 * d
	+ 0.30 * Wf
	+ np.random.normal(0, 0.006, samples)
)

BW1 = 0.62 + 0.07 * d + 0.03 * Wf + np.random.normal(0, 0.0015, samples)
BW2 = 0.95 + 0.06 * d + 0.04 * Wf + np.random.normal(0, 0.0015, samples)
BW3 = 0.78 + 0.08 * d + 0.03 * Wf + np.random.normal(0, 0.0015, samples)

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