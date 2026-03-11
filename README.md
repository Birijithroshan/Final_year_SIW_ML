# Machine Learning-Based Prediction of SIW Antenna Parameters

**Project:** Final Year Project — Substrate Integrated Waveguide (SIW) Antenna Parameter Prediction using Machine Learning

---

## Overview

This project applies supervised machine learning to predict the resonance frequencies and bandwidths of a multi-ring SIW antenna directly from its geometric design parameters. Instead of running computationally expensive electromagnetic simulations for every new antenna design, trained ML models can predict the antenna's performance characteristics instantly from its physical dimensions.

---

## Project Workflow

### Step 1 — Dataset Generation (`generate_dataset.py`)

A synthetic dataset of **600 antenna samples** is generated using physics-inspired analytical relationships between antenna geometry and electromagnetic performance.

**Input parameters (10 features):**

| Parameter | Description | Range |
|-----------|-------------|-------|
| R1 | Radius of ring 1 (mm) | 0.8 – 1.2 |
| R2 | Radius of ring 2 (mm) | 1.2 – 1.6 |
| R3 | Radius of ring 3 (mm) | 1.8 – 2.2 |
| R4 | Radius of ring 4 (mm) | 2.4 – 2.8 |
| R5 | Radius of ring 5 (mm) | 3.0 – 3.4 |
| R6 | Radius of ring 6 (mm) | 3.4 – 3.8 |
| R7 | Radius of ring 7 (mm) | 4.0 – 4.4 |
| R8 | Radius of ring 8 (mm) | 4.8 – 5.6 |
| d  | Substrate thickness (mm) | 0.45 – 0.65 |
| Wf | Feed line width (mm) | 1.0 – 1.6 |

**Output targets (6 labels):**

| Parameter | Description |
|-----------|-------------|
| F1, F2, F3 | Resonance frequencies at three bands (GHz) |
| BW1, BW2, BW3 | Bandwidths at three resonance bands (GHz) |

The targets are computed using linear antenna design equations with a small Gaussian noise term (σ = 0.001) to simulate real-world measurement variation.

---

### Step 2 — Model Training & Evaluation (`Train_predict.py`)

The dataset is split into **80% training (480 samples)** and **20% testing (120 samples)**. Six ML models are trained using `MultiOutputRegressor` to simultaneously predict all 6 output targets.

**Models trained and their average R² scores:**

| Rank | Model | Avg R² Score |
|------|-------|-------------|
| 1 | Linear Regression | 0.9776 |
| 2 | CatBoost          | 0.9697 |
| 3 | Gradient Boosting | 0.9686 |
| 4 | Extra Trees       | 0.9676 |
| 5 | Random Forest     | 0.9597 |
| 6 | Decision Tree     | 0.8926 |

**Linear Regression achieves the highest accuracy (97.76%)** because the underlying antenna relationships in the dataset are linear by design. After training, the script enters an interactive prediction mode that prompts the user to enter 10 antenna geometry values and returns the predicted F1, F2, F3, BW1, BW2, BW3 values using the best model.


## Files in This Project

| File | Description |
|------|-------------|
| `antenna_dataset.csv` | The dataset (600 rows × 16 columns) |
| `Train_predict.py` | Trains all 6 ML models, prints R² comparison, and accepts manual input for prediction |
| `graphs.py` | Generates and saves all 4 analysis figures as PNG files |
| `fig1_histograms.png` | Dataset parameter distributions with KDE curves |
| `fig2_error_accuracy.png` | Error and accuracy comparison across models |
| `fig3_freq_actual_vs_predicted.png` | Actual vs. predicted frequency scatter plots |
| `fig4_bw_actual_vs_predicted.png` | Actual vs. predicted bandwidth scatter plots |

---

## How to Run

```bash
# Step 1: Generate the dataset
python generate_dataset.py

# Step 2: Train models and make a prediction
python Train_predict.py

# Step 3: Generate all graphs
python graphs.py
```

---

## Dependencies

```
pandas
numpy
scikit-learn
catboost
scipy
matplotlib
```

Install all dependencies with:
```bash
pip install pandas numpy scikit-learn catboost scipy matplotlib
```

---

## Results Summary

The study demonstrates that **Linear Regression achieves the highest overall R² of 0.9776** (97.76%) for predicting all six antenna output parameters simultaneously. This is because the dataset relationships are fundamentally linear. Among non-linear models, **CatBoost (R² = 0.9697)** and **Gradient Boosting (R² = 0.9686)** perform best, confirming that ensemble gradient boosting methods are the most suitable for this type of antenna parameter regression task when the true underlying relationships are not known in advance.

---

*Project by Birijithroshan*

