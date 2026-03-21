import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error,
                             mean_squared_error, explained_variance_score)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor

# ── Load & split ──────────────────────────────────────────────────────────────
data = pd.read_csv("antenna_dataset.csv")

all_cols = ['R1','R2','R3','R4','R5','R6','R7','R8','d','Wf',
            'F1','F2','F3','BW1','BW2','BW3']

X = data[['R1','R2','R3','R4','R5','R6','R7','R8','d','Wf']]
y = data[['F1','F2','F3','BW1','BW2','BW3']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test_np = y_test.values

# ── Train all 6 models ────────────────────────────────────────────────────────
model_defs = [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree",     DecisionTreeRegressor()),
    ("Gradient Boosting", GradientBoostingRegressor()),
    ("Extra Trees",       ExtraTreesRegressor(n_estimators=200, random_state=42)),
    ("CatBoost",          CatBoostRegressor(verbose=0, random_seed=42)),
    ("Random Forest",     RandomForestRegressor(n_estimators=200, random_state=42)),
]

print("Training models...")
predictions = {}
for name, m in model_defs:
    mo = MultiOutputRegressor(m)
    mo.fit(X_train, y_train)
    predictions[name] = mo.predict(X_test)
    print(f"  {name} — done")

freq_idx = [0, 1, 2]   # F1, F2, F3
bw_idx   = [3, 4, 5]   # BW1, BW2, BW3

# ── Compute per-model metrics ─────────────────────────────────────────────────
def compute_metrics(pred, indices):
    yt = y_test_np[:, indices]
    yp = pred[:, indices]
    mae  = mean_absolute_error(yt, yp)
    mse  = mean_squared_error(yt, yp)
    rmse = np.sqrt(mse)
    r2   = r2_score(yt, yp)
    evs  = explained_variance_score(yt, yp)
    return mae, mse, rmse, r2, evs

metrics = {}
for name, _ in model_defs:
    metrics[name] = {
        'freq': compute_metrics(predictions[name], freq_idx),
        'bw':   compute_metrics(predictions[name], bw_idx),
    }

model_names = [n for n, _ in model_defs]
short_names  = ["LR", "DT", "GB", "ET", "CB", "RF"]

# ════════════════════════════════════════════════════════════════════════════════
# FIG 1 – Histograms of all 16 dataset parameters  (reference-image style)
# ════════════════════════════════════════════════════════════════════════════════
BAR_COLOR   = '#8eb9d9'   # steel-blue bars (matches reference)
EDGE_COLOR  = '#5a8fad'   # slightly darker bar edge
KDE_COLOR   = "#DEE4EB"   # dark blue KDE curve (matches reference overlay)
GRID_COLOR  = '#d4d4d4'   # light gray horizontal grid lines
BINS        = 25

fig1, axes1 = plt.subplots(4, 4, figsize=(20, 16))
fig1.patch.set_facecolor('white')
fig1.suptitle("Fig 1: K/Ka-Band Antenna Dataset Parameter Distributions",
              fontsize=16, fontweight='bold', y=1.005)

for idx, col in enumerate(all_cols):
    ax = axes1[idx // 4][idx % 4]
    ax.set_facecolor('white')

    values = data[col].dropna().values

    # Draw histogram (count on y-axis)
    counts, bin_edges, _ = ax.hist(
        values, bins=BINS,
        color=BAR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.6
    )

    # Overlay KDE curve scaled to match count axis
    bin_width = bin_edges[1] - bin_edges[0]
    kde = gaussian_kde(values, bw_method='scott')
    x_range = np.linspace(values.min() - 0.5 * bin_width,
                          values.max() + 0.5 * bin_width, 300)
    kde_scaled = kde(x_range) * len(values) * bin_width
    ax.plot(x_range, kde_scaled,
            color=KDE_COLOR, linewidth=2.0, zorder=4)

    ax.set_title(col, fontsize=12, fontweight='bold', pad=5, color='#222222')
    ax.set_xlabel('Value', fontsize=9, color='#444444')
    ax.set_ylabel('Count',  fontsize=9, color='#444444')

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.tick_params(axis='both', labelsize=8, colors='#444444', length=3)

    # Horizontal gray grid lines (matches reference)
    ax.yaxis.grid(True,  color=GRID_COLOR, linestyle='-', linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # All 4 spines visible (box style, matches reference)
    for spine in ax.spines.values():
        spine.set_edgecolor('#aaaaaa')
        spine.set_linewidth(0.8)
        spine.set_visible(True)

plt.tight_layout(pad=1.8, h_pad=2.5, w_pad=1.8)
plt.savefig('fig1_histograms.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print("Fig 1 ready.")

# ════════════════════════════════════════════════════════════════════════════════
# FIG 2 – ML Error and Accuracy Analysis (2 × 2 subplots)
# ════════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(18, 13))
fig2.suptitle("Fig 2: ML Error and Accuracy Analysis (K/Ka Resonance + Bandwidth)",
              fontsize=16, fontweight='bold')

x = np.arange(len(model_names))
w = 0.26   # bar width

panels = [
    ('a', 'Frequency', 'freq', 'error'),
    ('b', 'Frequency', 'freq', 'accuracy'),
    ('c', 'Bandwidth', 'bw',   'error'),
    ('d', 'Bandwidth', 'bw',   'accuracy'),
]

for (label, domain, key, kind), ax in zip(panels, axes2.flatten()):
    mae_v  = [metrics[n][key][0] for n in model_names]
    mse_v  = [metrics[n][key][1] for n in model_names]
    rmse_v = [metrics[n][key][2] for n in model_names]
    r2_v   = [metrics[n][key][3] * 100 for n in model_names]
    evs_v  = [metrics[n][key][4] * 100 for n in model_names]

    if kind == 'error':
        b1 = ax.bar(x - w,  mae_v,  w, label='MAE',  color='#EF5350', edgecolor='white', zorder=3)
        b2 = ax.bar(x,      mse_v,  w, label='MSE',  color='#FFA726', edgecolor='white', zorder=3)
        b3 = ax.bar(x + w,  rmse_v, w, label='RMSE', color='#AB47BC', edgecolor='white', zorder=3)
        ax.set_ylabel('Value', fontsize=11)
        for bar in [b1, b2, b3]:
            for rect in bar:
                h = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2, h + ax.get_ylim()[1]*0.005,
                        f'{h:.5f}', ha='center', va='bottom', fontsize=6.5, rotation=90)
        ax.set_title(f'({label}) {domain} – Error Metrics (MAE, MSE, RMSE)',
                     fontsize=12, fontweight='bold')

    else:
        b1 = ax.bar(x - w/2, r2_v,  w, label='R²',                color='#42A5F5', edgecolor='white', zorder=3)
        b2 = ax.bar(x + w/2, evs_v, w, label='Explained Variance', color='#26C6DA', edgecolor='white', zorder=3)
        for bar in [b1, b2]:
            for rect in bar:
                h = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2, h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_ylim(0, 115)
        ax.set_title(f'({label}) {domain} – Accuracy (R², Explained Variance)',
                     fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=12)
    ax.set_xlabel('Models', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.35, linestyle='--', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=2.5)
plt.savefig('fig2_error_accuracy.png', dpi=150, bbox_inches='tight')
print("Fig 2 ready.")

# ════════════════════════════════════════════════════════════════════════════════
# FIG 3 – Actual vs Predicted: Resonance Frequencies
# ════════════════════════════════════════════════════════════════════════════════
subplot_order  = ["Linear Regression", "Decision Tree", "Gradient Boosting",
                  "Extra Trees",       "CatBoost",      "Random Forest"]
subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']
display_names  = ["Linear Regression", "Decision Tree", "Gradient Boosting",
                  "Extra Tree",        "CatBoost",      "Random Forest"]

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
fig3.suptitle("Fig 3: Actual vs. Predicted Resonance Frequencies (K/Ka Bands)",
              fontsize=15, fontweight='bold')

for idx, (key, ax) in enumerate(zip(subplot_order, axes3.flatten())):
    pred   = predictions[key]
    actual = y_test_np[:, freq_idx].flatten()
    preds  = pred[:, freq_idx].flatten()
    r2_f   = r2_score(y_test_np[:, freq_idx], pred[:, freq_idx])

    lo = min(actual.min(), preds.min()) - 0.05
    hi = max(actual.max(), preds.max()) + 0.05

    ax.scatter(actual, preds, color='#1565C0', s=18, alpha=0.65,
               label='• Test Set', zorder=3)
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.8,
            label='–– Ideal Prediction', zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(f'({subplot_labels[idx]}) {display_names[idx]}\nR² = {r2_f:.4f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual Values (GHz)', fontsize=10)
    ax.set_ylabel('Predicted Values (GHz)', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=2.5)
plt.savefig('fig3_freq_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
print("Fig 3 ready.")

# ════════════════════════════════════════════════════════════════════════════════
# FIG 4 – Actual vs Predicted: Bandwidths
# ════════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(2, 3, figsize=(18, 12))
fig4.suptitle("Fig 4: Actual vs. Predicted Bandwidths Using Machine Learning Models",
              fontsize=15, fontweight='bold')

for idx, (key, ax) in enumerate(zip(subplot_order, axes4.flatten())):
    pred   = predictions[key]
    actual = y_test_np[:, bw_idx].flatten()
    preds  = pred[:, bw_idx].flatten()
    r2_bw  = r2_score(y_test_np[:, bw_idx], pred[:, bw_idx])

    lo = min(actual.min(), preds.min()) - 0.002
    hi = max(actual.max(), preds.max()) + 0.002

    ax.scatter(actual, preds, color='#B71C1C', s=18, alpha=0.65,
               label='• Test Set', zorder=3)
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.8,
            label='–– Ideal Prediction', zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(f'({subplot_labels[idx]}) {display_names[idx]}\nR² = {r2_bw:.4f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual Values (GHz)', fontsize=10)
    ax.set_ylabel('Predicted Values (GHz)', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=2.5)
plt.savefig('fig4_bw_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
print("Fig 4 ready.")

# ── Show all figures ──────────────────────────────────────────────────────────
print("\nAll figures saved as PNG. Displaying now...")
if plt.get_backend().lower() != 'agg':
    plt.show()
