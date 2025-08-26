# make_model_comparisons.py
# Creates charts + a PDF report comparing your 6 models.
# Uses only matplotlib (non-interactive) + pandas.

import os
os.environ["MPLBACKEND"] = "Agg"   # non-interactive
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------
# 1) Put your metrics here
# -----------------------------
data = [
    # name, accuracy, macro_f1, weighted_f1, c0_precision, c0_recall, c0_f1, c1_f1, cv_mean, cv_std, roc_auc
    ("XGBoost",            0.95, 0.84, 0.95, 0.83, 0.60, 0.70, 0.97, 0.8279, 0.0318, np.nan),
    ("Random Forest",      0.94, 0.79, 0.94, 0.76, 0.52, 0.62, 0.97, 0.7722, 0.0277, np.nan),
    ("KNN (scaled)",       0.94, 0.72, 0.92, 0.69, 0.36, 0.47, 0.96, 0.7030, 0.0260, np.nan),
    ("SVM (scaled)",       0.94, 0.70, 0.92, 1.00, 0.28, 0.44, 0.97, 0.7754, 0.0303, np.nan),
    ("LogReg (scaled)",    0.82, 0.66, 0.85, 0.30, 0.72, 0.42, 0.90, 0.6156, 0.0237, np.nan),
    ("TensorFlow (DL)",    0.9286, 0.70, 0.92, 0.73, 0.32, 0.44, 0.96, np.nan, np.nan, 0.6541),
    ("LSTM (DL)", 0.6222, 0.4720, 0.7036, 0.12, 0.50, 0.19, 0.75, np.nan, np.nan, 0.3476),
]

cols = ["Model","Accuracy","MacroF1","WeightedF1",
        "Class0_Precision","Class0_Recall","Class0_F1","Class1_F1",
        "CV_Mean_F1","CV_Std","ROC_AUC"]

df = pd.DataFrame(data, columns=cols)

# Optional: sort by MacroF1 (comment out if you want original order)
df = df.sort_values("MacroF1", ascending=False).reset_index(drop=True)

# -----------------------------
# 2) Output directory
# -----------------------------
OUTDIR = "model_viz_outputs"
os.makedirs(OUTDIR, exist_ok=True)
df.to_csv(os.path.join(OUTDIR, "model_metrics_table.csv"), index=False)

# -----------------------------
# 3) Helpers
# -----------------------------
def annotate_bars(ax, fmt="{:.2f}", size=9, rotation=0, offset=3):
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h): 
            continue
        ax.annotate(fmt.format(h),
                    (p.get_x() + p.get_width()/2., h),
                    ha='center', va='bottom', fontsize=size, rotation=rotation,
                    xytext=(0, offset), textcoords='offset points')

def save_bar(fig_name, x, y, title, ylabel):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=20, ha='right')
    annotate_bars(ax)
    fig.tight_layout()
    path = os.path.join(OUTDIR, fig_name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, path

def save_grouped_bar(fig_name, x, ys, labels, title, ylabel):
    # ys: list of arrays/series aligned with x
    n = len(x); g = len(ys)
    idx = np.arange(n)
    width = 0.8/g
    fig, ax = plt.subplots(figsize=(9,5))
    for i, y in enumerate(ys):
        ax.bar(idx + i*width, y, width, label=labels[i])
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.set_xticks(idx + (g-1)*width/2)
    ax.set_xticklabels(x, rotation=20, ha='right')
    ax.legend()
    # annotate each group
    for i, y in enumerate(ys):
        for j, val in enumerate(y):
            if np.isnan(val): continue
            ax.annotate(f"{val:.2f}", (idx[j] + i*width, val),
                        ha='center', va='bottom', fontsize=8, xytext=(0,3),
                        textcoords='offset points')
    fig.tight_layout()
    path = os.path.join(OUTDIR, fig_name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, path

def save_errorbar(fig_name, x, mean, std, title, ylabel):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.errorbar(x, mean, yerr=std, fmt='o-', capsize=4)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=20, ha='right')
    for i, m in enumerate(mean):
        if np.isnan(m): continue
        ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points",
                    xytext=(0,5), ha='center', fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUTDIR, fig_name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, path

def save_radar(fig_name, row, metrics, title):
    # row: pd.Series for a single model
    # metrics: list of metric column names
    values = row[metrics].values.astype(float)
    labels = metrics
    # Normalize to [0,1] for nicer radar (optional)
    # Here we just plot raw values.
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    values = np.concatenate([values, values[:1]])
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, marker='o')
    ax.fill(angles, values, alpha=0.15)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
    ax.set_title(title, y=1.08)
    ax.set_rlim(0, 1)  # y-axis from 0..1 (fits your metric ranges)
    fig.tight_layout()
    path = os.path.join(OUTDIR, fig_name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, path

# -----------------------------
# 4) Build figures
# -----------------------------
figs_paths = []

# (A) Topline grouped bars: Accuracy vs MacroF1 vs WeightedF1
xlabels = df["Model"].tolist()
fig, path = save_grouped_bar(
    "grouped_accuracy_macro_weighted.png",
    xlabels,
    [df["Accuracy"].values, df["MacroF1"].values, df["WeightedF1"].values],
    ["Accuracy", "Macro F1", "Weighted F1"],
    "Model Comparison: Accuracy vs MacroF1 vs WeightedF1",
    "Score"
)
figs_paths.append((fig, path))

# (B) Class-0 metrics
fig, path = save_grouped_bar(
    "class0_metrics.png",
    xlabels,
    [df["Class0_Precision"].values, df["Class0_Recall"].values, df["Class0_F1"].values],
    ["Class0 Precision", "Class0 Recall", "Class0 F1"],
    "Minority Class (Class-0) Metrics by Model",
    "Score"
)
figs_paths.append((fig, path))

# (C) Class-1 F1
fig, path = save_bar(
    "class1_f1.png",
    xlabels,
    df["Class1_F1"].values,
    "Class-1 F1 by Model",
    "F1"
)
figs_paths.append((fig, path))

# (D) CV mean ± std (only where available)
mask_cv = ~df["CV_Mean_F1"].isna()
x_cv = df.loc[mask_cv, "Model"].tolist()
mean_cv = df.loc[mask_cv, "CV_Mean_F1"].values
std_cv  = df.loc[mask_cv, "CV_Std"].values
fig, path = save_errorbar(
    "cv_f1_mean_std.png",
    x_cv, mean_cv, std_cv,
    "Cross-Validation Macro F1 (mean ± std)",
    "F1 (CV)"
)
figs_paths.append((fig, path))

# (E) ROC-AUC bars (only for those you have)
mask_roc = ~df["ROC_AUC"].isna()
if mask_roc.any():
    fig, path = save_bar(
        "roc_auc.png",
        df.loc[mask_roc,"Model"].tolist(),
        df.loc[mask_roc,"ROC_AUC"].values,
        "ROC-AUC by Model (where available)",
        "ROC-AUC"
    )
    figs_paths.append((fig, path))

# (F) Radar charts per model (Accuracy, MacroF1, Class0_Recall, Class0_Precision, Class0_F1)
radar_metrics = ["Accuracy","MacroF1","Class0_Recall","Class0_Precision","Class0_F1"]
for _, row in df.iterrows():
    fig, path = save_radar(
        f"radar_{row['Model'].replace(' ','_').replace('(','').replace(')','')}.png",
        row,
        radar_metrics,
        f"Radar: {row['Model']}"
    )
    figs_paths.append((fig, path))

# -----------------------------
# 5) One PDF with all pages
# -----------------------------
pdf_path = os.path.join(OUTDIR, "model_comparison_report.pdf")
with PdfPages(pdf_path) as pdf:
    for fig, _ in figs_paths:
        pdf.savefig(fig)
    # Add a table page
    fig_tbl, ax = plt.subplots(figsize=(10, 2 + 0.35*len(df)))
    ax.axis('off')
    table = ax.table(cellText=df.round(3).values,
                     colLabels=df.columns,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title("Model Metrics Table", pad=10)
    pdf.savefig(fig_tbl)
    plt.close(fig_tbl)

# Close figures (free memory)
for fig, _ in figs_paths:
    plt.close(fig)