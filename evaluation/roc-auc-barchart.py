import glob, os, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# adjust to your project; must return X, y (0/1)
from loader import load_data

def _score_for_auc(model, X):
    """Return a 1D score for ROC-AUC (probabilities preferred)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return np.ravel(model.decision_function(X))
    # Worst-case fallback
    return np.ravel(model.predict(X)).astype(float)

def compute_ml_aucs(model_files, X_test, y_test):
    aucs = {}
    for path in model_files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            model = joblib.load(path)
            y_score = _score_for_auc(model, X_test)
            aucs[name] = roc_auc_score(y_test, y_score)
        except Exception as e:
            print(f"[Skip] {name}: {e}")
    return aucs

def plot_auc_bars(auc_dict, title="ROC-AUC (higher is better)", out_path="results/roc_auc_bar.png"):
    # DataFrame for sorting + plotting
    df = pd.DataFrame({"Model": list(auc_dict.keys()), "ROC-AUC": list(auc_dict.values())})
    df = df.sort_values("ROC-AUC", ascending=True)  # for horizontal bars

    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(df["Model"], df["ROC-AUC"])
    for i, v in enumerate(df["ROC-AUC"]):
        plt.text(v + 0.01, i, f"{v:.2f}", va="center")  # label at end of each bar

    plt.xlim(0.0, 1.05)
    plt.xlabel("ROC-AUC")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.show()
    print(f"[Saved] {out_path}")

def main():
    # 1) Same split that used before
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2) Pick which ML model pickles to include (explicit list is safest)
    model_files = [
        "logistic_model.pkl",
        "knn_model.pkl",
        "rf_model.pkl",
        "svm_model.pkl",
        "xgb_model.pkl",
    ]
   
    # 3) Compute ML AUCs
    ml_aucs = compute_ml_aucs(model_files, X_test, y_test)

    # DL AUCs

    dl_aucs = {
         "Dense NN (TF)": 0.6541,
         "LSTM": 0.35,
    }
    all_aucs = {**ml_aucs, **dl_aucs}

    # 5) Plot
    plot_auc_bars(all_aucs, title="ROC-AUC Comparison (ML + DL)")

if __name__ == "__main__":
    main()
