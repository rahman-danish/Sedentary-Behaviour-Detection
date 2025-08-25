# lstm_sedentary.py
# LSTM for time-series classification (e.g., Fitbit minute-level -> day-level label)

import os, json, math, datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -----------------------------
# Config
# -----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

RESULTS_DIR = "results_lstm"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# (A) Provide X, y as sequences
# -----------------------------
def load_sequences():
    """
    Return:
        X: np.ndarray (n_samples, seq_len, n_features)
        y: np.ndarray (n_samples,)
        feature_names: list[str]
    Replace this stub with your actual loader that builds sequences.
    """
    # ---- EXAMPLE (delete after wiring your data) ----
    # Synthetic example: 300 days, each up to 1440 minutes, features=[steps, hr]
    n_samples = 300
    seq_len = 1440        # pad/trim to 1440 minutes per day
    n_features = 2        # e.g., steps, heart rate
    # Create random data and simple labels (just for a runnable example)
    X = np.random.rand(n_samples, seq_len, n_features).astype("float32")
    # Make class imbalance similar to your dataset (~90% class 1)
    y = np.zeros(n_samples, dtype=int)
    y[int(n_samples*0.08):] = 1
    feature_names = ["steps", "heart_rate"]
    return X, y, feature_names

# -----------------------------
# (B) Helper: build sequences from minute-level tables (optional)
# -----------------------------
def make_sequences_from_minutes(df_minutes, id_col, ts_col, feature_cols,
                                day_col=None, pad_to=1440):
    """
    df_minutes: DataFrame with one row per minute and features (e.g., steps/hr).
    id_col: identifier for a person/session if multiple; if single user, pass a constant.
    ts_col: timestamp column (datetime64)
    feature_cols: list of feature column names to include in sequence
    day_col: precomputed 'date' column; if None, uses ts_col.dt.date
    pad_to: target sequence length (minutes/day)

    Returns X (n_days, pad_to, n_features), y placeholder (zeros), and feature_names.
    You should replace y with your true labels by joining on date later.
    """
    df = df_minutes.copy()
    if day_col is None:
        df["date"] = pd.to_datetime(df[ts_col]).dt.date
    else:
        df["date"] = df[day_col]

    # Ensure per-day chronological order
    df = df.sort_values([id_col, "date", ts_col])

    X_list = []
    dates = []
    for d, grp in df.groupby(["date"]):
        arr = grp[feature_cols].to_numpy(dtype="float32")
        # trim or pad
        if arr.shape[0] >= pad_to:
            arr = arr[:pad_to]
        else:
            pad = np.zeros((pad_to - arr.shape[0], len(feature_cols)), dtype="float32")
            arr = np.vstack([arr, pad])
        X_list.append(arr)
        dates.append(d)

    X = np.stack(X_list, axis=0)
    y = np.zeros(len(dates), dtype=int)  # placeholder; map your true labels here
    return X, y, feature_cols, dates

# -----------------------------
# Build the model
# -----------------------------
def build_lstm(input_shape, lstm_units=64, dropout=0.3, recurrent_dropout=0.0):
    """
    input_shape: (seq_len, n_features)
    Masking assumes padded timesteps are zeros.
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(lstm_units, return_sequences=False,
                    dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")])
    return model

# -----------------------------
# Train/Evaluate
# -----------------------------
def main():
    # Load data
    X, y, feature_names = load_sequences()
    seq_len, n_features = X.shape[1], X.shape[2]

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    # Class weights (help with imbalance)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced",
                                         classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, class_weights)}

    # Model
    model = build_lstm(input_shape=(seq_len, n_features), lstm_units=64)

    # Callbacks
    run_name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(RESULTS_DIR, f"best_lstm_{run_name}.keras")
    log_dir = os.path.join(RESULTS_DIR, "tensorboard_logs", f"lstm_{run_name}")
    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_roc_auc",
                                  mode="max", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_roc_auc", mode="max",
                                patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=4, verbose=1),
        callbacks.TensorBoard(log_dir=log_dir)
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        class_weight=class_weight,
        callbacks=cbs,
        verbose=2
    )

    # Evaluate
    test_probs = model.predict(X_test).ravel()
    test_preds = (test_probs >= 0.5).astype(int)

    report = classification_report(y_test, test_preds, output_dict=True, digits=4)
    auc = roc_auc_score(y_test, test_probs)

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, f"lstm_metrics_{run_name}.json")
    with open(metrics_path, "w") as f:
        json.dump({"classification_report": report, "roc_auc": auc}, f, indent=2)

    # Plots
    # ROC
    RocCurveDisplay.from_predictions(y_test, test_probs)
    plt.title("LSTM ROC Curve")
    roc_path = os.path.join(RESULTS_DIR, f"lstm_roc_{run_name}.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    ConfusionMatrixDisplay(cm).plot(values_format="d")
    plt.title("LSTM Confusion Matrix")
    cm_path = os.path.join(RESULTS_DIR, f"lstm_cm_{run_name}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    # History curves
    def plot_history(hist, key, val_key=None, title=None, fname=None):
        val_key = val_key or f"val_{key}"
        plt.figure()
        plt.plot(hist.history[key], label=key)
        if val_key in hist.history:
            plt.plot(hist.history[val_key], label=val_key)
        plt.xlabel("Epoch"); plt.ylabel(key); 
        if title: plt.title(title)
        plt.legend()
        if fname:
            plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()

    plot_history(history, "loss", title="Loss", fname=f"lstm_loss_{run_name}.png")
    plot_history(history, "accuracy", title="Accuracy", fname=f"lstm_acc_{run_name}.png")
    if "roc_auc" in history.history:
        plot_history(history, "roc_auc", title="ROC AUC", fname=f"lstm_auc_{run_name}.png")

    # Save final model
    final_model_path = os.path.join(RESULTS_DIR, f"lstm_final_{run_name}.keras")
    model.save(final_model_path)

    # Print brief summary
    print("Saved:")
    print(f"  Best model:   {ckpt_path}")
    print(f"  Final model:  {final_model_path}")
    print(f"  Metrics JSON: {metrics_path}")
    print(f"  ROC plot:     {roc_path}")
    print(f"  CM plot:      {cm_path}")
    print(f"Test ROC-AUC: {auc:.4f}")
    print(pd.DataFrame(report).T)

if __name__ == "__main__":
    main()
