import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import shap
import tensorflow as tf  # ok to keep even if not used directly (Keras models)


def _predict_vec(model):
    """Return a function f(X)->(n_rows,) so SHAP never gets a scalar."""
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)[:, 1]  # prob of positive class
    if hasattr(model, "decision_function"):
        return lambda X: np.ravel(model.decision_function(X))
    return lambda X: np.ravel(model.predict(X)).astype(float)

def shap_explain_ml(model, X_train_df, X_sample_df):
    """
    Build a robust SHAP explainer:
    - ensures 2D DataFrames
    - uses an independent masker (small background)
    - returns shap.Explanation
    """
    import shap
    # Ensure DataFrames (keep same column order)
    if not isinstance(X_train_df, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train_df, columns=X_sample_df.columns)
    if not isinstance(X_sample_df, pd.DataFrame):
        X_sample_df = pd.DataFrame(X_sample_df, columns=X_train_df.columns)

    # Small background for speed/robustness
    bg = X_train_df.sample(min(100, len(X_train_df)), random_state=0)
    masker = shap.maskers.Independent(bg)

    f = _predict_vec(model)          # always returns (n_rows,)
    explainer = shap.Explainer(f, masker)
    return explainer(X_sample_df)

def shap_explain_dl(model, X_train_df, X_sample_df):
    """
    SHAP for Keras models (tabular). Uses a small independent masker
    and a model-agnostic explainer.
    """
    import shap, numpy as np

    # make sure we keep feature names
    if not isinstance(X_train_df, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train_df, columns=required_features)
    if not isinstance(X_sample_df, pd.DataFrame):
        X_sample_df = pd.DataFrame(X_sample_df, columns=X_train_df.columns)

    bg = X_train_df.sample(min(100, len(X_train_df)), random_state=0)
    masker = shap.maskers.Independent(bg)

    f = lambda X: np.ravel(model.predict(np.asarray(X), verbose=0))
    explainer = shap.Explainer(f, masker)
    return explainer(X_sample_df)

st.set_page_config(page_title="Smart Sedentary Behaviour App", layout="wide")
st.title("🤔 Sedentary Behaviour Detection APP")
st.write("Upload any Fitbit CSV file(s), and this app will automatically detect your activity level.")

# -----------------------------
# Expected features (daily)
# -----------------------------
required_features = [
    'TotalSteps', 'TotalDistance', 'TrackerDistance', 'LoggedActivitiesDistance',
    'VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance',
    'SedentaryActiveDistance', 'VeryActiveMinutes', 'FairlyActiveMinutes',
    'LightlyActiveMinutes', 'Calories', 'TotalSleepRecords',
    'TotalMinutesAsleep', 'TotalTimeInBed', 'AvgHeartRate', 'AvgStepsPerMinute'
]

# -----------------------------
# Upload
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload one or more Fitbit CSV files (activity, sleep, heart rate, steps, etc.):",
    type="csv",
    accept_multiple_files=True
)

# -----------------------------
# Helpers
# -----------------------------
def extract_avg_heartrate(df):
    if 'Value' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Time'])
        df['Date'] = df['Datetime'].dt.date
        return df.groupby('Date')['Value'].mean().reset_index(name='AvgHeartRate')
    return pd.DataFrame()

def extract_avg_steps(df):
    if 'Steps' in df.columns:
        df['Datetime'] = pd.to_datetime(df['ActivityMinute'])
        df['Date'] = df['Datetime'].dt.date
        return df.groupby('Date')['Steps'].mean().reset_index(name='AvgStepsPerMinute')
    elif 'StepTotal' in df.columns:
        df['Datetime'] = pd.to_datetime(df['ActivityMinute'])
        df['Date'] = df['Datetime'].dt.date
        return df.groupby('Date')['StepTotal'].mean().reset_index(name='AvgStepsPerMinute')
    return pd.DataFrame()

def process_sleep(df):
    if 'date' in df.columns and 'time' in df.columns and 'value' in df.columns:
        df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df['Date'] = df['Datetime'].dt.date
        summary = df.groupby('Date').agg({'value': ['count', 'sum']})
        summary.columns = ['TotalSleepRecords', 'TotalMinutesAsleep']
        summary.reset_index(inplace=True)
        return summary
    return pd.DataFrame()

def build_flexible_dataframe(files):
    base_df = None
    temp_frames = []

    for file in files:
        df = pd.read_csv(file)
        cols = df.columns

        if 'ActivityDate' in cols and 'Calories' in cols:
            df['ActivityDate'] = pd.to_datetime(df['ActivityDate'], errors='coerce').dt.date
            if base_df is None:
                base_df = df
            else:
                base_df = pd.merge(base_df, df, on='ActivityDate', how='outer')

        elif 'date' in cols and 'time' in cols and 'value' in cols:
            sleep_df = process_sleep(df)
            if not sleep_df.empty:
                temp_frames.append(sleep_df)

        elif 'Time' in cols and 'Value' in cols:
            hr_df = extract_avg_heartrate(df)
            if not hr_df.empty:
                temp_frames.append(hr_df)

        elif 'ActivityMinute' in cols and ('Steps' in cols or 'StepTotal' in cols):
            steps_df = extract_avg_steps(df)
            if not steps_df.empty:
                temp_frames.append(steps_df)

    if base_df is None:
        if temp_frames:
            base_df = temp_frames[0]
            temp_frames = temp_frames[1:]
        else:
            return pd.DataFrame(), []

    for temp in temp_frames:
        key_left = 'ActivityDate' if 'ActivityDate' in base_df.columns else 'Date'
        key_right = 'Date' if 'Date' in temp.columns else 'ActivityDate'
        base_df = pd.merge(base_df, temp, left_on=key_left, right_on=key_right, how='outer')

    if 'Date' in base_df.columns and 'ActivityDate' not in base_df.columns:
        base_df.rename(columns={'Date': 'ActivityDate'}, inplace=True)

    # Ensure all required features exist (create then clean)
    for col in required_features:
        if col not in base_df.columns:
            base_df[col] = np.nan

    # ---- Make features numeric and NaN-free (this was causing SVM/KNN/LogReg errors) ----
    num_cols = [c for c in required_features if c in base_df.columns]
    base_df[num_cols] = base_df[num_cols].apply(pd.to_numeric, errors="coerce")
    base_df[num_cols] = base_df[num_cols].replace([np.inf, -np.inf], np.nan)
    # choose your imputation: median is usually safer than zeros
    med = base_df[num_cols].median(numeric_only=True)
    base_df[num_cols] = base_df[num_cols].fillna(med).fillna(0.0)

    actual_features = num_cols
    cols_out = actual_features + (['ActivityDate'] if 'ActivityDate' in base_df.columns else [])
    return base_df[cols_out], actual_features

# -----------------------------
# Model loaders (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".ubj"):
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(path)
        return m
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_dl_model(path: str):
    import os, joblib, tensorflow as tf

    if not os.path.exists(path):
        return f"ERROR: File not found: {path}"

    ext = os.path.splitext(path)[1].lower()
    # Portable formats first
    if os.path.isdir(path) or ext in {".keras", ".h5"}:
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            return f"ERROR: Failed to load Keras model: {e}"

    # Legacy pickle (not recommended, but kept as fallback)
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        if "keras.src.saving.pickle_utils" in str(e):
            return ("ERROR: This DL model was pickled with Keras 3. "
                    "Use the converted .keras file instead.")
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"

# -----------------------------
# UI: Two sections (tabs)
# -----------------------------
ml_tab, dl_tab = st.tabs(["🧠 Machine Learning (choose one)", "🧪 Deep Learning (TensorFlow)"])

with ml_tab:
    st.subheader("Select an ML model")
    model_map = {
        "Random Forest": "rf_model.pkl",
        "SVM": "svm_model.pkl",
        "XGBoost": "xgb_sklearn.json",
        "KNN": "knn_model.pkl",
        "Logistic Regression": "logistic_model.pkl",
    }
    ml_choice = st.selectbox("Model", list(model_map.keys()))
    run_ml = st.button("🔢 Predict with ML", type="primary")

with dl_tab:
    st.subheader("TensorFlow (DL)")
    dl_path = st.text_input("DL model path", "F:/Dessertation/dataset/tensorflow_sedentary_model.keras"
)
    dl_threshold = st.slider("Sedentary threshold (DL)", 0.10, 0.90, 0.50, 0.01)
    run_dl = st.button("🔢 Predict with DL", type="primary")

# -----------------------------
# Prediction handlers
# -----------------------------
def render_summary_chart(verdict_series, title="📊 Summary Chart"):
    st.subheader(title)
    chart_data = verdict_series.value_counts()
    fig, ax = plt.subplots()
    ax.pie(chart_data, labels=chart_data.index, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig)

def render_single_day_block(df, engine_label, extra_cols=None):
    st.subheader(f"📆 Prediction for Single Day ({engine_label})")
    if 'ActivityDate' in df.columns:
        st.markdown(f"**Date:** `{df['ActivityDate'].iloc[0]}`")
    st.markdown(f"**Status:** {df['Verdict'].iloc[0]}")
    if 'Confidence' in df.columns:
        st.markdown(f"**Confidence:** `{df['Confidence'].iloc[0]:.2f}%`")
    if extra_cols:
        for label, value_fmt in extra_cols:
            st.markdown(f"**{label}:** `{value_fmt}`")

    st.subheader("📊 Feature Values")
    st.dataframe(df[required_features].T.rename(columns={df.index[0]: "Value"}))

# ---- Matplotlib style helpers for smaller SHAP plots ----
import matplotlib as mpl
import matplotlib.pyplot as plt

_SMALL_RC = {
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

class small_fonts:
    """Temporarily shrink Matplotlib fonts (restores after use)."""
    def __enter__(self):
        self._old = mpl.rcParams.copy()
        mpl.rcParams.update(_SMALL_RC)
    def __exit__(self, exc_type, exc, tb):
        mpl.rcParams.update(self._old)

# ---- ML run ----
if run_ml:
    if not uploaded_files:
        st.error("👇 Upload one or more Fitbit CSV files to begin.")
    else:
        df, actual_features = build_flexible_dataframe(uploaded_files)
        if df.empty:
            st.error("❌ Could not process uploaded files. Please ensure correct format.")
        else:
            model_path = model_map[ml_choice]
            ml_model = load_model(model_path)
            if isinstance(ml_model, str) and ml_model.startswith("ERROR"):
                st.error(f"❌ Failed to load '{model_path}': {ml_model}")
            else:
                try:
                    X = (
                            df[required_features]
                            .apply(pd.to_numeric, errors="coerce")        # coerce bad strings to NaN
                            .replace([np.inf, -np.inf], np.nan)           # guard against infs
                            .fillna(0.0)                                   # or .fillna(df.median()) if you prefer
                            .astype(float)
                        )

                    preds = ml_model.predict(X)
                    if hasattr(ml_model, "predict_proba"):
                        conf = ml_model.predict_proba(X).max(axis=1) * 100
                    else:
                        conf = np.full(len(preds), 100.0)
                except Exception as e:
                    st.error(f"ML prediction failed: {e}")
                    st.stop()

                out = df.copy()
                out['Prediction'] = preds
                out['Confidence'] = conf
                out['Verdict'] = out['Prediction'].apply(lambda x: '✅ Active' if x == 0 else '❌ Sedentary')

                if len(out) == 1:
                    render_single_day_block(out.iloc[[0]], f"ML - {ml_choice}")
                    st.subheader("🧠 Why This Prediction? (ML)")
                    try:
                        X_train_like = X
                        X_one = X.iloc[[0]]  # keep 2D
                        sv = shap_explain_ml(ml_model, X_train_like, X_one)

                        with small_fonts():
                            fig = plt.figure(figsize=(6, 3.8))             # smaller canvas
                            shap.plots.waterfall(sv[0], show=False)        # draw on current fig
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)
                    except Exception as e:
                        st.info(f"SHAP explanation unavailable: {e}")

                else:
                    st.subheader("📅 Daily Predictions (ML)")
                    st.dataframe(out[['ActivityDate', 'Verdict', 'Confidence']])
                    render_summary_chart(out['Verdict'])

                    st.subheader("🧠 Why This Prediction? (ML)")
                    try:
                        batch = X.iloc[:min(50, len(X))].copy()
                        sv = shap_explain_ml(ml_model, X, batch)

                        with small_fonts():
                            # Global importance bar
                            fig = plt.figure(figsize=(6, 3.8))
                            shap.plots.bar(sv, max_display=10, show=False)
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)

                            # Optional: beeswarm (also smaller)
                            fig = plt.figure(figsize=(6, 4.2))
                            shap.plots.beeswarm(sv, max_display=15, show=False)
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)
                    except Exception as e:
                        st.info(f"SHAP explanation unavailable: {e}")


# ---- DL run ----
# ---- DL run ----
if run_dl:
    # 1) make sure the path exists before loading
    if not os.path.exists(dl_path):
        st.error(f"File not found: {dl_path}")
        st.stop()

    if not uploaded_files:
        st.error("👇 Upload one or more Fitbit CSV files to begin.")
    else:
        df, actual_features = build_flexible_dataframe(uploaded_files)
        if df.empty:
            st.error("❌ Could not process uploaded files. Please ensure correct format.")
        else:
            dl_model = load_dl_model(dl_path)   # safe to call now
            if isinstance(dl_model, str) and dl_model.startswith("ERROR"):
                st.error(f"❌ Failed to load DL model '{dl_path}': {dl_model}")
                st.stop()
            else:
                try:
                    # Build clean DF with feature names
                    X_df = (
                        df[required_features]
                        .apply(pd.to_numeric, errors="coerce")
                        .replace([np.inf, -np.inf], np.nan)
                    )
                    # simple impute (or keep your median logic if you prefer)
                    X_df = X_df.fillna(0.0)

                    # If you saved a StandardScaler, apply it (recommended)
                    scaler_path = "results/scaler.joblib"
                    if os.path.exists(scaler_path):
                        scaler = joblib.load(scaler_path)
                        X_scaled = pd.DataFrame(
                            scaler.transform(X_df), columns=X_df.columns, index=X_df.index
                        )
                    else:
                        X_scaled = X_df

                    preds = dl_model.predict(X_scaled.to_numpy(), verbose=0) if hasattr(dl_model, "predict") else None
                    if preds is None:
                        raise ValueError("Loaded DL object has no .predict method.")
                except Exception as e:
                    st.error(f"DL prediction failed: {e}")
                    st.stop()

                # Convert to sedentary probability
                preds = np.asarray(preds)
                if preds.ndim == 2 and preds.shape[1] == 1:
                    prob_sed = preds[:, 0]
                elif preds.ndim == 2 and preds.shape[1] == 2:
                    # assume column 1 = sedentary (change if your label order differs)
                    prob_sed = preds[:, 1]
                else:
                    st.error(f"Unexpected DL output shape: {preds.shape}. Expect (B,1) or (B,2).")
                    st.stop()

                out = df.copy()
                out['Prob_Sedentary'] = prob_sed
                out['Prediction'] = (out['Prob_Sedentary'] >= dl_threshold).astype(int)  # 1=sedentary
                out['Confidence'] = (np.maximum(out['Prob_Sedentary'], 1 - out['Prob_Sedentary']) * 100).round(2)
                out['Verdict'] = out['Prediction'].apply(lambda x: '✅ Active' if x == 0 else '❌ Sedentary')

                if len(out) == 1:
                    # Single-day DL block with extras
                    extra = [
                        ("P(Sedentary)", f"{out['Prob_Sedentary'].iloc[0]:.3f}")
                    ]
                    render_single_day_block(out.iloc[[0]], "DL - TensorFlow", extra_cols=extra)
                    st.subheader("🧠 Why This Prediction? (DL)")
                    try:
                        X_one = X_scaled.iloc[[0]]             # keep 2D
                        sv = shap_explain_dl(dl_model, X_scaled, X_one)
                        with small_fonts():
                            fig = plt.figure(figsize=(6, 3.8))
                            shap.plots.waterfall(sv[0], show=False)
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)
                    except Exception as e:
                        st.info(f"DL SHAP unavailable: {e}")

                else:
                    st.subheader("📅 Daily Predictions (DL)")
                    st.dataframe(out[['ActivityDate', 'Verdict', 'Prob_Sedentary', 'Confidence']])
                    render_summary_chart(out['Verdict'])

                    st.subheader("🧠 Why This Prediction? (DL)")
                    try:
                        batch = X_scaled.iloc[:min(50, len(X_scaled))].copy()
                        sv = shap_explain_dl(dl_model, X_scaled, batch)
                        with small_fonts():
                            fig = plt.figure(figsize=(6, 3.8))
                            shap.plots.bar(sv, max_display=10, show=False)
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)

                            # optional beeswarm
                            fig = plt.figure(figsize=(6, 4.2))
                            shap.plots.beeswarm(sv, max_display=15, show=False)
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)
                    except Exception as e:
                        st.info(f"DL SHAP unavailable: {e}")


# -----------------------------
# Empty states
# -----------------------------
if not uploaded_files and not (('run_ml' in locals() and run_ml) or ('run_dl' in locals() and run_dl)):
    st.info("👇 Upload one or more Fitbit CSV files to begin.")
