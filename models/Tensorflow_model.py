import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
import sys
import numpy as np
import joblib
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 📌 Setup results and logs folder
import os, datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Where this script is
RESULTS_DIR = os.path.join(BASE_DIR, "results")        # Folder for outputs
LOGS_DIR = os.path.join(RESULTS_DIR, "tensorboard_logs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

from loader import load_data

# ✅ Load data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.joblib")

# ✅ TensorBoard + EarlyStopping
log_dir = os.path.join(LOGS_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ✅ Build model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ✅ Train model and store history
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tensorboard_callback, early_stop]
)

# ✅ Predict and evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy_val = accuracy_score(y_test, y_pred)
roc_auc_val = roc_auc_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("✅ TensorFlow Accuracy:", accuracy_val)
print("✅ ROC-AUC Score:", roc_auc_val)
print(report)

# ✅ Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# ===== 📌 SAVE RESULTS =====

# 📌 1. Save Accuracy & Loss plots
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy per Epoch')
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_plot.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss per Epoch')
plt.savefig(os.path.join(RESULTS_DIR,"loss_plot.png"))
plt.close()

# 📌 2. Save Confusion Matrix as PNG
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULTS_DIR,"confusion_matrix.png"))
plt.close()

# 📌 3. Save metrics to a text file (UTF-8 to avoid encoding errors)
with open("model_metrics.txt", "w", encoding="utf-8") as f:
    f.write("✅ TensorFlow Model Evaluation\n")
    f.write(f"Accuracy: {accuracy_val:.4f}\n")
    f.write(f"ROC-AUC Score: {roc_auc_val:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# 📌 4. Combine plots & metrics in a PDF
# make sure RESULTS_DIR exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Accuracy per Epoch (PNG + optional PDF) ---
    fig, ax = plt.subplots()
    ax.plot(history.history["accuracy"], label="Train Accuracy")
    ax.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax.set_title("Accuracy per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()

    # Save to PNG with the exact name you want
    png_path = os.path.join(RESULTS_DIR, "Accuracy_per_Epoch.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    # If you also want it in a PDF report:
    pdf_path = os.path.join(RESULTS_DIR, "training_report.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)   # note: no filename here

    plt.close(fig)


    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    fig.savefig(os.path.join(RESULTS_DIR,"Accuracy_per_Epoch.png"))
    pdf.savefig(fig)
    plt.close()

    # Confusion matrix
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    fig.savefig(os.path.join(RESULTS_DIR,"Confusion_matrix.png"))
    pdf.savefig(fig)
    plt.close()

    # Metrics as text
    plt.figure()
    plt.axis('off')
    metrics_text = f"Accuracy: {accuracy_val:.4f}\nROC-AUC: {roc_auc_val:.4f}\n\n{report}"
    plt.text(0, 0.5, metrics_text, fontsize=10, va='center')
    fig.savefig(os.path.join(RESULTS_DIR,"TF_model_accuracy_score.png"))
    pdf.savefig(fig)
    plt.close()

# Save scaler
joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.joblib"))

# Save the Keras model in the modern format (portable)
model.save(os.path.join(RESULTS_DIR, "tensorflow_sedentary_model.keras"))

# save as .h5 backup (modern Keras format)
model.save(os.path.join(RESULTS_DIR, "tensorflow_sedentary_model.h5"))

print("✅ Model, plots, metrics, and reports saved successfully.")

print("\n📊 To view TensorBoard logs, run this command in your terminal:")
print(f"tensorboard --logdir {LOGS_DIR}")
print("Then open the shown URL in your browser.")
