import sys
import os
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Automatically add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loader import load_data

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ----------- Train KNN Model with standard scaler for feature scaling-----------
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance'))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------- Classification Report -----------
print("K-Nearest Neighbors Results:")
print(classification_report(y_test, y_pred))

# ----------- Cross-Validation -----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_cv = KNeighborsClassifier()
scores = cross_val_score(model_cv, X, y, cv=cv, scoring='f1_macro')
print("KNN - F1 Macro Scores:", scores)
print(f"Mean F1: {scores.mean():.4f}, Std: {scores.std():.4f}")

# ----------- Confusion Matrix -----------
cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("F:/Dessertation/dataset/KNN/conf_matrix_KNN.png")
plt.show()

# ----------- ROC Curve (Binary classification only) -----------
if len(labels) == 2 and hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("F:/Dessertation/dataset/KNN/roc_curve_KNN.png")
    plt.show()
else:
    print("ROC curve is only available for binary classification with probability estimates.")

# ----------- Save Trained Model -----------
joblib.dump(model, "knn_model.pkl")