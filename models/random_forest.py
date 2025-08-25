import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib



# Automatically add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loader import load_data


X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Random Forest Results:")
print(classification_report(y_test, y_pred))

#Cross_Validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(class_weight='balanced')
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
print("Random Forest - F1 Macro Scores:", scores)
print(f"Mean F1: {scores.mean():.4f}, Std: {scores.std():.4f}")

# Feature Importance
model.fit(X, y)
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("F:/Dessertation/dataset/Random Forest/random_forest_feature_importance.png")
plt.show()

#Confusion Matrix (with Heatmap)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("F:/Dessertation/dataset/Random Forest/conf_matrix_Random_Forest.png")
plt.show()


#ROC Curve (Receiver Operating Characteristic)

if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)[:, 1]
else: 
    y_score = model.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("F:/Dessertation/dataset/Random Forest/roc_curve_Random_Forest.png")
plt.show()
joblib.dump(model, "rf_model.pkl")