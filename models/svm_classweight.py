import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Automatically add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loader import load_data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC

model = SVC(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Support Vector Machine Results:")
print(classification_report(y_test, y_pred))
