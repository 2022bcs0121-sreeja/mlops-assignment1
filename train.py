import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
import joblib

# Simulated dataset
np.random.seed(42)

data = pd.DataFrame({
    "ticket_7d": np.random.randint(0, 5, 500),
    "ticket_30d": np.random.randint(0, 10, 500),
    "ticket_90d": np.random.randint(0, 20, 500),
    "sentiment_score": np.random.uniform(-1, 1, 500),
    "monthly_change": np.random.uniform(-50, 50, 500),
    "churn": np.random.randint(0, 2, 500)
})

X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")