import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Load data
df = pd.read_csv("data/raw/diabetes.csv")

X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average="macro")

    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("f1_macro", f1)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    mlflow.sklearn.log_model(model, "model")

    print("Model trained. F1:", f1)
