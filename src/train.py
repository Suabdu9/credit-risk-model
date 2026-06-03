import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


RANDOM_STATE = 42

DATA_PATH = "data/processed/model_data.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    return metrics


def main():

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    mlflow.set_experiment("credit-risk-model")

    best_model = None
    best_auc = 0

    models = {
        "LogisticRegression": (
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            ),
            {
                "C": [0.01, 0.1, 1, 10]
            }
        ),

        "RandomForest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None]
            }
        )
    }

    for model_name, (model, params) in models.items():

        with mlflow.start_run(run_name=model_name):

            search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1
            )

            search.fit(X_train, y_train)

            best_estimator = search.best_estimator_

            metrics = evaluate_model(
                best_estimator,
                X_test,
                y_test
            )

            mlflow.log_params(
                search.best_params_
            )

            mlflow.log_metrics(
                metrics
            )

            mlflow.sklearn.log_model(
                best_estimator,
                artifact_path=model_name
            )

            print(f"\n{model_name}")
            print(metrics)

            if metrics["roc_auc"] > best_auc:

                best_auc = metrics["roc_auc"]
                best_model = best_estimator

    joblib.dump(
        best_model,
        f"{MODEL_DIR}/best_model.pkl"
    )

    print("\nBest model saved.")


if __name__ == "__main__":
    main()