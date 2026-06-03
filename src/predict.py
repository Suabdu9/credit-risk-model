import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_risk(data: dict):

    df = pd.DataFrame([data])

    probability = model.predict_proba(df)[0][1]

    prediction = int(probability >= 0.5)

    return {
        "risk_probability": float(probability),
        "prediction": prediction
    }