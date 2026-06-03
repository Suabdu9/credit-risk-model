from fastapi import FastAPI

from src.predict import predict_risk

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse
)

app = FastAPI(
    title="Credit Risk API"
)


@app.get("/")
def root():

    return {
        "message": "Credit Risk Model API"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    result = predict_risk(
        request.model_dump()
    )

    return result