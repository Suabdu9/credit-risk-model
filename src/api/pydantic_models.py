from pydantic import BaseModel


class PredictionRequest(BaseModel):

    num__Amount: float
    num__Value: float
    num__PricingStrategy: float
    num__FraudResult: float


class PredictionResponse(BaseModel):

    risk_probability: float
    prediction: int
