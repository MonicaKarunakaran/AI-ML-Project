from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class PredictResponse(BaseModel):
    prediction: int
    species: str