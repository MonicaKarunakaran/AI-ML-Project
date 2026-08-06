from fastapi import FastAPI, HTTPException

from app.schemas import PredictRequest, PredictResponse
from app.services.model_loader import load_model
from app.services.predictor import predict
from app.crew.preprocess import preprocess
from app.services.ragas_evaluator import evaluate_prediction

app = FastAPI(
    title="FastAPI ML Model Serving",
    version="1.0.0",
    description="Serve an Iris Classification model using FastAPI and MLflow."
)


@app.on_event("startup")
def startup_event():
    load_model()
    print("Model loaded successfully.")


@app.get("/")
def home():
    return {
        "message": "Welcome to FastAPI ML Model Serving API"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):

    try:

        features = preprocess(request)

        result = predict(features)

        evaluate_prediction(result["prediction"])

        return PredictResponse(
            prediction=result["prediction"],
            species=result["species"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )