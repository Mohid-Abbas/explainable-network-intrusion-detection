from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from typing import Dict, Any, List
from src.api.schemas import (
    PredictRequest,
    BatchPredictRequest,
    PredictResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.api.inference import InferenceEngine

app = FastAPI(title="Explainable Network Intrusion Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InferenceEngine()


@app.on_event("startup")
def load_model() -> None:
    engine.load()
    if engine.model is not None and hasattr(engine.model, "feature_names_in_"):
        engine.feature_names = [str(c) for c in engine.model.feature_names_in_]
    elif engine.model is not None:
        engine.feature_names = [f"feature_{i}" for i in range(engine.model.n_features_in_)]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=engine.model is not None)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=engine.model_version,
        metrics_summary=None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        features = request.features.dict(by_alias=False, exclude_unset=True)
        result = engine.predict(features)
        return PredictResponse(**result)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest) -> List[PredictResponse]:
    results = []
    for flow in request.flows:
        try:
            features = flow.dict(by_alias=False, exclude_unset=True)
            result = engine.predict(features)
            results.append(PredictResponse(**result))
        except Exception as e:
            results.append(PredictResponse(
                prediction="error",
                confidence=0.0,
                is_malicious=False,
                top_features=[],
                model_version=engine.model_version,
            ))
    return results
