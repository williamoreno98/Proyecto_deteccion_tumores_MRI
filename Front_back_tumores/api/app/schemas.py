# api/app/schemas.py
from pydantic import BaseModel, Field
from typing import Dict, Literal

# Acepta exactamente "ml" o "rf"
ModelName = Literal["ml", "rf"]

class PredictMLResponse(BaseModel):
    # el back responderá con el modelo que usó: "ml" o "rf"
    model: ModelName = Field(default="ml", description="Modelo que procesó la imagen")
    # clase superior (label ganador)
    top_class: str
    # probabilidades por clase, ejemplo: {"Glioma": 0.85, "Meningioma": 0.10, ...}
    probabilities: Dict[str, float]
