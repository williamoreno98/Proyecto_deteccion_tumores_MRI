from pydantic import BaseModel
from typing import Dict

class PredictMLResponse(BaseModel):
    model: str = "ml"
    top_class: str
    probabilities: Dict[str, float]
