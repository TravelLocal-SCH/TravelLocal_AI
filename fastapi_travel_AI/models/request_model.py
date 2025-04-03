from pydantic import BaseModel
from typing import List

class AnalyzeRequest(BaseModel):
    answers: List[str]