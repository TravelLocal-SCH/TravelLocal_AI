from pydantic import BaseModel
from typing import List
from typing import Optional

class AnalyzeRequest(BaseModel):
    answers: List[str]

class FeedbackRequest(BaseModel):
    user_answer_id: int
    is_agree: bool
    comment: str