from pydantic import BaseModel
from typing import List
from typing import Optional

class AnalyzeRequest(BaseModel):
    answers: List[str]

class FeedbackRequest(BaseModel):
    user_answer_id: int
    is_agree: Optional[bool] = None  # 동의 여부 (True, False, None)
    comment: Optional[str] = None    # 선택적 코멘트