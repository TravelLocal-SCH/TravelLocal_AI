# routers/feedback.py

from fastapi import APIRouter
from models.request_model import FeedbackRequest
from services.db_service import save_feedback,get_feedback_stats

router = APIRouter()

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        save_feedback(
            user_answer_id=feedback.user_answer_id,
            is_agree=feedback.is_agree,
            comment=feedback.comment
        )
        return {"message": "피드백이 성공적으로 저장되었습니다."}
    except Exception as e:
        return {"error": str(e)}
    

@router.get("/feedback/stats")
async def feedback_stats():
    try:
        stats = get_feedback_stats()
        return stats
    except Exception as e:
        return {"error": str(e)}