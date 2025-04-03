from fastapi import APIRouter
from services.db_service import  get_recent_answers

router = APIRouter()


@router.get("/stats/recent_answers")
async def recent_answers():
    return get_recent_answers()
