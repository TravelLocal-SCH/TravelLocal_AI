from fastapi import APIRouter
from models.request_model import AnalyzeRequest
from services.gemini_service import generate_rag_recommendation

router = APIRouter()

@router.post("/rag_recommend")
async def rag_recommendation(request: AnalyzeRequest):
    return await generate_rag_recommendation(request.answers)
