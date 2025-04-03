from fastapi import APIRouter
from services.gemini_service import generate_questions

router = APIRouter()

@router.get("/generate_question")
async def generate_question():
    return await generate_questions()