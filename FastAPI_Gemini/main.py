import os
import json
import re
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import mysql.connector


# .env에서 API 키 로딩
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# FastAPI 앱 초기화
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시엔 특정 도메인만 허용 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def get_mbti_info(mbti_type: str):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  # 본인 DB 계정
        password="78910",  # 본인 DB 비번
        database="travel_mbti"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM mbti_traits WHERE type = %s", (mbti_type,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result



def get_all_tags():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="78910",
        database="travel_mbti"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT tag FROM travel_tags")
    tags = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return tags



# travel_traits.json 로드
with open("travel_traits.json", "r", encoding="utf-8") as f:
    TRAVEL_TRAITS = json.load(f)

# 사용자 응답 데이터 모델
class AnalyzeRequest(BaseModel):
    answers: List[str]

# ✅ 1. 객관식 질문 생성 API
@app.get("/generate_question")
async def generate_question():
    prompt = """
여행자 성향을 분석하기 위한 객관식 질문을 5개 만들어 주세요.
- 질문은 활동, 예산, 동행 여부, 여행 스타일 등을 다양하게 포함해 주세요.
- 각 질문에는 4개의 선택지를 포함해 주세요.
- 아래 JSON 형식으로 응답해 주세요:

{
  "questions": [
    {
      "question": "질문1",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"]
    }
  ]
}
"""

    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        content = response.text.strip()

        # JSON 블록만 추출
        if "```" in content:
            match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        return json.loads(content)

    except Exception as e:
        return {
            "error": "질문 생성 실패",
            "details": str(e),
            "raw_response": response.text if 'response' in locals() else None
        }

# ✅ 2. 성향 분석 API
@app.post("/analyze")
async def analyze_traveler(request: AnalyzeRequest):
    answers = request.answers

    traits_text = """
A1: 조용한 자연파
A2: 도시 탐험가
A3: 문화 체험가
A4: 모험가
A5: 미식가
A6: 힐링 여행자
A7: 즉흥 여행가
A8: 역사 탐험가
B1: 혼자 여행 선호자
B2: 단체 여행 선호자
B3: 가족 중심 여행자
B4: 사진 애호가
B5: 경제형 여행자
B6: 럭셔리 여행자
B7: 활동가형 여행자
B8: 자유 방랑형
"""

    prompt = f"""
다음은 사용자의 객관식 설문 응답입니다:

{json.dumps(answers, ensure_ascii=False, indent=2)}

위 응답을 참고하여 16가지 성향 중 하나로 분류해주세요.

응답은 아래 JSON 형식으로 제공해주세요:
{{
  "type": "A1",
  "name": "조용한 자연파",
  "description": "설명",
  "recommended_places": ["추천지1", "추천지2", "추천지3"]
}}

성향 목록:
{traits_text}
"""

    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        content = response.text.strip()

        # JSON 추출
        if "```" in content:
            match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        return json.loads(content)

    except Exception as e:
        return {
            "error": "성향 분석 실패",
            "details": str(e),
            "raw_response": response.text if 'response' in locals() else None
        }

# ✅ 3. RAG + 감성 추천 메시지 생성 API
@app.post("/rag_recommend")
async def generate_rag_recommendation(request: AnalyzeRequest):
    answers = request.answers

    # 1. Gemini로 MBTI 예측
    analyze_prompt = f"""
다음은 여행자 객관식 설문 응답입니다:

{json.dumps(answers, ensure_ascii=False, indent=2)}

이 여행자의 MBTI 유형을 예측해주세요. 응답 형식은 JSON으로:
{{
  "mbti": "ENFP"
}}
"""

    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content(analyze_prompt)
        result_text = response.text.strip()

        if "```" in result_text:
            match = re.search(r"```json\n(.*?)```", result_text, re.DOTALL)
            if match:
                result_text = match.group(1).strip()

        result = json.loads(result_text)

        # 2. DB에서 MBTI 설명 가져오기
        trait = get_mbti_info(result["mbti"])
        if not trait:
            return {"error": f"{result['mbti']} 유형 정보가 DB에 없습니다."}

        # 3. 감성 메시지 생성 프롬프트
        rag_prompt = f"""
당신은 여행 성향 추천 전문가입니다.

MBTI 유형: {trait['type']}
설명: {trait['description']}

이 사용자에게 감성적이고 친근한 여행 추천 메시지를 작성해주세요.
"""

        rag_response = model.generate_content(rag_prompt)

        return {
            "mbti": result["mbti"],
            "trait": trait,
            "recommendation": rag_response.text
        }

    except Exception as e:
        return {
            "error": "RAG 추천 생성 실패",
            "details": str(e),
            "raw_response": response.text if 'response' in locals() else None
        }
    

@app.post("/recommend_tags")
async def recommend_tags(request: AnalyzeRequest):
    user_answers = request.answers
    tags = get_all_tags()

    prompt = f"""
당신은 여행 해시태그 추천 전문가입니다.

사용자의 여행 성향을 다음 답변에서 유추해보세요:

{json.dumps(user_answers, ensure_ascii=False, indent=2)}

아래 50개의 국내 여행 해시태그 중에서 이 사용자에게 어울리는 10개를 골라주세요:

{json.dumps(tags, ensure_ascii=False, indent=2)}

응답은 다음 형식으로 해주세요:
{{
  "tags": ["#힐링여행", "#계획없이떠나기", "#감성사진", ...]
}}
"""

    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if "```" in result_text:
            match = re.search(r"```json\n(.*?)```", result_text, re.DOTALL)
            if match:
                result_text = match.group(1).strip()

        result = json.loads(result_text)
        return result

    except Exception as e:
        return {
            "error": "해시태그 추천 실패",
            "details": str(e),
            "raw_response": response.text if 'response' in locals() else None
        }
    

@app.post("/analyze_and_recommend")
async def analyze_and_recommend(request: AnalyzeRequest):
    answers = request.answers

    # ✅ MBTI 성향 분석
    mbti_prompt = f"""
다음은 여행자 객관식 설문 응답입니다:

{json.dumps(answers, ensure_ascii=False, indent=2)}

이 여행자의 MBTI 유형을 예측해주세요. 응답 형식은 JSON으로:
{{
  "mbti": "ENFP"
}}
"""

    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content(mbti_prompt)
        result_text = response.text.strip()

        if "```" in result_text:
            match = re.search(r"```json\n(.*?)```", result_text, re.DOTALL)
            if match:
                result_text = match.group(1).strip()
        mbti_result = json.loads(result_text)

        # ✅ 성향 정보 조회
        trait = get_mbti_info(mbti_result["mbti"])
        if not trait:
            return {"error": f"{mbti_result['mbti']} 유형 정보가 DB에 없습니다."}

        # ✅ 감성 추천 메시지 생성
        reco_prompt = f"""
당신은 여행 심리 전문가입니다.

MBTI 유형과 해당 설명을 보고, 이 유형의 여행 성향, 특징, 선호하는 여행 방식에 대해 요약된 분석 내용을 작성해주세요.

MBTI 유형: {trait['type']}
설명: {trait['description']}

요청사항:
- 문장 형식으로 3~5문장 정도
- 해당 MBTI 유형이 어떤 여행 스타일을 좋아하고 어떤 방식으로 여행을 즐기는지를 알려주세요
- 너무 딱딱하지 않지만, 정보 중심으로 설명해주세요
- 감성적 표현은 피하고, 분석/설명 중심으로 작성해주세요

예시 형식:
"ENFP 유형은 즉흥적인 여행을 선호하며, 낯선 장소에서도 빠르게 적응합니다. 여행 중 다양한 사람들과 교류하는 것을 즐기며, 계획보다는 분위기를 따라 움직이는 경우가 많습니다."

이제 작성해주세요:
"""
        reco_response = model.generate_content(reco_prompt)

        # ✅ 해시태그 추천
        tags = get_all_tags()
        tag_prompt = f"""
사용자의 여행 응답은 다음과 같습니다:
{json.dumps(answers, ensure_ascii=False, indent=2)}

아래 국내 여행 해시태그 중 이 사용자에게 어울리는 10개를 골라주세요:

{json.dumps(tags, ensure_ascii=False, indent=2)}

응답 형식:
{{
  "tags": ["#힐링여행", "#감성사진", ...]
}}
"""
        tag_response = model.generate_content(tag_prompt)
        tag_text = tag_response.text.strip()
        if "```" in tag_text:
            match = re.search(r"```json\n(.*?)```", tag_text, re.DOTALL)
            if match:
                tag_text = match.group(1).strip()
        tag_result = json.loads(tag_text)
    
            # ✅ 지역 추천 프롬프트 추가 (Gemini)
        region_prompt = f"""
아래는 {trait['type']} 유형의 여행 성향 설명입니다:

"{trait['description']}"

이 유형에게 어울리는 대한민국 국내 도시(시 단위)를 3곳 추천해주세요.
조건:
- 한국의 시 단위 도시 이름만 콤마(,)로 구분해서 반환해주세요.
- 예시: 서울, 부산, 강릉
- 설명은 필요 없습니다.
"""
        region_response = model.generate_content(region_prompt)
        region_text = region_response.text.strip()

        # 콤마로 나눈 3개 도시만 추출
        recommended_regions = [city.strip() for city in region_text.split(",")[:3]]            

        # ✅ 최종 결과 반환
        return {
            "mbti": mbti_result["mbti"],
            "trait": trait,
            "recommendation": reco_response.text.strip(),
            "tags": tag_result["tags"],
            "recommended_regions": recommended_regions
        }

    except Exception as e:
        return {
            "error": "전체 추천 실패",
            "details": str(e),
            "raw_response": response.text if 'response' in locals() else None
        }    

    ## 지역으로/ 엠비티아이 형식/ 데베 mysql/기본 틀 키워드와 해시태그  종류 고민

    #성향 mbti 해시태그