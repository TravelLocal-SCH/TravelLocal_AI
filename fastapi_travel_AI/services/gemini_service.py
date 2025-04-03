import json
import re
from config.settings import GEMINI_API_KEY
import google.generativeai as genai
from services.db_service import get_mbti_info, get_all_tags


genai.configure(api_key=GEMINI_API_KEY)

async def generate_questions():
    prompt = """
여행 성향 분석을 위한 객관식 질문 5개를 아래 형식으로 생성하세요:
{
  "questions": [
    {
      "question": "당신은 어떤 여행을 선호하나요?",
      "options": ["자연", "도시", "음식", "즉흥"]
    }
  ]
}
"""
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    content = response.text.strip()
    match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
    if match:
        content = match.group(1)
    return json.loads(content)

async def generate_rag_recommendation(answers):
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    # 1. MBTI 예측
    mbti_prompt = f"""
{json.dumps(answers, ensure_ascii=False, indent=2)}
이 응답 기반으로 MBTI를 추측해 주세요. JSON으로 주세요: {{ "mbti": "ENFP" }}
"""
    response = model.generate_content(mbti_prompt)
    mbti_text = re.search(r"```json\n(.*?)```", response.text, re.DOTALL)
    mbti_json = json.loads(mbti_text.group(1)) if mbti_text else json.loads(response.text)
    mbti_type = mbti_json["mbti"]
 

    # 2. DB에서 설명 가져오기
    trait = get_mbti_info(mbti_type)


    if not trait:
        return {"error": f"{mbti_type} 유형 정보가 DB에 없습니다."}

    # 3. 분석 메시지 생성
    rag_prompt = f"""
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
    reco_response = model.generate_content(rag_prompt)

    
    # 4. 해시태그 추천
    tags = get_all_tags()
    tag_prompt = f"""
다음은 사용자의 여행 성향 응답입니다:

{json.dumps(answers, ensure_ascii=False, indent=2)}

다음 국내 여행 해시태그 중에서 이 사용자에게 어울리는 10개를 골라주세요:

{json.dumps(tags, ensure_ascii=False, indent=2)}

요청사항:
- 응답은 반드시 JSON 형식으로
- 코드 블럭으로 감싸 주세요
- 형식 예시:
```json
{{ "tags": ["#힐링여행", "#맛집투어", ...] }}
"""
    tag_response = model.generate_content(tag_prompt)
    tag_text = tag_response.text.strip()

    
    # 코드 블럭으로 감싸져 있을 경우
    match = re.search(r"```json\n(.*?)```", tag_text, re.DOTALL)
    if match:
        tag_result = json.loads(match.group(1).strip())
    else:
        tag_result = json.loads(tag_text)

    recommended_tags = tag_result["tags"]



    # 5. 지역 추천
    region_prompt = f"""
MBTI: {trait['type']}
설명: {trait['description']}
이 유형에게 어울리는 한국 도시 3개를 콤마로 제시해 주세요.
요청사항: 
- 유명 도시말고도 이 유형에게 어울리는 다양한 한국 도시를 추천해줘도 좋아요.
"""
    region_response = model.generate_content(region_prompt)
    region_list = [r.strip() for r in region_response.text.split(",")[:3]]

    
    return {
        "mbti": mbti_type,
        "trait": trait,
        "recommendation": reco_response.text.strip(),
        "tags": recommended_tags,#tag_result["tags"],
        "recommended_regions": region_list
    }
