# 🧭 FastAPI 기반 Travel MBTI RAG 분석 API

이 프로젝트는 여행 관련 설문 응답을 기반으로 사용자의 MBTI를 예측하고, MBTI별 성향에 맞는 감성 메시지, 해시태그, 국내 여행지를 추천하는 **RAG 기반 API 서버**입니다.  
Google Gemini API와 MySQL DB를 활용하여 사용자 맞춤형 여행 성향 분석을 제공합니다.

---

## 🗺️ 시스템 구성도 (RAG 구조)

[ 사용자의 설문 응답 ]
↓
[ Gemini API ]

- MBTI 예측
  ↓
  [ MySQL DB 검색 (MBTI 설명, 해시태그) ]
  ↓
  [ Gemini API ]
- 상세 설명 생성
- 추천 지역 생성
- 해시태그 선택
  ↓
  [ 최종 결과 종합 ]
  MBTI + 상세 설명 + 해시태그 + 지역 추천
  ↓
  [ 사용자에게 반환 ]

**RAG(Retrieval-Augmented Generation)** 구조는 다음과 같은 흐름으로 작동합니다:

1. **사용자 설문 응답** → FastAPI에 전달
2. **Gemini API**로 MBTI 유형 예측
3. 예측된 MBTI에 따라 **MySQL DB에서 관련 설명/성향 조회**
4. Gemini에 DB 정보를 포함한 프롬프트로 **감성 메시지, 해시태그, 도시 추천 생성**
5. 종합된 결과를 사용자에게 JSON 형식으로 응답

---

## 📂 폴더 구조

```
fastapi_travel_AI/
│
├── config/
│   └── settings.py           # 환경변수 및 DB 설정 관리
│
├── models/
│   └── request_model.py      # Pydantic 기반 Request 모델 정의
│
├── routers/
│   ├── question.py           # 객관식 질문 생성 API
│   └── rag.py                # MBTI 분석 + 감성 추천 API
│
├── services/
│   ├── db_service.py         # MySQL 연동 함수 모음
│   └── gemini_service.py     # Gemini API 호출 함수 모음
│
├── .env                      # 환경 변수 저장 파일 (API 키 및 DB 정보)
├── main.py                   # FastAPI 애플리케이션 실행 진입점
└── requirements.txt          # 패키지 의존성 목록
```

---

## 🚀 주요 기능

| 기능                  | 설명                                                              | 엔드포인트               |
| --------------------- | ----------------------------------------------------------------- | ------------------------ |
| 🔸 객관식 질문 생성   | 여행 성향을 분석하기 위한 질문 5개 생성                           | `GET /generate_question` |
| 🔸 RAG 기반 성향 분석 | 설문 응답 기반으로 MBTI 예측 + 감성 메시지 + 해시태그 + 지역 추천 | `POST /rag_recommend`    |

---

## 🧪 API 테스트

### ✅ 예시 요청 (`/rag_recommend`)

```json
POST /rag_recommend
{
  "answers": [
    "나는 자연을 좋아해",
    "혼자 여행을 즐긴다",
    "즉흥적인 여행을 좋아해",
    "계획은 최소한으로",
    "현지 음식을 즐긴다"
  ]
}
```

### ✅ 예시 응답

```json
{
  "mbti": "ESFJ",
  "trait": {
    "type": "ESFJ",
    "main_title": "사교적인 배려왕",
    "description": "친화력 1등! 혼자 여행가도 문제없어 ..."
  },
  "recommendation": "ESFJ님은 계획적이면서도 타인을 잘 챙기는 여행 스타일입니다. ...",
  "tags": ["#우정여행", "#도시탐방", "#감성사진", ...],
  "recommended_regions": ["부산", "여수", "전주"]
}
```

---

## ⚙️ 설치 및 실행 방법

### 1️⃣ 프로젝트 클론 및 진입

```bash
git clone https://github.com/your-name/fastapi_travel_project.git
cd fastapi_travel_project
```

### 2️⃣ 가상환경 설정 (선택)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성하고 아래 내용을 작성합니다.

```env
GEMINI_API_KEY=your-gemini-api-key

DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your-password
DB_NAME=travel_mbti
```

### 4️⃣ 패키지 설치

```bash
pip install -r requirements.txt
```

### 5️⃣ 서버 실행

```bash
uvicorn main:app --reload
```

- 로컬 접속: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🗃️ DB 테이블 구조

### 🔹 `mbti_traits`

| 필드        | 설명                 |
| ----------- | -------------------- |
| type        | MBTI 유형 (ex. ENFP) |
| main_title  | 성향 이름 요약       |
| sub_title   | 짧은 설명            |
| description | 상세 성향 설명       |
| created_at  | 생성일               |

### 🔹 `travel_tags`

| 필드 | 설명                                 |
| ---- | ------------------------------------ |
| tag  | 추천 해시태그 문자열 (예: #감성사진) |

---

## 📦 requirements.txt 예시

```txt
fastapi
uvicorn
pydantic
python-dotenv
google-generativeai
mysql-connector-python
```

---

## 📌 참고 사항

- Google Cloud Console에서 Gemini API 사용 설정 및 API 키 발급 필요
- `.env`는 Git에 절대 커밋하지 마세요 (민감 정보 포함)
- 프론트엔드에서 CORS 허용을 위해 모든 origin을 열어두었으나, 배포 시에는 도메인 제한 권장

---

## ✨ 기여 및 확장

이 프로젝트는 학습 및 실습, 실제 서비스로도 확장이 가능합니다.  
PR, 피드백, 기능 추가 언제든 환영합니다 😊
