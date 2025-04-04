import json
from config.settings import DB_CONFIG
import mysql.connector

def get_mbti_info(mbti_type: str):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM mbti_traits WHERE type = %s", (mbti_type,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result

def get_all_tags():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT tag FROM travel_tags")
    tags = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return tags

def save_user_answer(answers: list, predicted_mbti: str):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    query = "INSERT INTO user_answers (answers, predicted_mbti) VALUES (%s, %s)"
    cursor.execute(query, (json.dumps(answers, ensure_ascii=False), predicted_mbti))
    conn.commit()
    inserted_id = cursor.lastrowid  
    cursor.close()
    conn.close()
    return inserted_id
    
#피드백을 사용하지않을경우
def get_recent_answers(limit=10):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT id, predicted_mbti, created_at
        FROM user_answers
        ORDER BY created_at DESC
        LIMIT %s
    """
    cursor.execute(query, (limit,))
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

# services/db_service.py
def get_recent_user_answers(limit=50):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT answers, predicted_mbti
        FROM user_answers
        ORDER BY created_at DESC
        LIMIT %s
    """
    cursor.execute(query, (limit,))
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

def save_feedback(user_answer_id, is_agree, comment=None):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    query = """
        INSERT INTO user_feedback (user_answer_id, is_agree, comment)
        VALUES (%s, %s, %s)
    """
    cursor.execute(query, (user_answer_id, is_agree, comment))
    conn.commit()
    cursor.close()
    conn.close()

def get_feedback_stats():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 전체 피드백 수, 동의 수, 비동의 수 계산
    query = """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN is_agree = TRUE THEN 1 ELSE 0 END) AS agree,
            SUM(CASE WHEN is_agree = FALSE THEN 1 ELSE 0 END) AS disagree,
            SUM(CASE WHEN is_agree IS NULL THEN 1 ELSE 0 END) AS no_response
        FROM user_feedback
    """
    cursor.execute(query)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    return {
        "total_feedback": row[0],
        "agree": row[1],
        "disagree": row[2],
        "no_response": row[3],
        "agree_ratio": round(row[1] / row[0] * 100, 2) if row[0] else 0,
        "disagree_ratio": round(row[2] / row[0] * 100, 2) if row[0] else 0,
        "no_response_ratio": round(row[3] / row[0] * 100, 2) if row[0] else 0,
    }


def get_feedbacked_answers(limit=50):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT ua.answers, ua.predicted_mbti, uf.is_agree
        FROM user_answers ua
        JOIN user_feedback uf ON ua.id = uf.user_answer_id
        WHERE uf.is_agree IS NOT NULL
        ORDER BY uf.created_at DESC
        LIMIT %s
    """
    cursor.execute(query, (limit,))
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result