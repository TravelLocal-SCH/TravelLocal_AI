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
    
