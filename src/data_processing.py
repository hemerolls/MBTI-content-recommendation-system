"""
Пайплайн подготовки текстовых данных MBTI.
Очистка → Нарезка на чанки → Токенизация → Train/Test split по user_id.
"""
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Tuple


# Паттерн для удаления упоминаний MBTI-типов (борьба с data leakage)
MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISTP", "ESTJ", "ESTP", "ISFJ", "ISFP", "ESFJ", "ESFP",
]
RE_MBTI = re.compile(r"\b(" + "|".join(MBTI_TYPES) + r")\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    """
    Очистка текста от мусора и утечки данных.
    
    Удаляет: URL, email, @упоминания, хэштеги, MBTI-типы,
    разделители |||, длинные числа, спецсимволы.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = RE_MBTI.sub(" ", text)
    text = re.sub(r"\|{2,}|#{2,}", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\b\d{5,}\b", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_tokens(
    token_ids: List[int], 
    body_tokens: int = 510, 
    stride: int = 128
) -> List[List[int]]:
    """
    Sliding-window нарезка токенов на чанки.
    
    Args:
        token_ids: список токенов
        body_tokens: макс. токенов без [CLS]/[SEP]
        stride: перекрытие между окнами
    """
    n = len(token_ids)
    if n <= body_tokens:
        return [token_ids]
    
    chunks = []
    step = body_tokens - stride
    start = 0
    while start + body_tokens < n:
        chunks.append(token_ids[start:start + body_tokens])
        start += step
    # Последний чанк — сдвигаем назад до полной длины
    chunks.append(token_ids[n - body_tokens:])
    return chunks


def type_to_dims(mbti_type: str) -> List[int]:
    """'INTJ' → [0, 1, 1, 0] (E/I, S/N, T/F, J/P)"""
    dim_map = [{"E": 1, "I": 0}, {"S": 0, "N": 1}, {"T": 1, "F": 0}, {"J": 0, "P": 1}]
    t = mbti_type.strip().upper()
    return [dim_map[i][ch] for i, ch in enumerate(t)]
