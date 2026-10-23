"""
Анализ утечки данных (Data Leakage Detection) в датасетах MBTI.

Ключевой научный вклад ВКР: обнаружение, что 87% образцов содержат
прямые упоминания собственного типа, что приводит к ложной точности 90%.
После очистки реальная точность падает до 21%.
"""
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


MBTI_TYPES = [
    "INFP", "INFJ", "INTP", "INTJ", "ENTP", "ENFP", "ISTP", "ISFP",
    "ENTJ", "ISTJ", "ENFJ", "ISFJ", "ESTP", "ESFP", "ESFJ", "ESTJ",
]


def detect_leakage(df: pd.DataFrame, text_col: str = "text", 
                   type_col: str = "type") -> pd.DataFrame:
    """
    Обнаружение образцов с утечкой данных.
    
    Утечка = текст содержит прямое упоминание собственного MBTI-типа.
    
    Returns:
        DataFrame с образцами, содержащими утечку
    """
    pattern = re.compile(r"\b(" + "|".join(MBTI_TYPES) + r")\b", re.IGNORECASE)
    
    leakage_rows = []
    for idx, row in df.iterrows():
        text = str(row[text_col])
        found = [m.upper() for m in pattern.findall(text)]
        own_type = str(row[type_col]).upper().strip()
        own_mentions = [f for f in found if f == own_type]
        
        if own_mentions:
            leakage_rows.append({
                "idx": idx,
                "type": own_type,
                "own_count": len(own_mentions),
                "all_mentions": len(found),
                "text_snippet": text[:200],
            })
    
    return pd.DataFrame(leakage_rows)


def compare_accuracy_with_without_leakage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    has_leakage: np.ndarray,
) -> Dict[str, float]:
    """
    Сравнение точности на образцах с утечкой и без.
    
    Returns:
        dict с accuracy_leak, accuracy_clean, delta
    """
    from sklearn.metrics import accuracy_score
    
    results = {}
    if has_leakage.any():
        results["accuracy_leak"] = accuracy_score(
            y_true[has_leakage], y_pred[has_leakage]
        )
        results["n_leak"] = int(has_leakage.sum())
    if (~has_leakage).any():
        results["accuracy_clean"] = accuracy_score(
            y_true[~has_leakage], y_pred[~has_leakage]
        )
        results["n_clean"] = int((~has_leakage).sum())
    if "accuracy_leak" in results and "accuracy_clean" in results:
        results["delta"] = results["accuracy_leak"] - results["accuracy_clean"]
        
    return results
