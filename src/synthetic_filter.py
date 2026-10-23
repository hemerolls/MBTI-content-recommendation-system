"""
Классификатор-фильтр для валидации синтетических данных.

Принимает тексты, сгенерированные Qwen3.5-flash, и пропускает
только те, где модель уверенно предсказывает целевой тип.
Это обеспечивает замкнутый цикл положительной обратной связи.
"""
import torch
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional, Tuple


class SyntheticDataFilter:
    """
    Фильтр синтетических данных на основе обученного классификатора.
    
    Критерий прохождения: хотя бы одна голова (type16 или dichotomy)
    правильно предсказывает целевой тип.
    """
    
    TYPE_LIST = [
        "IE", "IN", "IT", "IP",  # битовая кодировка
        # Полная таблица генерируется динамически
    ]
    
    def __init__(self, model, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Генерация таблицы типов
        self.type_list = []
        for i in range(16):
            ie = "E" if (i >> 3) & 1 else "I"
            sn = "N" if (i >> 2) & 1 else "S"
            tf = "T" if (i >> 1) & 1 else "F"
            jp = "P" if i & 1 else "J"
            self.type_list.append(ie + sn + tf + jp)

    @torch.no_grad()
    def filter_texts(
        self,
        df: pd.DataFrame,
        text_col: str = "post",
        type_col: str = "type",
        max_tokens: int = 512,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Фильтрация синтетических текстов.
        
        Returns:
            DataFrame с прошедшими проверку текстами + метаданные
        """
        verified_rows = []
        
        for idx, row in df.iterrows():
            true_type = str(row[type_col]).upper().strip()
            text = str(row[text_col])
            
            # Инференс (упрощённый, без чанкинга для скорости)
            enc = self.tokenizer(
                text, return_tensors="pt",
                max_length=max_tokens, truncation=True, padding=True
            ).to(self.device)
            
            output = self.model(**enc)
            
            # Проверка type16 head
            head_match = False
            if output.type16_logits is not None:
                pred_idx = output.type16_logits.argmax(-1).item()
                pred_type = self.type_list[pred_idx]
                head_match = (pred_type == true_type)
            
            # Проверка dichotomy head
            dich_match = False
            if output.dich_logits is not None:
                dich_pred = (torch.sigmoid(output.dich_logits) > 0.5).long()
                # Конвертация в тип
                flat = (dich_pred[0, 0] * 8 + dich_pred[0, 1] * 4 +
                        dich_pred[0, 2] * 2 + dich_pred[0, 3]).item()
                dich_type = self.type_list[flat]
                dich_match = (dich_type == true_type)
            
            if head_match or dich_match:
                pass_level = "both" if (head_match and dich_match) else \
                             "head_only" if head_match else "dich_only"
                verified_rows.append({
                    "text": text,
                    "true_type": true_type,
                    "pass_level": pass_level,
                    "head_match": head_match,
                    "dich_match": dich_match,
                })
        
        return pd.DataFrame(verified_rows)
