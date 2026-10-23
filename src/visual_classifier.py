"""
Визуальный классификатор MBTI: CLIP ViT-B/32 + Stacking Ensemble
Финальная версия из ВКР (точность 64% по дихотомиям).
"""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple


class VisualMBTIClassifier:
    """
    Визуальный классификатор на основе CLIP + Stacking Ensemble.
    
    Извлекает эмбеддинги через CLIP ViT-B/32, затем использует
    ансамбль бустинга (XGBoost + LightGBM + CatBoost + MLP) 
    для классификации по 4 дихотомиям.
    """
    
    AXES = ["E/I", "S/N", "T/F", "J/P"]
    
    def __init__(self, clip_model_id: str = "openai/clip-vit-base-patch32"):
        self.clip_model_id = clip_model_id
        self.processor: Optional[CLIPProcessor] = None
        self.clip_model: Optional[CLIPModel] = None
        self.ensembles: Dict[str, object] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
    def load_clip(self, device: str = "auto") -> None:
        """Загрузка CLIP модели."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(self.clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_id).to(device)
        self.clip_model.eval()
        self.device = device
        
    def extract_embeddings(
        self, image_paths: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """Извлечение нормализованных CLIP-эмбеддингов."""
        all_embs = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            images = []
            for p in batch:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224)))
                    
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            all_embs.append(feats.cpu().numpy())
            
        return np.vstack(all_embs) if all_embs else np.array([])
    
    def predict(self, embeddings: np.ndarray) -> List[str]:
        """Предсказание MBTI-типов по эмбеддингам."""
        predictions = {}
        for axis in self.AXES:
            if axis in self.ensembles:
                pred = self.ensembles[axis].predict(embeddings)
                le = self.label_encoders[axis]
                predictions[axis] = le.inverse_transform(pred)
            else:
                raise ValueError(f"Ensemble for {axis} not loaded")
                
        # Сборка типа из дихотомий
        types = []
        for i in range(len(embeddings)):
            t = "".join(predictions[ax][i] for ax in self.AXES)
            types.append(t)
        return types
