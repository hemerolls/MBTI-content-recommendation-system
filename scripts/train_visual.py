#!/usr/bin/env python3
"""
Обучение визуального классификатора MBTI.
CLIP ViT-B/32 → Feature Engineering → Stacking Ensemble (XGBoost+LGBM+CatBoost+MLP).

Usage:
    python scripts/train_visual.py --config configs/default.yaml
"""
import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visual_classifier import VisualMBTIClassifier


AXES = ["E/I", "S/N", "T/F", "J/P"]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class StackingEnsemble:
    """L0: XGBoost + LightGBM + CatBoost + MLP → L1: LogisticRegression."""
    
    def __init__(self, xgb_params=None, lgbm_params=None, seed=42):
        self.templates = {
            "xgb": XGBClassifier(**(xgb_params or {}), objective="binary:logistic",
                                 eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0),
            "lgbm": LGBMClassifier(**(lgbm_params or {}), random_state=seed, n_jobs=-1, verbose=-1),
            "cat": CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05,
                                      random_state=seed, verbose=0, auto_class_weights="Balanced"),
            "mlp": MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation="relu",
                                 max_iter=500, early_stopping=True, validation_fraction=0.15,
                                 random_state=seed),
        }
        self.meta = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        self.fitted = {}
    
    def fit(self, X, y):
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        oof = np.zeros((len(y), len(self.templates)))
        for ti, vi in skf.split(X, y):
            for mi, (name, tmpl) in enumerate(self.templates.items()):
                m = copy.deepcopy(tmpl)
                m.fit(X[ti], y[ti])
                oof[vi, mi] = m.predict_proba(X[vi])[:, 1]
        self.meta.fit(oof, y)
        for name, tmpl in self.templates.items():
            m = copy.deepcopy(tmpl)
            m.fit(X, y)
            self.fitted[name] = m
    
    def predict(self, X):
        l0 = np.column_stack([m.predict_proba(X)[:, 1] for m in self.fitted.values()])
        return self.meta.predict(l0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    vc = cfg["visual"]
    
    # 1. Загрузка данных
    df = pd.read_excel(cfg["data"]["image_dataset_path"])
    df = df.dropna(subset=["image_path"])
    df = df[df["personality_type"].astype(str).str.match(r"^[IE][NS][TF][JP]$")]
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    
    for ax in AXES:
        idx = {"E/I": 0, "S/N": 1, "T/F": 2, "J/P": 3}[ax]
        df[ax] = df["personality_type"].str[idx]
    
    print(f"Rows: {len(df)}")
    
    # 2. CLIP embeddings
    clip_clf = VisualMBTIClassifier(vc["clip_model_id"])
    clip_clf.load_clip()
    embeddings = clip_clf.extract_embeddings(df["image_path"].tolist(), batch_size=vc["batch_size"])
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 3. Labels
    label_encoders = {}
    y_data = {}
    for ax in AXES:
        le = LabelEncoder()
        y_data[ax] = le.fit_transform(df[ax])
        label_encoders[ax] = le
    
    # 4. Scaling + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=vc["pca_variance"], random_state=cfg["seed"])
    X_pca = pca.fit_transform(X_scaled)
    print(f"After PCA: {X_pca.shape}")
    
    # 5. Split
    mbti_full = df["personality_type"].values
    _, _, idx_train, idx_test = train_test_split(
        X_pca, np.arange(len(df)), test_size=cfg["data"]["test_size"],
        random_state=cfg["seed"], stratify=mbti_full,
    )
    X_train, X_test = X_pca[idx_train], X_pca[idx_test]
    
    # 6. Train per-axis ensembles
    os.makedirs("checkpoints/visual", exist_ok=True)
    results = []
    
    for ax in AXES:
        print(f"\n{'='*50}\n  Axis: {ax}\n{'='*50}")
        y_tr = y_data[ax][idx_train]
        y_te = y_data[ax][idx_test]
        
        xgb_params = {"n_estimators": vc["xgb_n_estimators"],
                       "max_depth": vc["xgb_max_depth"],
                       "learning_rate": vc["xgb_learning_rate"]}
        
        ens = StackingEnsemble(xgb_params=xgb_params, seed=cfg["seed"])
        ens.fit(X_train, y_tr)
        
        y_pred = ens.predict(X_test)
        le = label_encoders[ax]
        acc = accuracy_score(le.inverse_transform(y_te), le.inverse_transform(y_pred))
        f1 = f1_score(y_te, y_pred, average="weighted")
        print(f"  Accuracy: {acc:.4f}  F1: {f1:.4f}")
        results.append({"Axis": ax, "Accuracy": acc, "F1": f1})
        
        joblib.dump(ens, f"checkpoints/visual/ensemble_{ax.replace('/', '_')}.pkl")
    
    # Save preprocessors
    joblib.dump(scaler, "checkpoints/visual/scaler.pkl")
    joblib.dump(pca, "checkpoints/visual/pca.pkl")
    joblib.dump(label_encoders, "checkpoints/visual/label_encoders.pkl")
    
    res_df = pd.DataFrame(results)
    print(f"\nMean Accuracy: {res_df['Accuracy'].mean():.4f}")
    print(f"Mean F1:       {res_df['F1'].mean():.4f}")
    print("\n✅ Models saved to checkpoints/visual/")


if __name__ == "__main__":
    main()
