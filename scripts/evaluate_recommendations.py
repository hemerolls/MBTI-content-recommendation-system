#!/usr/bin/env python3
"""
Оценка рекомендательной системы.
Сравнивает: Personality Centroid, Dichotomy Blend, Global Mean, Random, Ranking Model.

Usage:
    python scripts/evaluate_recommendations.py --config configs/default.yaml
"""
import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recommendation import ColdStartRecommender


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ranking_metrics(user_emb, test_embs, test_types, target_type, k_values):
    sims = cosine_similarity(user_emb.reshape(1, -1), test_embs)[0]
    order = np.argsort(-sims)
    rel = (test_types == target_type).astype(float)
    total_rel = rel.sum()
    res = {}
    for k in k_values:
        top = order[:k]
        hits = rel[top].sum()
        res[f"P@{k}"] = hits / k
        dcg = sum(rel[order[i]] / np.log2(i + 2) for i in range(k))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, int(total_rel))))
        res[f"NDCG@{k}"] = dcg / idcg if idcg > 0 else 0.0
    for i, idx in enumerate(order):
        if rel[idx] == 1:
            res["MRR"] = 1.0 / (i + 1)
            break
    else:
        res["MRR"] = 0.0
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    rc = cfg["recommendation"]
    
    # Загрузка эмбеддингов
    embeddings = np.load("checkpoints/visual/image_embeddings.npy")
    metadata = pd.read_csv("checkpoints/visual/image_metadata.csv")
    types = metadata["personality_type"].values
    
    unique_types = sorted(set(types))
    k_values = rc["k_values"]
    
    # Split
    train_idx, test_idx = train_test_split(
        np.arange(len(types)), test_size=0.3,
        random_state=cfg["seed"], stratify=types,
    )
    train_embs, test_embs = embeddings[train_idx], embeddings[test_idx]
    train_types, test_types = types[train_idx], types[test_idx]
    
    recommender = ColdStartRecommender(train_embs, train_types)
    
    strategies = {
        "Personality Centroid": recommender.personality_centroid,
        "Dichotomy Blend": recommender.dichotomy_blend,
        "Global Mean": lambda _: recommender.global_mean(),
        "Random": lambda _: np.random.randn(embeddings.shape[1]) / 
                            np.linalg.norm(np.random.randn(embeddings.shape[1])),
    }
    
    all_results = {s: defaultdict(list) for s in strategies}
    
    for ptype in tqdm(unique_types, desc="Evaluating"):
        for sname, sfn in strategies.items():
            if sname == "Random":
                runs = defaultdict(list)
                for _ in range(rc["n_random_runs"]):
                    m = ranking_metrics(sfn(ptype), test_embs, test_types, ptype, k_values)
                    for k, v in m.items():
                        runs[k].append(v)
                for k, vs in runs.items():
                    all_results[sname][k].append(np.mean(vs))
            else:
                m = ranking_metrics(sfn(ptype), test_embs, test_types, ptype, k_values)
                for k, v in m.items():
                    all_results[sname][k].append(v)
    
    # Summary
    summary = {}
    for sname in strategies:
        summary[sname] = {metric: np.mean(vals) for metric, vals in all_results[sname].items()}
    
    summary_df = pd.DataFrame(summary).T
    cols = [c for c in ["P@5", "P@10", "P@20", "NDCG@10", "MRR"] if c in summary_df.columns]
    print("\n" + "=" * 70)
    print("  RECOMMENDATION SYSTEM EVALUATION")
    print("=" * 70)
    print(summary_df[cols].round(4).to_string())
    
    # Gains
    print("\n📈 Gains vs Global Mean:")
    glob = summary.get("Global Mean", {})
    pers = summary.get("Personality Centroid", {})
    for metric in cols:
        g = glob.get(metric, 0)
        p = pers.get(metric, 0)
        gain = ((p - g) / g * 100) if g > 0 else float("inf")
        print(f"  {metric}: {p:.4f} vs {g:.4f} → +{gain:.1f}%")
    
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
