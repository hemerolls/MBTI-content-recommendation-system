#!/usr/bin/env python3
"""
Обучение текстового классификатора MBTI.
Двухголовая архитектура + Focal Loss + WeightedRandomSampler + OneCycleLR.

Usage:
    python scripts/train_text.py --config configs/default.yaml
"""
import argparse
import math
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from tqdm import tqdm
import pandas as pd
import re

# Импорт из src
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.text_classifier import MBTIClassifier, focal_bce_with_logits


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Очистка текста от утечки данных и мусора."""
    if not isinstance(text, str):
        return ""
    mbti_pat = (
        r"\b(?:INFP|INFJ|INTP|INTJ|ENTP|ENFP|ISTP|ISFP"
        r"|ENTJ|ISTJ|ENFJ|ISFJ|ESTP|ESFP|ESFJ|ESTJ)\b"
    )
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\|{3,}", "", text)
    text = text.replace('"', "").replace("'", "")
    text = re.sub(mbti_pat, "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip()


MBTI_TYPES = sorted([
    "INFP", "INFJ", "INTP", "INTJ", "ENTP", "ENFP", "ISTP", "ISFP",
    "ENTJ", "ISTJ", "ENFJ", "ISFJ", "ESTP", "ESFP", "ESFJ", "ESTJ",
])
type2id = {t: i for i, t in enumerate(MBTI_TYPES)}
NUM_CLASSES = len(MBTI_TYPES)


def type_to_dich(t: str) -> list[int]:
    return [int(t[0] == "E"), int(t[1] == "S"),
            int(t[2] == "T"), int(t[3] == "J")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    tc = cfg["text"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Head mode: {tc['head_mode']}")
    
    # ── 1. Данные ──
    df = pd.read_csv(cfg["data"]["text_dataset_path"])
    df["text"] = df["post"].apply(clean_text)
    df["label"] = df["type"].map(type2id)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained(tc["model_name"], use_fast=True)
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].values, df["label"].values,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["seed"], stratify=df["label"].values,
    )
    
    # Токенизация (упрощённая, без чанкинга для brevity)
    from datasets import Dataset
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})
    
    def tokenize_fn(examples):
        enc = tokenizer(
            examples["text"], padding="max_length", truncation=True,
            max_length=tc["max_length"], return_tensors="pt",
        )
        enc["labels"] = examples["label"]
        # Добавляем dich_label
        dich_labels = [type_to_dich(MBTI_TYPES[l]) for l in examples["label"]]
        enc["dich_label"] = torch.tensor(dich_labels, dtype=torch.float32)
        return enc
    
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")
    test_ds.set_format("torch")
    
    # ── 2. Sampler + DataLoader ──
    sampler = None
    if tc["use_sampler"]:
        counts = Counter(train_labels)
        weights = [1.0 / (counts[l] + tc["sampler_alpha"]) for l in train_labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    train_dl = DataLoader(train_ds, batch_size=tc["batch_size"],
                          sampler=sampler, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=tc["batch_size"],
                         shuffle=False, num_workers=0)
    
    # ── 3. Модель ──
    config = AutoConfig.from_pretrained(tc["model_name"])
    base_model = AutoModel.from_pretrained(tc["model_name"])
    lora_cfg = LoraConfig(
        r=tc["lora_r"], lora_alpha=tc["lora_alpha"],
        target_modules=tc["target_modules"],
        lora_dropout=tc["lora_dropout"], bias="none", use_dora=tc["use_dora"],
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    model = MBTIClassifier(peft_model, config.hidden_size, mode=tc["head_mode"]).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    
    # ── 4. Loss weights ──
    dich_data = np.array([type_to_dich(MBTI_TYPES[l]) for l in train_labels])
    pos_counts = dich_data.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    dich_weights = np.clip(neg_counts / (pos_counts + 1e-6), *tc["dich_weight_clip"])
    dich_weights_tensor = torch.FloatTensor(dich_weights).to(device)
    
    criterion_ce = nn.CrossEntropyLoss()
    
    # ── 5. Optimizer + Scheduler ──
    head_params = list(model.proj.parameters())
    if tc["head_mode"] in ("type16", "both"):
        head_params += list(model.type16_head.parameters())
    if tc["head_mode"] in ("dichotomy", "both"):
        head_params += list(model.dich_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": tc["lr"]},
        {"params": head_params, "lr": tc["lr"] * tc["head_lr_mult"]},
    ], weight_decay=0.01)
    
    steps_per_epoch = math.ceil(len(train_dl) / tc["grad_accum_steps"])
    total_steps = steps_per_epoch * tc["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[tc["lr"], tc["lr"] * tc["head_lr_mult"]],
        total_steps=total_steps, pct_start=0.1, anneal_strategy="cos",
    )
    
    # ── 6. Training loop ──
    best_f1 = 0.0
    os.makedirs(os.path.dirname(tc["checkpoint_path"]), exist_ok=True)
    
    for epoch in range(tc["epochs"]):
        model.train()
        run_loss = 0.0
        correct = total = 0
        
        pbar = tqdm(train_dl, desc=f"Ep {epoch+1}/{tc['epochs']}")
        for step, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            dich_labs = batch["dich_label"].to(device)
            
            output = model(ids, mask)
            
            loss = 0.0
            if output.type16_logits is not None:
                loss += tc["type16_loss_weight"] * criterion_ce(output.type16_logits, labs)
            if output.dich_logits is not None:
                loss += tc["dich_loss_weight"] * focal_bce_with_logits(
                    output.dich_logits, dich_labs,
                    pos_weight=dich_weights_tensor, gamma=tc["focal_gamma"],
                )
            loss = loss / tc["grad_accum_steps"]
            loss.backward()
            
            if (step + 1) % tc["grad_accum_steps"] == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            run_loss += loss.item() * tc["grad_accum_steps"]
            if output.type16_logits is not None:
                preds = output.type16_logits.argmax(1)
                correct += (preds == labs).sum().item()
                total += labs.size(0)
            
            pbar.set_postfix(loss=f"{run_loss/(step+1):.4f}", acc=f"{correct/max(total,1):.4f}")
        
        # Eval
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for batch in test_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                output = model(ids, mask)
                if output.type16_logits is not None:
                    all_p.extend(output.type16_logits.argmax(1).cpu().numpy())
                    all_l.extend(batch["labels"].numpy())
        
        v_acc = accuracy_score(all_l, all_p)
        v_f1 = f1_score(all_l, all_p, average="macro")
        print(f"  Val acc={v_acc:.4f}  F1={v_f1:.4f}")
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), tc["checkpoint_path"])
            print(f"  💾 Saved (F1={best_f1:.4f})")
    
    print(f"\n✅ Done. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
