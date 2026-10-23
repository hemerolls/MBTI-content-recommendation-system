"""
Текстовый классификатор MBTI: DistilBERT + DoRA + 2-Head Architecture
Финальная версия из ВКР (точность 48% с LLM-аугментацией).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MBTIOutput:
    """Контейнер выходов модели."""
    type16_logits: Optional[torch.Tensor] = None   # (B, 16)
    dich_logits: Optional[torch.Tensor] = None     # (B, 4)
    attentions: Optional[Tuple] = None


class MBTIClassifier(nn.Module):
    """
    Двухголовый классификатор MBTI.
    
    Архитектура: encoder → shared projection → {type16_head, dich_head}
    
    Args:
        encoder: HuggingFace model (DistilBERT + LoRA/DoRA)
        hidden: размер скрытого слоя (обычно 768)
        mode: "type16" | "dichotomy" | "both"
    """
    def __init__(self, encoder: nn.Module, hidden: int, mode: str = "both"):
        super().__init__()
        assert mode in ("type16", "dichotomy", "both"), f"Invalid mode: {mode}"
        self.encoder = encoder
        self.mode = mode
        
        # Shared projection
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(0.25),
        )
        
        # Heads (создаются всегда для совместимости save/load)
        self.type16_head = nn.Linear(hidden, 16)
        self.dich_head = nn.Linear(hidden, 4)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> MBTIOutput:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        h = self.proj(out.last_hidden_state[:, 0])  # [CLS] token
        
        res = MBTIOutput()
        if self.mode in ("type16", "both"):
            res.type16_logits = self.type16_head(h)
        if self.mode in ("dichotomy", "both"):
            res.dich_logits = self.dich_head(h)
        if output_attentions:
            res.attentions = out.attentions
            
        return res


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Binary Focal Loss с pos_weight.
    
    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)
    
    Args:
        logits: (B, 4) raw logits
        targets: (B, 4) binary labels
        pos_weight: (4,) веса для положительных классов
        gamma: фокусирующий параметр (0 = обычная BCE)
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    with torch.no_grad():
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_w = (1 - p_t).pow(gamma)
    return (bce * focal_w).mean()
