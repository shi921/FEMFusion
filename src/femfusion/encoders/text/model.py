"""Frozen Hugging Face token-level Transformer + learnable numeric/text projections."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from femfusion.common.contracts import EncoderOutput
from .metadata import INPUT_FIELDS

DEFAULT_BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"


def load_pretrained(model_name: str = DEFAULT_BACKBONE, local_files_only: bool = True):
    """Never fall back to randomly initialized transformer weights."""
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError as exc:
        raise RuntimeError("Install Text dependencies: python -m pip install -e '.[text]' ") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"Pretrained tokenizer/model '{model_name}' not available in local cache. "
            "With Internet access prefetch once: python -c \"from transformers import "
            "AutoTokenizer,AutoModel; name='sentence-transformers/all-MiniLM-L6-v2'; "
            "AutoTokenizer.from_pretrained(name); AutoModel.from_pretrained(name)\". "
            "Re-run with cached weights; random fallback is forbidden."
        ) from exc
    return tokenizer, backbone


class EngineeringTextEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int | None = None, embedding_dim: int = 128,
                 freeze_backbone: bool = True, numeric_fields: tuple[str, ...] = INPUT_FIELDS):
        super().__init__()
        self.backbone = backbone
        if hidden_size is None:
            hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        if hidden_size is None or int(hidden_size) <= 0:
            raise ValueError("backbone config.hidden_size or explicit hidden_size required")
        self.numeric_fields = tuple(numeric_fields)
        self.freeze_backbone = bool(freeze_backbone)
        for param in self.backbone.parameters():
            param.requires_grad_(not self.freeze_backbone)
        self.text_projection = nn.Linear(int(hidden_size), embedding_dim)
        self.token_projection = nn.Linear(int(hidden_size), embedding_dim)
        self.numeric_mlp = nn.Sequential(nn.Linear(2 * len(self.numeric_fields), 128), nn.GELU(),
                                         nn.Linear(128, embedding_dim))
        self.fusion = nn.Sequential(nn.Linear(2 * embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim))
        self.field_embedding = nn.Embedding(len(self.numeric_fields), 16)
        self.masked_field_head = nn.Sequential(nn.Linear(embedding_dim + 16, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, batch: dict[str, Any]) -> EncoderOutput:
        ids, attn = batch["input_ids"], batch["attention_mask"]
        numeric, observed = batch["numeric"], batch["numeric_observed"]
        if ids.ndim != 2 or attn.shape != ids.shape:
            raise ValueError("input_ids/attention_mask must both be [B,L]")
        mask = attn.bool()
        if (~mask.any(dim=1)).any():
            raise ValueError("all-pad text sequence")
        if numeric.shape != (ids.shape[0], len(self.numeric_fields)) or observed.shape != numeric.shape:
            raise ValueError("invalid numeric feature dimensions")
        if self.freeze_backbone:
            self.backbone.eval()  # disable dropout despite enclosing model.train()
            with torch.no_grad():
                result = self.backbone(input_ids=ids, attention_mask=attn)
        else:
            result = self.backbone(input_ids=ids, attention_mask=attn)
        hidden = result.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[:2] != ids.shape:
            raise ValueError("backbone must return token-level last_hidden_state [B,L,H]")
        valid = mask.unsqueeze(-1).to(hidden.dtype)
        text_mean = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.)
        locals_ = self.token_projection(hidden) * valid
        # Share the learned local projection with the global objective; otherwise
        # an unfrozen token projector would receive no gradients at all.
        text = self.text_projection(text_mean) + locals_.sum(dim=1) / valid.sum(dim=1).clamp_min(1.)
        numeric_repr = self.numeric_mlp(torch.cat([numeric * observed, observed], dim=-1))
        global_token = self.fusion(torch.cat([text, numeric_repr], dim=-1))
        # Token positions refer to input_ids and remain reconstructable including specials.
        entity_ids = [[f"tok:{j}:id:{int(ids[i,j].item())}" for j in range(ids.shape[1]) if mask[i,j]]
                      for i in range(ids.shape[0])]
        return EncoderOutput(global_token=global_token, local_tokens=locals_, local_mask=mask,
                             sample_ids=list(batch["sample_ids"]), local_entity_ids=entity_ids,
                             aux={"texts": list(batch["texts"]), "template_ids": list(batch["template_ids"]),
                                  "numeric_fields": list(self.numeric_fields), "numeric_values": batch["raw_numerics"],
                                  "units": batch["units"], "masked_fields": batch["masked_fields"],
                                  "input_ids": ids.detach().cpu().tolist()})

    def predict_masked(self, embedding: torch.Tensor, field_indices: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2 or field_indices.shape != (embedding.shape[0],):
            raise ValueError("invalid masked field head inputs")
        return self.masked_field_head(torch.cat([embedding, self.field_embedding(field_indices)], dim=-1)).squeeze(-1)


def symmetric_infonce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Cross-template same-case positives, other case IDs within batch negatives."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("contrastive tensors must agree [B,D]")
    if a.shape[0] < 2:
        return (a.sum() + b.sum()) * 0
    logits = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5
