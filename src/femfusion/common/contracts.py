"""Shared output contract; ready to use across the three encoders."""
from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class EncoderOutput:
    """Shared batched representation.

    local_mask=True marks valid tokens. local_entity_ids[b] contains exactly
    the IDs of valid tokens in local_tokens[b], in order.
    """
    global_token: torch.Tensor
    local_tokens: torch.Tensor
    local_mask: torch.Tensor
    sample_ids: list[str]
    local_entity_ids: list[list[str | int]]
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def embedding_dim(self) -> int:
        return int(self.global_token.shape[-1])

    def validate(self) -> None:
        if self.global_token.ndim != 2:
            raise ValueError("global_token must have shape [B,D]")
        if self.local_tokens.ndim != 3:
            raise ValueError("local_tokens must have shape [B,N,D]")
        if self.local_mask.ndim != 2 or self.local_mask.dtype != torch.bool:
            raise ValueError("local_mask must be a bool tensor [B,N]")
        b, d = self.global_token.shape
        if self.local_tokens.shape[0] != b or self.local_tokens.shape[2] != d:
            raise ValueError("batch size or embedding dimension mismatch")
        if self.local_mask.shape != self.local_tokens.shape[:2]:
            raise ValueError("local_mask shape mismatch")
        if len(self.sample_ids) != b or len(self.local_entity_ids) != b:
            raise ValueError("sample_ids/local_entity_ids length must equal batch size")
        if not torch.is_floating_point(self.global_token) or not torch.is_floating_point(self.local_tokens):
            raise TypeError("global_token and local_tokens must have floating dtype")
        if self.global_token.device != self.local_tokens.device or self.global_token.device != self.local_mask.device:
            raise ValueError("encoder tensors must be on the same device")
        for i, ids in enumerate(self.local_entity_ids):
            n_valid = int(self.local_mask[i].sum().item())
            if len(ids) != n_valid:
                raise ValueError(f"local_entity_ids[{i}] has {len(ids)} IDs, expected {n_valid}")
