"""HF-tokenized two-view dataset with unrounded named engineering numerics."""
from __future__ import annotations

import random
from typing import Any

import torch
from torch.utils.data import Dataset

from .metadata import EngineeringMetadata, INPUT_FIELDS, UNITS
from .templates import TEMPLATE_IDS, render


class EngineeringTextDataset(Dataset):
    def __init__(self, metadata: EngineeringMetadata, split: str, tokenizer: Any,
                 max_length: int = 192, seed: int = 42, template_id: str = "concise",
                 augment: bool = False):
        if split not in metadata.splits:
            raise ValueError(f"unknown split {split}")
        if template_id not in TEMPLATE_IDS:
            raise ValueError(f"unknown template {template_id}")
        if metadata.normalizer is None:
            raise ValueError("fit/load train normalizer before dataset")
        self.metadata, self.records, self.tokenizer = metadata, metadata.splits[split], tokenizer
        self.split, self.max_length, self.seed, self.epoch = split, max_length, seed, 0
        self.template_id, self.augment = template_id, bool(augment and split == "train")
        if max_length < 8:
            raise ValueError("max_length must be >= 8")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _template_for(self, index: int, view: int) -> str:
        if not self.augment:
            return self.template_id if view == 0 else TEMPLATE_IDS[(TEMPLATE_IDS.index(self.template_id) + 1) % len(TEMPLATE_IDS)]
        # Deterministic per sample/epoch/view regardless of dataloader shuffle/worker count.
        rng = random.Random(self.seed + 104729 * self.epoch + 997 * index)
        first = rng.randrange(len(TEMPLATE_IDS))
        return TEMPLATE_IDS[(first + view) % len(TEMPLATE_IDS)]

    def view(self, index: int, template_id: str, masked_field: str | None = None) -> dict[str, Any]:
        record = self.records[index]
        description = render(record, self.metadata, template_id, masked_field=masked_field)
        token = self.tokenizer(description.text, truncation=True, max_length=self.max_length,
                               add_special_tokens=True, return_attention_mask=True)
        ids, attention = token["input_ids"], token["attention_mask"]
        if not isinstance(ids, list) or not isinstance(attention, list) or len(ids) != len(attention):
            raise ValueError("tokenizer must return unbatched aligned input_ids and attention_mask lists")
        norm = self.metadata.normalize(record.values)
        observed = [1.0] * len(INPUT_FIELDS)
        if masked_field is not None:
            p = INPUT_FIELDS.index(masked_field)
            norm[p] = 0.0
            observed[p] = 0.0
        return {"case_id": description.case_id, "template_id": description.template_id,
                "text": description.text, "input_ids": list(ids), "attention_mask": list(attention),
                "numeric": norm.tolist(), "numeric_observed": observed,
                "raw_numeric": description.values, "units": description.units, "masked_field": masked_field}

    def __getitem__(self, index: int) -> dict[str, Any]:
        first = self.view(index, self._template_for(index, 0))
        other = self.view(index, self._template_for(index, 1))
        masked_field = INPUT_FIELDS[(index + (self.epoch if self.augment else 0)) % len(INPUT_FIELDS)]
        masked = self.masked_view(index, self._template_for(index, 0), masked_field)
        return {"first": first, "second": other, "masked": masked}

    def masked_view(self, index: int, template_id: str, masked_field: str) -> dict[str, Any]:
        return self.view(index, template_id, masked_field=masked_field)


def collate_views(views: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    if not views:
        raise ValueError("empty text batch")
    length = max(len(v["input_ids"]) for v in views)
    tokens = torch.full((len(views), length), int(pad_token_id), dtype=torch.long)
    attn = torch.zeros(len(views), length, dtype=torch.long)
    nums = torch.zeros(len(views), len(INPUT_FIELDS), dtype=torch.float32)
    observed = torch.zeros_like(nums)
    for i, v in enumerate(views):
        n = len(v["input_ids"])
        tokens[i, :n] = torch.as_tensor(v["input_ids"], dtype=torch.long)
        attn[i, :n] = torch.as_tensor(v["attention_mask"], dtype=torch.long)
        nums[i] = torch.as_tensor(v["numeric"], dtype=torch.float32)
        observed[i] = torch.as_tensor(v["numeric_observed"], dtype=torch.float32)
    return {"input_ids": tokens, "attention_mask": attn, "numeric": nums,
            "numeric_observed": observed, "sample_ids": [v["case_id"] for v in views],
            "template_ids": [v["template_id"] for v in views], "texts": [v["text"] for v in views],
            "raw_numerics": [v["raw_numeric"] for v in views], "units": [v["units"] for v in views],
            "masked_fields": [v["masked_field"] for v in views]}


def collate_pairs(pairs: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, dict[str, Any]]:
    return {"first": collate_views([p["first"] for p in pairs], pad_token_id),
            "second": collate_views([p["second"] for p in pairs], pad_token_id),
            "masked": collate_views([p["masked"] for p in pairs], pad_token_id)}
