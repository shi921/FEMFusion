"""Single-modality template invariance + properly masked attribute completion."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from .dataset import EngineeringTextDataset, collate_pairs
from .metadata import EngineeringMetadata, INPUT_FIELDS
from .model import EngineeringTextEncoder, load_pretrained, symmetric_infonce


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(view: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in view.items()}


def step_losses(model: EngineeringTextEncoder, batch: dict[str, dict[str, Any]],
                temperature: float = 0.1, masked_weight: float = 0.1,
                device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    a = to_device(batch["first"], torch.device(device))
    b = to_device(batch["second"], torch.device(device))
    m = to_device(batch["masked"], torch.device(device))
    if len(set(a["sample_ids"])) != len(a["sample_ids"]):
        raise ValueError("contrastive batch contains duplicated case IDs")
    # Drop numeric features for the second contrastive view: prevents an identical
    # structured numeric vector being the sufficient positive-pair shortcut.
    b["numeric_observed"] = torch.zeros_like(b["numeric_observed"])
    first, second = model(a), model(b)
    contrastive = symmetric_infonce(first.global_token, second.global_token, temperature)
    masked_output = model(m)
    field_indices = torch.as_tensor([INPUT_FIELDS.index(f) for f in m["masked_fields"]],
                                    device=device, dtype=torch.long)
    # Supervised field target lives only in the *unmasked* ground truth; masked
    # template contains [unspecified] and numeric_observed=0 for that field.
    targets = a["numeric"].gather(1, field_indices[:, None]).squeeze(1)
    masked_loss = F.smooth_l1_loss(model.predict_masked(masked_output.global_token, field_indices), targets)
    return {"loss": contrastive + masked_weight * masked_loss,
            "contrastive": contrastive, "masked_field_huber": masked_loss}


def run_epoch(model: EngineeringTextEncoder, loader: DataLoader, optimizer: Any,
              device: torch.device, temperature: float, masked_weight: float) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    sums: dict[str, float] = {}
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            losses = step_losses(model, batch, temperature, masked_weight, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
            batch_count = len(batch["first"]["sample_ids"])
            count += batch_count
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * batch_count
    return {k: v / max(1, count) for k, v in sums.items()}


def create_loader(dataset: EngineeringTextDataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=lambda rows: collate_pairs(rows, dataset.tokenizer.pad_token_id or 0))


def train(config: dict[str, Any], data_root: str, checkpoint_dir: str,
          device: str = "cpu", local_files_only: bool = True) -> dict[str, Any]:
    cfg, tcfg = config["encoder"], config["training"]
    seed_everything(int(tcfg["seed"]))
    metadata = EngineeringMetadata.from_directory(data_root, config["context"], config.get("expected_counts"))
    metadata.fit_normalizer()  # train ONLY, no validation/test leakage
    tokenizer, backbone = load_pretrained(cfg["pretrained_model"], local_files_only=local_files_only)
    model = EngineeringTextEncoder(backbone, embedding_dim=int(cfg["embedding_dim"]),
                                   freeze_backbone=bool(cfg["freeze_backbone"])).to(device)
    ds_train = EngineeringTextDataset(metadata, "train", tokenizer, int(cfg["max_length"]),
                                      int(tcfg["seed"]), augment=True)
    ds_val = EngineeringTextDataset(metadata, "validation", tokenizer, int(cfg["max_length"]),
                                    int(tcfg["seed"]), augment=False)
    train_loader = create_loader(ds_train, int(tcfg["batch_size"]), shuffle=True)
    val_loader = create_loader(ds_val, int(tcfg["batch_size"]))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(tcfg["lr"]))
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata.save_normalizer(path / "normalizer.json")
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    best, no_improvement, history = float("inf"), 0, []
    epochs = int(tcfg["epochs"])
    for epoch in range(epochs):
        ds_train.set_epoch(epoch)
        tr = run_epoch(model, train_loader, optimizer, torch.device(device),
                       float(tcfg["temperature"]), float(tcfg["masked_weight"]))
        val = run_epoch(model, val_loader, None, torch.device(device),
                        float(tcfg["temperature"]), float(tcfg["masked_weight"]))
        history.append({"epoch": epoch + 1, "train": tr, "validation": val})
        print(json.dumps(history[-1]), flush=True)
        if val["loss"] < best - float(tcfg.get("min_delta", 1e-5)):
            best, no_improvement = val["loss"], 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1,
                        "best_validation_loss": best, "pretrained_model": cfg["pretrained_model"],
                        "freeze_backbone": cfg["freeze_backbone"], "embedding_dim": cfg["embedding_dim"],
                        "input_fields": list(INPUT_FIELDS)}, path / "best.pt")
        else:
            no_improvement += 1
            if no_improvement >= int(tcfg["patience"]):
                break
    summary = {"best_validation_loss": best, "epochs_run": len(history),
               "best_checkpoint": str(path / "best.pt"), "note": "proxy templates, not real technical documents"}
    (path / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (path / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train isolated FEMFusion proxy text encoder")
    parser.add_argument("--config", default="configs/encoders/text.yaml")
    parser.add_argument("--data-root", default="data/raw/agentfem")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit HF pretrained download")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(train(cfg, args.data_root, args.checkpoint_dir or cfg["training"]["checkpoint_dir"],
                           device=args.device, local_files_only=not args.allow_download), indent=2))


if __name__ == "__main__":
    main()
