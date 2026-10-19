"""Isolated proxy-text diagnostics; no CAD/FEM alignment claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .dataset import EngineeringTextDataset, collate_views
from .metadata import EngineeringMetadata, INPUT_FIELDS
from .model import EngineeringTextEncoder, load_pretrained
from .train import seed_everything, to_device
from .templates import TEMPLATE_IDS


@torch.no_grad()
def evaluate_split(model: EngineeringTextEncoder, dataset: EngineeringTextDataset,
                   device: str = "cpu", batch_size: int = 8) -> dict[str, Any]:
    model.eval()
    if not dataset.records:
        raise ValueError("cannot evaluate empty dataset")
    # Always evaluate all three controlled phrasings, regardless of training template.
    embeddings: list[np.ndarray] = []
    embeddings_text_only: list[np.ndarray] = []
    total_tokens, total_padding, checked = 0, 0, 0
    for template in TEMPLATE_IDS:
        out_rows, only_rows = [], []
        for start in range(0, len(dataset), batch_size):
            views = [dataset.view(i, template) for i in range(start, min(start + batch_size, len(dataset)))]
            batch = to_device(collate_views(views, dataset.tokenizer.pad_token_id or 0), torch.device(device))
            enc = model(batch)
            enc.validate()
            if enc.global_token.shape != (len(views), 128) or enc.local_tokens.shape[:2] != enc.local_mask.shape:
                raise ValueError("text output shape mismatch")
            if not torch.equal(enc.local_mask, batch["attention_mask"].bool()):
                raise ValueError("text attention_mask mismatch")
            if not all(len(ids) == int(mask.sum()) for ids, mask in zip(enc.local_entity_ids, enc.local_mask)):
                raise ValueError("text entity-id count mismatch")
            total_tokens += int(enc.local_mask.sum().item())
            total_padding += int((~enc.local_mask).sum().item())
            checked += len(views)
            out_rows.append(enc.global_token.cpu())
            batch["numeric_observed"] = torch.zeros_like(batch["numeric_observed"])
            only_rows.append(model(batch).global_token.cpu())
        embeddings.append(torch.cat(out_rows).numpy())
        embeddings_text_only.append(torch.cat(only_rows).numpy())

    def similarities(matrices: list[np.ndarray]) -> dict[str, float | int | None]:
        mats = np.stack(matrices, axis=0)  # [T,C,D]
        mats /= np.maximum(np.linalg.norm(mats, axis=-1, keepdims=True), 1e-12)
        same = [float((mats[0] * mats[1]).sum(-1).mean()),
                float((mats[0] * mats[2]).sum(-1).mean()),
                float((mats[1] * mats[2]).sum(-1).mean())]
        n = mats.shape[1]
        # Negatives compared across different case_ids, not different templates of same case.
        sim = mats[0] @ mats[1].T
        negatives = sim[~np.eye(n, dtype=bool)]
        correct = float(np.mean(np.argmax(sim, axis=1) == np.arange(n)))
        different = float(negatives.mean()) if negatives.size else None
        return {"same_case_cross_template_cosine": float(np.mean(same)),
                "different_case_cross_template_cosine": different,
                "cosine_separation_gap": float(np.mean(same) - different) if different is not None else None,
                "template0_to_template1_top1_case_retrieval": correct,
                "unique_cases": n}

    # Mask a *different* field in each sample. Neither numerical branch nor text
    # contains that field's target; report field units from the real normalizer.
    errors: dict[str, list[float]] = {k: [] for k in INPUT_FIELDS}
    for start in range(0, len(dataset), batch_size):
        indexes = range(start, min(start + batch_size, len(dataset)))
        for f in INPUT_FIELDS:
            views = [dataset.masked_view(i, "concise", f) for i in indexes]
            batch = to_device(collate_views(views, dataset.tokenizer.pad_token_id or 0), torch.device(device))
            encoded = model(batch)
            pos = INPUT_FIELDS.index(f)
            which = torch.full((len(views),), pos, dtype=torch.long, device=device)
            prediction = model.predict_masked(encoded.global_token, which).cpu().numpy()
            mean = dataset.metadata.normalizer["mean"][pos]
            scale = dataset.metadata.normalizer["std"][pos]
            reference = np.asarray([v["raw_numeric"][f] for v in views])
            errors[f].extend(((prediction * scale + mean) - reference).tolist())
    mae = {f: {"mae": float(np.mean(np.abs(v))), "rmse": float(np.sqrt(np.mean(np.square(v)))),
               "unit": dataset.metadata.normalizer["units"][f]} for f, v in errors.items()}
    return {"split": dataset.split, "cases": len(dataset), "template_count": len(TEMPLATE_IDS),
            "full_input": similarities(embeddings), "numeric_ablated": similarities(embeddings_text_only),
            "masked_field_recovery": mae, "contract": {
                "global_shape": "[B,128]", "local_shape": "[B,Nmax,128]",
                "mask_is_bool_and_matches_attention": True,
                "entity_ids_match_mask": True, "views_validated": checked,
                "real_tokens": total_tokens, "pad_positions": total_padding},
            "scope": "controlled synthetic-template engineering prose only, NOT technical-document comprehension"}


def evaluate(config: dict[str, Any], data_root: str, checkpoint_dir: str,
             split: str = "test", device: str = "cpu", batch_size: int = 8,
             output_dir: str | None = None) -> dict[str, Any]:
    path = Path(checkpoint_dir)
    metadata = EngineeringMetadata.from_directory(data_root, config["context"], config.get("expected_counts"))
    metadata.load_normalizer(path / "normalizer.json")
    tokenizer, backbone = load_pretrained(config["encoder"]["pretrained_model"])
    model = EngineeringTextEncoder(backbone, embedding_dim=config["encoder"]["embedding_dim"],
                                   freeze_backbone=config["encoder"]["freeze_backbone"]).to(device)
    ckpt = torch.load(path / "best.pt", map_location=device, weights_only=True)
    if ckpt.get("input_fields") != list(INPUT_FIELDS) or ckpt.get("pretrained_model") != config["encoder"]["pretrained_model"]:
        raise ValueError("checkpoint fields or pretrained backbone differs from config")
    model.load_state_dict(ckpt["model_state"])
    dataset = EngineeringTextDataset(metadata, split, tokenizer, config["encoder"]["max_length"],
                                     config["training"]["seed"])
    result = evaluate_split(model, dataset, device, batch_size)
    result["checkpoint_epoch"] = ckpt["epoch"]
    destination = Path(output_dir) if output_dir is not None else path
    destination.mkdir(parents=True, exist_ok=True)
    destination.joinpath(f"evaluation_{split}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate proxy-template text encoder only")
    parser.add_argument("--config", default="configs/encoders/text.yaml")
    parser.add_argument("--data-root", default="data/raw/agentfem")
    parser.add_argument("--checkpoint-dir", default="outputs/text/training")
    parser.add_argument("--output-dir", default="outputs/text/evaluation")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(config["training"]["seed"]))
    result = evaluate(config, args.data_root, args.checkpoint_dir, args.split, args.device, args.batch_size,
                      args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
