#!/usr/bin/env python3
"""Fetch only the official CSV/protocol and the STEP/FEM field archives on demand."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ID = "HaomingLuo/AgentFEM-Bracket-Elasticity"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "raw" / "agentfem"
SPLITS = ("train", "validation", "test")


def required_files(modality: str) -> list[str]:
    if modality not in ("all", "cad", "fem", "text"):
        raise ValueError(f"Unknown modality: {modality}")
    result = [f"{split}.csv" for split in SPLITS] + ["protocol.json"]
    if modality in ("all", "cad"):
        result.extend(
            f"geometry-{split}-{part:02}.zip"
            for split in SPLITS
            for part in range(4 if split == "train" else 1)
        )
    if modality in ("all", "fem"):
        result.extend(
            f"fields-{split}-{part:02}.zip"
            for split in SPLITS
            for part in range(4 if split == "train" else 1)
        )
    return result


def check(modality: str, data_dir: Path) -> list[str]:
    missing = [name for name in required_files(modality) if not (data_dir / name).is_file() or not (data_dir / name).stat().st_size]
    if missing:
        print(f"缺少 {len(missing)} 个 {modality} 模态数据文件：{data_dir}", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print(f"请先执行: ./run.sh download {modality}", file=sys.stderr)
    else:
        print(f"数据文件已齐全：{modality} ({len(required_files(modality))} 个文件)")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the official AgentFEM data (not included in source archive)")
    parser.add_argument("modality", nargs="?", default="all", choices=("all", "cad", "fem", "text"))
    parser.add_argument("--data-dir", type=Path, default=DESTINATION)
    parser.add_argument("--revision", default=None, help="Optional Hugging Face commit SHA / revision")
    parser.add_argument("--check", action="store_true", help="Check required local files without networking")
    args = parser.parse_args(argv)
    if args.check:
        return 1 if check(args.modality, args.data_dir) else 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        parser.error("请先运行 ./run.sh setup，或安装 huggingface-hub")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    filenames = required_files(args.modality)
    # Download official dataset license when present (not a model input).
    for index, name in enumerate(filenames, 1):
        print(f"[{index}/{len(filenames)}] {name}", flush=True)
        hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=name,
            local_dir=str(args.data_dir), revision=args.revision,
        )
    return 1 if check(args.modality, args.data_dir) else 0


if __name__ == "__main__":
    raise SystemExit(main())
