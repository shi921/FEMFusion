#!/usr/bin/env python3
"""No-training verification of three independently integrated AgentFEM encoders.

Run from the repository root after `pip install -e .`.
No download, training, checkpoint loading, or generated cache required.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile

SPLITS = ("train", "validation", "test")
EXPECTED = {"train": 128, "validation": 16, "test": 32}


def run(root: Path, require_optional: bool = False) -> dict:
    report = {"dependencies": {}, "data": {}, "checks": {}, "errors": [], "warnings": []}
    for name in ("torch", "numpy", "pandas", "yaml", "OCP", "torch_geometric", "transformers"):
        try:
            importlib.import_module(name)
            report["dependencies"][name] = "ok"
        except Exception as exc:
            report["dependencies"][name] = f"missing or import failed: {type(exc).__name__}: {exc}"
            if name in ("torch", "numpy", "pandas", "yaml") or require_optional:
                report["errors"].append(f"Dependency {name}: {exc}")
            else:
                report["warnings"].append(f"Optional dependency {name}: {exc}")
    if report["errors"]:
        return report

    try:
        from femfusion.common.contracts import EncoderOutput
        report["checks"]["shared_contract"] = EncoderOutput.__name__
    except Exception as exc:
        report["errors"].append(f"Shared contract: {exc}")
        return report

    if not all((root / (s + ".csv")).exists() for s in SPLITS):
        report["warnings"].append(f"Official CSVs not found in {root}; run ./run.sh download to fetch data")
        return report

    try:
        from femfusion.encoders.text.metadata import EngineeringMetadata
        text = EngineeringMetadata.from_directory(root, expected_counts=EXPECTED)
        report["data"]["text"] = {s: len(text.splits[s]) for s in SPLITS}
        text_ids = {s: {r.case_id for r in text.splits[s]} for s in SPLITS}
        text.fit_normalizer()
        report["checks"]["text_train_only_normalizer"] = text.normalizer["fitted_split"]
    except Exception as exc:
        report["errors"].append(f"Text metadata: {type(exc).__name__}: {exc}")
        text_ids = None

    try:
        from femfusion.encoders.fem.dataset import case_records
        records = {s: case_records(root, s) for s in SPLITS}
        report["data"]["fem"] = {s: len(records[s]) for s in SPLITS}
        if text_ids is not None:
            assert all({r.case_id for r in records[s]} == text_ids[s] for s in SPLITS)
            report["checks"]["fem_text_case_id_alignment"] = "ok"
    except Exception as exc:
        report["warnings"].append(f"FEM fields ZIP check: {type(exc).__name__}: {exc}")

    try:
        from femfusion.encoders.cad.brep_parser import step_sources, match_case_id
        from femfusion.encoders.cad.dataset import read_splits
        splits = read_splits(root)
        sources = step_sources(root)
        mapped = {s: [match_case_id(cid, sources) for cid in splits[s]] for s in SPLITS}
        report["data"]["cad"] = {s: len(mapped[s]) for s in SPLITS}
        if text_ids is not None:
            assert all(set(splits[s]) == text_ids[s] for s in SPLITS)
            report["checks"]["cad_text_case_id_alignment"] = "ok"
        if report["dependencies"]["OCP"] == "ok":
            from femfusion.encoders.cad.brep_parser import parse_source
            cid = splits["train"][0]
            with tempfile.TemporaryDirectory(prefix="agentfem-verify-step-") as tmp:
                geom = parse_source(mapped["train"][0], tmp)
            report["checks"]["real_cad_step"] = {"case_id": cid, "faces": len(geom.faces), "edges": len(geom.edges)}
    except Exception as exc:
        message = f"CAD STEP ZIP/parse check: {type(exc).__name__}: {exc}"
        # With --require-optional, a real STEP parse error must fail verification,
        # not be silently downgraded to a warning while returning exit code 0.
        target = "errors" if require_optional else "warnings"
        report[target].append(message)
    return report


def main():
    p = argparse.ArgumentParser(description="Verify AgentFEM CAD/FEM/Text encoder environment & case IDs")
    p.add_argument("--data-dir", default="data/raw/agentfem")
    p.add_argument("--require-optional", action="store_true", help="Fail if optional dependencies are missing or real CAD STEP parsing fails")
    args = p.parse_args()
    report = run(Path(args.data_dir), args.require_optional)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
