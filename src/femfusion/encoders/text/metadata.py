"""Validated, case-ID keyed AgentFEM tabular conditions (never FEM targets)."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Units checked against official reproduce/bracket_lab/operator_data.py lines 13-18.
INPUT_FIELDS = ("depth", "radius", "waist", "bow", "physical_load_N")
UNITS = {"depth": "m", "radius": "m", "waist": "1", "bow": "m", "physical_load_N": "N"}
TARGET_ONLY_FIELDS = frozenset({"mass_kg", "seat_deflection_mm_at_10kN", "mesh_volume_relative_error"})
SPLITS = ("train", "validation", "test")
# Official reproduce/bracket_lab/operator_data.py PARAMETER_BOUNDS.
PARAMETER_BOUNDS = {"depth": (0.015, 0.032), "radius": (0.010, 0.023),
                    "waist": (0.0, 0.4), "bow": (0.005, 0.030)}
# Sources checked against local official README.md and reproduce/bracket_lab/case.py.
DEFAULT_CONTEXT: dict[str, Any] = {
    "component": "symmetric double-arm support bracket",
    "young_modulus_Pa": 70e9,
    "poisson_ratio": 0.33,
    "density_kg_m3": 2700.0,
    "height_m": 0.189,
    "fixed_boundary": "foot undersides fixed in all displacement directions",
    "loading": "uniform downward traction on the top seat (-z)",
    "source": "official data/raw/agentfem/README.md; reproduce/bracket_lab/case.py lines 70-84",
    "field_source": "official reproduce/bracket_lab/operator_data.py lines 13-18; physical_load_N from split CSV",
}


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    split: str
    values: dict[str, float]


class EngineeringMetadata:
    """3 official CSV splits; never exposes target columns as model features."""

    def __init__(self, splits: dict[str, list[CaseRecord]], context: dict[str, Any] | None = None):
        self.splits = splits
        self.context = dict(DEFAULT_CONTEXT if context is None else context)
        self._validate_context()
        self.by_id = {record.case_id: record for rows in splits.values() for record in rows}
        if len(self.by_id) != sum(map(len, splits.values())):
            raise ValueError("duplicate case_id across splits")
        self.normalizer: dict[str, Any] | None = None

    def _validate_context(self) -> None:
        keys = set(DEFAULT_CONTEXT)
        if set(self.context) != keys or not self.context["source"] or not self.context["field_source"]:
            raise ValueError("engineering context requires all verified constants and nonempty source references")
        for name in ("young_modulus_Pa", "density_kg_m3", "height_m", "poisson_ratio"):
            value = self.context[name]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"invalid constant: {name}")
        if not (self.context["young_modulus_Pa"] > 0 and self.context["density_kg_m3"] > 0
                and self.context["height_m"] > 0 and -1 < self.context["poisson_ratio"] < 0.5):
            raise ValueError("invalid engineering material or geometry constants")
        for name in ("component", "fixed_boundary", "loading"):
            if not isinstance(self.context[name], str) or not self.context[name].strip():
                raise ValueError(f"missing context string: {name}")

    @classmethod
    def from_directory(cls, data_root: str | Path, context: dict[str, Any] | None = None,
                       expected_counts: dict[str, int] | None = None) -> "EngineeringMetadata":
        root = Path(data_root)
        splits: dict[str, list[CaseRecord]] = {}
        global_ids: set[str] = set()
        for split in SPLITS:
            path = root / f"{split}.csv"
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                required = {"case_id", *INPUT_FIELDS}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}")
                if len(set(reader.fieldnames)) != len(reader.fieldnames):
                    raise ValueError(f"{path}: duplicate CSV header")
                rows = []
                for line, row in enumerate(reader, 2):
                    cid = row["case_id"]
                    if not cid or not cid.strip() or cid.strip() != cid:
                        raise ValueError(f"{path}:{line}: invalid case_id")
                    if cid in global_ids:
                        raise ValueError(f"{path}:{line}: duplicate case_id {cid}")
                    if "split" in row and row["split"] != split:
                        raise ValueError(f"{path}:{line}: split mismatch for {cid}")
                    values: dict[str, float] = {}
                    for field in INPUT_FIELDS:
                        raw = row[field]
                        if raw is None or raw.strip() == "":
                            raise ValueError(f"{path}:{line} case_id={cid}: missing {field}")
                        try:
                            v = float(raw)
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f"{path}:{line} case_id={cid}: invalid {field}={raw}") from exc
                        if not math.isfinite(v):
                            raise ValueError(f"{path}:{line} case_id={cid}: nonfinite {field}")
                        values[field] = v
                    if any(values[k] <= 0 for k in ("depth", "radius", "bow", "physical_load_N")):
                        raise ValueError(f"{path}:{line} case_id={cid}: nonpositive length/load")
                    for field, (lower, upper) in PARAMETER_BOUNDS.items():
                        if not lower <= values[field] <= upper:
                            raise ValueError(f"{path}:{line} case_id={cid}: {field} outside official range [{lower},{upper}] {UNITS[field]}")
                    global_ids.add(cid)
                    rows.append(CaseRecord(cid, split, values))
                splits[split] = rows
                if expected_counts is not None and len(rows) != expected_counts[split]:
                    raise ValueError(f"{split}: expected {expected_counts[split]} rows, found {len(rows)}")
        result = cls(splits, context)
        protocol_file = root / "protocol.json"
        if protocol_file.exists():
            protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
            entries = protocol.get("cases")
            if not isinstance(entries, list):
                raise ValueError("protocol.json: missing cases list")
            keyed = {entry["id"]: entry for entry in entries}
            if len(keyed) != len(entries) or set(keyed) != set(result.by_id):
                raise ValueError("protocol.json: case_id mismatch with CSV splits")
            for cid, row in result.by_id.items():
                entry = keyed[cid]
                if entry.get("partition") != row.split:
                    raise ValueError(f"protocol.json: {cid} split mismatch")
                for field in INPUT_FIELDS:
                    expected = (entry["physical_load_N"] if field == "physical_load_N" else entry["design"][field])
                    if not math.isclose(float(expected), row.values[field], rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError(f"protocol.json: {cid} {field} mismatch")
        return result

    def fit_normalizer(self) -> dict[str, Any]:
        train = self.splits["train"]
        if not train:
            raise ValueError("empty train split")
        values = np.asarray([[r.values[f] for f in INPUT_FIELDS] for r in train], dtype=np.float64)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-12] = 1.0
        self.normalizer = {
            "input_fields": list(INPUT_FIELDS), "units": dict(UNITS), "fitted_split": "train",
            "count": len(train), "mean": mean.tolist(), "std": std.tolist(),
        }
        return self.normalizer

    def save_normalizer(self, path: str | Path) -> None:
        if self.normalizer is None:
            raise ValueError("call fit_normalizer first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.normalizer, indent=2) + "\n", encoding="utf-8")

    def load_normalizer(self, path: str | Path) -> dict[str, Any]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("fitted_split") != "train" or data.get("input_fields") != list(INPUT_FIELDS) or data.get("units") != UNITS:
            raise ValueError("normalizer schema/units/train origin mismatch")
        for name in ("mean", "std"):
            if len(data.get(name, [])) != len(INPUT_FIELDS) or not all(math.isfinite(float(x)) for x in data[name]):
                raise ValueError(f"invalid normalizer {name}")
        if any(float(x) <= 0 for x in data["std"]):
            raise ValueError("normalizer standard deviations must be positive")
        self.normalizer = data
        return data

    def normalize(self, values: dict[str, float]) -> np.ndarray:
        if self.normalizer is None:
            raise ValueError("normalizer not fitted/loaded")
        return ((np.asarray([values[f] for f in INPUT_FIELDS], dtype=np.float64)
                 - np.asarray(self.normalizer["mean"])) / np.asarray(self.normalizer["std"])).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AgentFEM CSVs and fit train-only text numeric normalizer")
    parser.add_argument("--data-root", default="data/raw/agentfem")
    parser.add_argument("--output", default="data/processed/text/normalizer.json")
    parser.add_argument("--config", default="configs/encoders/text.yaml")
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = EngineeringMetadata.from_directory(args.data_root, cfg["context"], expected_counts=cfg.get("expected_counts"))
    data.fit_normalizer()
    data.save_normalizer(args.output)
    print(json.dumps({"counts": {s: len(data.splits[s]) for s in SPLITS}, "input_fields": list(INPUT_FIELDS),
                      "units": UNITS, "normalizer": args.output, "context_source": data.context["source"]}, indent=2))


if __name__ == "__main__":
    main()
