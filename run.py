#!/usr/bin/env python3
"""Cross-platform entrypoint for installing, downloading and running each encoder."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MODALITIES = ("cad", "fem", "text")


def python_executable() -> str:
    if os.name == "nt":
        path = VENV / "Scripts" / "python.exe"
    else:
        path = VENV / "bin" / "python"
    return str(path) if path.is_file() else sys.executable


def run(args: list[str], *, use_venv: bool = True) -> None:
    argv = [python_executable() if use_venv else sys.executable, *args]
    print("\n$ " + " ".join(argv), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(argv, cwd=ROOT, env=env, check=True)


def setup(modality: str) -> None:
    if not VENV.exists():
        print(f"创建虚拟环境: {VENV}", flush=True)
        venv.EnvBuilder(with_pip=True).create(str(VENV))
    extra = "all" if modality == "all" else modality
    run(["-m", "pip", "install", "--upgrade", "pip"])
    run(["-m", "pip", "install", "-e", f".[{extra},download]"])
    print("依赖安装结束。使用 ./run.sh check 检查本机运行环境。", flush=True)


def load_config(modality: str) -> dict:
    """Load one encoder configuration, keeping paths in YAML as the source of truth."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Please run ./run.sh setup to install PyYAML first") from exc
    return yaml.safe_load((ROOT / f"configs/encoders/{modality}.yaml").read_text(encoding="utf-8"))


def verify_data(modality: str) -> None:
    run(["scripts/download_data.py", modality, "--check"])


def train(modality: str, *, device: str) -> None:
    verify_data(modality)
    if modality == "cad":
        run(["-m", "femfusion.encoders.cad.dataset", "--data-dir", "data/raw/agentfem",
             "--cache-dir", "data/interim/cad"])
        run(["-m", "femfusion.encoders.cad.train", "--config", "configs/encoders/cad.yaml",
             "--device", device])
    elif modality == "fem":
        # Keep CLI arguments aligned with configs/encoders/fem.yaml.
        cfg = load_config("fem")
        d, e, t = cfg["data"], cfg["encoder"], cfg["training"]
        cmd = ["-m", "femfusion.encoders.fem.train", "--data-root", d["root"],
               "--cache-dir", d["cache_dir"], "--run-dir", t["run_dir"],
               "--epochs", str(t["epochs"]), "--batch-size", str(t["batch_size"]),
               "--lr", str(t["lr"]), "--weight-decay", str(t["weight_decay"]),
               "--seed", str(t["seed"]), "--hidden-dim", str(e["hidden_dim"]),
               "--layers", str(e["num_layers"]), "--dropout", str(e["dropout"]),
               "--max-regions", str(d["max_regions"]), "--max-nodes", str(d["max_nodes"]),
               "--reference-load-N", str(d["reference_load_N"]),
               "--mask-ratio", str(t["mask_ratio"]), "--device", device]
        if t.get("fixed_train_mask", False):
            cmd.append("--fixed-train-mask")
        if t.get("resume_checkpoint"):
            cmd.extend(["--resume", str(t["resume_checkpoint"])])
        run(cmd)
    elif modality == "text":
        run(["-m", "femfusion.encoders.text.train", "--config", "configs/encoders/text.yaml",
             "--data-root", "data/raw/agentfem", "--checkpoint-dir", load_config("text")["training"]["checkpoint_dir"],
             "--device", device, "--allow-download"])


def evaluate(modality: str, *, device: str) -> None:
    verify_data(modality)
    if modality == "cad":
        checkpoint = str(Path(load_config("cad")["output_dir"]) / "best.pt")
        command = ["-m", "femfusion.encoders.cad.evaluate", "--checkpoint", checkpoint,
                   "--output-dir", "outputs/cad/evaluation"]
    elif modality == "fem":
        cfg = load_config("fem")
        checkpoint = str(Path(cfg["training"]["run_dir"]) / "best.pt")
        command = ["-m", "femfusion.encoders.fem.evaluate", "--data-root", cfg["data"]["root"],
                   "--checkpoint", checkpoint, "--cache-dir", cfg["data"]["cache_dir"],
                   "--output-dir", "outputs/fem/evaluation", "--device", device]
    else:
        checkpoint = str(Path(load_config("text")["training"]["checkpoint_dir"]) / "best.pt")
        command = ["-m", "femfusion.encoders.text.evaluate", "--config", "configs/encoders/text.yaml",
                   "--data-root", "data/raw/agentfem", "--checkpoint-dir", load_config("text")["training"]["checkpoint_dir"],
                   "--split", "test", "--output-dir", "outputs/text/evaluation", "--device", device]
    if not (ROOT / checkpoint).is_file():
        raise FileNotFoundError(f"找不到模型权重 {checkpoint}；先执行 ./run.sh train {modality}")
    run(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="FEMFusion V1: download, verify, train and evaluate encoders")
    parser.add_argument("action", choices=("setup", "download", "check", "train", "evaluate", "full"),
                        help="full = download + train all + evaluate all")
    parser.add_argument("modality", nargs="?", choices=("all", *MODALITIES), default="all")
    parser.add_argument("--modality", dest="modality_flag", choices=("all", *MODALITIES), default=None,
                        help="Optional alternative to positional modality, e.g. setup --modality cad")
    parser.add_argument("--device", default="cpu", help="Torch device: cpu (default), mps or cuda")
    parser.add_argument("--revision", help="Hugging Face dataset revision (download/full only)")
    args = parser.parse_args()
    modality = args.modality_flag or args.modality
    selected = MODALITIES if modality == "all" else (modality,)

    try:
        if args.action == "setup":
            setup(modality)
        elif args.action in ("download", "full"):
            cmd = ["scripts/download_data.py", modality]
            if args.revision:
                cmd += ["--revision", args.revision]
            run(cmd)
        if args.action == "check":
            run(["-m", "compileall", "-q", "src", "scripts", "run.py"])
            run(["scripts/verify_encoders.py", "--data-dir", "data/raw/agentfem"])
        elif args.action in ("train", "full"):
            for m in selected:
                train(m, device=args.device)
        if args.action in ("evaluate", "full"):
            for m in selected:
                evaluate(m, device=args.device)
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"运行失败：{exc}\n")


if __name__ == "__main__":
    main()
