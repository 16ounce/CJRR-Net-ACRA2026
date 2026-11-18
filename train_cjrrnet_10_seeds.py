#!/usr/bin/env python3
"""Purpose: Prepare ten-seed CJRR-Net runs.

Main functionality: Print or execute the paper-configuration training commands.

Usage:
    python train_cjrrnet_10_seeds.py --data-root DATA --output-root RUNS
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = [5, 7, 55, 17, 13, 6, 11, 9, 22, 44]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Directory containing latency_shift_train/val/test")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable,
                    help="Python executable in an environment with PyTorch and NumPy")
    ap.add_argument("--execute", action="store_true",
                    help="Run commands sequentially; default only prints them")
    args = ap.parse_args()
    entry = ROOT / "models/cjrrnet/train.py"
    lifter = ROOT / "weights/lifter_bypassed/lift3d_gaussian_mlp_best.pt"
    commands = []
    for seed in SEEDS:
        commands.append([
            args.python, str(entry), "--arm", "rand_full", "--config", "K",
            "--lambda_regret", "0.5", "--lambda_pair", "0.4",
            "--state_npz", "st_nlfonly.npz",
            "--train_data", str(args.data_root / "latency_shift_train"),
            "--val_data", str(args.data_root / "latency_shift_val"),
            "--test_data", str(args.data_root / "latency_shift_test"),
            "--save_dir", str(args.output_root / f"cjrrnet_seed{seed}"),
            "--lift3d_ckpt", str(lifter), "--regret_scale", "-1",
            "--epochs", "100", "--batch_size", "256", "--lr", "1e-3",
            "--weight_decay", "1e-4", "--dropout_prob", "0.1",
            "--patience", "15", "--lambda_lr", "0.1",
            "--rare_boost", "1.0", "--seed", str(seed),
        ])
    for command in commands:
        print(shlex.join(command), flush=True)
    if not args.execute:
        return
    for split in ("train", "val", "test"):
        directory = args.data_root / f"latency_shift_{split}"
        if not (directory / "st_nlfonly.npz").is_file():
            ap.error(f"missing {directory / 'st_nlfonly.npz'}")
    if not lifter.is_file():
        ap.error(f"missing {lifter}")
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
