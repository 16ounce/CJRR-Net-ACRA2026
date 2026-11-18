#!/usr/bin/env python3
"""Purpose: Prepare ten-seed adapted baseline runs.

Main functionality: Print or execute PoseErrNet and ActiveMoCap training commands.

Usage:
    python train_adapted_baselines_10_seeds.py --data-root DATA --output-root RUNS
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["poseerrnet", "activemocap", "both"],
                        default="both")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true",
                        help="Run the commands; the default prints them only")
    args = parser.parse_args()

    splits = [
        flag for split in ("train", "val", "test")
        for flag in (f"--{split}_data", str(args.data_root / f"latency_shift_{split}"))
    ]
    lifter = ROOT / "weights/lifter_bypassed/lift3d_gaussian_mlp_best.pt"
    methods = (["poseerrnet", "activemocap"] if args.method == "both"
               else [args.method])
    commands = []
    for method in methods:
        for seed in SEEDS:
            if method == "poseerrnet":
                command = [
                    args.python,
                    str(ROOT / "baselines/poseerrnet_adapted/train.py"),
                    "--arm", "poseerr",
                    *splits,
                    "--save_dir", str(args.output_root / f"poseerrnet_adapted_seed{seed}"),
                    "--seed", str(seed),
                ]
            else:
                command = [
                    args.python,
                    str(ROOT / "baselines/activemocap_inspired_planner/train.py"),
                    "--config", "F", "--utility_mode", "global_logdet",
                    "--lift3d_ckpt", str(lifter),
                    "--epochs", "40", "--batch_size", "256",
                    *splits,
                    "--save_dir", str(args.output_root / f"activemocap_inspired_seed{seed}"),
                    "--seed", str(seed),
                ]
            commands.append(command)

    for command in commands:
        print(shlex.join(command), flush=True)
    if not args.execute:
        return

    for split in ("train", "val", "test"):
        directory = args.data_root / f"latency_shift_{split}"
        for filename in ("keypoints_mv.npy", "mpjpe_per_cam.npy"):
            if not (directory / filename).is_file():
                parser.error(f"missing {directory / filename}")
        if "activemocap" in methods:
            for filename in ("occlusion_mask.npy", "lift3d_target_mv.npy"):
                if not (directory / filename).is_file():
                    parser.error(f"missing {directory / filename}")
    if "activemocap" in methods and not lifter.is_file():
        parser.error(f"missing {lifter}")
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
