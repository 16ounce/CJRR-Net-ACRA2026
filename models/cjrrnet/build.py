"""Purpose: Construct paper-configuration CJRR-Net.

Main functionality: Set the configuration-K network options and instantiate the ranker.
"""

from __future__ import annotations

import sys
from pathlib import Path


def build_cjrrnet(**kwargs):
    """Construct the published CJRR-Net architecture with configuration K."""
    source_root = Path(__file__).resolve().parents[2]
    for directory in (source_root, source_root / "utils" / "cjrrnet"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from models.cjrrnet.cjrrnet import (
        JointTokenReprojVisRankNet,
    )

    fixed = {
        "joint_tokens": True,
        "aux_head": False,
        "film_mode": "input",
        "pool_mode": "attn",
    }
    for key, required in fixed.items():
        if key in kwargs and kwargs[key] != required:
            raise ValueError(f"CJRR-Net requires {key}={required!r}")
        kwargs[key] = required
    return JointTokenReprojVisRankNet(**kwargs)
