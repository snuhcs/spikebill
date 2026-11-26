"""Run bookkeeping shared by all experiments: seeding and a config.json with the commit next to each output."""
import json
import os
import platform
import random
import subprocess
import sys
import time


def git_commit():
    """Short hash of HEAD, with '+dirty' if tracked files under code/ differ from HEAD; 'unknown' outside git."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", ".."], cwd=here, stderr=subprocess.DEVNULL).returncode != 0
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def seed_everything(seed):
    """Seed python, numpy and torch (whichever are installed)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_config(out_dir, args, **extra):
    """Write out_dir/config.json with args (dict or argparse.Namespace), commit, command line, host and time."""
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(
        args=args if isinstance(args, dict) else vars(args),
        commit=git_commit(),
        argv=sys.argv,
        host=platform.node(),
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1, default=str)
    return cfg


def save_result(out_dir, result):
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
