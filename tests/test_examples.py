"""The scripts in examples/ run (QUICK mode) and write their outputs."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EX = os.path.join(ROOT, "examples")


@pytest.mark.parametrize("script", ["quickstart.py", "from_torch_snntorch.py", "zoo_compare.py", "chip_compare.py"])
def test_example_runs(script, tmp_path):
    if script == "from_torch_snntorch.py":
        pytest.importorskip("snntorch")
    if script in ("quickstart.py", "chip_compare.py"):
        pytest.importorskip("matplotlib")
    env = dict(os.environ, QUICK="1", PYTHONPATH=os.path.join(ROOT, "src"), MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, os.path.join(EX, script), str(tmp_path)], env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "energy / inference" in r.stdout
    assert any(tmp_path.iterdir()) or script == "zoo_compare.py"


def test_reproduce_paper_subset(tmp_path):
    """examples/reproduce_paper.py: one model's Table-transformers numbers match the paper (full run: ~3 min, 111 numbers)."""
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    out = tmp_path / "rep.json"
    r = subprocess.run([sys.executable, os.path.join(EX, "reproduce_paper.py"), "--only", "table.sdt_v1_8_512",
                        "--workers", "2", "--out", str(out)], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "12 / 12 numbers match the paper" in r.stdout


def test_readme_quickstart_runs(tmp_path):
    """Every Python block of the public README runs as written (snnTorch model on random spikes; NIR file)."""
    pytest.importorskip("snntorch")
    pytest.importorskip("nir")
    pytest.importorskip("matplotlib")
    readme = next(p for p in (os.path.join(ROOT, "tools", "release", "README.public.md"), os.path.join(ROOT, "README.md"))
                  if os.path.exists(p))
    blocks = [b.split("```", 1)[0] for b in open(readme, encoding="utf-8").read().split("```python\n")[1:]]
    assert len(blocks) >= 2
    src = os.path.join(ROOT, "src") if os.path.isdir(os.path.join(ROOT, "src")) else os.path.join(ROOT, "src")
    env = dict(os.environ, PYTHONPATH=src)
    for code in blocks:
        if "report" in code and "report =" not in code:  # a follow-up block: runs after the Quickstart
            code = blocks[0] + "\n" + code
        if os.path.isdir(os.path.join(ROOT, "src")):    # this repo: the README imports the release name
            code = code.replace("from spikebill import", "from spikebill import")
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300,
                           cwd=tmp_path)
        assert r.returncode == 0, r.stderr[-2000:]
        assert "energy / inference" in r.stdout
    assert (tmp_path / "figs" / "chip_load.png").exists() and (tmp_path / "step.html").exists()
