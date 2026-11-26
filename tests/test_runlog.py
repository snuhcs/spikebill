import json

from spikebill import runlog


def test_save_config_writes_args_and_commit(tmp_path):
    cfg = runlog.save_config(tmp_path, {"seed": 3, "lr": 0.1})
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["args"] == {"seed": 3, "lr": 0.1}
    assert on_disk["commit"] == cfg["commit"] and on_disk["commit"]


def test_seed_everything_is_deterministic():
    import random
    runlog.seed_everything(0); a = random.random()
    runlog.seed_everything(0); b = random.random()
    assert a == b
