"""CLI user-facing behaviour: one-line errors, --version, activity dicts checked against the model."""
import pytest
def test_user_errors_are_one_line(capsys, tmp_path):
    """Release gap 38: user errors exit 2 with one line, no traceback; typo'd population names are not ignored."""
    from spikebill import cli
    bad = tmp_path / "r.json"
    bad.write_text('{"no_such_pop": 0.1}')
    for argv, needle in ((["zoo:sdt_v1"], "unknown zoo model"),
                         (["zoo:sdt_v1_8_512", "--chip", "loihi3"], "unknown chip 'loihi3'"),
                         (["zoo:sdt_v1_8_512.foo"], "no variant 'foo'"),
                         (["missing.json"], "No such file"),
                         (["zoo:sdt_v1_8_512", "--no-balance", "--rates", str(bad)], "not in model")):
        assert cli.entry(argv) == 2
        err = capsys.readouterr().err
        assert needle in err and "Traceback" not in err and len(err.strip().splitlines()) == 1, err


def test_version(capsys):
    from spikebill import cli, __version__
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_rates_validated():
    import spikebill
    from spikebill import NetBuilder
    b = NetBuilder("t")
    x = b.input("in", 100, rate=0.1)
    b.dense(x, "hidden_layer", 50)
    sim = spikebill.Simulator(b.graph(), chip="loihi2")
    with pytest.raises(ValueError, match="did you mean 'hidden_layer'"):
        sim.run(rates={"hiden_layer": 0.1}, balance=False)
    with pytest.raises(ValueError, match="must be a number >= 0"):
        sim.run(rates={"hidden_layer": -0.1}, balance=False)
    with pytest.raises(ValueError, match="<= 1"):
        sim.run(active={"hidden_layer": 2.0}, balance=False)
    sim.run(rates={"hidden_layer": 0.2}, active={"hidden_layer": 0.5}, balance=False)
