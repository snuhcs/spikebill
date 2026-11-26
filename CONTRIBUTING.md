# Contributing to SpikeBill

SpikeBill is maintained by the snuhcs group. Issues and pull requests are welcome.

## Code

- Run the tests before a pull request: `pip install -e .[dev]` then `pytest`.
- A change to the cost model needs a test that pins the new behaviour, and a line in `docs/cost_model.md` if the
  model changes.

## Chip files (the most useful contribution)

A chip file describes one chip: topology, per-event energies and times, static power, limits. Reports cite a chip
file by its `version` and sha256, so the rules are strict:

1. Start from `docs/chip_template.yaml` and put the file in `src/spikebill/chips/`.
2. Every parameter has a `value`, a `low` and a `high`, and a `source`. Tag the source:
   `[P]` published (paper, datasheet), `[C]` calibrated against hardware, `[D]` derived from other parameters,
   `[A]` assumption (give it a wide range). Cite the publication in the source string.
3. Values taken from another project carry the attribution in the source string, and in `NOTICE` if the project's
   licence asks for it. Copy numbers, not code or data files.
4. Any changed value bumps `version` and adds a line to `docs/chip_changelog.md`. Results computed with the old
   version keep their old numbers.
5. Add a test that loads the chip and runs a small network (`tests/test_model.py` has examples).

A maintainer reviews the sources before merging. Measurements on real hardware that tighten a range are especially
welcome; please describe how they were taken.
