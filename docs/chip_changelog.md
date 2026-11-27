# Chip parameter changelog

Reports cite `version` and the sha256 of the file; results before a version change keep their old numbers.

- loihi2 / loihi2_3d 2026.11.4-e6 (2026-11-27): no value changed; the `e_chip_hop` source says what the 10x
  multiplies (the Loihi 1 on-chip hop, 3.6 pJ). New sha256, identical results (zoo references recomputed).
- loihi2 / loihi2_3d 2026.11.3-e6, loihi1 2026.11.2 (2026-11-25): no value changed; source comments point to the
  public docs instead of internal notes (release preparation). New sha256, identical results.
- loihi2 / loihi2_3d 2026.11.2-e6 (2026-11-23): `p_static_chip` range 0.16-0.30 W -> 0.15-0.55 W (value 0.23 W
  unchanged), from measured Loihi 2 static power 0.15 W (Abreu 2025), 0.232 W (lava-dl PilotNet notebook),
  0.54-0.55 W (Parpart 2023). Changes only `ranges=True` results (high end).
- loihi2 / loihi2_3d 2026.11.1-e6 (2026-11-22): neuro-core energies 0.65 x Loihi 1 (E6), range 0.33-1.0.
- loihi1 2026.11.1: SANA-FE calibrated values (E1, E1b).
