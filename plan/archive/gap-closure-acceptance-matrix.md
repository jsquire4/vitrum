# Gap Closure Acceptance Matrix

Date: 2026-05-10

This matrix is the authoritative acceptance checklist for the current gap-closure wave across `vitrum` and the fork.

## Scenario Set (Deterministic)

| Scenario ID                     | Purpose                                                                             |  Seed | Resolution | Bounces |  SPP |
| ------------------------------- | ----------------------------------------------------------------------------------- | ----: | ---------: | ------: | ---: |
| `rfe03-layered-front-back`      | Validate front/back asymmetry (`frontLayer`/`backLayer`)                            |  1337 |   1280x720 |       8 |  512 |
| `rfe07-11-sss-mixed-panels`     | Validate `TRANSLUCENT_BIT` mixed-material gating                                    |  2027 |   1280x720 |       8 |  512 |
| `rfe08-13-spectral-payload`     | Validate spectral payload + hero wavelength pipeline                                |  4242 |   1280x720 |      10 | 1024 |
| `rfe14-thinfilm-angle-shift`    | Validate 35-layer thin-film angle-dependent response                                |  9001 |   1280x720 |      10 | 1024 |
| `rfe09-bridge-global-cmf`       | Validate CMF/CDF upload + per-material scalar non-override                          | 31415 |  1024x1024 |       8 |  256 |
| `rfe05-caustic-strategy`        | Validate transition from API-plumbed controls to runtime-verified strategy behavior | 27182 |   1280x720 |      10 | 1024 |
| `ptwgpu-parity-material-fields` | Validate WebGPU packed-field effects (spectral/volume/layered/thin-film)            |   777 |   1280x720 |       8 |  512 |

## RFE Acceptance Mapping

| RFE                    | Code closure                                                                     | Runtime closure target                                                    | Required evidence                                                            |
| ---------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 03 layered BSDF        | Implemented in fork packing + BSDF path                                          | Front/back view diverges as expected                                      | Before/after hash pair + visual delta note                                   |
| 05 caustics            | API options and param plumbing implemented; published capabilities remain `none` | Runtime-verified non-`none` strategy behavior with validated quality/perf | 3 mode captures (`none`, `manifold-nee`, `photon-map`) + perf + quality note |
| 09 uniform bridge      | Bridge uploads global spectral tables + strategy controls                        | CMF tables present; per-material scalar values remain fork-packed         | Uniform snapshot + regression tests                                          |
| 11 TRANSLUCENT_BIT     | Packed in fork flags + shader gating                                             | Mixed panel scene shows SSS only where flagged                            | A/B render and no-regression clear panel                                     |
| 13 payload restructure | Scalar throughput + wavelength path active                                       | Spectral fan smoother vs Sprint 8 baseline                                | Capture pair + perf delta                                                    |
| 14 thin-film TMM       | 35-layer TMM evaluator active                                                    | Dichroic angle shift visible and stable                                   | Angle sweep capture set + perf                                               |

## Sprint 12 Closure State (This Wave)

- Gap §1 payload restructure: code complete; runtime capture pending.
- Gap §2 spectral accumulation integration: code complete; runtime capture pending.
- Gap §3 hero-wavelength IOR path: code complete; runtime capture pending.
- Gap §4 thin-film TMM path: code complete; runtime capture pending.
- Gap §5 spectral attenuation Beer-Lambert path: code complete; runtime capture pending.

## Artifact Bundle Contract

For every scenario above, the verification report must include:

1. `scenarioId`
2. `seed`, `resolution`, `bounces`, `spp`
3. `beforeImageHash`
4. `afterImageHash`
5. `deltaSummary`
6. `perfBaselineMsPerSample`
7. `perfCandidateMsPerSample`
8. `passFail`
