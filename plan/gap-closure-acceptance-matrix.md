# Gap Closure Acceptance Matrix Baseline

Date: 2026-05-10

This file is the execution baseline for full gap closure across vitrum and the fork.

## RFEs and Closure Evidence Targets

| RFE | Current status baseline | Closure target | Evidence artifacts |
|---|---|---|---|
| 03 layered BSDF | Proposed | Applied + verified | Front/back asymmetric A/B renders; fork packing tests; shader branch tests |
| 05 caustics | Proposed | Applied + verified (both MNEE and photon-map) | Strategy-on/off renders, perf table, non-caustic regression suite |
| 09 uniform bridge | Applied (unverified) | Applied + verified | SSS/dispersion visual scenes; CMF upload check; regression tests |
| 11 TRANSLUCENT_BIT | Applied (unverified) | Applied + verified | Mixed-material SSS scene validation; no-regression render |
| 13 payload restructure | Partial (advanced) | Applied + verified | Smooth spectrum A/B vs Sprint 8; throughput perf (<30% target) |
| 14 thin-film TMM | Applied (unverified) | Applied + verified | Angle-shift iridescence A/B; perf validation on target scenes |

## Sprint 12 Gap Baseline

- Gap §1: APPLIED, runtime verification pending.
- Gap §2: APPLIED, runtime verification pending.
- Gap §3: APPLIED, runtime verification pending.
- Gap §4: APPLIED, runtime verification pending.
- Gap §5: NOT STARTED (Beer-Lambert spectral attenuation).

## pt-webgpu Baseline

- Area-light reciprocal MIS is present.
- Non-area and env reciprocal MIS parity is incomplete.
- HDRI path remains fallback-oriented.
- `causticStrategy` remains `none` in capabilities.

## Required End-of-Plan Verification Bundle

For each closure claim, produce:

1. Scenario ID and scene config.
2. Seed/settings and sample budget.
3. Before image hash.
4. After image hash.
5. Delta summary.
6. Performance summary.
7. Pass/fail against acceptance criterion.
