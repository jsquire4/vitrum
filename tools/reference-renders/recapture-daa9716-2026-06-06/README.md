# pt-webgpu baseline re-capture — closest-hit fix `daa9716` (2026-06-06)

Post-fix dzn (RTX 4090) re-captures of every committed pt-webgpu baseline, at each baseline's
committed params, via the same producing path. The committed originals in `../baseline/` are
**unchanged** — these live here for the lead to A/B and (for the stale ones) promote.

Fix: `daa9716` pt-webgpu `intersectionCore.wgsl.ts` closest-hit live upper bound (within-leaf
last-writer-wins → nearest-wins) — restored face-on thin Cornell walls that rendered black.
All committed baselines predate the fix and embed the old bug.

## Status vs the committed `../baseline/` originals (dzn, daa9716)

STALE — committed baseline differs, refresh recommended:
| file | PSNR vs old | %px changed |
|---|---|---|
| cornell-manylights.png | 3.57 dB | 50.7 |
| cornell-bdpt-on.png | 6.36 dB | 91.6 |
| ptwgpu-cauchy-dispersion.png | 6.36 dB | 90.5 |
| rfe05-caustic-strategy.png | 6.45 dB | 79.6 |
| ptwgpu-spectral-hero.png | 6.87 dB | 88.7 |
| rfe14-thinfilm-angle-shift.png | 7.18 dB | 98.5 |
| rfe07-11-sss-mixed-panels.png | 7.65 dB | 84.7 |
| rfe08-13-spectral-payload.png | 7.70 dB | 78.8 |
| ptwgpu-parity-material-fields.png | 8.41 dB | 68.6 |
| rfe03-layered-front-back.png | 10.03 dB | 67.6 |

NOT stale — byte-identical to committed (these copies match `../baseline/` exactly):
mnee-glass-slab.png, ptwgpu-thinfilm-angle.png, ptwgpu-layered-front.png, ptwgpu-sss-mixed-panels.png.

Full numbers, gate analysis, and method:
`~/projects/wsl-gpu/captures/g-sweep-2026-06-06/baseline-recapture/RESULTS.md`.
