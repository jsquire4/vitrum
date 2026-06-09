# WS6 — Validation: the real-GPU WebGL2 capture host + fork-vs-native A/B

> The one genuinely external blocker. Verified state: pt-webgl has **no automated hardware-GPU gate today** — the WGSL T1 smoke (lavapipe/dzn) is WebGPU-only and cannot run WebGL2; the WSL WebGL2 path (`wsl-gpu/webgl2-capture/`) runs only on SwiftShader (software GL). The native backend can't be fidelity-validated without a real-GPU WebGL2 host. **This gap exists regardless of Path B — it already blocks promoting the 8 fork fidelity rows out of `experimental`.** Closing it pays double.

## 1. The capture host (the gating dependency — do this FIRST in WS6)

Options, in preference order:
1. **Windows-Chrome host** — the fidelity-matrix's named missing piece (`tools/reference-renders/README.md:137`). A Windows machine running real-GPU Chrome, driven by the existing Playwright harness `tools/benchmark-runner/capturePtWebgl.mjs` (loads `two-engines-one-scene` `?mode=ptwebgl`, waits for `__vitrum.ptWebgl.converged`, screenshots `#c-pt`). Add a `?mode=ptwebgl2` once the new backend is wired into the example.
2. **Self-hosted GPU runner** — a Linux box with a real GPU + Chrome/headless-shell (NOT WSL SwiftShader). Same Playwright harness.
3. **Interim: SwiftShader (software GL)** — already works in WSL (`wsl-gpu/webgl2-capture/capture-webgl2.mjs` + `drive-scene.mjs`). Valid for **geometry/transform/consume-confirmation** correctness (S0–S2 silhouette/BVH checks, the V6 instanced-mesh PASS precedent) but **explicitly insufficient** for variance/out-of-gamut/perf (the fidelity-matrix promotion criteria). Use it to unblock S0–S2 correctness while a real-GPU host is procured.

**Deliverable:** a `?mode=ptwebgl2` capture path producing converged HDR PNGs on a real GPU, wired into `scripts/capture-all-refs.sh`.

## 2. The fork-vs-native A/B oracle

The fork's current output **is** the reference. `tools/reference-renders/pt-webgl-fidelity/*.baseline.png` (six scenarios: layered, caustic, SSS, spectral, CMF bridge, thin-film) are real-GPU fork captures. The gate already exists and is reusable — only the producer changes:
- `run-pt-webgl-fidelity-acceptance.mjs` — PSNR gate, default `minPsnr = 28`, consumed by the env-gated `fidelityAcceptance.test.ts` (`VITRUM_PTWEBGL_FIDELITY_ACCEPTANCE=1`).
- `tools/reference-renders/diff-baselines.mjs` — the HDR diff.

**Plan:** capture the native `pt-webgl2` output for the same six scenarios → PSNR-A/B against the existing fork baselines via `run-pt-webgl-fidelity-acceptance.mjs`. Add a `pt-webgl2` variant of the acceptance script (or parameterize the producer). A native-vs-fork PSNR ≥ 28 per scenario = no fidelity regression.

## 3. Per-feature deterministic oracles (where the fork has no independent ground truth)

The fork's vitrum extensions (spectral hero-λ, MNEE, BDPT, Jakob-Hanika, additive) are fork patches with no independent oracle — they're validated by self-consistency. For each, prefer a **deterministic math harness** (the lesson that self-validating harnesses land where scene-radiometric A/Bs dead-end):
- **MNEE caustic** → pt-webgpu's analytic-mirror + forward-traced-Snell A/Bs (`wsl-gpu/scripts/` — the manifold-NEE 98.7% energy references). Since S3 swaps in pt-webgpu's validated math, this oracle applies directly.
- **Spectral** → pt-webgpu's deterministic spectral A/B + the CMF-integral residual checks.
- **BVH traversal** → the CPU brute-force closest-hit oracle (WS3 §2) — gate at 100% before any render.
- **Additive accumulation** → a constant-shaded-quad unit test: `N` additive draws of value `v` with host-clear-to-0 must read back `N·v` (count in alpha) → mean `v`.

## 4. Unit-test tier (no GPU; runs in `npm test`)

Mirror pt-webgl's 37 CPU vitest files. Pin the THREE-free CPU logic:
- The packers (materials 85px, lights 6px, equirect CDF) against golden byte layouts (like `materialPackingCoreEquivalence`).
- The BVH texture adapter vs the CPU brute-force oracle.
- `packFrameParams` vs a golden std140 buffer.
- The program builder's `setDefine` change-gating + preamble generation (string assertions, no GL).
- The blend-regime selection logic.
- The boundary test (no `from 'three'`).

## 5. CI integration

- **Per-push (no GPU):** the unit tier above runs in `npm test`. Add the `pt-webgl2` boundary test + packer goldens.
- **Pre-push GPU smoke:** the existing T1 (lavapipe/dzn) is WebGPU-only — it does NOT cover pt-webgl2. The WebGL2 equivalent is the SwiftShader correctness smoke (geometry/BVH) — wire it into the pre-push hook as a `pt-webgl2` correctness gate (it catches the F-TLAS1/F-RC1 stride-bug class).
- **Nightly/gated real-GPU:** the fidelity A/B (§2) on the capture host — this is the promotion gate (out of `experimental`).

## 6. WS6 done-when
- A real-GPU `?mode=ptwebgl2` capture path produces converged HDR for the six fidelity scenarios.
- Native-vs-fork PSNR ≥ 28 on all six (S3 complete).
- The BVH adapter passes the brute-force oracle at 100% on the SwiftShader smoke (per-push).
- The 8 fork fidelity rows are re-captured/promoted as a side effect (the capture host closes that long-standing gap).
