# WebGPU PT Parity Program — Signoff (2026-05-26)

**Plan:** `plan/primary-release-and-webgpu-pt-parity-implementation-deep.md`  
**HEAD:** `1815517` and later on `main`

## Wave status

| Wave | Status | Evidence |
|------|--------|----------|
| WG-0 Visual harness | **Complete** | `tools/benchmark-runner/run-gap-closure-verification.mjs`, `capturePtWebgpu.mjs`, `tools/reference-renders/baseline/ptwgpu-parity-material-fields.png`, `npm run benchmark:gap-closure` |
| WG-1 OIDN-final | **Complete** | `packages/pt-webgpu/src/denoise/oidnFinalDispatcher.ts`, `oidnFinalIntegration.test.ts`, `getDenoisedFrame()` |
| WG-2 Hero λ + CMF MIS | **Complete (mechanical)** | `shared-samplers/wgsl/heroWavelength*.ts`, FrameParams extension, kernel wiring; opt-in `vitrum.ptWebgpu.spectralHeroWavelength`; `heroWavelengthPlumbing.test.ts` |
| WG-3 Cauchy dispersion | **Complete** | `dispersionAbbeNumber` in `materialPacking.ts`, `cauchyIorAtLambda` in WGSL |
| WG-4 Layered BSDF + transmission MIS | **Complete** | η² refraction PDF, `activeLayerWeightRgb`, connect/caustic `ior` plumbed; `wgslContract.test.ts` |
| WG-5 SSS / translucent | **Complete (mechanical)** | `isTranslucent` derived from transmission + scatteringCoeff; SSS gated in full/lite kernels |
| WG-6 Geometry incremental | **Complete** | `packSceneFromCore` via `shared-bvh`; `rebuildPrimitiveBlas` positions splice; ledger `positions: true` |
| WG-7 BDPT | **Deferred** | Explicit non-goal until product requires stained-glass PT BDPT on WebGPU |
| WG-8 createEngine opt-in | **Complete** | `prefer: 'quality-webgpu'`, `constructPathTracerWebGPU` in `@vitrum/engine` |
| WG-9 Extended denoisers | **Deferred** | `svgf-real` not wired on pt-webgpu (walkaround-hybrid only) |
| WG-10 Signoff | **This document** | See gates below |

## Definition-of-done gates

| ID | Gate | Result | Notes |
|----|------|--------|-------|
| WG-D1 | Fidelity matrix rows not **unsupported** / stale **approximate** where implemented | **Pass (mechanical)** | Updated `plan/renderer-fidelity-matrix.md` 2026-05-26 |
| WG-D2 | Gap-closure scenarios PASS | **Partial** | `ptwgpu-parity-material-fields` has committed baseline + strict mode on hardware; RFE rows (`rfe03`, `rfe07`, `rfe08`, …) require `VITRUM_GPU_CAPTURE=1` host run — presets tagged `backend: 'pt-webgpu'` where applicable |
| WG-D3 | `oidn-final` executes | **Pass** | Integration tests + runtime dispatcher (not warn-only) |
| WG-D4 | Hero λ + CMF MIS vs fork | **Pass (CPU/layout)** | CMF tables + integrals in FrameParams; full visual A/B vs fork is host workflow |
| WG-D5 | `quality-webgpu` documented; `auto` unchanged | **Pass** | `packages/engine/README.md`, `createEngineScale.ts` |
| WG-D6 | README parity section | **Pass** | `packages/pt-webgpu/README.md` limitations + implemented lists |
| WG-D7 | `npm run verify:mechanical` | **Pass** | 2026-05-26 run (~566s, shader-compile-ci 7/7 OK) |

## Host workflow (not CI-blocked on Linux lite tier)

```bash
# Seed / refresh WG-0 baseline (hardware with WebGPU)
npm run benchmark:seed-wg0

# Full gap-closure matrix (strict hashes on Windows GPU host)
VITRUM_STRICT_GAP_CLOSURE=1 npm run benchmark:gpu-windows -- run-gap-closure-verification.mjs

# Per-scenario pt-webgpu capture
VITRUM_GPU_CAPTURE=1 npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```

## Explicit non-goals (unchanged)

- pt-webgpu in `createEngine({ prefer: 'auto' })` until WG-D2 all-scenario PASS on hardware
- BDPT port (WG-7)
- Walkaround denoiser stack on pt-webgpu (WG-9)
