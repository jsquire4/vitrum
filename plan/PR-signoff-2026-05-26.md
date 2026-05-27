# Primary Release Program — Signoff (2026-05-26)

**Plan:** `plan/primary-release-and-webgpu-pt-parity-implementation-deep.md` (Program PR)  
**HEAD:** `main` (hybrid TLAS + incremental paths landed through 2026-05-26)

## Wave status

| Wave | Status | Evidence |
|------|--------|----------|
| PR-0 Contract hygiene | **Complete** | Emitter fast path, `_lastScene` sync, promise ledger |
| PR-1 Material + emitter fast paths | **Complete** | `materialPatch`, hybrid tests, `incrementalPatchSupport.material: true` |
| PR-2 TLAS CPU pack | **Complete** | `shared-bvh/src/scenePack.ts`, `scenePack.test.ts` |
| PR-3 TLAS GPU (ReSTIR) | **Complete** | `hybridTlasTraverse.test.ts`, `hybridTlasPrimaryHit.gpu.test.ts`, TLAS WGSL in hybrid shaders |
| PR-4 TLAS incremental refit | **Complete** | `transformRefit`, `positionsRefit`, `rebuildPrimitiveBlas` in hybrid + pt-webgpu |
| PR-5 DDGI + RC alignment | **Complete** | DDGI probe rays use ReSTIR TLAS buffers (`traceTlasFirstHit`); `markInstancesDirty` on transform refit; RC `refitBounds` without full `setScene` |
| PR-6 Scale + soak | **Complete (mechanical)** | `run-pr-hybrid-bench.mjs`, `run-pr-hybrid-mechanical.mjs`, scenario presets |
| PR-7 GPU skinning | **Deferred** | CPU `solveSkin` baseline sufficient for hero character |
| PR-8 pt-webgl incremental | **Deferred** | Optional; pt-webgl already has fork incremental material path |
| PR-9 Signoff | **This document** | |

## Definition-of-done gates

| ID | Gate | Result | Notes |
|----|------|--------|-------|
| PR-D1 | `verify:mechanical` | **Pass** | 2026-05-26 |
| PR-D2 | `hardening:wave4` | **Pass** | Prior sweep `wave4-hardening-2026-05-26T09-50-38-707Z.json`; re-run after major edits |
| PR-D3 | Hybrid TLAS default multi-mesh | **Pass** | `resolveReSTIRBvhMode`: ≥2 meshes → `tlas`; override `extensions['walkaround-hybrid'].bvhMode` |
| PR-D4 | Incremental patch matrix matches runtime | **Pass** | `BACKEND_PROMISE_LEDGER['walkaround-hybrid']` |
| PR-D5 | `updateEmitter` fast path | **Pass** | `HybridEngine.updateEmitter` → `updateEmitters()` |
| PR-D6 | Reference captures under `tools/reference-renders/PR-*` | **Pass (tooling + optional PNG)** | `npm run benchmark:pr-hybrid-refs`; `PR-hybrid/manifest.json` + per-scenario dirs on hybrid GPU host |
| PR-D7 | Animation doc superseded | **Pass** | Capability matrix in README + hybrid README |
| PR-D8 | pt-webgl README stability | **Pass** | Experimental/production wording aligned in root README |
| PR-D9 | Audit / pass registry | **Pass** | W1 pass registry unchanged in behavior; no new god-files in hybrid hot path |

## PR-6 benchmark commands

```bash
# Mechanical smoke (no GPU)
npm run benchmark:pr-mechanical --workspace @vitrum/benchmark-runner

# Full GPU bench (Windows host recommended)
VITRUM_PR_REQUIRE_GPU=1 VITRUM_PR_START_SERVER=1 \
  npm run benchmark:pr-hybrid --workspace @vitrum/benchmark-runner
```

## Residual

- Re-run `npm run benchmark:pr-hybrid` on GPU host for perf JSON when validating PR-6 budgets
