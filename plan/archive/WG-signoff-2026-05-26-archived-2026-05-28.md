> ARCHIVED 2026-05-28 — WebGPU-PT-parity program complete.

# WebGPU PT Parity Program — Signoff (2026-05-26)

**Plan:** `plan/primary-release-and-webgpu-pt-parity-implementation-deep.md`

## Wave status

| Wave | Status | Evidence |
|------|--------|----------|
| WG-0 Visual harness | **Complete** | `benchmark:gap-closure-mechanical`, committed baselines |
| WG-1 OIDN-final | **Complete** | `oidnFinalDispatcher.ts`, integration tests |
| WG-2–6 | **Complete** | See prior signoff + `wgslContract.test.ts` |
| WG-7 BDPT | **Complete (v1)** | `bdptLightPathBufferWebGPU`, CPU light-path fill, `evaluateBdptConnection` in full-tier kernel, `bdptAdvanceFrame`, `extensions['vitrum.ptWebgpu.bdpt']` |
| WG-8 createEngine opt-in | **Complete** | `quality-webgpu`; `auto` → pt-webgpu when WebGPU + large scenes |
| WG-9 Extended denoisers | **Complete** | `denoiser: 'svgf-real'` on pt-webgpu full tier (`SVGFRealDispatcher`) |
| WG-10 Signoff | **This document** | |

## Host workflows

```bash
npm run benchmark:gap-closure-mechanical
VITRUM_STRICT_GAP_CLOSURE=1 npm run benchmark:gpu-windows -- run-gap-closure-verification.mjs
npm run benchmark:pr-hybrid-gpu-windows
```
