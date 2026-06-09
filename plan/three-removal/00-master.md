# Path B — Master Implementation Plan: `@vitrum/pt-webgl2` (THREE-free native WebGL2 path tracer)

> Status: **IMPLEMENTATION PLAN (meticulous / near-code).** 2026-06-08.
> Companion to [`../three-removal-webgl2-backend.md`](../three-removal-webgl2-backend.md) (the A-vs-B scoping case — read first).
> Every signature, schema, texel layout, and `file:line` in this doc set was extracted verbatim from the codebase and spot-verified by the lead. Citations are load-bearing — follow them when implementing.

## 0. Thesis

Build `@vitrum/pt-webgl2`, a THREE-free WebGL2 converged path tracer that:
- **Keeps** the fork's ~4,663 LOC of GLSL path-tracing kernels (verified THREE-free: 0 imports / 0 `from 'three'` / 0 `#include`).
- **Ports** ~5,777 LOC of CPU driver + data-packers (re-target the `DataTexture` carrier → raw WebGL2 textures; the byte-packing math is unchanged).
- **Rewrites** ~2,764 LOC of THREE `WebGLRenderer` render-framework glue as a thin raw-WebGL2 layer (FBOs / programs / fullscreen-quad / blend).
- **Replaces** `three-mesh-bvh` with `@vitrum/shared-bvh` (node layout already byte-identical) + a ~200 LOC BVH texture-packing adapter + the ~545 LOC `BVHShaderGLSL` ported as strings.
- **Mirrors** `@vitrum/pt-webgpu`'s structure (Engine contract, `GlResources`↔`GpuResources`, `composeTraceGlsl`↔`composePtWebgpuTraceWgsl`, the `PT_WEBGL2_SUPPORT` single-source pattern).

End state: `pt-webgl` + `three-gpu-pathtracer` + `three-mesh-bvh` + `three` deleted from the runtime graph; `three-bindings` slimmed to an optional glTF→core loader.

## 1. Goals / non-goals

**Goals**
- A `@vitrum/pt-webgl2` backend implementing the `@vitrum/core` `Engine` contract with `presentationMode: 'offscreen-texture'`, returning a `WebGLTexture` as `FrameOutput.primaryRadiance`.
- Fidelity parity with the current fork output (PSNR ≥ 28 on the six `pt-webgl-fidelity` scenarios — the gate already exists).
- Zero THREE / three-mesh-bvh in the WebGL2 backend (boundary-test enforced, like `pt-webgl`'s `legacyThreeBoundary.test.ts`).
- The robustness + freebie wins from [`08`](./08-freebies-and-future.md): graceful tiered degradation, context-loss recovery, generated-layout codegen, capability single-source.

**Non-goals (explicit)**
- The Looking-Glass quilt renderer (`QuiltPathTracingRenderer.js`, 229 LOC) — drop it.
- Shared WGSL↔GLSL source (the cross-backend math unifier) — a *follow-on* (see [`08`](./08-freebies-and-future.md) §"Newly enabled"), not in scope for the cutover.
- WebGL2 realtime GI — long-horizon, not a deliverable here.
- Differentiable RT (`createInverseSession`) — pt-webgpu-only; the contract method stays optional/omitted.

## 2. Verified work inventory (recap)

| Cat | LOC | Disposition | Detail doc |
|---|---|---|---|
| GLSL kernels (33 `*.glsl.js`) | 4,663 | **Keep, port-as-string** | [`04`](./04-glsl-kernels.md) |
| Driver + CPU packers (~30 files) | 5,777 | **Port** (DataTexture→GL texture) | [`03`](./03-scene-bvh-packers.md) |
| THREE render framework | 2,764 | **Rewrite** raw WebGL2 | [`02`](./02-gl-framework.md) |
| BVH (three-mesh-bvh → shared-bvh) | ~750 new | **Adapter + GLSL port** | [`03`](./03-scene-bvh-packers.md) |

## 3. Workstreams + dependency graph

```
WS1 contract+skeleton ─┬─> WS2 gl-framework ─┬─> WS5 frame-loop+features ─> WS6 validation ─> WS7 cutover
 (01)                  │     (02)            │      (05)                      (06)              (07)
                       ├─> WS3 scene+bvh+pack┘
                       │     (03)
                       └─> WS4 glsl-kernels  ┘  (04, parallel with WS2/WS3 once the uniform/define schema is frozen)
WS8 freebies (08) — woven into WS1/WS2 at greenfield (cheap now, costly to retrofit)
```

Hard ordering: WS1 (the package + contract skeleton) unblocks everything. WS2 (the GL framework: program builder + FBO/blend) and WS3 (scene/BVH/packers) are independent and parallel. WS4 (GLSL port) can start once the **uniform + `#define` schema is frozen** (it is fully enumerated in [`04`](./04-glsl-kernels.md) §1 — it does not change during the port). WS5 wires the renderFrame loop + ports the feature extensions. WS6 (validation) needs a real-GPU WebGL2 capture host (the one external blocker). WS7 cuts over + deletes.

## 4. Phasing — vertical slices (each slice = a shippable A/B gate)

| Slice | Deliverable | Gate |
|---|---|---|
| **S0 — diffuse spine** | `createPTEngine_WebGL2` + `GlResources` + BVH texture adapter + a minimal Lambertian-only kernel rendering one core `Scene` to a `WebGLTexture` | A/B vs a fork diffuse Cornell capture (PSNR target; software-GL OK for geometry correctness) |
| **S1 — material parity** | port `MaterialsTexture` (85px) + the GGX/metal/transmission BSDF kernels + the material `struct` GLSL | A/B vs fork on a multi-material scene |
| **S2 — lights + IBL** | port `LightsInfoUniformStruct` (6px) + `EquirectHdrInfoUniform` (CDF) + light/equirect sampling GLSL | A/B vs fork on the `rfe`-emitter + HDRI scenes |
| **S3 — fork extensions** | spectral hero-λ, MNEE caustics, BDPT, Jakob-Hanika, additive accumulation — each its own A/B; opportunistically swap in pt-webgpu's validated math | the six `pt-webgl-fidelity` baselines all ≥ 28 PSNR |
| **S4 — cutover** | facade selects `pt-webgl2`; delete pt-webgl/fork/three-mesh-bvh/three; slim three-bindings | full `npm test` + the real-GPU fidelity gate green; `three` absent from `npm ls` for the WebGL2 path |

## 5. The single forced edit to `@vitrum/core`

`BackendId` (`packages/core/src/engine/promiseLedger.ts:11`) is a **closed union**:
```ts
export type BackendId = 'walkaround-hybrid' | 'pt-webgl' | 'pt-webgpu';
```
Add `'pt-webgl2'` and a `BACKEND_PROMISE_LEDGER['pt-webgl2']` row (clone the `pt-webgl` row at `promiseLedger.ts:272-345`; `presentationMode:'offscreen-texture'`, `accumulates:true`, `supportsAuxBuffers` per WS5, `methodPromises.debug:false`). The compile-time `_LedgerCoversCapabilities` guard (`promiseLedger.ts:74-89`) will TS-error if the row drops a guarded cap key. **This is the only change inside `@vitrum/core`.** During S0–S3 the new package may temporarily reuse `'pt-webgl'`'s ledger row to avoid churning core until the capability surface is final.

## 6. Effort + risk register

**Effort (single experienced dev; algorithms kept):** WS1 ~3–5 d · WS2 ~2–3 wk · WS3 ~2 wk · WS4 ~1 wk wiring (+debug tail) · WS5 ~2–3 wk · WS6 capture-host ~days–blocker + per-feature A/B tail. **Implementation ≈ 6–8 wk + the validation tail.**

| Risk | Severity | Mitigation |
|---|---|---|
| **No real-GPU WebGL2 capture host** (software-GL only in WSL) | HIGH (gating) | WS6 stands one up first; pays double (also promotes the 8 fork rows out of `experimental`). Until then S0–S2 use software-GL for geometry/transform/consume correctness only. |
| Float-blend accumulation needs `EXT_float_blend` | MED | `GlResources` probes it (`supportsFloatBlending`, `WebGLPathTracer.js:10-14`); fall back to the alpha-composite ping-pong regime (regime 2 in [`02`](./02-gl-framework.md) §3) when absent. |
| MRT G-buffer (`gNormalDepth`/`gAlbedo`) unsupported on a device | LOW | aux buffers are optional (`supportsAuxBuffers`); single-output fallback (the fork already documents this — `PhysicalPathTracingMaterial.js:240-243`). |
| Uniform/define schema drift during port | MED | Schema is frozen + fully enumerated in [`04`](./04-glsl-kernels.md) §1; adopt the generated-layout codegen ([`08`](./08-freebies-and-future.md)) so packer↔shader can't drift. |
| Per-feature fidelity regressions (spectral/MNEE/BDPT) | MED | Each is its own S3 A/B vs the fork baseline; where pt-webgpu has a validated peer (manifold-NEE caustic, spectral), use its deterministic math harness as an independent oracle. |

## 7. Definition of done (per phase gate)

Each slice gate requires: (a) `npm run typecheck` clean; (b) the slice's vitest unit suite green; (c) the slice's A/B PSNR gate met on the capture host; (d) no `from 'three'` outside the (eventually-deleted) compatibility shim — enforced by a `pt-webgl2` boundary test mirroring `pt-webgl/src/__tests__/legacyThreeBoundary.test.ts`. The final S4 gate additionally requires `three` absent from the WebGL2 runtime dependency graph.

## 8. Document map

- [`01-contract-and-skeleton.md`](./01-contract-and-skeleton.md) — package skeleton, the exact `Engine` impl, `StateSlot`, factory, capabilities, the ledger row, texture branding.
- [`02-gl-framework.md`](./02-gl-framework.md) — `GlResources`, the program builder (`#define`-recompile + uniform aliasing + GLSL3), FBO/texture/UBO, the fullscreen-quad pass driver, the 3 accumulation blend regimes, ping-pong.
- [`03-scene-bvh-packers.md`](./03-scene-bvh-packers.md) — shared-bvh reuse, the BVH→4-texture adapter, the ported packers (Materials 85px, Lights 6px, EquirectHdr CDF, attributes array).
- [`04-glsl-kernels.md`](./04-glsl-kernels.md) — the kernel port, the `BVHShaderGLSL` port, the **complete** uniform + `#define` schema, the binding-convention remap, the compose order.
- [`05-frame-loop-and-features.md`](./05-frame-loop-and-features.md) — the renderFrame accumulation loop, frameParams packer + generated layout, the WebGL2 trace-tier gate, and the per-feature port plan (spectral/MNEE/BDPT/Jakob-Hanika/additive).
- [`06-validation.md`](./06-validation.md) — the real-GPU WebGL2 capture host, the fork-vs-native A/B oracle, per-feature fidelity gates, CI.
- [`07-phasing-cutover-cleanup.md`](./07-phasing-cutover-cleanup.md) — slice gates, the facade switch, deletions, the three-bindings slim, the cleanup checklist.
- [`08-freebies-and-future.md`](./08-freebies-and-future.md) — greenfield-discipline freebies, robustness wins, the shared-source + WebGL2-realtime-GI horizons.
