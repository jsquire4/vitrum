# vitrum library architecture

## Design principles

1. **The contract is the thing that's fixed.** Backends are swappable; scene bindings are swappable; denoisers are composable. The public types in `@vitrum/core` are the load-bearing interface.
2. **The host owns lifecycle.** Engine accepts a device handle but does not own the device. Engine accepts frame inputs but does not own the cadence. This is the design choice that makes `@vitrum/*` survive Canvas remounts and other host-level lifecycle churn.
3. **Generalize over time.** Today's contract handles the most pressing concrete needs. Each Phase 6 sprint generalizes one more dimension. The `Material.extensions`, `EngineOptions.extensions`, and `AnalyticShape` discriminated union are the explicit extension points where generalization happens without breaking the contract.
4. **No upstream PRs (yet).** While vitrum is pre-prime-time, `@vitrum/pt-webgl` consumes the absorbed `packages/three-gpu-pathtracer` renderer package via `file:` (see `packages/pt-webgl/package.json`). Upstream contribution is a v1+ concern.

## Package responsibilities

### `@vitrum/core`

**Owns**: types only (Scene, Material, ScenePrimitive, SceneEmitter, Engine, FrameInput, FrameOutput, EngineCapabilities). No GPU code, no scene-binding code, no backend-specific code.

**Depends on**: nothing.

**Why separate**: every other package imports from core. Core's stability is everyone's stability.

### `@vitrum/three-bindings`

**Owns**: adapter from a `THREE.Scene` to `@vitrum/core`'s `Scene`. Material translation (Three.js `MeshPhysicalMaterial` → `@vitrum/core`'s `Material`). Texture handle wrapping. Light translation.

**Depends on**: `@vitrum/core`, `three`.

**Why separate**: `@vitrum/core` doesn't know about three.js. A future babylon binding, glTF binding, or raw-buffer binding implements the same `Scene`-construction contract.

### `@vitrum/stained-glass-extensions`

**Owns**: stained-glass host-domain seams extracted from generic packages: `SURFACE_TEXTURE_ID`, stained-glass `userData` key constants, and analytic came UBO packing helpers.

**Depends on**: nothing.

**Why separate**: keeps stainedGlass-specific contracts out of backend-agnostic surfaces while preserving an explicit opt-in extension seam.

### `@vitrum/shared-bvh`

**Owns**: software BVH compute. Two implementations: WebGPU compute pipeline (used by walkaround engines) and WebGL2 fragment-shader-readable texture (used by the WebGL2 PT backend, wrapping three-mesh-bvh's BVH but exposed via vitrum's contract).

**Depends on**: `@vitrum/core`, optionally `three-mesh-bvh` for the WebGL2 path.

### `@vitrum/shared-samplers`

**Owns**:
- Hammersley sampling + golden-ratio rotation
- Light tree CDF construction (Phase 6 Sprint 3)
- Mixture PDF combination (Phase 6 Sprint 3)
- Henyey-Greenstein phase function (Phase 6 Sprint 7)
- Equi-angular volume scatter PDF (Phase 6 Sprint 7)
- Jakob+Hanika spectral upsampling (Phase 6 Sprint 8)

**Depends on**: `@vitrum/core`.

**Why separate**: samplers are reused across PT backends and walkaround engines. Centralizing prevents the three-implementations-of-Welford-variance trap.

### `@vitrum/shared-denoisers`

**Owns**:
- À-trous wavelet (current walkaround denoiser, Phase 6 baseline)
- SVGF — **SHIPPED** (`svgf-real`, Schied 2017; `svgfRealWebGPU.ts` in `shared-denoisers`; available on `walkaround-hybrid` only — `unsupported` on BOTH converged backends, pt-webgl and pt-webgpu, which use `oidn-final`: SVGF is a real-time 1-spp filter, a regime mismatch for converged tracers)
- BMFR — **SHIPPED** (real Koskela-2019 Householder-QR feature regression in `@vitrum/shared-denoisers`; `BmfrDenoiser` in walkaround-hybrid, `denoiser: 'bmfr'`)
- OIDN final-pass via ONNX Runtime Web + WebNN execution provider (Phase 6 Sprint 10b)

**Depends on**: `@vitrum/core`.

### `@vitrum/pt-webgl`

**Owns**: implementation of the `Engine` contract via the forked three-gpu-pathtracer. Today: wraps the WebGL2 PT pipeline. Future: deprecated when `@vitrum/pt-webgpu` reaches feature parity.

**Depends on** (see `packages/pt-webgl/package.json`): `@vitrum/core`, `@vitrum/shared-samplers`, `@vitrum/shared-denoisers`, `@vitrum/three-bindings`, `three-gpu-pathtracer` (fork), `three-mesh-bvh`. BVH building blocks used indirectly via the fork and three.js stack (not a direct `@vitrum/shared-bvh` dependency); `@vitrum/shared-denoisers` IS a direct dependency (OIDN-final wire).

### `@vitrum/pt-webgpu` *(experimental backend, evolving toward Phase 7 goals)*

**Owns**: a from-scratch WebGPU-native path-tracer backend. Current implementation is an active experimental backend (progressive accumulation + CPU-built BVH + GPU traversal + multi-bounce diffuse/specular baseline), evolving toward hero-wavelength spectral, fuller Disney BSDF coverage, neural radiance caching (NRC), and other techniques that don't fit cleanly into the WebGL2 fragment-shader model.

**Depends on** (see `packages/pt-webgpu/package.json`): `@vitrum/core`, `@vitrum/shared-bvh`, `@vitrum/shared-samplers`, `@vitrum/shared-denoisers`. Notably **not** `three-gpu-pathtracer`.

### `@vitrum/walkaround-hybrid`

**Owns**: the WebGPU layered DDGI + RC + ReSTIR DI compute pipeline (the crown jewel — see `_staging/legacy-source/src/rendering/scene/walkaround/engines/restir/`). Implements the `Engine` contract for real-time GI use cases.

**Depends on**: `@vitrum/core`, `@vitrum/shared-bvh`, `@vitrum/shared-samplers`, `@vitrum/shared-denoisers`.

### `@vitrum/walkaround-rc`

**Owns**: Radiance Cascades subsystem — cascade pyramid, cascade buffer management, and `RCDispatcher` (raw-GPU and THREE-coupled entry points). Consumed by `@vitrum/walkaround-hybrid`.

**Depends on**: `@vitrum/core`.

### `@vitrum/scene-lighting`

**Owns**: host-side lighting-state primitives — time-of-day sky params, sun geometry, intensity tables, and `computeLightingState`. (Emitter packing lives in `pt-webgpu`; light-tree CDF construction lives in `@vitrum/shared-samplers` — not here.)

**Depends on**: `three` only (no `@vitrum/*` deps).

## How a host application consumes vitrum

After extraction, a host app's rendering layer looks like:

```typescript
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl';
import { createWalkaroundEngine_Hybrid } from '@vitrum/walkaround-hybrid';

// In a React effect:
const ptEngine = await createPTEngine_WebGL2({ device: glContext, samplesPerPixel: 192 });
const scene = sceneFromThreeJS(threeScene);
ptEngine.setScene(scene);

// In useFrame:
const output = ptEngine.renderFrame({
  viewMatrix, projMatrix, cameraPosition,
  viewport, frameIndex, frameSeed,
});
```

The host application retains:
- Domain composition logic (panel cell layout, came/solder generation, glass material profiles)
- Three.js scene assembly
- React lifecycle wrapping (`PathTracingLayer`, `WalkaroundStage`, etc. become thin host wrappers around `@vitrum/*` engines)
- UI controls

Everything else moves to `@vitrum/*`.

## Migration plan from the imported legacy source

Phase 6 Sprint 0 (next, 2–3 days) defines the contract and creates the package skeletons. This is already done — see `packages/core/src/*.ts`.

Subsequent sprints land their deliverables in vitrum packages, not the host app's source tree:

| Sprint | Work | Lands in |
|---|---|---|
| 0 (this one) | Contract + package skeletons | `@vitrum/core` |
| 1 | PT preview perf wins | `@vitrum/pt-webgl` config options |
| 2 | Per-cell luminance precompute | `@vitrum/shared-samplers` |
| 3 | Mixture PDF + light tree + back-face NEE | `@vitrum/shared-samplers` |
| 4 | BSDF cost reduction (lobeMask + lite BSDF + material LOD) | `@vitrum/pt-webgl` (fork-internal patches) |
| 5 | Analytic CSG came + MRT G-buffer | `@vitrum/pt-webgl` engine internals; CSG primitive type in `@vitrum/core` |
| 6 | Rough refraction + spatial denoiser | BSDF in `@vitrum/pt-webgl`; denoiser in `@vitrum/shared-denoisers` |
| 7 | Volume + SSS + equi-angular | `@vitrum/pt-webgl` (fork) + HG phase in `@vitrum/shared-samplers` |
| 8 | RGB-as-3λ + Jakob+Hanika | `@vitrum/pt-webgl` + spectral utility in `@vitrum/shared-samplers` |
| 9 | Adaptive sampling + checkerboard | `@vitrum/walkaround-hybrid` + Welford struct in `@vitrum/shared-samplers` |
| 10a | SVGF / BMFR — **SHIPPED** | `@vitrum/shared-denoisers` |
| 10b | OIDN ONNX final pass | `@vitrum/shared-denoisers/oidn-bridge` |
| 10c | Vanilla BDPT | `@vitrum/pt-webgl` |
| 11 | PPG — **SHIPPED** (W9) | `@vitrum/walkaround-hybrid` (`src/ppg/`; opt-in via `HybridEngineOptions.ppgEnabled`) |
| 12 | Hero spectral | `@vitrum/pt-webgl` AND/OR `@vitrum/pt-webgpu` |
| 13 | Custom WebGPU neural denoiser — **SHIPPED** | `@vitrum/walkaround-hybrid` (`src/neural/InferenceGraph.ts`; opt-in via `denoiser: 'neural'` + `neuralWeights`) |
| 6.5 | ReSTIR BDPT in walkaround | `@vitrum/walkaround-hybrid` extension or `@vitrum/walkaround-restir-bdpt` |

The host application's renderer subdirectory eventually empties as its files move to vitrum packages. Same for the PT pipeline files. The host's role narrows to "domain composition + UI + scene assembly."

## Versioning + release strategy

Pre-1.0: every release is a snapshot. Breaking changes in the core contract are expected. Versions track the monorepo (vitrum 0.0.0 → 0.0.1 → ... → 0.1.0 when first feature-stable).

Post-1.0: the public façade in `@vitrum/core` follows semver strictly. Backend implementations may release independently as they mature.

Local-only until prime time: no npm publish until the contract has been stable for at least 2 quarters of host-app consumption without breaking changes. Until then, packages are linked via `npm workspaces` (i.e., `file:../packages/core`).
