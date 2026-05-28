# vitrum — Phase 0 productization (implementation plan)

**Status:** implementation plan (file-level). **Date:** 2026-05-28.
**Scope:** roadmap `plan/roadmap.md` §5.1, §5.2, §4.1, §4.3 — three Phase-0 deliverables
plus the §5.1 `resolutionFactor` wiring. This doc is a **plan only**; it cites the real
code that exists today and specifies exact edits. It does not change any source.

Phase-0 north star (roadmap §0.5 item 3): graceful degradation across hardware Class A–E
via a **library-exported adapter probe**, **quality presets**, and a **hybrid lite tier**.

---

## 0. Ground-truth map (what exists today, verified by code-read)

| Concern | Lives at | Notes |
|---|---|---|
| Adapter caps probe (full/lite/hybrid booleans + limits) | `tools/benchmark-runner/launchWebGpuBrowser.mjs:42-58` (`readWebGpuAdapterCaps`) | Playwright-only, in-page `navigator.gpu` eval. The thresholds (10/5 full, 8/4 lite, 16/8 hybrid) are inlined here. |
| Core SwiftShader heuristic + limit reader | `packages/core/src/wgpuSupport.ts:25-33` (`isSwiftShaderAdapter`), `:98-136` (`probeWebGPU`) | Already in `@vitrum/core`; returns `{supported, vendor, architecture, features, limits, adapterKind}`. **Reusable foundation.** |
| Backend routing | `packages/engine/src/createEngineScale.ts:49-75` (`pickBackend`) | `prefer × hasWebGPU × triangleCount × needsTlas` → backend id. No limit awareness today. |
| pt-webgpu tier selection | `packages/pt-webgpu/src/traceTier.ts:16-39` (`selectPtWebgpuTraceTier`) + `webgpuLimits.ts:11-24` (threshold consts) + `ptwebgpuRequiredLimitsForAdapter` `webgpuLimits.ts:31-55` | Exported from `@vitrum/pt-webgpu/index.ts:77-83`. Full = 10 buf/5 tex; lite = 8 buf/4 tex. |
| Hybrid required limits | `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:125-132` (`HYBRID_WEBGPU_REQUIRED_LIMITS`) | `maxStorageBuffersPerShaderStage: 16`, `maxStorageTexturesPerShaderStage: 8`. Exported from package index `:14-17`. |
| Hybrid ctor options (~50 knobs) | `packages/walkaround-hybrid/src/HybridEngineOptions.ts:76-649` | The host-overridable surface. |
| Per-frame tunable table | `packages/walkaround-hybrid/src/HybridEngineTuning.ts:122-148` (`TUNABLE_DEFINITIONS`) | 21 number knobs + 2 init knobs; defaults Cornell-baseline. |
| Per-frame UBO layout (352 B) | `packages/walkaround-hybrid/src/shaders/walkaroundUbo.wgsl.ts:82-127` | Packed by `pipeline/uboUpdater.ts`. |
| Pass gating (runtime) | `pipeline/Pass.ts:39-46` (`PassGateOptions`), `:252` (`gates()`); loop at `WalkaroundGPUPipeline.ts:937-957` | `gates({denoiserMode, ppgEnabled})` per pass; non-gating passes return `true`. |
| Internal render size | `HybridEngine._width/_height` (`HybridEngine.ts:283-284`); set in ctor `:460-461`, mutated by `setSize()` `:1058-1073`; pipeline mirror `WalkaroundGPUPipeline.resize()` `:726-755`. | `runHybridEngineFrame` passes `screenWidth/Height = deps.width/height` to `pipeline.renderFrame` (`HybridEngineFrameOrchestrator.ts:297-298`). |
| Composite upscale | `pipeline/passes/CompositePass.ts:44-67` — fullscreen-triangle blit of `resolvedTexture` → `swapChainView` via `compositeSampler`. | **Sampler is `nearest`** (`createCommonFrameResources.ts:109-114`) — blocky upscale; relevant for `resolutionFactor`. |
| `FrameInput.quality.resolutionFactor` | `packages/core/src/frame.ts:46-47` | Contract present; hybrid currently ignores it (`frame.ts:75-86` documents the rejection). |

### Compile-time constants that §4.3 wants to vary but have **no runtime knob today**

These are WGSL `const`s baked into the shader strings — varying them at runtime means
either a UBO field + shader edit, a pipeline permutation, or (cheapest) leaving them fixed
and degrading the *adjacent* runtime knob instead:

- DI spatial neighbor count `NEIGHBORS = 5u` — `shaders/spatial.wgsl.ts:41`.
- DI spatial pass **count** (2 ping-pong) — hardcoded as two labels in `passes/SpatialReservoirPass.ts:19` + `passOrder.ts:59`.
- GI spatial neighbor count `K_SPATIAL_GI = 5u` — `shaders/spatialGi.wgsl.ts:32`.
- GI spatial pass count (2 ping-pong) — `passes/SpatialGIReservoirPass.ts` + `passOrder.ts:62`.
- RIS-GI base candidate count `M_GI_BASE = 8u`, scaled per-pixel by adaptive tier — `shaders/risGi.wgsl.ts:77,150-151`.
- DI RIS `M_LIGHT = 64u` — `shaders/ris.wgsl.ts:73`.
- DDGI round-robin coverage `probesPerFrame = ceil(totalProbes / 4)` — **hardcoded `/4`** in `ddgi/probeUpdateFrameParams.ts:63` + `ddgi/probeUpdateBlend` consumer. This is the only "DDGI probe update stride" lever and it is fixed.

**Design consequence (locked by maintainer preference — fidelity + flexibility, no hardcoded single-path):**
where §4.3 asks for a dimension that today is a compile-time `const`, the preset table below maps to
the **closest existing runtime knob** AND flags the const as a follow-up "promote to UBO" item.
Presets must not silently no-op a dimension; every row resolves to a real field write or an
explicit documented "fixed at N, not yet tunable" note.

---

## Deliverable 1 — `probeAdapterProfile()` as a library export

### 1.1 Decision: export from `@vitrum/engine`, type alias in `@vitrum/core`

**Reasoning (dependency-direction forced):** `@vitrum/core` has **zero** package deps
(verified — `packages/core/package.json` has no `dependencies` block). The limit thresholds
live in `@vitrum/walkaround-hybrid` (`HYBRID_WEBGPU_REQUIRED_LIMITS`) and `@vitrum/pt-webgpu`
(`PT_WEBGPU_FULL_*` / `PT_WEBGPU_LITE_*`). Core cannot import those without inverting the
dependency graph. `@vitrum/engine` already depends on **all three** (`packages/engine/package.json`
deps: core, walkaround-hybrid, pt-webgl, pt-webgpu) and already imports `ptWebgpuRequiredLimitsForAdapter`
(`createEngine.ts:33`). Therefore:

- The **`AdapterProfile` interface** (pure data shape, no thresholds) → `@vitrum/core`
  (alongside `WgpuProbeResult` in a new file or appended to `wgpuSupport.ts`). This keeps the
  shape on the contract surface so hosts can type-annotate without pulling in `@vitrum/engine`.
- The **`probeAdapterProfile()` function** (reuses the real thresholds) → `@vitrum/engine`,
  because only there can it import both packages' threshold constants without duplicating magic numbers.

This honors design principle #1 ("the contract is the thing that's fixed" — type in core)
while satisfying "don't duplicate the magic numbers" (function in engine, importing the consts).

### 1.2 Type shape (new — `packages/core/src/adapterProfile.ts`)

```ts
// @vitrum/core — graceful-degradation adapter capability report.
export type RealtimeTier = 'ultra' | 'high' | 'medium' | 'low' | 'unavailable';
export type HeroBackendRec = 'pt-webgpu-full' | 'pt-webgpu-lite' | 'pt-webgl' | 'none';
export type PtWebgpuTierRec = 'full' | 'lite' | 'none';

export interface AdapterProfile {
  readonly hasWebGPU: boolean;
  /** Meets HYBRID_WEBGPU_REQUIRED_LIMITS (16 buf / 8 tex). */
  readonly hybridCapable: boolean;
  /** Meets a reduced HYBRID_LITE_LIMITS (Deliverable 3). False ⇒ hybrid unavailable. */
  readonly hybridLiteCapable: boolean;
  readonly ptWebgpuTier: PtWebgpuTierRec;
  readonly maxStorageBuffersPerStage: number;
  readonly maxStorageTexturesPerStage: number;
  /** SwiftShader-class heuristic (vendor==='google' && arch==='swiftshader'). */
  readonly isSoftwareAdapter: boolean;
  readonly adapterKind: WgpuAdapterKind;      // re-exported from wgpuSupport
  readonly recommendedRealtimeTier: RealtimeTier;
  readonly recommendedHeroBackend: HeroBackendRec;
  /** Raw limit bag (forward-compat; lets hosts read anything not surfaced above). */
  readonly limits: Readonly<Record<string, number>>;
}
```

Export from `packages/core/src/index.ts` (append after line 12): `export type { AdapterProfile, RealtimeTier, HeroBackendRec, PtWebgpuTierRec } from './adapterProfile.js';`

### 1.3 Function (new — `packages/engine/src/adapterProfile.ts`)

Signature (overloaded to accept either a live `GPUDevice`, a `GPUAdapter`, or nothing):

```ts
export async function probeAdapterProfile(
  source?: GPUDevice | GPUAdapter
): Promise<AdapterProfile>;
```

Implementation steps:

1. **Resolve limits + info.**
   - No `source`: call `probeWebGPU()` from `@vitrum/core` (`wgpuSupport.ts:98`). If
     `!supported` → return the all-false `unavailable` profile (`hasWebGPU:false`,
     `recommendedRealtimeTier:'unavailable'`, `recommendedHeroBackend: hasWebGL2 ? 'pt-webgl' : 'none'`).
     (WebGL2 presence: reuse `detectGpu` from core, `gpuDetection.ts`, to avoid a second path.)
   - `source` is a `GPUDevice`: read `source.limits` + (best-effort) `source.adapterInfo`/none.
   - `source` is a `GPUAdapter`: read `adapter.limits` + `readAdapterInfo`-style info.
   Factor the limit-bag extraction so it matches `probeWebGPU` (`wgpuSupport.ts:109-114`) exactly.

2. **Compute capability booleans from REAL thresholds (no inline numbers):**
   ```ts
   import { HYBRID_WEBGPU_REQUIRED_LIMITS } from '@vitrum/walkaround-hybrid';
   import {
     selectPtWebgpuTraceTier,           // for GPUDevice input
     PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP,
     PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
     PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
     PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
   } from '@vitrum/pt-webgpu';
   import { HYBRID_LITE_LIMITS } from '@vitrum/walkaround-hybrid'; // NEW (Deliverable 3)

   const buf = limits.maxStorageBuffersPerShaderStage ?? 8;
   const tex = limits.maxStorageTexturesPerShaderStage ?? 4;
   const hybridCapable =
     buf >= HYBRID_WEBGPU_REQUIRED_LIMITS.maxStorageBuffersPerShaderStage &&
     tex >= HYBRID_WEBGPU_REQUIRED_LIMITS.maxStorageTexturesPerShaderStage;
   const hybridLiteCapable =
     buf >= HYBRID_LITE_LIMITS.maxStorageBuffersPerShaderStage &&
     tex >= HYBRID_LITE_LIMITS.maxStorageTexturesPerShaderStage;
   const ptWebgpuTier: PtWebgpuTierRec =
     (buf >= PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP && tex >= PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE) ? 'full'
     : (buf >= PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE && tex >= PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE) ? 'lite'
     : 'none';
   ```
   For a `GPUDevice` input, prefer `selectPtWebgpuTraceTier(device)` wrapped in try/catch
   (it *throws* below lite — map throw → `'none'`) so the function reuses pt-webgpu's own
   verdict logic rather than re-deriving it.

3. **`isSoftwareAdapter`:** `isSwiftShaderAdapter(info)` from `@vitrum/core` (`wgpuSupport.ts:25`).

4. **Recommended realtime tier** (maps Class A–E from roadmap §4.2 to a preset id; pure data,
   testable without a GPU):
   ```
   !hasWebGPU                       → 'unavailable'   (Class E)
   isSoftwareAdapter                → 'unavailable'   (Class D — never init hybrid on SwiftShader, §4.4/§10.4)
   !hybridLiteCapable               → 'unavailable'   (Class D)
   hybridCapable && !isSoftware     → 'ultra'  if buf≥16 && tex≥8 (Class A)
   hybridLiteCapable && !hybridCap. → 'medium' (Class B/C — lite tier path)
   ```
   (The host can always *downshift* via explicit `qualityTier`; this is a ceiling, not a lock.)

5. **Recommended hero backend:**
   ```
   ptWebgpuTier==='full' → 'pt-webgpu-full'
   ptWebgpuTier==='lite' → 'pt-webgpu-lite'
   hasWebGL2             → 'pt-webgl'
   else                  → 'none'
   ```

Export from `packages/engine/src/index.ts` (append to the createEngine block, `index.ts:6-14`):
`export { probeAdapterProfile } from './adapterProfile.js';` and re-export the `AdapterProfile`
type from core for one-import ergonomics.

### 1.4 `createEngine()` consumption

`createEngine` (`packages/engine/src/createEngine.ts:86-119`) today calls `detectGpu` then
`pickBackend`. Wire the profile in **without changing default behavior** (additive):

1. After `const gpu = await detectGpu(...)` (`:101`), if the backend will be `walkaround-hybrid`,
   call `probeAdapterProfile(adapter)` once (reuse the `adapter` already requested at
   `constructWalkaround:132`) **before** `requestDevice` and:
   - If `!profile.hybridLiteCapable` → throw the actionable Class-D error mandated by §4.4
     ("Never initialize hybrid on hybridCanRun:false… suggest `prefer:'quality'`"). Today the
     code blindly `requestDevice()`s and may fail opaquely.
   - Pass `profile.recommendedRealtimeTier` into the merged options as the **default**
     `qualityTier` (Deliverable 2) *only when* `opts.advanced` did not set one — satisfies
     roadmap §5.1 "createEngine() applies recommended tier when advanced omits overrides."
2. Add an optional `CreateEngineOptions.onAdapterProfile?: (p: AdapterProfile) => void` callback
   (`createEngine.ts:62-84`) so hosts can read the JSON for a HUD / CI artifact (§4.1 step 1,
   §10.3 "Adapter profile JSON attached").

### 1.5 Tests (`packages/engine/src/__tests__/adapterProfile.test.ts`)

All **unit-testable** (no GPU) by passing a fake `{limits:{...}}` device/adapter:
- 16/8 → `hybridCapable`, `ptWebgpuTier:'full'`, `recommendedRealtimeTier:'ultra'`.
- 10/5 → `!hybridCapable`, `hybridLiteCapable` (if ≥ lite limits), `ptWebgpuTier:'full'`.
- 8/4 → `ptWebgpuTier:'lite'`, hybrid-lite verdict depends on `HYBRID_LITE_LIMITS`.
- SwiftShader info (`vendor:'google', arch:'swiftshader'`) → `isSoftwareAdapter:true`,
  `recommendedRealtimeTier:'unavailable'` even if limits pass.
- No `navigator.gpu` → `hasWebGPU:false`, `recommendedHeroBackend` falls to `'pt-webgl'`/`'none'`.
- **Threshold-coupling guard:** assert the boolean flips exactly at the imported constant
  (e.g. `buf === HYBRID_WEBGPU_REQUIRED_LIMITS.maxStorageBuffersPerShaderStage - 1` ⇒ false),
  so the test fails if anyone forks the magic number instead of importing it.

GPU-validation-gated: the *real* `probeWebGPU()` path (no `source`) needs Playwright/Chrome —
covered by extending `tools/benchmark-runner/run-pt-webgpu-adapter-probe.mjs` to emit the new
`AdapterProfile` JSON (replace its hand-built `report` object `run-pt-webgpu-adapter-probe.mjs:45-63`
with a call into the shared verdict logic so the tool and library can't drift).

---

## Deliverable 2 — Quality presets (`ultra` / `high` / `medium` / `low`)

### 2.1 New option: `HybridEngineOptions.qualityTier`

Add to `HybridEngineOptions` (`HybridEngineOptions.ts`, near the library-generality block ~`:236`):
```ts
/** Coarse preset. Resolves to concrete tunable/UBO/pass-gate values per the
 *  roadmap §4.3 table. Explicit per-knob options in this same object OVERRIDE
 *  the preset (preset is a baseline, not a lock). Default: 'ultra' (= current
 *  Cornell-baseline behavior, byte-identical). */
readonly qualityTier?: 'ultra' | 'high' | 'medium' | 'low';
```

### 2.2 Preset → knob resolution module (new — `packages/walkaround-hybrid/src/HybridEngineQualityPreset.ts`)

A pure, table-driven resolver (mirrors `HybridEngineTuning.ts`'s table style):
```ts
export interface QualityPreset {
  readonly resolutionFactor: number;                 // → render-size scaling (Deliverable 4 path)
  readonly adaptiveSamplingThresholds: [number, number]; // → M_GI proxy (UBO offset 384? see below)
  readonly gtaoMode: 'on' | 'quarter' | 'off';       // → pass gate + GTAOPass half/quarter dispatch
  readonly denoiser: HybridEngineOptions['denoiser'];
  readonly targetFrameIntervalMs: number | null;
  readonly diSpatialPasses: 1 | 2;                   // → SpatialReservoirPass label count (needs pass-count knob)
  readonly giSpatialPasses: 1 | 2;
  readonly enableRcPpgNeuralByDefault: boolean;      // documentary; presets never force RC/PPG ON
}
export const QUALITY_PRESETS: Readonly<Record<'ultra'|'high'|'medium'|'low', QualityPreset>> = { ... };
```

The HybridEngine ctor calls `resolveQualityPreset(opts.qualityTier ?? 'ultra')` FIRST, then
applies per-knob overrides from `opts` on top (so `qualityTier:'low'` + explicit `gtao:{...}`
keeps the explicit gtao). Resolution feeds into the existing `readTunables` / `readInitTunables`
defaulting and the new pass-gate/resolution wiring below.

### 2.3 Preset → real-knob mapping table (grounded in actual field names)

| §4.3 dimension | Ultra | High | Medium | Low | **Real lever (file:line)** | Mechanism |
|---|---|---|---|---|---|---|
| Internal resolution scale | 1.0 | 0.85 | 0.67 | 0.5 | `HybridEngine._width/_height` → `pipeline.resize()` (`HybridEngine.ts:1058`, `WalkaroundGPUPipeline.ts:726`) | **Deliverable 4** — apply factor to render-target size; composite upscales. |
| ReSTIR spatial passes | 2 | 2 | 1 | 1 | `passes/SpatialReservoirPass.ts:19` `passLabels` (2 hardcoded) | **No runtime knob today.** Add `SpatialReservoirPass` ctor arg `passCount:1\|2` driven by preset; `passLabels` slices `['spatial-1','spatial-2']`. Pass-layout/timestamp must stay consistent (see Risk R2). |
| Spatial neighbors | 5 | 5 | 3 | 3 | `shaders/spatial.wgsl.ts:41` `NEIGHBORS=5u` (compile-time const) | **No runtime knob.** Map to **new UBO field `diSpatialNeighbors`** (UBO + shader read) OR leave fixed-at-5 and document. **Recommend: leave fixed at 5 for Phase 0**, degrade via pass count + resolution instead (avoids a UBO+shader edit for a marginal lever). Flag as P1 promote-to-UBO. |
| ReSTIR-GI M_GI scale (full/reduced/minimal) | full | full | reduced | minimal | `shaders/risGi.wgsl.ts:77,150-151` `M_GI = M_GI_BASE(8) * tier / 2`; tier from `sampleBudget.wgsl.ts:89-91` driven by `adaptiveSamplingThresholds` (`HybridEngineOptions.ts:405`, UBO offsets via `TUNABLE_DEFINITIONS` M2) | **Runtime knob exists.** Raise `adaptiveSamplingThresholds` so more pixels fall to tier 1 (M_GI→4) instead of tier 2/4 (M_GI 8/16). `reduced` = `[0.04,0.40]`, `minimal` = `[0.20,2.0]` (tune on GPU). This is the §4.3 "reduced/minimal" lever realized through the existing tier classifier. |
| DDGI probe update stride | default | default | 2× | 4× | `ddgi/probeUpdateFrameParams.ts:63` `probesPerFrame = ceil(totalProbes/4)` (hardcoded `/4`) | **No runtime knob.** Add `ddgiUpdateDivisor` param threaded from preset (default 4 → medium 8 → low 16, i.e. "2×/4× stride" relative to the default 4-frame cycle). One-line change in `packProbeUpdateFrameParams` + plumb through `DDGI.updateFrame` callsite. |
| GTAO | on | on | on | off / quarter-res | `GTAOPass.gates()` returns `true` always (`passes/GTAOPass.ts:49`); half-res dispatch `:77-80` | **Add `gates(opts)` check** on a new `PassGateOptions.gtaoEnabled` (off ⇒ skip GTAO + GTAOUpsample). Quarter-res = change the `/2` divisors at `GTAOPass.ts:77-78` to `/4` driven by a dispatch-scale field. |
| Denoiser | atrous-variance | atrous-variance | atrous-variance | atrous | `HybridEngineOptions.denoiser` (`:196`); registry lookup `WalkaroundGPUPipeline.ts:511`; gate via `PassGateOptions.denoiserMode` (`Pass.ts:41`) | **Direct field.** Preset sets `denoiser`. ('atrous' legacy mode already supported.) |
| RC / PPG / neural | opt-in | off default | off | off | `rcEnabled`/`ppgEnabled` (`:578,612`), `denoiser:'neural'` | **Documentary only** — presets NEVER force these ON (they need extra resources/weights). Preset just records the recommendation; default stays OFF. `enableRcPpgNeuralByDefault` is informational for host UI. |
| `targetFrameIntervalMs` | null | null | 20 | 33 | `HybridEngineOptions.targetFrameIntervalMs` (`:248`); enforced `HybridEngineFrameOrchestrator.ts:236-247` | **Direct field.** Preset sets it. |

**Net new knobs required by the table (Phase 0):**
1. `SpatialReservoirPass`/`SpatialGIReservoirPass` runtime pass-count (ctor arg).
2. `GTAOPass.gates()` honoring a `gtaoEnabled` flag + a dispatch-scale (half/quarter/off).
3. `ddgiUpdateDivisor` threaded into `packProbeUpdateFrameParams`.
4. (Resolution factor — Deliverable 4.)

Everything else uses fields that already exist. Spatial-neighbor count is **intentionally left
fixed at 5** for Phase 0 (documented), to avoid a UBO-layout + 2-shader edit for the smallest lever.

### 2.4 Wiring details

- **Pass-gate extension:** add `gtaoEnabled: boolean` to `PassGateOptions` (`Pass.ts:39-46`,
  already an open `[extension:string]:unknown` bag) and build it in the per-frame `gateOpts`
  (`WalkaroundGPUPipeline.ts:937-940`). `GTAOPass.gates(opts)` and `GTAOUpsamplePass.gates(opts)`
  return `opts.gtaoEnabled !== false`. Threading source: `HybridEngine` → pipeline field set at
  init (preset is fixed per engine instance, so store on the pipeline like `_denoiserMode`
  `WalkaroundGPUPipeline.ts:347`, not per-frame).
- **Spatial pass count:** give `SpatialReservoirPass` a `constructor(pipeline, passCount: 1|2 = 2)`
  and compute `passLabels = passCount === 1 ? ['spatial-2'] : ['spatial-1','spatial-2']`. The
  static `passOrder.ts:59` + `composePassLabels` + timestamp layout must reflect the **same**
  count, so `buildPassLayout` (`WalkaroundGPUPipeline.ts:437,852`) needs the count passed in.
  (See Risk R2 — keep timestamp `querySet` sizing consistent.)
- **DDGI divisor:** `DDGI.updateFrame` → `probeUpdatePass` already calls
  `packProbeUpdateFrameParams`; add the divisor to `ProbeUpdateFrameParamsInput`
  (`probeUpdateFrameParams.ts:49-55`) and to the blend params (`:78-85`, must match the
  consumer `probeUpdateBlend.wgsl.ts` expectation that blend and ray passes agree on coverage).

### 2.5 Tests (characterization — mostly **unit-testable**)

`packages/walkaround-hybrid/src/__tests__/qualityPreset.test.ts`:
- `resolveQualityPreset('ultra')` deep-equals the Cornell-baseline values (regression guard:
  ultra MUST be byte-identical to today's defaults — assert `adaptiveSamplingThresholds:[0.01,0.10]`,
  `targetFrameIntervalMs` default, `resolutionFactor:1.0`, `diSpatialPasses:2`).
- Each preset maps to the exact table values above.
- Per-knob override beats preset: `{qualityTier:'low', gtao:{radiusPx:99}}` ⇒ gtao radius 99
  retained while other low-tier values applied.
- **UBO characterization:** call the (pure) `uboUpdater` pack with each preset's resolved
  `adaptiveSamplingThresholds*` and assert the DataView bytes at the M2 offsets
  (`walkaroundUbo.wgsl.ts:134-135` thresholds map; offsets per `uboUpdater.ts` header) — this is
  the "preset → expected UBO flags" characterization the roadmap §5.1 calls for, runnable headless.
- **Pass-flag characterization:** `buildPassLayout({denoiserMode, gtaoEnabled:false, diSpatialPasses:1})`
  (the data-only layout builder, GPU-free per its own docstring `passOrder.ts:19-23`) ⇒ assert the
  label list omits `gtao`/`gtao-upsample` and contains a single `spatial-2`.

GPU-validation-gated: actual frame-time deltas per preset (`tools/benchmark-runner` budgets,
roadmap §5.3) and visual A/B of low vs ultra.

---

## Deliverable 3 — Hybrid lite tier

### 3.1 Decision: **runtime UBO/gate feature-gating, NOT a second shade pipeline**

**Reasoning (explicit, per the §5.2 risk note "prefer runtime UBO gating over N× pipeline objects"):**
The hybrid pipeline already gates optional passes at runtime (`gates(opts)` loop,
`WalkaroundGPUPipeline.ts:954-956`) and already gates stained-glass/RC physics via UBO **flag
bits** that short-circuit to `vec3f(0)` with no recompile (`stainedGlassFlags`
`walkaroundUbo.wgsl.ts:120-133`; RC `enabled==0u` short-circuit `bindGroupLayouts.ts:139-148`).
A second shade variant would fork `shade.wgsl` and double the compile/bind-layout maintenance.
So lite = **the same pipeline run with optional passes gated off + a lighter resource budget**,
selected by the adapter profile. This is identical in spirit to how pt-webgpu lite differs from
full (fewer aux buffers), but achieved without a WGSL fork.

### 3.2 What "lite" drops to fit a smaller storage-texture / buffer budget

The pressure points (verified):
- **Frame BGL storage textures:** slots 8/10/12/13/14 = **5 write storage textures**
  (`bindGroupDescriptors.ts:117,123,129,132,136`), plus GTAO/atrous/accum write 1 each in their
  own groups. The 8-tex hybrid requirement (`HYBRID_WEBGPU_REQUIRED_LIMITS` tex:8) is driven by
  shade writing 4 simultaneously + headroom (the comment at `WalkaroundGPUPipeline.ts:127-131`).
- **Frame+scene BGL storage buffers:** scene group has up to 11 (`bindGroupDescriptors.ts:144-154`,
  incl. 5 TLAS buffers), frame group has 4 reservoir buffers + RC's hybrid-layers read buffer.
  The 16-buffer requirement comes from the shade pass binding 13 simultaneously
  (`WalkaroundGPUPipeline.ts:109-113`).

Lite-tier reductions (the cheapest wins, in order):
1. **No TLAS in lite** — force `bvhMode:'merged'` (the 5 TLAS scene-group buffers 6-10
   `bindGroupDescriptors.ts:150-154` become 1-vec placeholders). Merged BVH already exists as
   the default pre-TLAS path. This alone removes 5 storage buffers from the scene group.
2. **No RC / PPG / neural in lite** — already default-off; lite *forbids* enabling them.
3. **Lower internal resolution default** (lite presets bias to medium/low resolutionFactor).
4. Storage-texture count is **structural to shade** (can't drop the 4 shade writes without a
   shade fork) — so the lite *texture* floor stays at what shade needs. The lite win is on the
   **buffer** axis (drop TLAS) + resource memory, not the texture axis.

### 3.3 `HYBRID_LITE_LIMITS` (new — `WalkaroundGPUPipeline.ts`, beside `HYBRID_WEBGPU_REQUIRED_LIMITS:125`)

```ts
/** Reduced device limits for the merged-BVH, no-TLAS, no-RC/PPG lite path.
 *  Buffers drop because the TLAS scene buffers are absent; textures stay at
 *  shade's structural floor (shade writes 4 storage textures + headroom). */
export const HYBRID_LITE_LIMITS: Record<string, number> = {
  maxStorageBuffersPerShaderStage: 10,   // merged path peak (no 5 TLAS buffers)
  maxStorageTexturesPerShaderStage: 5,   // shade's 4 writes + 1 (cannot go lower without shade fork)
};
```
(Exact numbers to be confirmed by counting the live merged-path bindings on a GPU device — the
8-tex/16-buf full numbers were themselves "lift for headroom" choices, not hard minima; see Risk R3.)
Export it from the package index (`walkaround-hybrid/src/index.ts:14-17`) alongside the full limits.

### 3.4 Engine plumbing

- Add `HybridEngineOptions.tier?: 'full' | 'lite'` (default `'full'`). When `'lite'`:
  - Force `extensions['walkaround-hybrid'].bvhMode = 'merged'` (override; the merge happens at
    `createEngineScale.ts:33-47` `mergeWalkaroundTlasExtension` — make lite take precedence over
    the auto-TLAS extension so a needs-TLAS scene still runs merged on a weak adapter, with a
    `console.warn` that fidelity for instanced scenes is reduced).
  - Reject `rcEnabled`/`ppgEnabled`/`denoiser:'neural'` (throw at ctor with actionable message).
  - Bias `qualityTier` default to `'medium'`.
- `createEngine()` selects lite when `profile.hybridCapable === false && profile.hybridLiteCapable === true`
  (the device is requested with `HYBRID_LITE_LIMITS` instead of `HYBRID_WEBGPU_REQUIRED_LIMITS`
  at `constructWalkaround:136` `adapter.requestDevice()` — currently it requests **no** limits,
  so full hybrid silently relies on defaults; this also fixes that latent gap).

### 3.5 `AdapterProfile` fields (already in Deliverable 1 type)

`hybridCapable` (≥16/8) and `hybridLiteCapable` (≥`HYBRID_LITE_LIMITS`). `createEngine` routing:
`hybridCapable` → full; else `hybridLiteCapable` → lite; else throw Class-D error.

### 3.6 Tests

- **Unit:** `HYBRID_LITE_LIMITS` values < `HYBRID_WEBGPU_REQUIRED_LIMITS` on both axes.
- **Unit:** ctor with `tier:'lite', rcEnabled:true` throws; `tier:'lite'` forces `bvhMode:'merged'`
  even when the TLAS-extension merge ran.
- **Unit:** `probeAdapterProfile` lite-vs-full booleans flip exactly at the imported lite/full consts.
- **GPU-validation-gated:** lite pipeline actually compiles + binds on a device requested with
  `HYBRID_LITE_LIMITS` (the real confirmation of §3.3's numbers). Add a benchmark-runner mode
  that launches with reduced limits and asserts `initialize()` succeeds + a frame renders.

---

## §5.1 add-on — wire `FrameInput.quality.resolutionFactor` into hybrid

### 4.1 Feasibility verdict: **feasible per-frame, with a guarded debounce** (not a clean rejection)

`runHybridEngineFrame` passes `screenWidth/Height = deps.width/height` to `pipeline.renderFrame`
(`HybridEngineFrameOrchestrator.ts:297-298`); the compute kernels dispatch from those, and the
**composite pass upscales `resolvedTexture` to the swap-chain view via a fullscreen-triangle blit**
(`CompositePass.ts:44-67`) — so an internal render at `swap × factor` already upscales for free.
The cost: changing the internal size means `pipeline.resize()` (`WalkaroundGPUPipeline.ts:726-755`),
which **destroys + reallocates all FrameResources and resets the temporal accumulator** (~5-30 ms +
a 1-frame history reset). Doing that every frame on a scrubbing `resolutionFactor` would thrash.

**Plan: honor `resolutionFactor` but debounce the actual reallocation.**

### 4.2 Steps

1. **Decouple "swap size" from "internal size".** Today `_width/_height` are both. Introduce
   `_swapWidth/_swapHeight` (set by `setSize`, the canvas size) and `_internalWidth/_internalHeight`
   (= `swap × resolutionFactor`, what the pipeline renders at). `setSize` updates swap dims and
   recomputes internal dims from the last-seen factor.
2. **Read the factor per frame.** In `runHybridEngineFrame`, read `input.quality?.resolutionFactor ?? 1.0`,
   clamp to `(0,1]`, compute target internal `W' = max(1, round(swapW*f))`, `H'` likewise.
3. **Debounced apply.** If `(W',H')` differs from current internal dims by more than a threshold
   (e.g. ≥ 2 px on either axis) AND it has been stable for N frames (or simply: apply immediately
   but coalesce — pick one; recommend "apply when changed, but never more than once per ~250 ms"
   guarded by a timestamp like the existing `targetFrameIntervalMs` guard at `:236`), call
   `pipeline.resize(W',H')`. The composite still targets the full swap-chain view, so no swap-chain
   reconfigure is needed.
4. **Composite sampler must be `linear`.** `compositeSampler` is currently `nearest`
   (`createCommonFrameResources.ts:109-114`) → reduced-res upscale would be blocky. Change `magFilter`
   to `'linear'` (keep `minFilter` as-is or also linear). The composite texture is `rgba16float`
   sampled as `unfilterable-float` in the composite BGL (`bindGroupLayouts.ts:101-108` →
   `bglEntriesFor('composite')`) — **verify** the composite BGL/texture allows a filtering sampler;
   if the format is unfilterable, either (a) use a separate resolve-to-rgba8 step, or (b) keep
   nearest and document that reduced-res hybrid is sharp-but-aliased. (Flag as the one GPU-gated
   sub-decision here — see Risk R4.)
5. **Update the contract docs** that currently say hybrid hard-ignores per-frame size
   (`frame.ts:75-86`, `HybridEngine.renderFrame` JSDoc `:1098-1103`): clarify that **viewport**
   (canvas size) still requires `setSize`, but **`quality.resolutionFactor`** is now honored
   per-frame via internal-resolution scaling. Keep the viewport-needs-setSize rule (resizing the
   *canvas* still needs the host to call setSize; only the internal scale is per-frame).

### 4.3 Tests

- **Unit (orchestrator):** extend `__tests__/hybridEngineFrameOrchestrator.test.ts` — a fake
  `pipeline` records `resize` calls; assert `resolutionFactor:0.5` triggers a resize to
  `round(swap*0.5)` and that re-passing the same factor does NOT re-resize (debounce/idempotence).
- **Unit:** `resolutionFactor` omitted ⇒ no resize, internal == swap (regression guard).
- **GPU-validation-gated:** visual upscale quality (linear vs nearest) + temporal-reset cost.

---

## Implementation order / DAG

```
D1.type  AdapterProfile interface in @vitrum/core            (no deps)
   │
D3.limits  HYBRID_LITE_LIMITS in walkaround-hybrid + export   (no deps)
   │
   ├──> D1.fn  probeAdapterProfile in @vitrum/engine          (needs D1.type + D3.limits + existing pt-webgpu consts)
   │        │
   │        └──> D1.consume  createEngine wiring + Class-D throw + onAdapterProfile
   │
D2.preset  HybridEngineQualityPreset module + qualityTier opt (independent of D1; pure table)
   │   ├── needs: SpatialReservoirPass passCount arg
   │   ├── needs: GTAOPass gates(gtaoEnabled) + dispatch-scale
   │   ├── needs: ddgiUpdateDivisor in probeUpdateFrameParams
   │   └── reuses: adaptiveSamplingThresholds / denoiser / targetFrameIntervalMs (exist)
   │        │
   │        └──> D1.consume also applies recommendedRealtimeTier as default qualityTier
   │
D4.resfactor  internal-vs-swap size split + per-frame factor + linear composite sampler
   │   (independent; can land before or after D2, but D2 'resolution scale' row delegates to D4)
   │
D3.tier  HybridEngineOptions.tier + lite plumbing + createEngine lite routing
        (needs D1.fn for routing, D3.limits for device request)
```

Suggested landing sequence (each independently green-typecheck + test):
1. `D1.type` + `D3.limits` (pure additions, no behavior change).
2. `D1.fn` + `adapterProfile.test.ts` + benchmark-runner JSON unification.
3. `D4.resfactor` (size split + linear sampler + orchestrator test).
4. `D2.preset` (table + 3 net-new knobs + characterization tests); ultra == today's bytes.
5. `D3.tier` (lite plumbing) + `D1.consume` (createEngine routing + Class-D throw + recommended tier).

---

## Test strategy summary — unit-testable vs GPU-validation-gated

| Area | Unit-testable (headless / happy-dom) | GPU-validation-gated |
|---|---|---|
| AdapterProfile verdicts | All boolean/tier/recommendation logic via fake limit bags + threshold-coupling guard | Real `probeWebGPU()` adapter info (Playwright); benchmark-runner JSON emit |
| Quality presets | `resolveQualityPreset` table; ultra==baseline; override-beats-preset; UBO byte characterization (M2 offsets); `buildPassLayout` label characterization | Per-preset frame-time budgets; low-vs-ultra visual A/B |
| Hybrid lite | `HYBRID_LITE_LIMITS < full`; ctor rejects RC/PPG/neural; forces merged BVH; profile flip-points | Lite pipeline compiles + binds + renders on a reduced-limit device |
| resolutionFactor | Orchestrator resize-call recording; debounce idempotence; omit ⇒ no resize | Linear upscale quality; reallocation cost; composite-format filterability (Risk R4) |

Mechanical gates (CLAUDE.md testing protocol): `npm run typecheck` + `npm test` must stay green
each landing. New characterization tests go in the existing `__tests__/` dirs noted above.

---

## Risks

- **R1 — Magic-number drift.** The whole point is to reuse `HYBRID_WEBGPU_REQUIRED_LIMITS` /
  `PT_WEBGPU_*` / new `HYBRID_LITE_LIMITS` rather than re-typing thresholds. Mitigation: the
  threshold-coupling unit guard (D1.5) fails if anyone inlines a number. The benchmark-runner
  probe (`run-pt-webgpu-adapter-probe.mjs:45-63`) currently *re-types* `10/5/8` — fold it into
  the shared verdict so the tool can't drift from the library.
- **R2 — Pass-layout / timestamp-querySet desync.** Making spatial pass count and GTAO gating
  runtime means `buildPassLayout` (`WalkaroundGPUPipeline.ts:437,852`), `composePassLabels`
  (`passOrder.ts:117`), and the timestamp `querySet` size must all agree for a given config.
  Mitigation: thread the *same* `{gtaoEnabled, diSpatialPasses, giSpatialPasses}` config into
  both `buildPassLayout` and the runtime registry, and add a parity test
  (à la `bindGroupDescriptorParity.test.ts`).
- **R3 — `HYBRID_LITE_LIMITS` numbers unverified.** The full 16/8 were "headroom" choices, not
  hard minima. The lite 10/5 is a hypothesis until a device requested with those limits compiles
  the merged-path shade pipeline. GPU-gated; ship the constant with a doc note + the compile test.
- **R4 — Composite sampler filterability for resolutionFactor upscale.** `resolvedTexture` is
  `rgba16float` and the composite BGL binds it `unfilterable-float`. A `linear` sampler on an
  unfilterable format is invalid. Either flip the composite input to a filterable path or accept
  nearest (sharp/aliased) reduced-res. GPU-gated sub-decision; the orchestrator wiring is
  independent of it.
- **R5 — resolutionFactor temporal reset thrash.** Every internal resize resets the accumulator
  (`WalkaroundGPUPipeline.resize:745-754`). A host that ramps factor continuously would never
  converge. Mitigation: debounce + snap to a small set of factors (e.g. presets' 1.0/0.85/0.67/0.5)
  rather than honoring arbitrary continuous values.
- **R6 — `qualityTier:'ultra'` must be byte-identical to today.** Any preset-resolution path that
  subtly changes a default (e.g. `targetFrameIntervalMs`) is a silent regression for existing
  hosts. Mitigation: the ultra==baseline deep-equal test (D2.5) is the gate; presets are additive.
- **R7 — Class-D hard-fail behavior change.** Adding the §4.4 "throw on hybrid-incapable adapter"
  changes `createEngine` from "try and fail opaquely" to "throw early." That is the *intended* UX
  per roadmap §4.4/§10.4, but it is a behavior change — gate it behind the profile and give the
  actionable `prefer:'quality'` message so hosts can recover.

---

## Files touched (summary)

**New:**
- `packages/core/src/adapterProfile.ts` (type + re-export `WgpuAdapterKind`).
- `packages/engine/src/adapterProfile.ts` (`probeAdapterProfile`).
- `packages/engine/src/__tests__/adapterProfile.test.ts`.
- `packages/walkaround-hybrid/src/HybridEngineQualityPreset.ts` (`QUALITY_PRESETS`, `resolveQualityPreset`).
- `packages/walkaround-hybrid/src/__tests__/qualityPreset.test.ts`.

**Edited:**
- `packages/core/src/index.ts` (export AdapterProfile types).
- `packages/engine/src/index.ts` (export `probeAdapterProfile`).
- `packages/engine/src/createEngine.ts` (profile probe, Class-D throw, lite routing, recommended tier, `onAdapterProfile`).
- `packages/engine/src/createEngineScale.ts` (lite precedence over auto-TLAS in `mergeWalkaroundTlasExtension`).
- `packages/walkaround-hybrid/src/HybridEngineOptions.ts` (`qualityTier`, `tier`).
- `packages/walkaround-hybrid/src/HybridEngine.ts` (preset resolve in ctor; `_swap*`/`_internal*` size split; resolutionFactor read; lite ctor rejections; gtao/spatial-pass-count plumb).
- `packages/walkaround-hybrid/src/HybridEngineFrameOrchestrator.ts` (read `quality.resolutionFactor`, debounced `pipeline.resize`).
- `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` (`HYBRID_LITE_LIMITS`; `gtaoEnabled`/`diSpatialPasses` config; pass-layout threading).
- `packages/walkaround-hybrid/src/pipeline/Pass.ts` (`PassGateOptions.gtaoEnabled`).
- `packages/walkaround-hybrid/src/pipeline/passes/{GTAOPass,GTAOUpsamplePass,SpatialReservoirPass,SpatialGIReservoirPass}.ts` (gates / pass-count).
- `packages/walkaround-hybrid/src/pipeline/passes/passOrder.ts` (config-driven `composePassLabels`).
- `packages/walkaround-hybrid/src/pipeline/frameResources/createCommonFrameResources.ts` (composite sampler `linear`, pending R4).
- `packages/walkaround-hybrid/src/ddgi/probeUpdateFrameParams.ts` (`ddgiUpdateDivisor`).
- `packages/walkaround-hybrid/src/index.ts` (export `HYBRID_LITE_LIMITS`).
- `packages/core/src/frame.ts` + `HybridEngine.renderFrame` JSDoc (resolutionFactor contract update).
- `tools/benchmark-runner/run-pt-webgpu-adapter-probe.mjs` (emit unified `AdapterProfile` JSON).
```
