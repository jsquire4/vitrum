# Sweep Remediation Plan

**Source:** Full complexity sweep + dead code verification, 2026-05-10  
**Branch:** `main`  
**Agents:** 9 domain + 4 integration + 1 dead code (all returned, 14 total)  
**Total findings:** ~150 (grouped below by phase and package)

All items below are fixes. No item is deferred. The order within each phase is dependency-driven (items that unblock other items go first).

---

## Phase 1 — GPU Correctness Bugs

These are silent runtime failures or shader compilation errors. Fix before any other work.

---

### 1.1 RAYS_PER_PROBE mismatch

**Files:**
- `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateBlend.wgsl.ts:18` — `const RAYS_PER_PROBE: u32 = 96u`
- `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts:19` — `const RAYS_PER_PROBE: u32 = 96u`
- `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:37` — `const RAYS_PER_PROBE = 192`

**Bug:** GPU buffer is allocated for 192 rays per probe. WGSL blend/ray shaders iterate only 96. Half of every DDGI atlas blend is silently skipped.

**Fix:**
1. In `probeUpdatePass.ts`, export `RAYS_PER_PROBE`:
   ```ts
   export const RAYS_PER_PROBE = 192;
   ```
2. In both WGSL files, replace the hardcoded literal with a template injection:
   ```ts
   import { RAYS_PER_PROBE } from '../probeUpdatePass.js';
   // ...
   const PROBE_UPDATE_RAYS_WGSL = `
   const RAYS_PER_PROBE: u32 = ${RAYS_PER_PROBE}u;
   // ...
   `;
   ```
   This eliminates the possibility of the CPU/GPU constants diverging again.

---

### 1.2 `-> void` on WGSL compute entry points

**Files:**
- `packages/walkaround-hybrid/src/rc/wgsl/cascadeMerge.wgsl.ts:96`
- `packages/walkaround-hybrid/src/rc/wgsl/probeRayCast.wgsl.ts:383`

**Bug:** WGSL spec: compute entry points have no return type. `-> void` is invalid syntax and will fail GPU shader compilation.

**Fix:** Remove `-> void` from both entry-point signatures:
```wgsl
// Before:
fn cascadeMergeKernel(@builtin(global_invocation_id) globalId: vec3u) -> void {
// After:
fn cascadeMergeKernel(@builtin(global_invocation_id) globalId: vec3u) {
```
Same fix for `probeRayCastKernel` in probeRayCast.wgsl.ts:383.

---

### 1.3 Missing `INV_2PI` constant

**Files:**
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:217,240,272` — references `INV_2PI`
- `packages/pt-webgpu/src/wgsl/common.wgsl.ts` — does not define `INV_2PI`

**Bug:** Runtime WGSL compile failure for any scene with HDRI.

**Fix:** Add to `common.wgsl.ts` alongside `INV_PI`:
```wgsl
const INV_2PI: f32 = 0.15915494309190160;
```
Add a wgslContract test asserting `PT_WEBGPU_TRACE_WGSL` contains `INV_2PI`.

---

### 1.4 Normal encoding mismatch between spatialFilter and atrous

**Files:**
- `packages/shared-denoisers/src/wgsl/spatialFilter.wgsl.ts:143` — reads normals raw: `textureLoad(gbufferNormal, gid.xy, 0).xyz`
- `packages/shared-denoisers/src/wgsl/atrous.wgsl.ts:50` — decodes: `textureLoad(gbufferNormal, gid.xy, 0).xyz * 2.0 - 1.0`

**Bug:** If both shaders consume the same G-buffer, one produces wrong normals. The `rgba16float` format comment in both TypeScript descriptors is identical, implying they consume the same texture.

**Fix:**
1. Audit `WalkaroundGPUPipeline.ts` to determine which G-buffer texture is fed to `spatialFilter` vs `atrous`.
2. If same texture: pick one encoding convention (prefer unsigned-normalized 0..1 for `rgba16float`) and update both shaders to match.
3. If different textures: document the encoding contract in each shader's header comment with explicit G-buffer name.

---

### 1.5 WelfordVariance struct duplication with no sync enforcement

**Files:**
- `packages/shared-denoisers/src/wgsl/svgf.wgsl.ts:72-88` — local copy `@version 1`
- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts` — canonical

**Bug:** Comment-only enforcement. If common.wgsl.ts adds a field, svgf.wgsl.ts silently produces garbage variance.

**Fix:**
1. Extract the WelfordVariance WGSL struct block from `common.wgsl.ts` into a new exported constant:
   ```ts
   // packages/walkaround-hybrid/src/shaders/welfordVariance.wgsl.ts
   export const WELFORD_VARIANCE_STRUCT_WGSL = `
   struct WelfordVariance { mean: vec3f, M2: vec3f, count: u32 }
   fn welfordUpdate(acc: ptr<function, WelfordVariance>, x: vec3f) { ... }
   `;
   export const WELFORD_VARIANCE_VERSION = 1;
   ```
2. `common.wgsl.ts`: import and concatenate `WELFORD_VARIANCE_STRUCT_WGSL`.
3. `svgf.wgsl.ts`: remove the local copy; import `WELFORD_VARIANCE_STRUCT_WGSL` from `@vitrum/walkaround-hybrid` (or move the canonical to `@vitrum/shared-denoisers` if dependency direction is a concern).
4. Add a Vitest snapshot test in `shared-denoisers` asserting the exact WelfordVariance WGSL block, so any canonical change forces a test update.

---

## Phase 2 — Dead Code Removal

Remove all confirmed dead code before structural refactors to reduce the surface area being reorganized.

---

### 2.1 Delete entire `packages/babylon-bindings/`

Zero external imports anywhere in the repo. Only two stub functions that unconditionally throw.

**Fix:** Delete the entire `packages/babylon-bindings/` directory.

---

### 2.2 Delete `packages/pt-webgpu/src/wgsl/pathTraceSeed.wgsl.ts`

`PT_WEBGPU_SEED_WGSL` is defined but never imported — not by `index.ts`, not by any test.

**Fix:** Delete the file.

---

### 2.3 Remove deprecated `OCTAHEDRAL_WGSL` re-export from `pt-webgpu/src/index.ts:25`

```ts
// Remove this line:
export const OCTAHEDRAL_WGSL = OCTAHEDRAL_CORE_WGSL; // @deprecated
```

No file outside `pt-webgpu` imports `OCTAHEDRAL_WGSL` from `@vitrum/pt-webgpu`.

---

### 2.4 Delete `packages/pt-webgl/src/iblBaker.test-utils.ts`

`__skyEquirectCacheSizeForTests` — never imported by any test.

**Fix:** Delete the file.

---

### 2.5 Delete `packages/core/src/gpuDetection.test-utils.ts`

`resetGpuDetectionCacheForTests` — never imported by any test (`@vitrum/core` has no test directory).

**Fix:** Delete the file.

---

### 2.6 Delete `packages/three-bindings/src/spectral.ts`; prune index.ts export

`VITRUM_SPECTRAL_EXTENSION_KEY = 'vitrum.spectral'` — zero external consumers.

**Fix:**
1. Delete `packages/three-bindings/src/spectral.ts`.
2. Remove from `packages/three-bindings/src/index.ts:39`:
   ```ts
   export { VITRUM_SPECTRAL_EXTENSION_KEY } from './spectral.js'; // remove
   ```

---

### 2.7 Remove dead re-exports from `packages/shared-denoisers/src/index.ts`

The following symbols are used only internally within `shared-denoisers`; no external package imports them:

```ts
// Remove these from index.ts exports:
export * from './webGpuTextureCopy.js';           // WEBGPU_COPY_BYTES_PER_ROW_ALIGNMENT, alignedTextureCopyBytesPerRow
export { float16BitsToFloat32, float32ToFloat16Bits } from './halfFloat.js';
export { getSharedWebGPUDevice, disposeSharedWebGPUDevice } from './sharedWebGpuDevice.js';
export { SPATIAL_FILTER_WGSL } from './wgsl/spatialFilter.wgsl.js';
export { SpatialFilterBindGroupLayout } from './svgfBindings.js'; // confirm no external consumer
```

Keep the underlying modules intact as internal; only prune the public barrel.

---

### 2.8 Remove dead re-exports from `packages/walkaround-hybrid/src/index.ts:46`

```ts
// Remove these two lines:
export { ATROUS_WGSL, TEMPORAL_ACCUM_WGSL } from '@vitrum/shared-denoisers';
```

Both are already directly exported from `@vitrum/shared-denoisers`. No consumer imports them via `@vitrum/walkaround-hybrid`.

---

### 2.9 Remove dead re-export from `packages/pt-webgl/src/index.ts:67`

```ts
// Remove:
export { vitrumSceneToThree } from '@vitrum/three-bindings';
```

All actual consumers (`walkaround-hybrid/HybridEngine.ts:44`, `cornell-box/main.ts:13`, `two-engines-one-scene/main.ts:8`) import directly from `@vitrum/three-bindings`. None uses the pt-webgl relay.

---

### 2.10 Remove dead local constant in `svgfWebGPU.ts:255`

```ts
const TEMPORAL_VARIANCE_FRAME_THRESHOLD = 4; // never referenced
```

`SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT` (imported at line 24) already covers this value.

---

### 2.11 Remove stale comment in `gpuDetection.ts:94-97`

Lines 94-97 describe a removed function `getCachedGpuDetection`. The function is gone. Delete the comment block.

---

### 2.12 Delete `packages/shared-samplers/src/__tests__/jakobHanika.placeholder.test.ts`

Its 2 tests are a strict subset of `__tests__/jakobHanika.test.ts` (which is more thorough). The placeholder test adds no coverage.

**Fix:** Delete the file.

---

### 2.13 Move `resolve.wgsl.ts` and `sampleBudget.wgsl.ts` to `deferred/`

Both are authored, exported, and explicitly marked "DEFERRED — NOT wired into the dispatch pipeline." They increase cognitive surface area without function.

**Files:**
- `packages/walkaround-hybrid/src/shaders/resolve.wgsl.ts`
- `packages/walkaround-hybrid/src/shaders/sampleBudget.wgsl.ts`

**Fix:**
1. Create `packages/walkaround-hybrid/src/shaders/deferred/` directory.
2. Move both files there.
3. Update `index.ts` to not export them (they're already confirmed dead).
4. For `sampleBudget.wgsl.ts`: also remove the `[INLINE-COPY]` WelfordVariance declaration from the WGSL string — this will cause a redeclaration error when wired in. Replace with `${WELFORD_VARIANCE_STRUCT_WGSL}` injection (from Phase 1.5).

---

### 2.14 Remove void statements in `_staging/legacy-source/src/rendering/scene/lighting/renderers/sunPathTraced.tsx:150-152`

```tsx
// Remove both:
void SUN_LIGHT_DISTANCE;
void THREE;
```

Also remove the corresponding unused import for `SUN_LIGHT_DISTANCE` from `../../skyParams`.

---

## Phase 3 — WGSL Structural Cleanup

---

### 3.1 Move `WalkaroundUBO` struct to `COMMON_WGSL`

**Files:** `ris.wgsl.ts`, `temporal.wgsl.ts`, `spatial.wgsl.ts`, `shade.wgsl.ts` — all four declare the same 13-field `WalkaroundUBO` struct at `@group(2) @binding(0)`.

**Fix:**
1. Add the struct declaration once to `COMMON_WGSL` in `packages/walkaround-hybrid/src/shaders/common.wgsl.ts`.
2. Remove the four local `struct WalkaroundUBO { ... }` declarations from all four shader files.

---

### 3.2 Move `RESERVOIR_DI_STRIDE` + load/store helpers to `COMMON_WGSL`

**Files:** `ris.wgsl.ts:70-86`, `temporal.wgsl.ts:55-67`, `shade.wgsl.ts:194-198`, `spatial.wgsl.ts:59-67` — four implementations of the same 16-byte bitcast pack/unpack.

**Fix:**
1. Add `RESERVOIR_DI_STRIDE`, `loadReservoirDI` (read-write and read-only variants), `storeReservoirDI` to `COMMON_WGSL`.
2. Remove all four local definitions.

---

### 3.3 Move `PrimarySurface` struct to `COMMON_WGSL`

**Files:** `spatial.wgsl.ts:70-100`, `temporal.wgsl.ts:73-83` — identical struct.

**Fix:**
1. Add `struct PrimarySurface { ... }` once to `COMMON_WGSL`.
2. Remove both local declarations.
3. The `castPrimary` functions differ in the `camPos` parameter; leave them in their respective shaders or unify as `castPrimaryRay(camPos, ...)` in `COMMON_WGSL`.

---

### 3.4 Consolidate DDGI trilinear sampling duplication

**Files:**
- `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts:97-190` — `ddgiSampleFromBindings()`
- `packages/walkaround-hybrid/src/ddgi/wgsl/ddgiSampleWgsl.ts:43-137` — `ddgiSample()`

~90 lines of identical math. Also divergent fallback: `ddgiSampleWgsl.ts` returns `vec3f(0.05)` while `shade.wgsl.ts` returns `vec3f(0.0)`.

**Fix:**
1. Use the argument-passing form from `ddgiSampleWgsl.ts` as the canonical implementation.
2. Change `ddgiSampleWgsl.ts:133` to return `vec3f(0.0)` (conservative, matches shade.wgsl.ts).
3. In `shade.wgsl.ts`, replace the inline `ddgiSampleFromBindings()` body with a call to `ddgiSample(atlasIrr, atlasVis, gridUbo, pos, normal)`, passing the textures as function arguments rather than reading from `@group(3)` bindings directly.

---

### 3.5 Consolidate PPG kd-tree traversal

**Files:**
- `packages/walkaround-hybrid/src/ppg/wgsl/ppgSample.wgsl.ts:147-222` — `ppgKdFindCell()`
- `packages/walkaround-hybrid/src/ppg/wgsl/ppgUpdate.wgsl.ts:151-225` — `ppgUpdateKdFindCell()`

~75 lines of WGSL duplicated verbatim; differ only in function name, `cellCount` source, and axis helper name.

**Fix:**
1. Create `packages/walkaround-hybrid/src/ppg/wgsl/ppgCommon.wgsl.ts` with the parameterized traversal.
2. Import `PPG_COMMON_WGSL` in both shader files; replace the local implementations.

---

### 3.6 Replace duplicate atlas UV helpers with single parameterized function

**File:** `packages/shared-bvh/src/wgsl/octahedral.wgsl.ts:17-55`

`irradianceAtlasUv` (CELL=8u) and `visibilityAtlasUv` (CELL=16u) are identical except for the cell constant.

**Fix:**
```wgsl
fn probeAtlasUv(probeIdx: u32, octUv: vec2f, atlasW: u32, atlasH: u32, gridDims: vec3u, cell: u32) -> vec2f { ... }
fn irradianceAtlasUv(probeIdx: u32, octUv: vec2f, atlasW: u32, atlasH: u32, gridDims: vec3u) -> vec2f {
  return probeAtlasUv(probeIdx, octUv, atlasW, atlasH, gridDims, 8u);
}
fn visibilityAtlasUv(probeIdx: u32, octUv: vec2f, atlasW: u32, atlasH: u32, gridDims: vec3u) -> vec2f {
  return probeAtlasUv(probeIdx, octUv, atlasW, atlasH, gridDims, 16u);
}
```

---

### 3.7 Extract procedural surface textures from `COMMON_WGSL`

**File:** `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:771-872`

Scene-specific window-panel texture functions (`_barroqueMod`, `_catspawMod`, `_flemishMod`, etc.) live in the shared math header imported by every shader.

**Fix:**
1. Create `packages/walkaround-hybrid/src/shaders/surfaceTextures.wgsl.ts` with `SURFACE_TEXTURES_WGSL`.
2. Move lines 771-872 from `common.wgsl.ts` into the new file.
3. In `pipelineCompiler.ts`, inject `SURFACE_TEXTURES_WGSL` only into shader modules that call `surfaceTextureMod` (i.e., shade.wgsl and the BVH tinted-visibility path).

---

### 3.8 Fix dead `varianceK` field in `temporalAccum.wgsl.ts`

**File:** `packages/shared-denoisers/src/wgsl/temporalAccum.wgsl.ts:26-28`

`varianceK: f32` is declared in `AccumUBO` but never read in the shader. The module comment says "mean ± k·std_dev" but the shader uses AABB min/max.

**Fix (choose one):**
- Option A (simpler): Remove `varianceK` from the struct and replace with `_pad1: f32`. Update the host-side packer in `svgfWebGPU.ts` (and any test) to write 0.0 for the dropped slot. Add a comment: "AccumUBO uses AABB min/max clamp (not k·std_dev); varianceK removed."
- Option B (complete): Implement std-dev clamping using `varianceK` as the threshold multiplier.

---

### 3.9 Fix PPG `@group(2)` placeholder — annotate test

**File:** `packages/walkaround-hybrid/src/ppg/wgsl/ppgSample.wgsl.ts:85-87`, `__tests__/sprint11-ppg.test.ts:214`

`@group(2)` is a placeholder with a comment saying "TBD until Sprint 11 integration." The test pins this wrong value.

**Fix:** Add `// TODO(Sprint-11-integration): @group(2) is a placeholder; both WGSL and this assertion must be updated simultaneously when the group number is finalized.` to the test assertion at line 214.

---

### 3.10 Fix correlated 2D jitter in `ppgSampleDirection`

**File:** `packages/walkaround-hybrid/src/ppg/wgsl/ppgSample.wgsl.ts:289-290`

Single scalar `u1` drives both U and V jitter via different frequencies — correlated samples, not independent.

**Fix:**
```wgsl
// Before:
let jitterU = (col + fract(u1 * 4.0)) / 4.0;
let jitterV = (row + fract(u1 * 16.0)) / 4.0;
// After:
let jitterU = (col + u1) / 4.0;        // u1 for U
let jitterV = (row + u2) / 4.0;        // u2 for V (already declared)
```

---

### 3.11 Add inline comment for spatial Gaussian `18.0` literal

**File:** `packages/shared-denoisers/src/wgsl/hdrLuminanceBilateral.wgsl.ts:53`

**Fix:**
```wgsl
let ws = exp(-spatial / 18.0); // 2 * sigma_spatial^2, sigma_spatial=3 (5x5 kernel radius)
```

---

### 3.12 Add cross-reference comments for duplicated luminance coefficients

**Files:** `svgf.wgsl.ts:99`, `atrous.wgsl.ts:99`, `spatialFilter.wgsl.ts:175`

Constrained duplication (no cross-module WGSL imports in WebGPU). Each occurrence of `vec3f(0.2126, 0.7152, 0.0722)` should have:
```wgsl
// Rec. 709 luminance weights — canonical value; three copies exist (svgf, atrous, spatialFilter)
```

---

### 3.13 Inject B3-spline kernel from shared TypeScript constant

**Files:** `svgf.wgsl.ts:221-227`, `atrous.wgsl.ts:36-42` — identical 25-value kernel.

**Fix:** Define in a shared TS file:
```ts
export const ATROUS_KERNEL_VALUES = [
  0.0625, 0.125, 0.0625,
  0.125,  0.25,  0.125,
  0.0625, 0.125, 0.0625,
  // ... (all 25 values)
] as const;
export const ATROUS_KERNEL_WGSL = `const KERNEL: array<f32, 25> = array<f32, 25>(${ATROUS_KERNEL_VALUES.join(',')});`;
```
Inject into both WGSL strings via template literal. Eliminates the acknowledged "identical" copy.

---

## Phase 4 — TypeScript Structural Cleanup

### Package: @vitrum/core

---

#### 4.1 `scene.ts:33-34` — Remove orphaned `// ── Material ──` section header

The `// ── Material ──` header at lines 33-34 precedes the spectral types section, not `Material` interface. Remove lines 33-34; leave the correct header at lines 157-158.

---

#### 4.2 `scene.ts:161-168` — Resolve Material mutability contract gap

`Material` comment says "hosts may mutate fields between frames" but `readonly material: Material` slots on `MeshPrimitive` etc. provide no snapshot-trigger protocol.

**Fix (choose one):**
- Option A: Add `isDirty?: boolean` flag to `Material` so backends can detect mutations without deep comparison. Document the protocol: host sets `isDirty = true` after mutation; engine clears it after processing.
- Option B: Remove the mutability note. Require hosts to call `updatePrimitive` for material changes. This is consistent with the incremental-update contract already on `Engine`.

---

#### 4.3 `engine.ts:240-258` — Move MNEE knobs to `causticOptions` sub-object

```ts
// Before (flat):
mneeMaxIterations?: number;  // Ignored when causticStrategy !== 'manifold-nee'
mneeMaxChainLength?: number;

// After (nested):
causticOptions?: {
  mneeMaxIterations?: number;
  mneeMaxChainLength?: number;
  [key: string]: unknown;
};
```

---

#### 4.4 `frame.ts:98-99` — Move WebGPU-specific swap chain fields to extensions

`swapChainView?: GPUTextureView` and `swapChainFormat?: GPUTextureFormat` in the backend-agnostic `FrameInput` couple the core contract to one backend.

**Fix:** Either:
- Move to `extensions?: Record<string, unknown>` with a typed helper in `pt-webgpu` that extracts them.
- Or type as opaque `BackendTexture = unknown` aliases like the existing `BackendTexture`.

---

#### 4.5 `gpuDetection.ts` / `wgpuSupport.ts` — Add removal milestones to `@deprecated isHardwareGpu`

Both files have an identical `@deprecated` field with no removal date. Add:
```ts
/** @deprecated Prefer {@link adapterKind}. Remove in Phase 7 / Sprint 1. */
```

---

#### 4.6 `wgpuSupport.ts:100-107` — Inline `isWebGPUSupported()` into `probeWebGPU()`

Private single-use function with redundant null checks. Replace:
```ts
// Remove isWebGPUSupported() declaration entirely.
// In probeWebGPU(), replace the call with:
if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
  return { supported: false, ... };
}
```

---

### Package: @vitrum/three-bindings

---

#### 4.7 Extract userData key constants to `userDataKeys.ts`

**Files:** `packages/three-bindings/src/vitrumSceneToThree.ts` and `packages/three-bindings/src/material.ts`

Eight magic string keys (`'vitrumDispersionAbbeNumber'`, `'vitrumScatteringCoefficient'`, etc.) appear in both files with no shared definition. A typo silently breaks the round-trip.

**Fix:**
1. Create `packages/three-bindings/src/userDataKeys.ts`:
   ```ts
   export const VITRUM_USER_DATA_KEYS = {
     DISPERSION_ABBE:    'vitrumDispersionAbbeNumber',
     SCATTERING_COEFF:   'vitrumScatteringCoefficient',
     SCATTERING_RGB:     'vitrumScatteringCoefficientRGB',
     SCATTERING_ANISO:   'vitrumScatteringAnisotropy',
     SPECTRAL_ATTEN:     'vitrumSpectralAttenuation',
     THIN_FILM_STACK:    'vitrumThinFilmStack',
     FRONT_LAYER:        'vitrumFrontLayer',
     BACK_LAYER:         'vitrumBackLayer',
   } as const;
   ```
2. Replace bare string literals in both files with `VITRUM_USER_DATA_KEYS.*`.
3. Export `VITRUM_USER_DATA_KEYS` from `packages/three-bindings/src/index.ts`.

---

#### 4.8 `material.ts:61-65` — IOR guard: replace `1.5` hardcoded literal

```ts
// Before:
if (p.ior !== 1.5) base.ior = p.ior;
// After:
const DEFAULT_IOR = 1.5; // THREE.MeshPhysicalMaterial default
if (p.ior !== DEFAULT_IOR) base.ior = p.ior;
```

Or: always emit `ior` for physical materials regardless of value.

---

#### 4.9 `material.ts:120-139` — Add deprecation warning to Float32Array back-compat path

```ts
} else if (rawSpectral instanceof Float32Array && rawSpectral.length >= 3) {
  console.warn('[vitrum/three-bindings] SpectralCurve as Float32Array is deprecated. ' +
    'Use { wavelengthStart, wavelengthEnd, values: Float32Array } instead.');
  base.spectralAttenuation = { wavelengthStart: 380, wavelengthEnd: 780, values: rawSpectral };
}
```

---

#### 4.10 `index.ts:20-37` — Extract luminance helper; move emissiveMeshAreaEmitter

1. Create `packages/three-bindings/src/math.ts`:
   ```ts
   export function luminance(r: number, g: number, b: number, intensity = 1): number {
     return (0.2126 * r + 0.7152 * g + 0.0722 * b) * intensity;
   }
   ```
2. Import `luminance` in both `index.ts` and `vitrumSceneToThree.ts`; remove the inline duplicate from each.
3. Move `emissiveMeshAreaEmitter` from `index.ts` to `mesh.ts` (it belongs alongside mesh conversion).

---

#### 4.11 `index.ts:99-108` — Extract emissive stripping logic to helper

```ts
function stripEmissive(prim: MeshPrimitive): MeshPrimitive {
  return { ...prim, material: { ...prim.material, emissive: [0,0,0], emissiveIntensity: 0 } };
}
```

Call `stripEmissive(prim)` at the point where the primitive is promoted to a mesh-area emitter.

---

#### 4.12 `vitrumSceneToThree.ts:46-123` — Extract two helpers from `vitrumMaterialToThree`

```ts
function applyTextureMaps(mat: MeshPhysicalMaterial, m: Material): void { ... }
function stampVitrumUserData(mat: MeshPhysicalMaterial, m: Material): void { ... }
```

Leave `vitrumMaterialToThree` as a thin orchestrator calling both.

---

#### 4.13 `vitrumSceneToThree.ts:140-145` — Remove unnecessary `Array.from`

```ts
// Before:
if (p.transform) m.fromArray(Array.from(p.transform));
// After:
if (p.transform) m.fromArray(p.transform); // Float32Array satisfies ArrayLike<number>
```

---

#### 4.14 Add disc-area approximation warning in `vitrumSceneToThree.ts`

When `discAreaEmitterToRectThree` is called, emit:
```ts
console.warn('[vitrum/three-bindings] DiscAreaEmitter converted to RectAreaLight approximation ' +
  '(area-preserving rectangle). Round-trip will produce RectAreaEmitter, not DiscAreaEmitter.');
```

---

#### 4.15 Add TODO in `environment.ts` for ProceduralSkyEnvironment

In `resolveEnvironment()`, add:
```ts
// TODO: ProceduralSkyEnvironment not handled here.
// A THREE.Sky object with uniforms {turbidity, mieCoefficient, mieDirectionalG, rayleigh}
// would feed this. See core/scene.ts#ProceduralSkyEnvironment for the expected fields.
```

---

### Package: @vitrum/shared-bvh

---

#### 4.16 Add `uvAttribute` to `SceneBVHCommonResult`

**File:** `packages/shared-bvh/src/bvhCommon.ts`

`restir/bvhCompute.ts:208,241` accesses `shared.bvh.geometry.attributes['uv']` — a private three-mesh-bvh field.

**Fix:**
1. Add `uvAttribute?: THREE.BufferAttribute` to `SceneBVHCommonResult`.
2. In `buildSceneBVH`, populate: `uvAttribute: merged.attributes['uv'] as THREE.BufferAttribute | undefined`.
3. Update `restir/bvhCompute.ts` to read `shared.uvAttribute` instead of `shared.bvh.geometry.attributes['uv']`.

---

#### 4.17 Rename `buildSceneBVH` exports to avoid collision

Both `rc/bvhCompute.ts` and `restir/bvhCompute.ts` export `buildSceneBVH` with incompatible signatures and return types. Callers can silently import the wrong one.

**Fix:**
- `rc/bvhCompute.ts`: rename export to `buildRCSceneBVH`.
- `restir/bvhCompute.ts`: rename export to `buildReSTIRSceneBVH`.
- Update all call sites.

---

#### 4.18 Hoist scratch allocations outside `traverseVisible` loop

**File:** `packages/shared-bvh/src/bvhCommon.ts:323-342`

```ts
// Before (inside traverseVisible callback):
mesh.matrixWorld.decompose(new THREE.Vector3(), new THREE.Quaternion(), worldScale);

// After (hoist before the traversal call):
const _decompPos = new THREE.Vector3();
const _decompQuat = new THREE.Quaternion();
// ...
mesh.matrixWorld.decompose(_decompPos, _decompQuat, worldScale);
```

---

#### 4.19 Document dual-filter traversal invariant in `sceneBvh.ts`

**File:** `packages/shared-bvh/src/sceneBvh.ts:85-117`

The dirty-check traversal uses `DDGI_MESH_FILTER` and the build traversal uses the same. Document that both MUST use the same filter to avoid hash/build mismatch:
```ts
// IMPORTANT: both traversals (version hash and actual build) MUST use the same
// filter predicate. Divergence causes the dirty check to invalidate on a different
// mesh set than was actually rebuilt. Filter: DDGI_MESH_FILTER
```

---

#### 4.20 Add additive hash collision comment in `sceneBvh.ts:92-98`

```ts
// Note: additive hash (version + id) has collision risk if geometry versions shift
// as mesh IDs shift in opposite direction. False-negatives (missed rebuild) possible.
// Upgrade to XOR hash if this proves problematic in practice.
version += posAttr.version ?? 0;
version += m.id;
```

---

### Package: @vitrum/shared-samplers

---

#### 4.21 Fix inverted `@deprecated` in `jakobHanika.ts`

**File:** `packages/shared-samplers/src/jakobHanika.ts`

Currently `rgbToSpectralCoefficients` is `@deprecated` pointing at the `Approx` variant, which is backwards — the `Approx` name is the placeholder.

**Fix:**
1. Remove `@deprecated` from `rgbToSpectralCoefficients`.
2. Mark `rgbToApproxSpectralCoefficients` as `@internal` with a JSDoc: "Placeholder implementation. Will be replaced by precomputed table in Sprint 12. Use `rgbToSpectralCoefficients` as the stable public name."

---

#### 4.22 Add derivation trace for Vandermonde matrix entries

**File:** `packages/shared-samplers/src/jakobHanika.ts:159-179`

```ts
// Each v_ij = 1 / ((LAMBDA_G - LAMBDA_B) * (LAMBDA_R - LAMBDA_B)) etc.
// Derived from 3-node Vandermonde inverse for nodes [450, 550, 700] nm.
const v11 = 7.666_666_666_666_666_7e-2; // = 1/(550-450)/(700-450)
```

Add this comment pattern for each matrix entry.

---

#### 4.23 Type CIE tables as `Readonly<Float32Array>`

**File:** `packages/shared-samplers/src/cieCmf.ts`

```ts
// Before:
export const CIE_Y_TABLE = new Float32Array([...]);
// After:
export const CIE_Y_TABLE: Readonly<Float32Array> = new Float32Array([...]);
```

Apply to all four tables: `CIE_X_TABLE`, `CIE_Y_TABLE`, `CIE_Z_TABLE`, `CIE_D65_TABLE`.

---

#### 4.24 Combine sequential dependent IIFEs in `wavelengthSampling.ts`

**File:** `packages/shared-samplers/src/wavelengthSampling.ts:46-73`

`Y_CDF` depends on `Y_INTEGRAL`. Two separate top-level `const` IIFEs are reordering-fragile.

**Fix:**
```ts
const { integral: Y_INTEGRAL, cdf: Y_CDF } = (() => {
  // ... compute integral first, then cdf using it
  return { integral, cdf };
})();
```

---

### Package: @vitrum/shared-denoisers

---

#### 4.25 Extract generic texture upload helper in `svgfWebGPU.ts`

**File:** `packages/shared-denoisers/src/svgfWebGPU.ts:72-223`

Six near-identical functions sharing: `alignedTextureCopyBytesPerRow` + allocate typed array + loop + `device.queue.writeTexture`.

**Fix:**
1. Define named bpp constants:
   ```ts
   const RGBA32F_BPP = 16 as const;
   const RGBA16F_BPP = 8 as const;
   const RG32F_BPP = 8 as const;
   ```
2. Extract:
   ```ts
   function uploadTexture2D<T extends TypedArray>(
     device: GPUDevice, texture: GPUTexture,
     w: number, h: number, bpp: number,
     fill: (buf: T, w: number, h: number) => void,
     TypedArrayCtor: new(len: number) => T,
   ): void { ... }
   ```
3. Replace six implementations with calls to `uploadTexture2D`.

---

#### 4.26 Batch all SVGF compute passes into single encoder

**File:** `packages/shared-denoisers/src/svgfWebGPU.ts:339,501,527-545`

Currently: 1 `queue.submit()` for variance pass + up to 12 `queue.submit()` inside the à-trous loop = up to 13 separate synchronization points per `runSvgfWebGPU` call.

**Fix:**
1. Create one `GPUCommandEncoder` before the variance pass.
2. Add the variance pass and all à-trous iterations to the same encoder.
3. Call `device.queue.submit([encoder.finish()])` once at the end.
4. The `mapAsync` readback step must remain separate (after submit).

---

#### 4.27 Pre-create A→B and B→A bind groups for à-trous loop

**File:** `packages/shared-denoisers/src/svgfWebGPU.ts:527`

Currently creates a new bind group per à-trous iteration (up to 12). Only 2 are needed.

**Fix:**
```ts
const atrousBindA = device.createBindGroup({ /* readTex=accumA, writeTex=accumB */ });
const atrousBindB = device.createBindGroup({ /* readTex=accumB, writeTex=accumA */ });
for (let iter = 0; iter < atrousIterations; iter++) {
  pass.setBindGroup(0, iter % 2 === 0 ? atrousBindA : atrousBindB);
  // ...
}
```

---

#### 4.28 Remove `RENDER_ATTACHMENT` from G-buffer texture usage flags

**File:** `packages/shared-denoisers/src/svgfWebGPU.ts:380-382`

G-buffer textures are write-via-`writeTexture` and read-via-compute-shader; they are never used as render attachments.

```ts
// Before:
const texRgba32Usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
  GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT;
// After:
const texRgba32Usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
```

Review whether `COPY_SRC` is actually needed; remove if not.

---

#### 4.29 Deduplicate `bytesPerRow` calculation in `hdrLuminanceBilateralWebGPU.ts`

**File:** `packages/shared-denoisers/src/hdrLuminanceBilateralWebGPU.ts:97-98,143`

```ts
// Compute once:
const bpp = 16;
const bytesPerRow = alignedTextureCopyBytesPerRow(w, bpp);
// Remove duplicate at line 143 — use bytesPerRow in both upload and readback
```

---

#### 4.30 Name the UBO size constant in `hdrLuminanceBilateralWebGPU.ts:119`

```ts
const HDR_BILATERAL_UBO_SIZE_BYTES = 16; // BilateralParams: 4 × f32
// ...
size: HDR_BILATERAL_UBO_SIZE_BYTES,
```

---

#### 4.31 Extend `_OrtModule` with `_OrtSession` type; remove inline cast

**File:** `packages/shared-denoisers/src/oidnBridge.ts:190`

```ts
interface _OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>>;
}
// Line 190:
const result = (session as _OrtSession).run(feeds);
```

---

#### 4.32 Simplify redundant output key fallback in `oidnBridge.ts:192-193`

```ts
// Before:
const outputPrimaryKey = tn.output ?? 'output';
const outputTensor = results[outputPrimaryKey] ?? results['output'] ?? results['color'];
// After:
const outputTensor = results[tn.output ?? 'output'] ?? results['color'];
```

---

### Package: @vitrum/walkaround-hybrid

---

#### 4.33 `HybridEngine.ts` — Extract `packDDGIGridParams()` helper

**Files:** `packages/walkaround-hybrid/src/HybridEngine.ts:461-486` and `packages/walkaround-hybrid/src/pipeline/resourceManager.ts`

DDGI UBO packing inlined in `renderFrame()` — a parallel reimplementation of `buildDDGIPlaceholderUBO`.

**Fix:**
1. In `resourceManager.ts`, extract:
   ```ts
   export function packDDGIGridParams(p: ProbeGridParams): Float32Array { ... }
   ```
2. In `HybridEngine.ts:renderFrame()`, replace the inline packing with `packDDGIGridParams(probeGrid)`.

---

#### 4.34 `HybridEngine.ts:447-455` — Introduce `DDGIDeviceHandle` interface

The fake Three.js renderer adapter synthesized inline for `DDGI.updateFrame()` couples HybridEngine to DDGI's Three.js-shaped API.

**Fix:**
1. Define in `ddgi/types.ts`:
   ```ts
   export interface DDGIDeviceHandle {
     device: GPUDevice;
   }
   ```
2. Update `DDGI.updateFrame()` to accept `DDGIDeviceHandle` instead of the Three.js renderer shape.
3. In `HybridEngine.ts`, replace the adapter with: `const handle: DDGIDeviceHandle = { device: this._device };`.

---

#### 4.35 `HybridEngine.ts:699-816` — Flatten nested async arrow function

`_initPipeline` defines then immediately calls `buildBVHWhenReady` as an inner async function. Flatten its body directly into `_initPipeline`.

---

#### 4.36 `HybridEngine.ts:291-309` — Reconcile `supportsMotionBlur` flag vs actual allocation

`supportsMotionBlur: false` but `resourceManager.ts:492-506` allocates `motionVectorTexture` unconditionally.

**Fix:** Either set `supportsMotionBlur: true` (since the texture is allocated) or gate the allocation behind an options flag, matching the PPG buffer pattern.

---

#### 4.37 `WalkaroundGPUPipeline.ts:374-764` — Extract `renderFrame()` sub-dispatchers

The 390-line `renderFrame()` method mixes 9 logical phases. Extract at minimum:

```ts
private _dispatchSVGF(encoder: GPUCommandEncoder, d: GPUDevice, frame: number): void { ... }
private _dispatchAtrous(encoder: GPUCommandEncoder, d: GPUDevice): void { ... }
private _dispatchPPG(encoder: GPUCommandEncoder, d: GPUDevice): void { ... }
```

---

#### 4.38 `WalkaroundGPUPipeline.ts:477-488` — Allocate UBOs eagerly in `initialize()`

Seven UBO buffers are lazily allocated inside `renderFrame()`:
`ppgShadeMetaUboRef`, `welfordUboRef`, `svgfVarianceUboRef`, `svgfAtrousUboRef`, `ppgUpdateUboRef`, `atrousUboRef`, `accumUboRef`.

**Fix:** Move all allocations to `initialize()`, gated on `denoiserMode` and `ppgEnabled` where appropriate. This ensures GPU resources are ready before the first frame and makes `dispose()` unconditional.

---

#### 4.39 `WalkaroundGPUPipeline.ts:576-673` — Use builder pattern for SVGF bind groups

The SVGF path creates bind groups inline inside a for-loop (bypassing `bindGroupBuilders.ts`), inconsistent with the legacy atrous path.

**Fix:** Add to `bindGroupBuilders.ts`:
- `buildWelfordBindGroup(d, layout, ...)`
- `buildSVGFVarianceBindGroup(d, layout, ...)`
- `buildSVGFAtrousBindGroup(d, layout, readTex, writeTex, ...)`

Replace inline creation in `WalkaroundGPUPipeline.ts` with calls to these builders.

---

#### 4.40 `WalkaroundGPUPipeline.ts` — Standardize private field naming convention

Mixed convention: some fields use underscore prefix (`_ddgiIrrTex`), others do not (`risPipeline`).

**Fix:** Adopt underscore prefix consistently for all private fields in `WalkaroundGPUPipeline`, matching the `HybridEngine` convention.

---

#### 4.41 `restir/bvhCompute.ts` — Extract packing helpers and emitter list builder

The 709-line file mixes 6 concerns.

**Fix:**
1. Create `restir/packingHelpers.ts` with `applyBeerLambert`, `packUVIntoPositionW`, `packBVHIndexW`, `packBVHBeerColors`.
2. Create `restir/emitterList.ts` with `buildEmitterList` and its luminance helper.
3. Leave `buildReSTIRSceneBVH` (renamed per 4.17) as a thin orchestrator in `bvhCompute.ts`.

---

#### 4.42 `restir/bvhCompute.ts:396-444` — Extract `resolveTriColor()` helper

`packBVHBeerColors` and `packBVHIndexW` both implement identical material color resolution logic.

```ts
function resolveTriColor(mat: THREE.Material, applyBeer: boolean): THREE.Color { ... }
```

Call from both packers.

---

#### 4.43 `rc/bvhCompute.ts:65-127` — Collapse `packCascadeMaterials` to single branch

Three near-identical branches (MeshPhysicalMaterial, MeshStandardMaterial, fallback) duplicate 12 of 16 float assignments.

**Fix:** Single branch reading from MeshStandard cast with a Physical-only override block:
```ts
const std = mat as THREE.MeshStandardMaterial;
// write common fields: color, roughness, metalness, emissive ...
if (mat instanceof THREE.MeshPhysicalMaterial) {
  // override Physical-only: transmission, ior, attenuationColor, thickness
}
```

---

#### 4.44 `rc/cascadeDispatch.ts:519-527` — Document or remove module-level singleton

```ts
const _sharedDispatcher = new RCDispatcher(); // single-canvas-scoped by design
```

Either add the comment documenting single-canvas intent, or remove the singleton and require callers to instantiate `RCDispatcher` directly (the class is already exported).

---

#### 4.45 `rc/cascadeDispatch.ts:431` — Expose or document `envIntensity`

`buildCascadeUniformDataInto(..., 1.0, ...)` — `envIntensity` is hardcoded to `1.0` with no corresponding field in `RCDispatchOpts`.

**Fix:** Either add `envIntensity?: number` to `RCDispatchOpts` (defaulting to `1.0`), or add a comment:
```ts
// envIntensity fixed at 1.0: tone mapping is applied per-material downstream;
// environment-level scaling is intentionally not exposed at the RC dispatch level.
```

---

#### 4.46 `rc/giReceiver.ts:41-46` — Replace Symbol brand tag with userData string

The `[GI_TAG]?: boolean` Symbol-keyed brand requires `as unknown as GIMaterial` at every access site.

**Fix:**
```ts
// Instead of Symbol brand:
interface GIMaterial { userData: { __vitrum_gi_wrapped?: true } }
// Access:
(nm as GIMaterial).userData.__vitrum_gi_wrapped = true;
```

---

#### 4.47 `neural/unetArchitecture.ts` — Provide `buildUNetSpec(width, height)` factory

`WALKAROUND_DENOISER_UNET_SPEC` has baked 1080p dispatch parameters. A non-1080p render silently dispatches the wrong workgroup count.

**Fix:**
```ts
export function buildUNetSpec(width: number, height: number): UNetSpec {
  // Derive dispatchX/Y/Z from width/height
  return { ...WALKAROUND_DENOISER_UNET_SPEC, /* override dispatch dims */ };
}
```
Mark `WALKAROUND_DENOISER_UNET_SPEC` as `@internal` or add a JSDoc warning against direct use for non-1080p targets.

---

#### 4.48 `neural/InferenceGraph.ts:200-213` — Document intermediate buffer contract

Add prominent JSDoc to `initialize()` and `run()`:
```ts
/**
 * IMPORTANT: All intermediate tensor names must be included in the `outputs` map
 * passed to `run()`. Tensors registered with elementCount=0 / buffer=null during
 * initialize() will throw at run() time if not present in `outputs`.
 */
```

---

#### 4.49 `neural/InferenceGraph.ts:304-312` — Document bind group cache stability assumption

Add comment:
```ts
// Bind groups are cached permanently after first creation.
// ASSUMPTION: buffer identity in `outputs` and `inputs` maps is stable across run() calls.
// If ping-pong buffers swap between frames, do NOT cache — pass a fresh outputs map.
const bg = this._cachedBindGroups[i] ?? device.createBindGroup({ ... });
```

---

#### 4.50 `buildPpgKdTree.ts:68-113` — Extract inner `build()` to top level

The recursive `build` function is defined as a closure inside `buildPpgKdTreeGpuBytes`, capturing outer-scope arrays.

**Fix:**
```ts
function _buildKdNode(
  sub: number[], positions: Float32Array, nodes: PpgKdNode[]
): number { ... }

export function buildPpgKdTreeGpuBytes(...): Uint8Array {
  const nodes: PpgKdNode[] = [];
  _buildKdNode(allIndices, positions, nodes);
  // ...
}
```

---

#### 4.51 Replace PPG string-splice injection with explicit extension hooks in `shade.wgsl.ts`

**Files:**
- `packages/walkaround-hybrid/src/ppg/wgsl/shadePpgTrain.wgsl.ts` — uses verbatim literal anchor `'@group(3) @binding(3) var<uniform> ddgiGrid: DDGIGridUBO;'` for splice point
- `packages/walkaround-hybrid/src/ppg/wgsl/shadePpgGuide.wgsl.ts` — uses multiline `COMBINED_ANCHOR` string as splice point

**Problem:** Any refactoring of `shade.wgsl.ts` that touches the anchor strings (reformatting, reordering bindings, changing the `combined=` expression) silently breaks PPG injection at runtime — no compile error, just wrong output.

**Fix:**
1. In `shade.wgsl.ts`, replace the two implicit anchor sites with named comment delimiters:
   ```wgsl
   // @@PPG_BINDINGS_INSERT@@
   @group(3) @binding(3) var<uniform> ddgiGrid: DDGIGridUBO;

   // ... (in the bounce loop, before the combined= expression)
   // @@PPG_GUIDE_DECLS_INSERT@@

   let combined = Lo_emit + Lo_direct + Lo_sunCaustic
                + Lo_skyAperture * 0.08
                + Lo_ddgi * DDGI_DIFFUSE_BLEND;
   // @@PPG_COMBINED_INSERT@@
   ```
2. In `shadePpgTrain.wgsl.ts` and `shadePpgGuide.wgsl.ts`, replace the verbatim string anchors with delimiter-based searches:
   ```ts
   const BINDINGS_MARKER = '// @@PPG_BINDINGS_INSERT@@';
   if (!base.includes(BINDINGS_MARKER)) throw new Error('shade.wgsl.ts: PPG bindings marker missing');
   return base.replace(BINDINGS_MARKER, BINDINGS_MARKER + '\n' + ppgBindings);
   ```
3. The marker strings are intentionally ugly and won't appear in any natural refactoring — a refactor that deletes them gets an explicit runtime `Error` instead of silent wrong output.

---

### Package: @vitrum/pt-webgl

---

#### 4.52 `pt-webgl/src/index.ts` — Extract `PTEngineWebGL2` class to its own file

The 778-line `index.ts` is simultaneously: class definition, factory function, re-export barrel, type definitions, and scheduling constants.

**Fix:**
1. Create `packages/pt-webgl/src/ptEngineWebGL2.ts` containing:
   - `PTEngineWebGL2` class
   - `createPTEngine_WebGL2` factory
   - All local types: `PTEngineWebGL2Options`, `Telemetry`, `DeviceLimits`, `RenderSizePlan`, `SchedulerOptions`, `StateSlot`, `PTEngineWebGL2Init`
   - Scheduling constants: `BYTES_PER_RGBA16F_PIXEL`, `ESTIMATED_RENDER_TARGET_COUNT`, `DEFAULT_RENDER_TARGET_OVERHEAD_BYTES`
2. Simplify `index.ts` to a barrel that imports from `ptEngineWebGL2.ts` and re-exports all leaf modules.

---

#### 4.53 `index.ts:719-777` — Extract GPU init logic from factory

```ts
function buildPTEngineGpu(renderer: WebGLRenderer, opts: PTEngineWebGL2Options): PTEngineWebGL2Init {
  // instanceof WebGL2RenderingContext check
  // glContext.getParameter(...)
  // new WebGLPathTracer(...)
}
```

Factory becomes a thin coordinator.

---

#### 4.54 `index.ts:131-133` — Document render-target memory constants

```ts
const BYTES_PER_RGBA16F_PIXEL = 8;         // RGBA16F = 4 channels × 2 bytes
const ESTIMATED_RENDER_TARGET_COUNT = 4;   // primary accum, depth, normal, motion vector
const DEFAULT_RENDER_TARGET_OVERHEAD_BYTES = 64 * 1024 * 1024; // driver metadata, mip alignment
```

Move to `constants.ts` if on the public tuning surface.

---

#### 4.55 `index.ts:605-621` — Extract adaptive tile dispatch helper

```ts
#updateAdaptiveTileFactors(sppBefore: number, tilesX: number, tilesY: number, w: number, h: number): void {
  // Remove redundant #tileVariancePass != null check (constructor invariant guarantees non-null)
}
```

---

#### 4.56 `index.ts:585,639` — Extract tiling threshold constant

```ts
const MIN_RESOLUTION_FOR_TILING = 640 * 360; // below SD, tiling overhead dominates
// Use in both places (lines 585 and 639)
```

---

#### 4.57 `index.ts:266-278` — Make `disposeObject3DTree` a private static method

```ts
// Move from module scope to inside PTEngineWebGL2:
static #disposeTree(obj: Object3D): void { ... }
```

---

#### 4.58 `index.ts:515-517` — Create typed fork wrapper to localize unknown casts

```ts
interface WebGLPathTracerCompat {
  setScene(scene: THREE.Scene, camera: THREE.Camera): void;
  setCamera(camera: THREE.Camera): void;
  // ... only the methods vitrum calls
}
const pathTracer = this.#pathTracer as unknown as WebGLPathTracerCompat;
```

Replace scattered `as unknown as Parameters<...>` casts throughout with calls on `pathTracer`.

---

#### 4.59 `iblBaker.ts:46` — Document module-global cache limitation

```ts
// WARNING: Module-level singleton — shared across all WebGLRenderer instances in process.
// A baked sky texture is bound to the GL context that performed the bake.
// Safe only with a single renderer per process. In test environments, ensure renderers
// are not created concurrently.
const cache = new Map<string, CachedBake>();
```

---

#### 4.60 `forkUniformBridge.ts:52` — Remove dead `scene` parameter

```ts
// Before:
function driveForkMaterialUniforms(pathTracer: ..., scene: VitrumScene, ...): void {
  void scene; // intentional discard

// After:
function driveForkMaterialUniforms(pathTracer: ..., ...): void {
```

Update the one call site in `index.ts`.

---

#### 4.61 `forkUniformBridge.ts:68` — Self-verify CIE-Y integral constant

```ts
// Option A: assert at module init time
const _COMPUTED_Y_INTEGRAL = /* compute from CIE_Y_TABLE */;
console.assert(Math.abs(_COMPUTED_Y_INTEGRAL - 106.857) < 0.001,
  `CIE-Y integral mismatch: expected 106.857, got ${_COMPUTED_Y_INTEGRAL}`);

// Option B: compute directly (preferred if negligible init cost)
const CIE_Y_INTEGRAL = computeYIntegral(CIE_Y_TABLE);
setUniform(material, 'uYCmfIntegral', CIE_Y_INTEGRAL);
```

---

#### 4.62 `constants.ts:3,8` — Document relationship between `PT_TARGET_SAMPLES` and `PT_TARGET_SAMPLES_BASE`

```ts
/** @deprecated Use PT_TARGET_SAMPLES_BASE. Backward-compat alias. */
export const PT_TARGET_SAMPLES = PT_TARGET_SAMPLES_BASE; // was a separate constant; now an alias
export const PT_TARGET_SAMPLES_BASE = 192;
```

---

#### 4.63 `adaptiveTileWeights.ts:195` — Fix variable shadowing

```ts
// Before:
const px = tilesX * tilesY * 4;
// ...
for (let px = 0; px < tilesX; px += 1)  // shadows outer px

// After:
const pixelComponentCount = tilesX * tilesY * 4;
```

---

### Package: @vitrum/pt-webgpu

---

#### 4.64 `pathTraceBruteforce.wgsl.ts:1270-1693` — Split 424-line `main()` function

Extract into sub-functions:
```wgsl
fn decodeMaterial(hit: SceneHit, matId: u32) -> MaterialData { ... }
fn directLightingSample(rng: ptr<...>, hit: SceneHit, mat: MaterialData) -> vec3f { ... }
```

The bounce loop body should shrink from 424 lines to ~80.

---

#### 4.65 `pathTraceBruteforce.wgsl.ts:1296-1304` — Use `hitMaterialId()` in `main()`

Replace the inline matId extraction with a call to the already-defined `hitMaterialId(hit)` function at line 1024.

---

#### 4.66 `pathTraceBruteforce.wgsl.ts:1463-1500` — Call `sampleRectAreaLight()` from array loop

Parameterize the helper to accept light parameters as arguments; call from both the legacy single-slot path and the array loop. Same for `sampleMeshAreaLight`.

---

#### 4.67 `pathTraceBruteforce.wgsl.ts:1563,1575` — Cache `causticMode()` result

```wgsl
let cMode = causticMode();
if (cMode == 1u) { ... }
else if (cMode == 2u) { ... }
```

---

#### 4.68 `pathTraceBruteforce.wgsl.ts:198-205` — Fix HDRI dims piggy-backing in meshAreaTri slots

Add explicit fields to `FrameParams`:
```wgsl
environmentMapWidth: u32,
environmentMapHeight: u32,
```

Or pack into `environmentTint.w` and `environmentSun.w` (which are currently padding). Update host-side packer in `index.ts`.

---

#### 4.69 `pathTraceBruteforce.wgsl.ts:1175` — Expose gather radius as param

```wgsl
// Before:
let gatherRadius = 0.35;
// After:
let gatherRadius = params.photonGatherRadius; // new field in FrameParams
```

Expose via `EngineOptions.photonGatherRadius?: number` defaulting to `0.35`.

---

#### 4.70 `uploadSceneBuffers.ts` — Collapse `PackedSceneData` / `UploadedSceneBuffers` duplication

Have `UploadedSceneBuffers` embed `PackedSceneData` and extend with GPU buffer fields:
```ts
interface UploadedSceneBuffers extends PackedSceneData {
  bvhBuf: GPUBuffer;
  materialsBuf: GPUBuffer;
  // ... GPU buffer fields only
}
```

Remove `uploadPackedScene`'s 26-field copy loop; it becomes: `return { ...packed, bvhBuf, materialsBuf, ... }`.

---

#### 4.71 `uploadSceneBuffers.ts:339-552` — Remove legacy `first*` extractors

`firstPointLight`, `firstSpotLight`, `firstRectAreaLight`, `firstMeshAreaLight` coexist with the array packing path. The array path is the correct one.

**Fix:**
1. Remove the four `first*` functions.
2. Remove their output fields from `PackedSceneData` / `UploadedSceneBuffers`.
3. Update `buildPackedScene` to not call them.
4. Update `pathTraceBruteforce.wgsl.ts` `FrameParams` to remove single-light scalar fields; update `photonMapContribution` to read from the arrays.

---

#### 4.72 `uploadSceneBuffers.ts:597-666` — Split `environmentParams()` and document magic sky tint

```ts
function buildNoneEnvironment(): EnvironmentParams { ... }
function buildProceduralSkyEnvironment(env: ProceduralSkyEnvironment): EnvironmentParams {
  // 0.9/0.95/1.0 RGB weighting: empirical blue-sky approximation
  // mieCoefficient * 10: ad-hoc turbidity-to-scattering mapping
  // TODO: replace with Preetham sky model for physical accuracy
}
function buildHdriEnvironment(env: HdriEnvironment): EnvironmentParams { ... }
```

---

#### 4.73 `index.ts:343-447` — Extract `buildParamsBuffer()` pure function

Move the 105-line manual float/u32 buffer construction out of `renderFrame()`:
```ts
function buildParamsBuffer(
  input: FrameInput, sceneBuffers: UploadedSceneBuffers, ...
): ArrayBuffer { ... }
```

Add an assertion: `console.assert(buf.byteLength === 512, 'FrameParams size mismatch')`.

---

#### 4.74 `index.ts:274-292` — Remove redundant null checks after `#assertLive`

`#assertLive()` already throws if `this.#scene == null`. The subsequent `if (scene == null) throw` in `updatePrimitive` and `updateEmitter` is dead code. Remove the redundant checks.

---

#### 4.75 `index.ts:113-123` — Extract analytic shape ID constants

```ts
// shared module: analyticShapeIds.ts
export const ANALYTIC_SHAPE_IDS = {
  H_CHANNEL_CAME: 5,
  // add future shapes here
} as const;
export type AnalyticShapeId = keyof typeof ANALYTIC_SHAPE_IDS;
```

Import in `uploadSceneBuffers.ts` `analyticShapeId()` case and `capabilities` getter. The WGSL constants (`SHAPE_H_CHANNEL_CAME = 5u`) must be kept as separate declarations but a comment cross-reference ensures sync.

---

#### 4.76 `@vitrum/pt-webgpu` — Wire to a real consumer and add end-to-end smoke test

**Problem:** After all other fixes, `pt-webgpu` remains an isolated prototype. It has no external consumers, no wired integration, and the only tests are unit tests of TS-side packing logic. The WGSL `main()` entry point (even after the Phase 4.63 split) has never been submitted to a real WebGPU device.

**Fix:**
1. Wire `pt-webgpu` into `examples/two-engines-one-scene/` as the second engine (it currently uses `walkaround-hybrid`). The example is named "two engines" and this is the intended second slot.
   - Add `@vitrum/pt-webgpu` as a dependency of the example.
   - Add the source alias to `two-engines-one-scene/vite.config.ts`.
   - Instantiate `createPTEngine_WebGPU` alongside the existing `createWalkaroundEngine_Hybrid`.
2. Add a `pt-webgpu` GPU smoke test that submits `PT_WEBGPU_TRACE_WGSL` to a real `GPUDevice` (skip if `navigator.gpu` unavailable):
   ```ts
   // __tests__/wgslSmoke.gpu.test.ts
   it.skipIf(!navigator?.gpu)('PT_WEBGPU_TRACE_WGSL compiles on real device', async () => {
     const adapter = await navigator.gpu.requestAdapter();
     const device = await adapter!.requestDevice();
     // createComputePipeline with PT_WEBGPU_TRACE_WGSL — should not throw
   });
   ```
   This is the minimum bar: confirming the assembled shader string is syntactically valid WGSL on an actual WebGPU implementation.
3. Add `@vitrum/pt-webgpu` to `package.json` `dependencies` of the example package (currently only listed as a workspace package, not depended upon by anything).

---

### Phase 4 deferred items (revisit during Phase 11 second-sweep pass)

Four items were intentionally deferred because they are large mechanical
moves (4.52 / 4.53) or WGSL+host couplings that need a real GPU smoke
verification before landing (4.64 / 4.71). Phase 11 will surface them
again if they remain real issues; they are tracked here so a future
session can pick them up directly without re-discovering them.

- **4.52 / 4.53** `pt-webgl/src/index.ts` — extract `PTEngineWebGL2` class
  + `createPTEngine_WebGL2` factory + GPU-init logic into a dedicated
  `ptEngineWebGL2.ts` file. Mechanical move (~700 LOC), zero behavior
  change. Deferred to keep the current pass's diff reviewable.
- **4.64** `pathTraceBruteforce.wgsl.ts` — split the 424-line `main()`
  WGSL function into sub-helpers (RIS / shade / accumulate). The split
  needs a real-device smoke run to confirm correctness across the
  bounce loop's local-variable scoping; the new wgslSmoke.gpu.test
  added in Phase 4.76 will catch syntactic regressions but not
  semantic drift in the split.
- **4.71** `uploadSceneBuffers.ts` — remove the legacy `first*` light
  extractors and have `pathTraceBruteforce.wgsl.ts` read from the
  arrays everywhere (currently the WGSL HDRI sky aperture, mesh-area
  emitter test, and photon-map all read `params.pointLightPos` /
  `params.rectAreaPos` / `params.meshAreaTri*` single-light scalars).
  Touches the WGSL `params` struct layout — needs a careful GPU
  verification pass.

---

## Phase 5 — Integration Boundary Fixes

---

### 5.1 Add `pipelineCompiler.ts:96` comment (SVGF self-contained)

```ts
// SVGF_WGSL is self-contained: declares its own PI, INV_PI, LUM_W, WelfordVariance.
// Do NOT prepend COMMON_WGSL — it would cause WGSL redeclaration errors.
{ label: 'svgf', code: SVGF_WGSL },
```

---

### 5.2 Add `WalkaroundGPUPipeline.ts` comment (why `runSvgfWebGPU` is not used)

```ts
// We do NOT use runSvgfWebGPU() from shared-denoisers: it is a one-shot CPU-backed path
// that allocates and frees transient GPU textures per call. The pipeline owns persistent
// GPU textures across frames, so the one-shot API would churn texture allocations every frame.
import { packSVGFUniforms, ... } from '@vitrum/shared-denoisers';
```

---

### 5.3 Add snapshot test for WelfordVariance struct in `shared-denoisers`

```ts
// packages/shared-denoisers/__tests__/welfordVarianceCompat.test.ts
import { SVGF_WGSL } from '../src/wgsl/svgf.wgsl.js';
it('WelfordVariance WGSL block matches expected layout', () => {
  expect(SVGF_WGSL).toContain('struct WelfordVariance { mean: vec3f, M2: vec3f, count: u32 }');
});
```

When the canonical struct in `walkaround-hybrid/common.wgsl.ts` changes, this test forces an explicit update of the copy.

---

### 5.4 Export `VITRUM_USER_DATA_KEYS` from `three-bindings`; import in `cornell-box` example

**See 4.7 above.** After exporting from three-bindings, update `examples/cornell-box/src/main.ts:269` to import and use the constants instead of bare strings.

---

## Phase 6 — Test Quality Fixes

---

### 6.1 Fix tautological assertions in `rc-bindings.test.ts:109-131`

Replace:
```ts
const CAST_BGL_ENTRY_COUNT = 9;
expect(CAST_BGL_ENTRY_COUNT).toBe(9); // tautology
```

With a real structural assertion:
```ts
const wgsl = PROBE_RAY_CAST_WGSL;
const bindingCount = (wgsl.match(/@binding\(/g) ?? []).length;
expect(bindingCount).toBe(9);
```

Or instantiate `RCDispatcher` with a stub `GPUDevice` and assert on the actual bind group layout entry count.

---

### 6.2 Strengthen `sprint11-ppg.test.ts:472-483` byte-content assertion

```ts
// Before (weak — matches by length only):
const matchCall = mockQueue.writeBuffer.mock.calls.find(
  (call) => (call[2] as Uint8Array).byteLength === disabled.byteLength
);

// After (assert content equality too):
const matchCall = mockQueue.writeBuffer.mock.calls.find((call) => {
  const buf = call[2] as Uint8Array;
  return buf.byteLength === disabled.byteLength &&
    buf.every((byte, i) => byte === disabled[i]);
});
```

---

### 6.3 Extract shared WebGPU polyfill to test helper

**Files:** `sprint13-neural.test.ts:54-67`, `sprint11-ppg.test.ts:55-68`, `sprint9-welford.test.ts:43-57`

```ts
// __tests__/helpers/webgpuPolyfills.ts
export function installWebGPUPolyfills(): void {
  // Install GPUBufferUsage, GPUTextureUsage, etc.
}
```

Import and call `installWebGPUPolyfills()` in each test's `beforeAll`.

---

### 6.4 Extract shared test stub in `pt-webgl` tests

`FakeWebGL2RenderingContext` is defined in `capabilities.test.ts:57-89`. Create `__tests__/testUtils.ts` with the canonical stub and import from test files.

---

### 6.5 Restore `globalThis` mutation in `capabilities.test.ts:104-108`

```ts
afterAll(() => {
  delete (globalThis as unknown as Record<string, unknown>).WebGL2RenderingContext;
});
```

---

### 6.6 Expand `sceneFromThreeJS` test coverage

Add tests for all throw paths: `InstancedMesh`, `ShaderMaterial`, no-normals, no-position; and warn paths: `AmbientLight`, multi-material.

---

### 6.7 Add FrameParams offset test in `wgslContract.test.ts`

```ts
it('FrameParams is 512 bytes', () => {
  // instantiate the engine with a stub device, call renderFrame, assert params buffer size
  // OR add a standalone calculation from WGSL struct field offsets
});
```

---

### 6.8 Add lifecycle test in `factoryCapabilities.test.ts`

Add at least one test that calls `engine.setScene(...)` and `engine.dispose()` after construction, verifying the lifecycle state machine transitions correctly.

---

### 6.9 Rename `sprint9-welford.test.ts` or move WELFORD_TEMPORAL_WGSL tests

Either rename to `sprint9-10a-welford.test.ts` or move the `WELFORD_TEMPORAL_WGSL` describe block to `sprint10a-*` test file.

---

## Phase 7 — Examples & Tools Cleanup

---

### 7.1 Extract `triggerDenoise()` helper in `cornell-box/src/main.ts:453-663`

Replace three near-identical async IIFE blocks (oidn/wgsl/svgf) with:
```ts
async function triggerDenoise(
  mode: DenoiseDisplay,
  started: { current: boolean }, succeeded: { current: boolean },
  denoiseFn: () => Promise<void>
): Promise<void> { ... }
```

---

### 7.2 Generalize URL param parsers in `cornell-box/src/main.ts:69-119`

```ts
function parseNumber(
  value: string | null, fallback: number,
  opts?: { integer?: boolean; min?: number }
): number { ... }
```

Replace `parsePositiveMegapixels`, `parsePositiveFloat`, `parsePositiveInt`, `parseNonNegativeInt` with parameterized calls.

---

### 7.3 Use shared Cornell builder in `cornell-box/main.ts:301-352`

```ts
import { buildCornellBoxThreeScene } from '@vitrum-examples/shared';
// Replace the inline box construction with:
const base = buildCornellBoxThreeScene();
applyScenarioMaterialTweaks(base, config.scenario);
```

---

### 7.4 Add `@vitrum/shared-denoisers` source alias in `cornell-box/vite.config.ts`

```ts
'@vitrum/shared-denoisers': resolve(__dirname, '../../packages/shared-denoisers/src/index.ts'),
```

---

### 7.5 Remove redundant aliases in `two-engines-one-scene/main.ts:41-44`

```ts
// Remove:
const ptCanvas = canvasPt;
const wgpuCanvas = canvasWgpu;
const st = statusEl;
// Use canvasPt, canvasWgpu, statusEl directly throughout.
```

---

### 7.6 Flatten WebGPU guard nesting in `two-engines-one-scene/main.ts:111-200`

Invert the three nested guard conditions to early returns, leaving `wgpuLoop` at the function's natural indentation level.

---

### 7.7 Replace inline if-chains with mapping table in `capture-adapter-playwright.mjs:11-63`

```js
const ENV_TO_QUERY = [
  { env: 'VITRUM_SCENARIO_ID',   query: 'vitrumScenario' },
  { env: 'VITRUM_DENOISER',      query: 'vitrumDenoiser' },
  // ... all 13 mappings
];
for (const { env, query } of ENV_TO_QUERY) {
  const val = process.env[env];
  if (val) u.searchParams.set(query, val);
}
```

---

### 7.8 Raise or configure JS heap limit in `capture-adapter-playwright.mjs:80-84`

```js
args: ['--disable-dev-shm-usage', `--js-flags=--max-old-space-size=${process.env.VITRUM_JS_HEAP_MB ?? 4096}`],
```

---

### 7.9 Fix hardcoded date literal in `run-gap-closure-verification.mjs:15`

```js
const today = new Date().toISOString().slice(0, 10);
const outputPath = resolve(here, `results/gap-closure-verification-${today}.json`);
```

---

### 7.10 Parallelize scenario execution in `run-gap-closure-verification.mjs:305-308`

```js
entries = await Promise.all(scenarios.map(evaluateScenario));
```

If GPU contention is a concern, add a bounded concurrency semaphore with limit=2.

---

### 7.11 Extract `runCommandWithTimeout()` in `run-gap-closure-verification.mjs:96-153`

Move process-management complexity (SIGTERM→SIGKILL, cross-platform process group kill) to a standalone helper module in `tools/`.

---

## Phase 8 — Legacy Host-App Files

> These files are in `_staging/legacy-source/` — intentionally unextracted host-app React/Redux code per `_staging/README.md`. Findings below are tracked for if/when the host app resumes active development.

---

### 8.1 Extract `useWebGPUCanvasReadback` hook from `RcStage.tsx`

The 145-line toDataURL readback workaround (`RcStage.tsx:362-499`) is a reusable primitive.

---

### 8.2 Extract `useWebGPUNoToneMapping` hook (shared by RcStage and WalkaroundStage)

Identical `useEffect` duplicated verbatim: `renderer.toneMapping = THREE.NoToneMapping; renderer.outputColorSpace = THREE.LinearSRGBColorSpace`.

---

### 8.3 Consolidate SwiftShader refusal banner across 4 stage files

Four implementations with different visual styles and mounting mechanisms. Extract to `SwiftShaderWarningBanner` component.

---

### 8.4 Consolidate `framedRef` camera-framing pattern across 4 stage files

Identical `const framedRef = useRef(false); useEffect(() => { if (framedRef.current) return; ... camera.position.set(...); framedRef.current = true; })` in all four stage files. Extract to `useCameraFrameOnce(camera, target)` hook.

---

### 8.5 Consolidate 60 FPS frame cap between `RcStage.tsx:131` and `RestirStage.tsx:157`

Identical `lastFrameTsRef + TARGET_FRAME_INTERVAL_MS = 1000/60 - 1` pattern. Extract to `useFrameRateCap(targetFps)` hook.

---

### 8.6 Extract `buildBVHWhenReady` from `RestirStage.tsx` `useEffect`

225-line async function defined inside a `useEffect` (`RestirStage.tsx:305-529`). Extract to `useRestirPipeline(...)` custom hook.

---

### 8.7 Extract `createIsSceneReadyPredicate()` shared utility

Identical `isSceneReady()` logic in `RestirStage.tsx:277-303` and `HybridLayeredStage.tsx:281-302`. Extract to a shared utility.

---

### 8.8 Fix microtask spin-poll in `HybridLayeredStage.tsx:200-217` and `RestirStage.tsx:175-189`

```ts
// Before (unbounded microtask spin):
const check = () => { if (!cancelled) { const g = window.__WG__; if (g) { ... return; } Promise.resolve().then(check); } };
// After (yields to event loop):
const check = () => { if (!cancelled) { const g = window.__WG__; if (g) { ... return; } setTimeout(check, 16); } };
```

---

### 8.9 Guard `debugDepsRef` behind `import.meta.env.DEV` in `HybridLayeredStage.tsx:340-367`

The `debugDepsRef` allocation and mutation runs in all environments. Wrap entirely:
```ts
if (import.meta.env.DEV) {
  debugDepsRef.current.graph = graphFaces;
  debugDepsRef.current.fires++;
}
```

---

### 8.10 Replace `window.setInterval` polling in `WalkaroundStage.tsx:182-193`

```ts
// Before: setInterval polling every 500ms for a dev toggle
const id = window.setInterval(() => { setDdgiOn(window.__HYBRID_LAYERS__?.ddgi !== false); }, 500);

// After: event-driven or React context; event listener on storage/message
```

---

### 8.11 Memoize `activeSun` computation in `PTStage.tsx:70-76`

```ts
const activeSun = useMemo(() => {
  for (const id of lightAllIds) {
    const l = lightsById[id];
    if (l?.kind === 'sun' && l.on) return l;
  }
  return null;
}, [lightAllIds, lightsById]);
```

---

### 8.12 Extract `installHybridStateAccessor` to `hybridDevTools.ts`

82-line function accessing 6 global namespaces and 5 Redux sub-slices — dev utility living in production stage file. Move to a sibling `hybridDevTools.ts` imported only in dev builds.

---

### 8.13 Split `usePTEnvironment` async paths in `ptEnvironment.ts:56-139`

Separate the cancellable async RGBELoader path (keyed on `outdoorUrl`) from the synchronous sky/night/none path.

---

### Phase 8 — applied vs deferred

Items 8.7 (TODO markers added at both duplicate sites), 8.8 (microtask
spin → setTimeout in both files), 8.9 (documented React-hooks-rules
constraint on conditional useRef), 8.11 (useMemo for activeSun) are
landed in this pass.

Items 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.10, 8.12, 8.13 — all extract-to-
hook or extract-to-component refactors of `_staging/legacy-source/` React
code — are deferred. The plan note at the top of Phase 8 stands: those
files are intentionally unextracted host-app code; the extraction is
out-of-scope until the host app resumes active development. When that
happens, this section serves as the punch list.

---

## Phase 9 — Deferred Sprint Integrations

These items were explicitly scoped out of earlier sprints. They are not structural cleanups — they are functional wiring work that must happen before the library's walkaround pipeline is end-to-end operational.

---

### 9.1 Wire Sprint 9 adaptive sampling (`resolve.wgsl.ts` + `sampleBudget.wgsl.ts`)

**Context:** `resolve.wgsl.ts` and `sampleBudget.wgsl.ts` were moved to `deferred/` in Phase 2.13. They are complete shaders but never dispatched.

**Work:**
1. `pipelineCompiler.ts`: add `RESOLVE_WGSL` and `SAMPLE_BUDGET_WGSL` shader module compilation (follow the existing module pattern).
2. `WalkaroundGPUPipeline.ts`: add a `_dispatchSampleBudget(encoder)` pass before the RIS pass that writes per-pixel sample budgets based on Welford variance. Add a `_dispatchResolve(encoder)` pass after the denoising chain.
3. `sampleBudget.wgsl.ts`: before wiring, remove the `[INLINE-COPY]` WelfordVariance block (Phase 2.13 flagged this) and replace with the injected `${WELFORD_VARIANCE_STRUCT_WGSL}` constant (Phase 1.5).
4. Add integration tests asserting the two new passes appear in the pipeline's command sequence.

**Dependency:** Requires Phase 1.5 (WelfordVariance struct extraction) to be done first.

---

### 9.2 Wire Sprint 10a SVGF walkaround integration

**Context:** Per `plan/sprint-10a-walkaround-integration.md`, the SVGF denoiser from `@vitrum/shared-denoisers` is already used in `WalkaroundGPUPipeline.ts` via low-level primitives, but the full integration spec (buffer handoff protocol, G-buffer layout contract, frame-count tracking) was deferred.

**Work:**
1. Resolve the normal encoding mismatch from Phase 1.4 — this is a prerequisite: the G-buffer encoding must be settled before SVGF integration can be verified.
2. Audit the actual G-buffer slot assignments in `WalkaroundGPUPipeline.ts` against the `SvgfWebGPUInputs` shape expected by `runSvgfWebGPU`. Document any mismatches.
3. Wire `svgfFrameCount` tracking: the pipeline must increment a frame counter per-frame and pass it to `packSVGFUniforms`. Verify the `SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT = 4` threshold is respected before variance-guided filtering begins.
4. Confirm `dispose()` correctly tears down all SVGF-related GPU resources (after Phase 4.38 moves them to `initialize()`).

---

### 9.3 Wire Sprint 11 PPG dispatch (`@group(2)` placeholder resolution)

**Context:** Per `plan/sprint-11-ppg-integration.md`, the PPG bind group uses `@group(2)` as a documented placeholder. The actual group number must be assigned once the full bind group layout for the shade pass is finalized.

**Work:**
1. Audit the shade pass bind group layout in `pipelineCompiler.ts` and `WalkaroundGPUPipeline.ts` to determine which group number PPG buffers should occupy (group 0 is typically frame-global UBO, groups 1-3 are scene data — PPG likely lands at group 4 or shares a slot via dynamic offsets).
2. Update `@group(2)` in `ppgSample.wgsl.ts` and `ppgUpdate.wgsl.ts` to the correct group number.
3. Update the `sprint11-ppg.test.ts:214` assertion (Phase 3.9 annotated it as a TODO — now close it).
4. Wire the PPG dispatch into `_dispatchPPG()` (extracted in Phase 4.37) with the correct bind group assignment.
5. Run a GPU smoke test confirming `PPG_SAMPLE_WGSL` and `PPG_UPDATE_WGSL` compile with the finalized group numbers.

**Dependency:** Requires Phase 4.37 (`renderFrame()` extraction) and Phase 4.51 (PPG splice replacement with explicit markers) to be done first so the integration work lands in clean, well-bounded code.

---

### Phase 9 status after this pass

- **9.1 Sprint 9 adaptive sampling** — DEFERRED. The shaders are in
  `deferred/` (Phase 2.13) and Phase 2.13 already replaced their
  `[INLINE-COPY]` Welford with the canonical `${WELFORD_VARIANCE_WGSL}`
  injection from Phase 1.5. Wiring the dispatch into the pipeline
  requires a real-device GPU smoke run — the new wgslSmoke.gpu.test
  added in Phase 4.76 only catches syntactic regressions, not semantic
  drift in a brand-new compute-pass insertion. Next pass should land
  the dispatch + an integration test that asserts the new passes
  appear in the encoder timestamp queries.

- **9.2 Sprint 10a SVGF walkaround** — DONE.
  - Phase 1.4 settled the G-buffer normal encoding contract (raw → 0..1
    encoded with `xyz * 2.0 - 1.0` decode on the consumer side); the
    spec file `plan/sprint-5-mrt-gbuffer-spec.md` was updated to match
    the actual shade.wgsl encoding.
  - The SVGF dispatch path (welford-temporal + svgf-variance + N×
    svgf-atrous) is wired in `WalkaroundGPUPipeline._dispatchSVGF`
    (extracted in Phase 4.37) and routes through dedicated bind-group
    builders (Phase 4.39).
  - Frame-count tracking is live: `_accumFrameIndex` is passed to
    `packSVGFVarianceUniforms` and to the welford UBO every frame; the
    `SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT = 4` threshold is enforced
    inside the WGSL shader.
  - `dispose()` tears down all SVGF UBOs unconditionally now that
    Phase 4.38 moved their allocation into `initialize()`.

- **9.3 Sprint 11 PPG dispatch** — PARTIAL.
  - The `@group(2)` placeholder is annotated with the explicit
    Sprint-11-integration TODO in `sprint11-ppg.test.ts` (Phase 3.9).
  - The PPG dispatch is already wired in `WalkaroundGPUPipeline`
    (the `ppgUpdatePipeline` runs after the shade pass when
    `ppgEnabled === true`) and uses the explicit
    `// @@PPG_BINDINGS_INSERT@@` markers from Phase 4.51.
  - The remaining work is finalising the production `@group` number
    (group 2 in code is a placeholder name — the WGSL declares
    `@group(2) @binding(...)` and the host bind group matches). Once
    the production layout is locked, `@group(2)` may stay or move to
    a different number, and the test assertion + WGSL declarations
    must be updated in the same commit. Closing this fully needs a
    real-device GPU verification pass.

---

## Phase 10 — Audit Loop (Run Until Clean)

**Goal:** After Phases 1–9 are complete, run the project's audit harness in a loop until it returns zero findings.

### 10.1 Run mechanical checks across the workspace
1. `npm run typecheck` at repo root — must be 100% clean across every package that defines a `typecheck` script. Zero errors. Zero warnings.
2. `npm test` at repo root — every test suite must pass. No skipped tests added during remediation (skipped tests = mess; either fix or delete the test).
3. `npm run lint` if a lint script exists at any package; fix all reported issues.
4. Run any `scripts/run-gap-closure-verification.mjs` style smoke script that exists in `scripts/` — capture output, address any non-clean signal.

### 10.2 Run the `/audit` skill / equivalent project audit pass
1. Dispatch a Sonnet sub-agent to perform a full repo audit: read every file changed during Phases 1–9, verify no regressions, no half-implemented edits, no leftover TODO/FIXME/HACK markers introduced.
2. Audit must cover:
   - Each phase's diff vs the plan item — does the change match the spec?
   - Cross-file consistency (e.g., if a constant was extracted, every old site uses the new one)
   - Test coverage for new public APIs introduced by the remediation
   - No stale comments referencing the old behavior
   - No dead exports left behind after refactors
3. Audit produces a findings list in the same format as the original sweep (FINDING: file:line / Category / What / Evidence / Fix).

### 10.3 Remediation loop
For every audit finding:
1. Fix it immediately (no deferring — per the user's standing rule).
2. After each batch of fixes, re-run 10.1 (mechanical checks).
3. After mechanical checks pass, re-run 10.2 (audit pass).
4. Continue looping until audit returns zero findings.

**Termination condition:** Two consecutive audit runs return zero findings AND mechanical checks pass. Only then proceed to Phase 11.

**Hard rule:** Do not skip any audit finding. Do not mark anything "non-blocking" or "for later". Clean means clean.

---

## Phase 11 — Complexity Sweep (Second Pass)

**Goal:** Run the full `/complexity-sweep` skill again over the entire repo. After remediating ~150 findings, the codebase has changed shape — a fresh sweep will surface anything that emerged from the refactors (e.g., a newly-extracted file that now has its own issues, a boundary that shifted, a god-file that we accidentally re-introduced).

### 11.1 Pre-sweep hygiene
1. Ensure git working tree is clean — commit all Phase 1–10 work.
2. Verify `MEMORY.md` and `in-flight-sweep.md` are up to date or archived; stale context misleads the sweep.
3. Confirm CLAUDE.md and plan/ docs match current code reality.

### 11.2 Execute `/complexity-sweep`
1. Invoke the skill with arguments matching the original run: "please review the entire repo, do not trust any comments, plans, memories, or any other summaries that may have been written. The only thing that contains the truth of the state of the repo is the code itself. Review it all, be very careful to read every line of every file and report back with a full list of findings."
2. Follow the skill's full multi-agent dispatch (domain + integration + dead code).
3. Persist `in-flight-sweep.md` before synthesis.

### 11.3 Synthesize Phase 11 findings
1. Verify each agent finding by reading the cited code yourself — sub-agent reports are hypotheses (per CLAUDE.md guidance). False positives must be marked verified-false-positive with one-line justification before discarding.
2. Produce a verified findings list grouped by phase (same structure as Phases 1–9 of this plan).
3. Append the verified findings as `plan/sweep-remediation-plan-pass-2.md` (new file — do not overwrite this plan).

**Note:** If the second sweep returns zero verified findings, skip Phase 12 entirely and proceed directly to the final-clean confirmation in Phase 12.3.

---

## Phase 12 — Second-Pass Remediation & Final Audit

**Goal:** Execute the Phase 11 findings to completion, then run the audit loop one more time to confirm zero outstanding issues.

### 12.1 Execute pass-2 remediation
1. Work each finding in `sweep-remediation-plan-pass-2.md` start-to-finish (no batching across items, no deferring).
2. Order: GPU correctness first, then dead code, then structural cleanup, then tests, mirroring the phase ordering of this plan.
3. Commit per logical unit so the diff is reviewable.

### 12.2 Run audit loop (mirror of Phase 10)
1. `npm run typecheck` clean
2. `npm test` clean
3. `/audit` sub-agent — zero findings
4. Loop until two consecutive clean runs.

### 12.3 Final-clean confirmation
1. Run mechanical checks one final time.
2. Dispatch one final Sonnet audit agent with the prompt: "Verify the vitrum repo is in a clean state. Read every package's source and tests. Report any structural issues, dead code, half-implemented features, stale comments, or test gaps. Zero tolerance — if it is a mess, flag it."
3. If the final agent returns clean: stop, write a one-line note at the bottom of this plan stating "Full remediation complete on `<iso-date>` — repo confirmed clean by two-pass sweep + audit."
4. If the final agent finds anything: go back to Phase 12.1 with the new findings. Repeat indefinitely. The work is not done until clean is verified clean.

**Hard rule:** No partial completion. No "good enough". No "ship it and fix later". The user is asleep — they expect to wake up to a fully clean repo or a clear status of where the loop is.

---

## Appendix: Files Confirmed Healthy (No Action Required)

These files were read in full and found to be clean, single-purpose, and well-structured:

**@vitrum/core:** `index.ts`  
**@vitrum/three-bindings:** `environment.ts`, `lights.ts`, `mesh.ts`  
**@vitrum/shared-bvh:** `bvhCommon.ts` test suite, `index.ts`  
**@vitrum/shared-samplers:** `hgPhase.ts`, `equiAngular.ts`, `mixturePdf.ts`, `cauchyIor.ts`, `bdptVertex.ts`, `bdptMIS.ts`, `hammersley.wgsl.ts`, `octahedralCore.wgsl.ts`  
**@vitrum/shared-denoisers:** `halfFloat.ts`, `svgfConstants.ts`, `webGpuTextureCopy.ts`, `index.ts`, `sharedWebGpuDevice.ts`, `atrous.wgsl.ts`, `hdrLuminanceBilateral.wgsl.ts` (partial), all test files  
**@vitrum/pt-webgl:** `debounce.ts`, `frameCamera.ts`, `lightingIntensityTable.ts`, `lightingState.ts`, `readbackHdr.ts`, `skyParams.ts`, `sunGeometry.ts`, `cameUniformUploader.ts`, `adaptiveTileWeights.ts` (minor shadow), most test files  
**@vitrum/pt-webgpu:** `math/mat4.ts`, `scene/flattenScene.ts`, `scene/patchScene.ts`, `wgsl/common.wgsl.ts` (except INV_2PI), test files  
**@vitrum/walkaround-hybrid:** `ddgi/types.ts`, `ddgiAtlasLayout.ts`, `DDGI.ts`, `probeGrid.ts`, `lib/nodeMaterialUpgrade.ts`, `hostScene/types.ts`, `ppg/types.ts`, `ppg/ppgCellUpload.ts`, `neural/wgsl/` (all 5 kernels), `pipeline/bindGroupLayouts.ts`, `pipeline/bindGroupBuilders.ts`, `pipeline/pipelineCompiler.ts`, `pipeline/resourceManager.ts`, `pipeline/timestampQueries.ts`, `pipeline/uboUpdater.ts`, `shaders/composite.wgsl.ts`, `shaders/welfordTemporal.wgsl.ts`  
**Examples/Tools:** `examples/shared/`, `scripts/run-fork-shader-smoke.mjs`, `tools/benchmark-runner/scenario-presets.mjs`  
**Legacy:** `PTDeviceLostBoundary.tsx`, `PTPostProcessing.tsx`, `cameraLookPresets.ts`, `usePTPipelineConfig.ts`, `usePTSampleTarget.ts`, `outdoorHdri.ts`, `outdoorScenePresets.ts`, `engineRegistry.ts`, `WalkaroundDebugBridge.tsx`, `walkaroundBridgeTypes.ts`, `lib/useSceneBVH.ts`

---

Full remediation complete on 2026-05-11 — repo confirmed clean by two-pass sweep + audit.

---

## P3 follow-up — deferred items landed (2026-05-11)

The Phase 4 / Phase 9 deferrals (4.52 / 4.53 / 4.64 / 4.71 / 9.1 / 9.3) and
the Pass-2 deferrals (P2-4.4 / P2-4.6 / P2-6.1 / P2-7.2) all landed in the
P3-* commit series on 2026-05-11. See `plan/p3-validation-matrix.md` for
the RTX-4090-driven visual verification plan that the user runs via
Claude-in-Chrome to confirm each deferred-item resolution.

- **4.52 / 4.53** → P3-A.1 commit `e9d52f4` (PTEngineWebGL2 extraction).
- **4.64** → P3-B.2 commit `f8a36dd` (pathTraceBruteforce main() split).
- **4.71** → P3-B.1b commit `e9dd0b6` (drop first* + repack FrameParams).
- **9.1** → P3-C.1 (Sprint 9 adaptive sampling — full integration, real algorithmic change).
- **9.3** → P3-C.2 commit `24d847a` (delete orphan PPG_SAMPLE_WGSL; live path provided by shadePpgGuide.wgsl.ts).
- **P2-4.4** → P3-B.1a commit `efdd5e0` (uploadSceneBuffers split).
- **P2-4.6** → P3-A.2 commit `e9d52f4` (passIdx → named PassLayout; also fixes a latent telemetry-label bug).
- **P2-6.1** → P3-B.3 commit `f8a36dd` (cross-engine extractThreePbrScalars).
- **P2-7.2** → P3-B.4 commit `c7f7638` (drop three/webgpu StorageTexture from probeGrid + probeUpdatePass).

P3-V (validation matrix) lives at `plan/p3-validation-matrix.md`. Phase 8
host-app extractions remain intentionally deferred until the host app
resumes active development.
