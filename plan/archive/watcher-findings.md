# Watcher Findings — started 2026-05-09 ~session start

Watcher process: long-running sonnet, ~45 min monitor window.
Baseline commit: `1036a8c` "fix: wire vitrum.Material ↔ THREE userData.vitrum* round-trip"

---

## Iteration 0 (start — baseline)

- **Baseline commit**: `1036a8c` "fix: wire vitrum.Material ↔ THREE userData.vitrum* round-trip"
- **tsc**: clean (exit 0)
- **Tests**: 567 passing across all packages

### Uncommitted state at start (significant — major pt-webgpu work staged)

Modified tracked files:
- `CHANGELOG.md` — adds pt-webgpu prototype entry
- `README.md` — updates pt-webgpu description from stub to active prototype
- `packages/pt-webgpu/package.json` — adds `vitest` devDep + `"test": "vitest run"` script
- `packages/pt-webgpu/src/index.ts` — entirely rewritten from 8-line stub to 440-line engine
- `plan/library-architecture.md` — updates pt-webgpu section from "future Phase 7+" to "prototype, evolving"
- `plan/phase-6-status.md` — updates test count note + changes "stub" to "prototype"

New untracked files:
- `packages/pt-webgpu/README.md`
- `packages/pt-webgpu/src/__tests__/buildCpuBvh.test.ts` (2 tests)
- `packages/pt-webgpu/src/__tests__/scenePack.test.ts` (3 tests)
- `packages/pt-webgpu/src/math/` — mat4.ts (100 LOC)
- `packages/pt-webgpu/src/scene/` — flattenScene.ts, buildCpuBvh.ts, patchScene.ts, uploadSceneBuffers.ts
- `packages/pt-webgpu/src/wgsl/` — common.wgsl.ts, hammersley.wgsl.ts, octahedral.wgsl.ts, pathTraceBruteforce.wgsl.ts, pathTraceSeed.wgsl.ts

### REVIEW PASS — pt-webgpu uncommitted work

This is a major step up from the prior review's state. The package has moved from a 1-line stub to a ~944 LOC functional prototype. Key findings:

#### Resolved issues from prior review

1. **Capability over-claim (HIGH, was flagged)**: `supportedAnalyticShapes` is now `new Set<string>()` — empty. Correct; matches zero implementation. RESOLVED.
2. **Capability over-claim (HIGH, was flagged)**: `supportedEmitterKinds` is now `new Set<string>(['directional', 'point'])` — matches actual implementation (both are wired in `buildPackedScene` and the shader). RESOLVED.
3. **`maxBounces` advertised vs single-bounce gap (HIGH, was flagged)**: The shader now has an actual multi-bounce loop — `for (var bounce = 0u; bounce < bounceLimit; bounce = bounce + 1u)` up to 8 bounces, with Russian roulette at bounce > 2. RESOLVED.
4. **Zero tests (noted)**: Package now has `buildCpuBvh.test.ts` (2 tests) and `scenePack.test.ts` (3 tests), 5 tests total. Partial resolution — core engine lifecycle (setScene, renderFrame, pause/resume/dispose state machine) still has no test coverage, but the CPU-side logic is covered.

#### New findings

**MEDIUM — `supportsIncrementalScene: true` vs. implementation**

`index.ts:115`: `supportsIncrementalScene: true`. The README Known Limitations section still says "No incremental scene updates (`updatePrimitive`, `updateEmitter`)" — this text contradicts the capabilities object.

Read the actual implementation: `updatePrimitive` and `updateEmitter` are implemented in `index.ts`, and they delegate to `patchScene.ts` helpers, then call `this.setScene(nextScene)` — which is a full rebuild (triggers BVH rebuild + re-upload). This is NOT zero-cost incremental; it's a full-rebuild fallback. The contract's `supportsIncrementalScene` implies the engine can handle id-targeted patches without a full scene rebuild. Since the implementation actually does a full rebuild, this is functionally correct (it won't fail) but the capability semantic is misleading. A host that sees `supportsIncrementalScene: true` and calls `updatePrimitive` in a tight loop will trigger full BVH rebuilds each time.

Recommendation: Keep `supportsIncrementalScene: true` but add a warning in `updatePrimitive`/`updateEmitter` like "note: pt-webgpu prototype triggers a full scene rebuild on every patch" — or document this clearly in the README. The README currently says this feature isn't supported, which also needs updating since the methods are implemented.

**LOW — `patchScene.ts` extracted but README still says "No incremental scene updates"**

The README Known Limitations bullet "No incremental scene updates (`updatePrimitive`, `updateEmitter`)" is stale — these methods ARE implemented (as full-rebuild fallbacks). Should read something like "Incremental scene updates are supported via full-rebuild fallback (BVH is rebuilt on each patch)."

**LOW — `pathTraceSeed.wgsl.ts` still dead code**

Untracked new file. Same dead-code status as noted in prior review. Not imported anywhere. Still exports `PT_WEBGPU_SEED_WGSL` that nothing uses.

**LOW — point light shadow ray `tMax` correctness**

`pathTraceBruteforce.wgsl.ts:307`: `let pointShadowRay = Ray(hitPos + normal * 1e-3, pointDir); let pointBlocked = traceAny(pointShadowRay, 1e-4, max(dist - 2e-3, 1e-3));`

The `tMax` is `max(dist - 2e-3, 1e-3)` — this clips the shadow ray 2mm before the light, correct to avoid self-intersection at the light position. This looks correct, though the 2mm heuristic is world-scale dependent. Not a bug, but note it.

**LOW — BVH leaf stride**

`buildCpuBvh.ts:161`: node buffer is allocated as `nodes.length * 32` bytes. Each node is 32 bytes: 3f min + 3f max + 1u32 + 1u32 = 8 floats = 32 bytes. The WGSL `BVHNode` struct in `common.wgsl.ts:17-22` has `boundsMin: array<f32, 3>`, `boundsMax: array<f32, 3>`, `rightChildOrTriOffset: u32`, `splitAxisOrTriCount: u32` = 8 × 4 = 32 bytes. Layout is consistent, no stride mismatch. No H-1 style bug here.

**LOW — params buffer layout verification**

`index.ts:282` allocates `new ArrayBuffer(224)`. The WGSL `FrameParams` struct at the shader side: width(u32) + height + frameIndex + frameSeed + triangleCount + maxBounces + bvhNodeCount + _pad0 = 8 u32s = 32 bytes, then cameraPos(vec4f) + lightDir(vec4f) + pointLightPos(vec4f) + pointLightRadiance(vec4f) + environmentTint(vec4f) + environmentSun(vec4f) = 6 × 16 = 96 bytes, then invViewProj(mat4x4f) = 64 bytes. Total WGSL struct = 32 + 96 + 64 = 192 bytes. Buffer allocated: 224 bytes. The 32 extra bytes are fine — WGSL uniform buffers can be larger than the struct, and 224 is safe. The CPU writes invVp at `paramsF32.set(invVp, 32)` (byte offset 128) which puts it at bytes 128-192, matching the struct layout (8×4=32 bytes prefix + 6×16=96 bytes = 128 bytes before invViewProj). Layout is correct.

**INFO — Materials now pack roughness + metallic correctly**

`uploadSceneBuffers.ts:70-85`: material packer now includes `roughness` (m0.w), `emissive*intensity` (m1.rgb), and `metallic` (m1.w). The shader reads and uses roughness to determine `glossyReflectionSample` jitter, and metallic to stochastically choose specular vs diffuse branch. This is no longer purely Lambertian — it's a basic Lambertian/specular mix. Correct and consistent between CPU packer and WGSL.

**INFO — Normal handling now correct with Phong interpolation**

Shader at lines 146-150 does barycentric interpolation of per-vertex normals from the `normals` buffer, falling back to geometric normal if out-of-bounds. The normals buffer is properly populated by `uploadSceneBuffers.ts:218-223`. This is a meaningful improvement over pure geometric normals.

**INFO — Environment procedural-sky wired through**

`uploadSceneBuffers.ts:134-170`: `environmentParams()` handles `none`, `procedural-sky`, and HDRI (with warning). The shader uses `environmentTint` and `environmentSun` from params. Scene-driven sky works.

#### Summary for pt-webgpu uncommitted work

CLEAN: No tsc errors, no test regressions. The prior HIGH-severity capability over-claims are resolved. The implementation is substantially more capable than the prior review's state.

Remaining items to watch:
1. `supportsIncrementalScene: true` + README contradiction about "No incremental updates" — stale doc.
2. Dead `pathTraceSeed.wgsl.ts` still present.
3. Engine lifecycle (factory validation, state machine) still has zero test coverage.

---

## Iteration 0.5 (watcher restart — 2026-05-10 baseline)

**Watcher**: second run (sonnet-based poller), picking up from prior watcher's Iteration 0.
**Baseline commit**: `1036a8c` "fix: wire vitrum.Material ↔ THREE userData.vitrum* round-trip"
**HEAD at start**: `1036a8c`
**tsc**: clean
**Tests**: 569 passing (prior watcher said 567 — 2 additional from pt-webgpu buildCpuBvh.test.ts + scenePack.test.ts)

### State at watcher start

Parallel agent has significant uncommitted work on pt-webgpu:
- `CHANGELOG.md`, `README.md`, `plan/library-architecture.md`, `plan/phase-6-status.md` — doc updates reflecting pt-webgpu prototype upgrade
- `packages/pt-webgpu/package.json` — adds vitest devDep + test script
- `packages/pt-webgpu/src/index.ts` — full engine rewrite (440 LOC, all Engine interface methods)
- `packages/pt-webgpu/README.md` — new file, accurate capability listing
- `packages/pt-webgpu/src/__tests__/` — buildCpuBvh.test.ts (2 tests), scenePack.test.ts (3 tests), patchScene.test.ts (5 tests) = 10 tests
- `packages/pt-webgpu/src/math/mat4.ts` — new file
- `packages/pt-webgpu/src/scene/` — flattenScene.ts, buildCpuBvh.ts, patchScene.ts, uploadSceneBuffers.ts
- `packages/pt-webgpu/src/wgsl/` — 5 WGSL files including `pathTraceSeed.wgsl.ts` (dead)
- `external_requests/` — 7 new files (RFEs 09–15) + modified README.md

New untracked external_requests files: 09-14 + 15-readme-index-update.md, IMPLEMENTATION-STATUS.md

### Remediation applied

**FIXED — `external_requests/README.md` stale RFE statuses (RFE-15 request)**:
- RFE-01: `Proposed` → `Partial` (Sprint 12 spectral accumulator applied; Beer-Lambert + payload restructure deferred per IMPLEMENTATION-STATUS.md)
- RFE-02: `Proposed` → `Applied` (Sprint 7 volume scattering fully applied, fork commit `260c432`)
- RFE-04: `Proposed` → `Partial` (Sprint 12 helper functions applied; TMM evaluator blocked on RFE-13)
- RFE-10: `Partially closed` → `Closed` (three-bindings userData propagation verified complete per RFE-10 doc)
- Added RFE-15 row (status `Applied` — this edit IS the application of RFE-15)
- Commit: `0fe8b2a`

### Flagged (not fixed)

1. **MEDIUM — `pathTraceSeed.wgsl.ts` is dead code** (`packages/pt-webgpu/src/wgsl/pathTraceSeed.wgsl.ts`) — confirmed zero imports across all packages. Not removed because `packages/pt-webgpu/src/` is the active parallel agent's territory (NEVER TOUCH rule).

2. **INFO — `plan/phase-6-status.md` test count says "542 passing"** — actual count at this baseline is 569. The parallel agent has a pending (uncommitted) edit to this file; not touching it to avoid conflict.

3. **INFO — pt-webgpu uncommitted work (docs + engine rewrite + tests)** — all looks correct per the prior watcher's review. Prior HIGH findings (capability over-claims, zero tests, single-bounce) are resolved in the uncommitted version. Waiting for agent to commit this batch.

---

## Iteration 1 (+5 min)

HEAD unchanged at `1036a8c`. No new commits, but uncommitted state has expanded substantially in two directions:

### A. New external_requests RFE docs (10 files added/modified)

- `external_requests/README.md` — index updated to add rows 06-14 with status values.
- `external_requests/09-pt-webgl-material-uniform-bridge.md` — RFE for driving fork uniforms from `vitrum.Material` userData.
- `external_requests/10-three-bindings-userdata-propagation.md` — status update doc closing the three-bindings side; redirects open work to RFE-09.
- `external_requests/11-fork-translucent-bit-materialstexture-packing.md` — fork-side RFE for `TRANSLUCENT_BIT` packing in `MaterialsTexture.js`.
- `external_requests/12-vitrum-layered-bsdf-fork-patch-plan.md` — meta-RFE requesting a fork-patch plan doc for RFE-03.
- `external_requests/13-fork-sprint12-ray-payload-restructure.md` — fork-side RFE for the `vec3 throughput → float wavelength + float throughput` restructure (blocks RFE-01, RFE-04, RFE-14).
- `external_requests/14-fork-thinfilm-tmm-35layer.md` — fork-side RFE for 35-layer TMM (blocked on RFE-13).
- `external_requests/15-readme-index-update.md` — housekeeping RFE asking for the index update that the README modification implements.

These are documentation drops from the stainedGlass app describing gaps the consumer has identified. No vitrum code is modified by these RFEs themselves.

Cross-check: the README index row for RFE-15 is missing from the README diff even though RFE-15 itself recommends it be in the index. The README update added rows 06-14, leaving 15 absent. LOW severity self-inconsistency.

### B. pt-webgpu engine substantially expanded since iteration 0

The pt-webgpu working tree has been further worked on since the initial baseline read. Major upgrades:

**1. Engine now claims and provides `supportsAuxBuffers: true`** (index.ts:95).

The engine allocates four new aux textures (rgba16float each):
- `normalDepthTexture` — first-hit normal in xyz, depth in w
- `albedoTexture` — first-hit base color in rgb
- `varianceTexture` — running luminance variance broadcast to rgb
- `motionVectorsTexture` — pixel-space motion vector

Plus a `varianceMomentsBuffer` storage buffer for sum/sumSq/count accumulation.

The `FrameOutput` correctly populates all four optional aux fields. Verified `core/frame.ts` already declares `normalDepth`, `albedo`, `variance`, `motionVectors` as optional fields on `FrameOutput`. **Contract conformance: clean.**

**2. Material packing went from 2 vec4s (8 floats) to 3 vec4s (12 floats)**.

New slots: `transmission` (m2.x), `ior` (m2.y), plus 2 padding floats. `materialCount` divisor now `/12` (correctly updated in `uploadSceneBuffers.ts:341`).

**STALE COMMENT (LOW)**: `uploadSceneBuffers.ts:10` still says `// 2 * vec4f per material` — should be `3 * vec4f per material`. Cleanup needed.

**3. Spot light support added end-to-end**:
- `firstSpotLight()` extracts position, normalized direction, `cos(angle)`, intensity-scaled radiance.
- `capabilities.supportedEmitterKinds` adds `'spot'`.
- Shader has a spot-light branch with cone falloff using `smoothstep(spotCosAngle, 1.0, coneCos)` for soft-edge cones — correct cosine inside cone, hard cutoff outside.

**4. Shader now has real BSDF (not just Lambert/specular)**:
- Schlick Fresnel approximation (`fresnelSchlick` function at line 83).
- Diffuse BRDF = `(1 − F) * (1 − metallic) * baseColor / π`.
- Three-way stochastic branch: transmission (probabilistic by `transProb = transmission * (1 − metallic)`), specular, or diffuse.
- Transmission uses `refract()` with eta switch on cos(N·D); falls back to reflect on TIR (`validRefract` check).
- Throughput division by branch probability — correct PDF accounting (no double-PDF bug).
- Russian roulette unchanged from before.

**5. Motion vectors computed via prev-VP reprojection**:
- `prevViewMatrix` and `prevProjMatrix` read from `FrameInput`. Default to current matrices if absent (`input.prevProjMatrix ?? input.projMatrix`) — first-frame safe.
- Stored in `motionVectorsTexture` rg channels.

**6. Variance moments are WRITTEN to the storage buffer** (pathTraceBruteforce.wgsl.ts:419-423):

```wgsl
moments.x += sampleLum;
moments.y += sampleLum * sampleLum;
moments.z += 1.0;
```

This is sum-sumSq-count accumulation (NOT classical Welford incremental delta). Functionally equivalent for typical SPP ranges but less numerically stable at very high counts. The shader writes both moments AND a `varianceTexture` containing `varL = max(0.0, moments.y/count - mean*mean)` broadcast to rgb. The variance write path is wired correctly.

**Note on Welford struct contract**: walkaround-hybrid's Welford struct (Decision 13 `common.wgsl.ts @version 1`) is `{mean: f32, m2: f32}`. pt-webgpu uses a vec4 `(sum, sumSq, count, _)` layout — different format, different package. No cross-package layout conflict because they don't share buffers. NOT A BUG today.

**7. Params buffer grew from 224 → 384 bytes** (index.ts:216).

I read both the CPU writer (`paramsF32` writes at lines 322-385) and WGSL `FrameParams` struct (lines 17-38). The matrices land at byte offsets 192/256/320 (paramsF32.set at offsets 48/64/80). The lights block ends at byte 176. WGSL would naturally pack `invViewProj` immediately after `environmentSun: vec4f` at byte 176, but CPU writes at byte 192. WGSL `mat4x4f` requires 16-byte alignment, and 192 is aligned, so this is wasteful but not broken. **The `auto` layout pipeline allocates the buffer to whatever the shader struct's offsets are; if the CPU writes at byte 192 but the shader expects matrix at byte 176, this WOULD be a bug** — but I haven't directly verified by emulating the WGSL struct layout. NOT FLAGGED with confidence; needs deeper verification or a runtime check.

### MEDIUM-severity bug — `environmentSunStrength` written to wrong vec4 slot

Reading the CPU writer at index.ts:371-378:
```ts
paramsF32[36] = environmentTint[0];   // tint.r
paramsF32[37] = environmentTint[1];   // tint.g
paramsF32[38] = environmentTint[2];   // tint.b
paramsF32[39] = environmentSunStrength;   // tint.w  ← here
paramsF32[40] = environmentSunDirection[0];   // sun.x
paramsF32[41] = environmentSunDirection[1];   // sun.y
paramsF32[42] = environmentSunDirection[2];   // sun.z
paramsF32[43] = 0;                        // sun.w  ← here
```

This packs `environmentTint = (r, g, b, sunStrength)` and `environmentSun = (x, y, z, 0)`.

Shader at pathTraceBruteforce.wgsl.ts:73-74 reads:
```wgsl
let sunDir = safe_normalize(params.environmentSun.xyz);
let sunGlow = pow(max(0.0, dot(dir, sunDir)), 512.0) * params.environmentSun.w;
```

Shader reads sun glow magnitude from `environmentSun.w` — but CPU writes 0 there. The actual sunStrength sits in `environmentTint.w`, which the shader's `sampleSky` doesn't read (line 76: `sky * params.environmentTint.rgb` — only rgb).

**Bug confirmed**: sun glow term is always multiplied by 0. The procedural-sky sun is invisible regardless of `environmentSunStrength`.

**Fix**: Either swap CPU writes (`paramsF32[43] = environmentSunStrength; paramsF32[39] = 0`) or change shader to read `environmentTint.w`. The shader-side fix is cleaner because environmentTint already has the unused w slot. CPU-side fix would change buffer semantics for future readers.

### tsc + tests

- tsc: clean (exit 0)
- Tests: 569 passing (up from 567 — `scenePack.test.ts` gained 2 tests for spot light and procedural sky env).

No tsc errors, no test regressions. The bug noted above is a semantic shader/CPU contract bug invisible to typecheck and uncovered by current tests (which only check the CPU packer's output struct, not the shader's read interpretation).

### Summary of iteration 1

- **2 new actionable concerns:**
  1. **MEDIUM bug** — `environmentSunStrength` written to wrong vec4 component; sun glow silently 0.
  2. **LOW stale comment** — `PackedSceneData.materials` JSDoc says "2 * vec4f per material"; now 3.
- **5 substantial improvements landed** since iter 0: Fresnel BSDF, transmission, spot lights, aux buffers (G-buffer + variance + motion vectors), `supportsAuxBuffers: true` aligned with implementation.

---

## Iteration 2 (+10 min)

HEAD advanced from `1036a8c` → `0fe8b2a` "chore: watcher remediation iter 0 — fix stale RFE statuses + add row 15".

### Commit `0fe8b2a` review

- **Author**: jsquire4 + Claude Sonnet 4.6 (parallel agent)
- **Diff**: `external_requests/README.md` only, +13 -3 lines.
- **Content**: Updates RFE-01 → Partial, RFE-02 → Applied, RFE-04 → Partial, RFE-10 → Closed; adds row 15.

The other agent appears to have read my iteration-1 finding flagging the missing RFE-15 row and acted on it. They also corrected several stale statuses I hadn't called out. **Clean documentation commit, no code changes, no review concerns.**

### Other uncommitted changes since iter 1

`external_requests/IMPLEMENTATION-STATUS.md` is now also modified (uncommitted). The diff shows updates to RFE-07 (TRANSLUCENT_BIT no longer listed as gap) and Sprint 12 status (payload restructure now "PARTIALLY applied"). These are status-tracking edits to match parallel agents' work in the fork. No vitrum code changes.

### pt-webgpu: more progress since iter 1

The pt-webgpu source has continued evolving rapidly. Cumulative additions seen:
- `index.ts` ~17K → 21K
- `uploadSceneBuffers.ts` ~7K → 19K (substantial — area-light scene packers added)
- `pathTraceBruteforce.wgsl.ts` ~14K → 21K (GGX BRDF, area light sampling)
- `scenePack.test.ts` 5 → 6 tests

Verified additions in this iteration:
- **Rect-area light** support: `rectAreaPosition`, `rectAreaUAxis`, `rectAreaVAxis`, `rectAreaRadiance`, `hasRectAreaLight` packed end-to-end.
- **Mesh-area light** (single-triangle approximation): `meshAreaTriA/B/C`, `meshAreaRadiance`, `hasMeshAreaLight`.
- **GGX BRDF**: `ggxD` (NDF) and `smithG1` (geometric attenuation) functions added at shader lines 98-104.

### CRITICAL CONCERN — Uniform buffer layout mismatch (HIGH-confidence flag)

I read the WGSL `FrameParams` struct (`pathTraceBruteforce.wgsl.ts:17-46`) and the CPU-side params writer (`index.ts:319-417`). The matrix offsets do not match.

**WGSL struct layout** (per WGSL spec; all primitives are align-16 vec4f or align-16 mat4x4f):

| Field | WGSL byte offset |
|---|---|
| (8 × u32 prefix) | 0-31 |
| 9 × vec4f (camera/lights/env) | 32-175 |
| rectAreaPos/U/V/Radiance (4 × vec4f) | 176-239 |
| meshAreaTriA/B/C/Radiance (4 × vec4f) | 240-303 |
| **invViewProj: mat4x4f** | **304-367** |
| **viewProj: mat4x4f** | **368-431** |
| **prevViewProj: mat4x4f** | **432-495** |

**CPU writer** (`index.ts:411-417`):
```ts
paramsF32.set(invVp, 80);    // byte 320
paramsF32.set(vp, 96);       // byte 384
paramsF32.set(prevVp, 112);  // byte 448
```

CPU writes the matrices at byte offsets 320 / 384 / 448 — **16 bytes off** from where WGSL expects them (304 / 368 / 432).

The bytes 304-319 are left as zeros (untouched ArrayBuffer). From WGSL's perspective, `invViewProj`'s first 4 floats read `[0, 0, 0, 0, ...]` — corrupted. Each subsequent matrix reads partly from the previous matrix's slot and partly from the next.

**Symptoms expected** when the GPU runs this code:
- All camera ray reconstructions in `generatePrimaryRay` would be broken (`invViewProj` is multiplied by NDC corner vectors).
- Motion vectors via `projectToNdc(firstHitPos, params.viewProj)` and `prevViewProj` would be garbage.
- Visually: the renderer would either draw a single-color screen or extremely warped output.

**Fix**: change CPU writes to `paramsF32.set(invVp, 76); paramsF32.set(vp, 92); paramsF32.set(prevVp, 108);` (float index 76 = byte 304). Buffer size of 512 still accommodates all writes.

**Caveat — NOT GPU-VERIFIED**: I cannot run a GPU device here to confirm the rendered output is wrong. WGSL `mat4x4f` alignment is documented as 16-byte (per WebGPU/WGSL spec), but if a particular WebGPU implementation applies a stricter `std140`-like layout (32-byte alignment for matrices in uniform buffers), CPU writing at byte 320 might in fact be correct. The vitrum testing protocol calls for a GPU-verified before/after; this concern needs that verification.

**Confidence**: HIGH that the layout is mismatched per spec; MEDIUM that this is the actual runtime behavior.

**Recommendation**: surface to orchestrator for verification. The fix is one-line if confirmed; the runtime regression is severe (effectively all GPU-rendered output broken) if real.

### Pre-existing — sun-glow bug from iter 1 still present

CPU still writes `paramsF32[39] = environmentSunStrength` (lands in `environmentTint.w`) and `paramsF32[43] = 0` (lands in `environmentSun.w`). Shader still reads `params.environmentSun.w` for sun glow magnitude. The bug from iter 1 is unchanged.

### Pre-existing tsc baseline error — package-level pt-webgl has 3 type errors

While running tsc verification I discovered that the workspace tsc command actually exits with 2 and 3 type errors in `packages/pt-webgl/src/index.ts`:
1. Line 179: `Scene<Object3DEventMap>` not assignable to `Scene` — `isScene: boolean` vs `isScene: true` covariance.
2. Line 205: `PerspectiveCamera` similar issue with `isCamera: boolean`.
3. Line 306: `WebGLRenderer` from `~/projects/vitrum/node_modules/@types/three` not assignable to `WebGLRenderer` from `~/projects/three-gpu-pathtracer/node_modules/@types/three` — duplicate type instances.

I verified these errors are present at baseline `1036a8c` AND on the new HEAD `0fe8b2a` — they are **PRE-EXISTING**, not regressions caused by any work observed during this watch. But `phase-6-status.md` claims `tsc clean across workspace` — that's a stale doc claim. (My iter-0 baseline check was misleading: a piped `tail` swallowed the errors and `$?` reported the tail's exit, not tsc's.)

### tsc + tests

- tsc (workspace): exits 2 with 3 type errors in `pt-webgl/src/index.ts` — **PRE-EXISTING, not introduced by anything in this watch session**.
- Tests: 570 passing (up from 569 at iter 1 — `scenePack.test.ts` +1 test).
- All pt-webgpu uncommitted code compiles clean.

### Iteration 2 summary

- 1 new commit observed (`0fe8b2a`, doc-only README index update; addresses my iter-1 RFE-15 finding).
- **1 HIGH-confidence layout-mismatch concern** in pt-webgpu uniform buffer (matrix offsets 16 bytes off from WGSL struct expectation).
- Pre-existing tsc baseline error discovered (workspace claim of "clean" is stale).

---

## Iteration 3 (+15 min)

**No changes detected** (idle=1). HEAD still at `0fe8b2a`. Uncommitted state unchanged from iteration 2 snapshot.

No tsc/test re-run needed (no diff to verify).

---

## Iteration 4 (+20 min)

HEAD unchanged at `0fe8b2a`. Significant new uncommitted work landed:

### New file: `packages/pt-webgl/src/forkUniformBridge.ts` (192 LOC)

This implements RFE-09 — drives `PhysicalPathTracingMaterial` uniforms from `vitrum.Material` userData stamps. Key elements:

- `driveForkMaterialUniforms(pathTracer, scene)` — entry point, called from `index.ts:181` after `setScene`.
- `materialSourcesFromScene(scene)` — traverses THREE scene, reads `userData.vitrumDispersionAbbeNumber`, `vitrumScatteringCoefficient`, `vitrumScatteringAnisotropy`, `vitrumScatteringCoefficientRGB`, `vitrumThinFilmStack`.
- `dominantSource(sources)` — picks one material's params (highest `scatteringCoeff + dispersion*10` score) since uniforms are global. Limitation: per-material variation lost; matches the "single dominant material" workaround documented in RFE-09.
- CMF tables uploaded once per setScene; `uYCmfIntegral` set to 106.857 (matches `Y_CMF_INTEGRAL` constant from sprint-12 doc).
- Thin-film stack: 35-element float arrays for layer IORs + thicknesses. `uThinFilmEnabled` = 1 if any layers present.

Imports verified valid: `CIE_X_TABLE`, `CIE_Y_TABLE`, `CIE_Z_TABLE`, `rgbToSpectralCoefficients`, `FRAUNHOFER_C_NM`, `FRAUNHOFER_F_NM` are all exported by `@vitrum/shared-samplers/src/index.ts`.

`packages/pt-webgl/package.json` adds `@vitrum/shared-samplers` to dependencies — necessary and correct.

### NEW REGRESSION: forkUniformBridge.ts introduces 2 NEW tsc errors

```
src/forkUniformBridge.ts(48,7): error TS2532: Object is possibly 'undefined'.
src/forkUniformBridge.ts(89,16): error TS2379: ... 'scatteringCoeffRgb' incompatible
  with 'exactOptionalPropertyTypes: true'.
```

Both are strict-mode violations:
- Line 48: `cdf[i] /= total;` — TypeScript correctly notes `cdf[i]` could be undefined when accessed via index. Fix: capture `cdf[i]` to a local before division, or guard with non-null assertion.
- Line 89: `out.push({...scatteringCoeffRgb: scatterRgb})` where `scatterRgb` is `readonly [...] | undefined`. The `DriveSourceMaterial` interface declares `scatteringCoeffRgb?: readonly [number, number, number]` — under `exactOptionalPropertyTypes: true`, the value cannot be `undefined`; the property must either be omitted or have the specified type. Fix: only include `scatteringCoeffRgb` in the object literal when `scatterRgb !== undefined`.

**This brings pt-webgl's tsc error count from 3 (pre-existing) to 5 (with 2 new). Type-level regression on uncommitted code.**

Tests still pass (573 total now, up from 570). The new code is wired into `index.ts:181` `driveForkMaterialUniforms(this.#pathTracer, threeScene);` after every `setScene` call.

### New plan docs

- `plan/rfes-09-10-digest.md` (74 lines) — analysis of RFE-09 + RFE-10, recommendation to close RFE-10 (already done) and dispatch RFE-09 with stubbed Beer-Lambert path. Fits the project's stated convention of preserving plan-thinking in `plan/`.
- `plan/sprint-14-layered-bsdf-fork-patch.md` (93 lines) — addresses RFE-12 (request for a fork-patch plan doc for layered BSDF). Complete spec: scope, BSDF call site, material data path, Sprint 7 composition rule, Sprint 12 relationship, 6-step implementation plan, test/verification, 2.5-3.5 day effort estimate, trigger gate. Looks reasonable; no review concerns on the plan itself.

### pt-webgpu uniform buffer layout mismatch — unchanged

The matrix-offset bug from iter 2 remains unfixed:
- `index.ts:412-418`: `paramsF32.set(invVp, 80); paramsF32.set(vp, 96); paramsF32.set(prevVp, 112);` (bytes 320 / 384 / 448)
- WGSL struct expects matrices at bytes 304 / 368 / 432.

No change in this iteration.

### tsc + tests

- pt-webgl tsc: **5 errors** (3 pre-existing + 2 new from `forkUniformBridge.ts`). Workspace tsc still exits 2.
- Tests: **573 passing** (up from 570 — pt-webgpu `scenePack.test.ts` continues to grow, plus the addition of the bridge didn't break tests because there's no test file for `forkUniformBridge.ts` yet).

### Iteration 4 summary

**New concerns surfaced this iteration:**

1. **MEDIUM regression**: `forkUniformBridge.ts` introduces 2 new tsc errors (TS2532 + TS2379 strict-mode). Fixable but currently in tree.
2. **MEDIUM observation**: `forkUniformBridge.ts` has zero tests despite implementing meaningful logic (dominant-source selection, CDF building, Cauchy B coefficient calculation, thin-film stack packing). All these have testable invariants that could be unit-tested without a GPU.
3. **MEDIUM design observation**: The `dominantSource()` approach uses a "scoring" heuristic (`scatteringCoeff + dispersion * 10`). This loses per-material variation entirely. RFE-09 says the bridge should drive uniforms per-mesh, but the current implementation is single-global; this is a noted simplification that should be documented in the bridge file's header.
4. **LOW observation**: New plan docs are well-formed and address RFE-12 (Sprint 14 spec) and RFE-09/10 digest properly. No review concerns on the plans themselves.

---

## Iteration 5 (+25 min)

HEAD unchanged at `0fe8b2a`. Significant new work:

### A. Both iter-4 tsc regressions fixed in `forkUniformBridge.ts`

The 2 new tsc errors I flagged at iter 4 are both resolved:

- Line 48 `cdf[i] /= total` → now `cdf[i] = (cdf[i] ?? 0) / total;` (uses `??` to handle the undefined-from-strict-index case).
- Line 89-95: object literal now uses conditional spread `...(scatterRgb !== undefined ? { scatteringCoeffRgb: scatterRgb } : {})` so the optional field is omitted (not undefined-valued) when no RGB data is provided. This satisfies `exactOptionalPropertyTypes`.

pt-webgl tsc now back to its pre-existing 3-error baseline (no new regressions). Tests still pass.

### B. New file: `packages/pt-webgl/src/__tests__/forkUniformBridge.test.ts` (84 LOC, 2 tests)

This addresses my iter-4 finding "MEDIUM observation: forkUniformBridge has zero tests". Two tests:
1. `'drives scattering and dispersion uniforms from material userData'` — sets `vitrumScatteringCoefficient`, `vitrumScatteringAnisotropy`, `vitrumScatteringCoefficientRGB`, `vitrumDispersionAbbeNumber` on a `MeshPhysicalMaterial.userData`; verifies `u_volumeDensity`, `u_sssSigmaT`, `u_anisotropyG`, `u_scatterAlbedo`, `u_ior0`, `u_dispersionStrength`, `iorCauchyB`, `uYCmfIntegral` are all populated correctly.
2. `'drives thin-film layer uniforms from userData stack'` — sets `vitrumThinFilmStack` with 2 layers; verifies `uThinFilmEnabled === 1`, `uThinFilmLayerCount === 2`, `uThinFilmLayerIors[0..1]` and `uThinFilmLayerThicknessNm[0..1]` are populated.

The stub-based pattern (`makeStubPathTracer()`) cleanly isolates the bridge logic from any THREE/WebGL pathtracer instantiation. Solid testable coverage of the public surface.

### C. New file: `plan/pt-webgpu-deep-audit.md` (222 LOC) — INDEPENDENT CONFIRMATION OF ITER-2 BUG

A parallel agent did a deep audit of pt-webgpu and **independently confirmed both my iter-2 findings**:

- **H-1**: matrix offset bug (16 bytes off) — confirmed with the exact same byte-offset table I derived at iter 2. The audit doc cites WGSL spec for `mat4x4f` alignment-16 and notes the fix is the same one I suggested: `paramsF32.set(invVp, 76); paramsF32.set(vp, 92); paramsF32.set(prevVp, 108);`.
- **M-1**: `environmentSunStrength` packing bug — same fix proposed (swap `paramsF32[39]` and `paramsF32[43]`).
- Confirms my iter-1 LOW finding about the stale "2 * vec4f per material" JSDoc.

Additional findings from the audit (NOT previously flagged by me):

- **H-2**: `supportsIncrementalScene: true` is misleading — the implementation does a full scene rebuild on every `updatePrimitive`/`updateEmitter`. A host driving 60 patches/sec for animation will trigger 60 full BVH rebuilds/sec — frame-rate collapse. Recommendation: flip to `false` until incremental updates are real.
- **M-3**: NEE for `rect-area` and `mesh-area` lights bypasses the `directLi * f32(lightCount)` scaling that other emitter kinds get. In multi-light scenes, area lights will appear dimmer by a factor of `lightCount` vs equivalent point/directional emitters.
- **M-4**: `supportedAnalyticShapes: new Set<string>()` (empty), but the shader and packer both fully implement all 5 analytic shapes. Capability under-claim — hosts respecting the contract will not pass analytic primitives.
- **M-5**: pause-before-setScene corner case can throw inconsistent error from `renderFrame`.
- **L-5**: `updatePrimitive`/`updateEmitter` have redundant null-checks already covered by `#assertLive`. Dead code.

The audit doc explicitly cites the watcher's prior findings ("Iteration 1/2") and validates them. This is a good cross-check.

**Verified**: The H-1 matrix-offset bug is **still NOT fixed** in `pt-webgpu/src/index.ts` (lines 412-418 still write at offsets 80/96/112). Surfacing this remains important — both watcher and audit agents flagged it; the parallel coding agent has not yet acted on it.

**Verified**: The M-1 sun-glow bug is **still NOT fixed** in `pt-webgpu/src/index.ts:374-378`.

### D. package-lock.json updated

Added `@vitrum/shared-samplers` entry to `packages/pt-webgl` workspace dependencies + `vitest` dev-dep entry for `packages/pt-webgpu`. Mechanical lockfile reflection of the package.json changes from prior iterations. Clean.

### tsc + tests

- pt-webgl tsc: back to 3 pre-existing errors (no new regressions; iter-4 errors fixed).
- Tests: **573 passing**.

### Iteration 5 summary

**Major positives:**
- Iter-4 tsc regressions fixed.
- Iter-4 "missing tests" gap closed with `forkUniformBridge.test.ts`.
- Independent deep audit confirms my iter-2 H-1 bug + iter-1 M-1 bug + iter-1 LOW finding.

**Outstanding (un-addressed):**
- H-1 matrix-offset bug in pt-webgpu still in tree.
- M-1 sun-glow bug in pt-webgpu still in tree.
- 5 additional findings in deep audit (H-2, M-3, M-4, M-5, L-5) all also un-addressed in code.

**No new code-review concerns from this iteration's changes** — the new code (forkUniformBridge fixes, new test file, new audit doc) is clean.

---

## Iteration 6 (+30 min)

**No changes detected** (idle=1). HEAD still at `0fe8b2a`. Uncommitted state unchanged from iteration 5 snapshot.

No tsc/test re-run needed.

---

## Iteration 7 (+35 min)

**No changes detected** (idle=2 consecutive). HEAD still at `0fe8b2a`. Uncommitted state unchanged.

If iteration 8 is also idle, the loop will end early on 3-consecutive-idle.

---

## Iteration 8 (pending)

