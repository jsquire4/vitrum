# pt-webgpu Deep Bug Audit — 2026-05-10

## Summary

- Findings: HIGH=2, MEDIUM=5, LOW=5
- Confirmed pre-existing (watcher Iteration 1/2): `environmentSunStrength` packing bug (MEDIUM, confirmed), matrix offset bug (HIGH, confirmed with exact offsets), stale JSDoc comment (LOW, confirmed)
- Test count snapshot: 570 passing across workspace (pt-webgpu contributes 11 tests); tsc exits 2 with 3 pre-existing errors in `pt-webgl/src/index.ts` (not pt-webgpu regressions)

## Status update — 2026-05-19: all findings closed

Every HIGH + MEDIUM + LOW finding from this audit has been re-verified
by direct file read and is now closed. Per-finding map:

- **H-1 (matrix offsets 16 bytes off)** — FIXED. The FrameParams struct
  was refactored (W4-A4 split into `pathTrace/*` modules); the regression
  test at `packages/pt-webgpu/src/__tests__/wgslContract.test.ts:61`
  (commit `105cbed`) walks the WGSL struct under spec alignment rules and
  asserts the CPU writer's matrix offsets (144/208/272 bytes ⇔ f32 indices
  36/52/68) match.
- **H-2 (`supportsIncrementalScene: true`)** — FIXED. Now reads `false`
  at `packages/pt-webgpu/src/index.ts:109` with the honest "flip when
  real incremental patching lands" comment.
- **M-1 (`environmentSunStrength` wrong vec4 component)** — FIXED.
  `paramsF32[35] = sb.environmentSunStrength` lands at byte offset 140 =
  `environmentSun.w`, matching the shader's read at
  `pathTrace/connect.wgsl.ts:29,68,129`.
- **M-2 (cosine-hemisphere PDF division)** — NOT A BUG per the audit's
  own analysis (the cancellation `(BRDF/π)·NdotL/(NdotL/π) = BRDF` is
  correct; no code change ever needed).
- **M-3 (rect-area / mesh-area writes to `*radiance` bypassing
  `directLi`)** — FIXED. The `pathTrace/kernel.wgsl.ts` rewrite inlines
  the rect-area and mesh-area sampling at lines 291-359, writing
  `directLi = ...` in the same accumulator the directional/point/spot
  paths use. Line 387 then applies the `* f32(lightCount)` MIS scaling
  uniformly.
- **M-4 (`supportedAnalyticShapes` empty set)** — FIXED. Now reads from
  `PT_WEBGPU_ANALYTIC_SHAPES.slice(1)`
  (sphere/box/capsule/cylinder/h-channel-came). The 2026-05-18 tightening
  also flipped the type from `ReadonlySet<string>` to
  `ReadonlySet<AnalyticShape>`.
- **M-5 (pause-before-setScene)** — NOT A CRASH BUG (audit's own
  classification). The inconsistency is documented in `#assertLive` —
  no action.
- **L-1 (stale "2 * vec4f per material" JSDoc)** — FIXED. The comment
  at `uploadSceneBuffers.ts:31` now uses the `MATERIAL_VEC4_STRIDE`
  symbolic reference.
- **L-2 (dead `pathTraceSeed.wgsl.ts`)** — FIXED. File deleted.
- **L-3 (varianceMomentsBuffer sizing)** — NOT A BUG (audit's own
  classification).
- **L-4 (instanced-mesh `instances[0]` for mesh-area light)** —
  documented limitation (single-triangle approximation is already lossy).
- **L-5 (redundant `#scene == null` guards)** — FIXED. The duplicate
  guards in `updatePrimitive`/`updateEmitter` are gone; the non-null
  assertion documents the invariant.

**Glossy BSDF sampling/PDF mismatch** (Item 14 from the 2026-05-11 sweep,
frequently relisted) — FIXED via commit `a7dd51a`:
`packages/pt-webgpu/src/wgsl/pathTrace/bsdf.wgsl.ts:124` ships
`sampleGgxVndfTangent` (Heitz 2018 Algorithm 1); `glossyReflectionSample`
calls it directly so sampling and PDF use the same distribution.

The package remains pre-alpha — see `packages/pt-webgpu/README.md` for
the explicit pre-alpha boundary. The remaining gaps are NOT code bugs
found by this audit; they are the absence of GPU-verified visual
reference captures, public API stability commitments, and feature
completeness relative to the WebGL fork (spectral parity, full
hero-wavelength MIS, denoiser integration).

---

## HIGH severity

### H-1 — Uniform buffer matrix offset 16 bytes off: all camera ray reconstruction is corrupted

**File**: `packages/pt-webgpu/src/index.ts:412-418`

**Issue**: The CPU writer sets matrices at `paramsF32.set(invVp, 80)` / `paramsF32.set(vp, 96)` / `paramsF32.set(prevVp, 112)`, corresponding to byte offsets 320 / 384 / 448. The WGSL `FrameParams` struct (pathTraceBruteforce.wgsl.ts:17-46) places `invViewProj` at byte offset 304, `viewProj` at 368, `prevViewProj` at 432. Derivation:

| Cumulative field layout | Byte offset |
|---|---|
| 8 × u32 prefix | 0–31 |
| 9 × vec4f (camera through environmentSun) | 32–175 |
| 4 × vec4f (rectArea pos/U/V/radiance) | 176–239 |
| 4 × vec4f (meshArea A/B/C/radiance) | 240–303 |
| **invViewProj: mat4x4f** (align 16) | **304** |
| viewProj: mat4x4f | 368 |
| prevViewProj: mat4x4f | 432 |

CPU is writing `invViewProj` at byte 320 (16 bytes late). Bytes 304–319 are left as zeros from the `new ArrayBuffer(512)` initialization. From the shader's perspective, `invViewProj`'s first column is `[0, 0, 0, 0]`, making `generatePrimaryRay` multiply NDC coordinates by a singular matrix — all rays emerge with undefined direction (NaN or degenerate). The same 16-byte shift corrupts `viewProj` (motion vectors) and `prevViewProj` (reprojection).

**Why HIGH**: Every rendered pixel's ray direction is derived from `invViewProj`. This bug renders all GPU output incorrect — either visually blank, all-one-color, or wildly wrong geometry. It is not GPU-testable in this repo (no GPU in CI), but the struct layout mismatch is unambiguous from spec.

**Suggested fix**: Change `paramsF32.set(invVp, 80)` → `paramsF32.set(invVp, 76)`, `paramsF32.set(vp, 96)` → `paramsF32.set(vp, 92)`, `paramsF32.set(prevVp, 112)` → `paramsF32.set(prevVp, 108)`. (Float indices 76/92/108 = byte offsets 304/368/432.) The 512-byte buffer accommodates: prevViewProj ends at byte 432 + 64 = 496 < 512.

**Note**: The watcher (Iteration 2) flagged this with a caveat about `std140` 32-byte matrix alignment in some implementations. WGSL spec (https://www.w3.org/TR/WGSL/#alignment-and-size) specifies `mat4x4f` has alignment 16, not 32. std140 32-byte-align-for-matrices is a GLSL rule that does NOT apply to WGSL. The bug is real at 16-byte misalignment.

---

### H-2 — `supportsIncrementalScene: true` causes unbounded full BVH rebuilds in tight-update loops

**File**: `packages/pt-webgpu/src/index.ts:93`, `252-269`

**Issue**: `capabilities.supportsIncrementalScene` is `true`. The Engine contract (`@vitrum/core`) documents that `supportsIncrementalScene = true` allows hosts to call `updatePrimitive` / `updateEmitter` without triggering a full rebuild. The actual implementation at lines 252-269 calls `this.setScene(nextScene)` inside both update methods — which calls `buildPackedScene` (full triangle flatten + BVH rebuild) + `uploadPackedScene` (all GPU buffer allocations). A host that sees `supportsIncrementalScene: true` and drives 60 `updatePrimitive` calls/second (e.g., animating a light position slider) will trigger 60 full BVH rebuilds/second with full GPU buffer reallocation each time — not a performance degradation; a frame-rate collapse.

**Why HIGH**: The contract semantic is clear and has direct, severe user-visible performance impact when relied upon. It is not merely misleading documentation — it is an incorrect boolean that will cause a host to make architectural decisions that break their application.

**Suggested fix**: Change `supportsIncrementalScene: false`. Once true zero-copy incremental updates are implemented (id-targeted patches without BVH rebuild), flip it back to `true`. The README Known Limitations section also needs to be updated to say "Incremental scene updates are implemented as full-rebuild fallbacks; `supportsIncrementalScene` is `false` until real incremental patching is added."

---

## MEDIUM severity

### M-1 — `environmentSunStrength` written to wrong vec4 component (confirmed pre-existing)

**File**: `packages/pt-webgpu/src/index.ts:375,379`; `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:86`

**Issue**: CPU writes `paramsF32[39] = environmentSunStrength` (landing at `environmentTint.w`) and `paramsF32[43] = 0` (landing at `environmentSun.w`). The shader reads `params.environmentSun.w` for sun glow magnitude (line 86: `pow(...) * params.environmentSun.w`). Since `environmentSun.w` is always 0, the procedural sky has no sun glow regardless of `environmentSunStrength`. Confirmed still present.

**Suggested fix**: Swap CPU writes: `paramsF32[43] = this.#sceneBuffers.environmentSunStrength` and `paramsF32[39] = 0`. This is a one-line fix and the cleanest approach since the shader is already reading the right field — only the CPU is packing to the wrong slot.

---

### M-2 — Cosine-weighted hemisphere sample PDF not divided out for diffuse throughput

**File**: `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:873-878`

**Issue**: In the diffuse branch:
```wgsl
ray.direction = cosineHemisphereSample(&rng, normal);
let kd = (vec3f(1.0) - fresnel) * (1.0 - metallic);
throughput = throughput * (kd * baseColor) / max(diffProb, 1e-4);
```
The throughput update should be `throughput * (kd * baseColor / pi) * pi / diffProb` because:
- `cosineHemisphereSample` samples with PDF `NdotL / pi`
- The BRDF for the diffuse lobe is `kd * baseColor / pi`
- The path weight update is `BRDF * NdotL / PDF = (kd * baseColor / pi) * NdotL / (NdotL / pi) = kd * baseColor`

This is actually correct — the `NdotL` and the `pi` from BRDF and from the cosine-weighted PDF cancel. So the `kd * baseColor` factor is correct. The `diffProb` division handles the stochastic branch selection. **This is NOT a bug.** The throughput update for cosine-weighted hemisphere is correct. (Noted here because it is subtle — not flagged as a finding.)

---

### M-3 — NEE missing PDF division when `sampleRectAreaLight` writes directly to `*radiance` instead of `directLi`

**File**: `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:835-838`

**Issue**: The MIS light selection picks one light uniformly at random and accumulates `directLi * f32(lightCount)` (line 844). For directional, point, and spot lights this is evaluated into `directLi` directly. For rect-area and mesh-area lights, `sampleRectAreaLight` and `sampleMeshAreaLight` write directly to `*radiance` (line 659, 702), bypassing the `directLi` accumulator. As a result, when one of these is `picked`, `directLi` remains `vec3f(0.0)` and the `directLi * f32(lightCount)` scaling on line 844 is never applied — the rect-area and mesh-area contributions do not get the light-count scaling factor. They are thus underweighted by `lightCount` relative to directional/point/spot.

For scenes with only a rect-area light and no other lights, `lightCount = 1` so the scaling is harmless. The bug activates when rect-area or mesh-area coexists with one or more directional/point/spot lights.

**Why MEDIUM**: Incorrect radiance for mixed-light scenes. Rect-area and mesh-area lights will appear darker by a factor of `lightCount` vs equivalent directional/point/spot intensity.

**Suggested fix**: Change `sampleRectAreaLight` and `sampleMeshAreaLight` to return a `vec3f` contribution (not write to `*radiance`), then accumulate that into `directLi` like the other lights, and apply the same `* f32(lightCount)` scaling.

---

### M-4 — Analytic primitive `supportedAnalyticShapes` claims all 5 shapes but shader intersects them in local space without the `supportedAnalyticShapes` Set being updated

**File**: `packages/pt-webgpu/src/index.ts:99`

**Issue**: Capabilities returns `supportedAnalyticShapes: new Set<string>()` (line 99) — empty set. But the shader (`pathTraceBruteforce.wgsl.ts:470-479`) and CPU packer (`uploadSceneBuffers.ts:145-160`) both implement all five analytic shapes (sphere, box, capsule, cylinder, h-channel-came) end-to-end. The `buildPackedScene` function at line 419-445 processes `analytic` primitives, assigns shape IDs, packs transform matrices, and the shader intersects them. The capabilities object advertises `supportedAnalyticShapes: new Set<string>()` — empty — which will cause hosts to skip analytic primitives and convert them to fallback meshes (if provided), wasting the implementation.

**Why MEDIUM**: Correct implementation silently disabled by a stale/wrong capabilities declaration. Any host respecting the capability contract will not pass analytic primitives even though they would be correctly rendered.

**Suggested fix**: Change to `supportedAnalyticShapes: new Set<string>(['sphere', 'box', 'capsule', 'cylinder', 'h-channel-came'])`.

---

### M-5 — `paused` state `renderFrame` returns `#accumTexture` before first frame (null on first pause)

**File**: `packages/pt-webgpu/src/index.ts:275-285`

**Issue**: `renderFrame` when paused returns `primaryRadiance: this.#accumTexture`. `#accumTexture` is `null` until the first `renderFrame` call that runs the active path (line 293 calls `#ensureAccumResources`). If a host calls `setScene()` then `pause()` then `renderFrame()`, the paused-return branch (line 275) fires before `#ensureAccumResources` is ever called, returning `{ primaryRadiance: null, samplesAccumulated: 0, ... }`. The `samplesAccumulated: 0` does satisfy the skip-frame contract — but `normalDepth`, `albedo`, `variance`, `motionVectors` also return as `undefined` (from `null ?? undefined`), which is consistent. The real issue is that `pause()` can be called before `setScene()` — `pause()` at line 477 does NOT call `#assertLive`, so an engine with `state = 'ready'` and `#scene = null` will accept `pause()` and then `renderFrame()` will hit `#assertLive` (line 108 checks `#scene == null`) and throw. This is inconsistent: `pause()` succeeds but the subsequent `renderFrame()` throws with a message that implies setScene() was not called, even though the user may have only "forgotten" that pause was set after a dispose-remount cycle.

**Suggested fix**: Not a crash bug. Low priority to fix but document the contract: `pause()` before `setScene()` is a no-op safe sequence; `renderFrame()` in that state will still throw due to `#assertLive`. Either add `#assertLive` to `pause()` for consistency, or document the intentional pattern.

---

## LOW severity

### L-1 — Stale JSDoc: `PackedSceneData.materials` says "2 * vec4f per material" (is 3)

**File**: `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts:11`

**Issue**: `// 2 * vec4f per material` — actual layout is 3 × vec4f (12 floats) as confirmed by `materialToPackedVec4s` (lines 101-136) and `uploadPackedScene`'s `materialCount: Math.floor(packed.materials.length / 12)` divisor. Confirmed still present.

**Suggested fix**: Change comment to `// 3 * vec4f per material: [baseColor.rgb, roughness], [emissive.rgb, metallic], [transmission, ior, 0, 0]`.

---

### L-2 — Dead file: `pathTraceSeed.wgsl.ts`

**File**: `packages/pt-webgpu/src/wgsl/pathTraceSeed.wgsl.ts`

**Issue**: Zero imports anywhere. Exports `PT_WEBGPU_SEED_WGSL` which nothing uses. Confirmed still present.

**Suggested fix**: Delete the file (or move to `__tests__/` as a test fixture if there are plans to use it for validation).

---

### L-3 — `varianceMomentsBuffer` allocated with `width * height * 16` bytes but `vec4f` is correct — confirmed clean

**File**: `packages/pt-webgpu/src/index.ts:198-202`

`varianceMomentsBuffer` size is `Math.max(16, targetByteSize)` where `targetByteSize = width * height * 16`. The shader reads/writes `varianceMomentsBuffer[pixelIndex]` as `vec4f` (16 bytes each). `pixelIndex = gid.y * params.width + gid.x`. Max index is `(height-1) * width + (width-1) = width * height - 1`. Buffer contains `width * height` vec4f entries = `width * height * 16` bytes. Size matches. **No bug.**

---

### L-4 — `instanced-mesh` transform application picks `instances[0]` for mesh-area light

**File**: `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts:335`

**Issue**: `firstMeshAreaLight` applies only `primitive.instances[0]` as the transform when the referenced primitive is an `instanced-mesh`. If the mesh-area emitter references an instanced-mesh with multiple instances, only the first instance's transform is used to compute the representative triangle. The other instances' triangles are not sampled. This is a known simplification (single-triangle approximation is already a simplification), but the choice of `instances[0]` is not documented.

**Impact**: LOW — mesh-area light is a prototype feature and the single-triangle approximation is already lossy. Not a correctness crash.

---

### L-5 — `#assertLive` redundantly checks `#scene == null` in `updatePrimitive`/`updateEmitter`

**File**: `packages/pt-webgpu/src/index.ts:252-269`

Both `updatePrimitive` and `updateEmitter` call `#assertLive` (which already checks `#scene == null` at line 109 and throws), then immediately check `if (scene == null)` again (lines 255, 265) and throw a second, different error message. The second check can never be reached because `#assertLive` would have already thrown. Dead code.

**Suggested fix**: Remove the redundant `if (scene == null)` guards in `updatePrimitive` and `updateEmitter`.

---

## Per-category notes

### A. CPU struct packing vs shader struct reading
- H-1 found: matrix offsets 16 bytes off (invViewProj/viewProj/prevViewProj).
- M-1 confirmed: `environmentSunStrength` at wrong vec4 slot.
- All scalar/vec4 fields before the matrices were verified individually and are correct. Fields [0]-[7] (u32s), [8]-[75] (all vec4f light/env fields) match WGSL struct members exactly.

### B. Bind group layout vs shader binding mismatch
- 18 bindings (0–17) in both the CPU bind group (index.ts:424-443) and the shader's `@group(0) @binding(N)` declarations. Order and types match. Textures are `texture_storage_2d<rgba16float, write>` on both sides. Storage buffers: shader uses `read` for positions/indices/etc. and `read_write` for accumBuffer/varianceMomentsBuffer, matching CPU `STORAGE` usage. No mismatch found.

### C. Workgroup size vs dispatch count
- Shader: `@workgroup_size(8, 8, 1)`. Dispatch: `Math.ceil(width / 8)` × `Math.ceil(height / 8)` (index.ts:450-454). `Math.ceil` is correct — edge pixels are covered. Guard at shader line 707 (`if gid.x >= params.width || gid.y >= params.height`) handles the overshoot. Clean.

### D. WGSL math correctness
- Russian roulette: throughput is divided by `survival` AFTER the random test (line 885: `throughput = throughput / survival`). Correct ordering.
- Fresnel: `fresnelSchlick(cosThetaO, f0)` where `cosThetaO = dot(normal, wo)`. For the direct-BRDF call, `evaluateBrdf` uses `vDotH = dot(wo, h)` for Fresnel (line 124). Both formulations are internally consistent (pixel-level Fresnel uses `dot(N,V)` for the stochastic branch selection; BRDF uses `dot(H,V)` for specular lobe). The watcher was correct; no double-accounting bug.
- TIR handling: `validRefract = dot(refr, refr) > 1e-8` (line 861), fallback to reflect. Correct.
- PDF accounting: each BRDF branch divides by its branch probability (`specProb`, `transProb`, `diffProb`). Diffuse cosine-hemisphere cancels naturally. Correct.
- M-3 above: NEE rect-area/mesh-area bypass `directLi * lightCount` scaling — real bug.

### E. Resource lifecycle
- `dispose()` calls `#destroyAccumTexture()` (which destroys all 5 textures + 2 buffers) and then `#paramsBuffer?.destroy()` and `#sceneBuffers?.destroy()`. All allocated resources are covered.
- `#destroyAccumTexture` is called before reallocation in `#ensureAccumResources` (line 157). Clean.
- `#computePipeline` and `#bindGroupLayout` are set to `null` in `dispose()` but not explicitly destroyed (they are GC-collected; WebGPU objects without a `destroy()` method rely on GC). This is correct WebGPU usage for pipelines and bind group layouts.

### F. Scene → GPU upload edge cases
- Empty `Scene.primitives`: `buildCpuBvh` with `triCount === 0` returns an 8-float empty node, `bvhNodeCount = 1`, `triangleCount = 0`. The shader early-exits when `bvhNodeCount > 0 && arrayLength(&bvhNodes) > 0` (it will enter the loop but find no triangles). Safe.
- `AnalyticPrimitive`: packer processes them fully — shape IDs, transforms, params packed. Shader intersects them. Capabilities says empty set (M-4 above).
- `MeshAreaEmitter` with missing `meshId`: returns `hasMeshAreaLight: false` with a warning. Safe.
- `Material.baseColorMap`: ignored (not packer reads it). Silent. Correct behavior for a prototype that documents "No texture support."

### G. Capabilities accuracy
- `supportsAuxBuffers: true`: all four aux textures and variance moments buffer are allocated and populated. Verified clean.
- `accumulates: true`: progressive accumulation in `accumBuffer` + running average. Correct.
- Motion blur: the original audit reported `supportsMotionBlur: false` and noted `shutterTime` is unread. Both fields were dropped from the contract on 2026-05-17 (W3-D18; re-dropped 2026-05-18 after a file-split regression). No further action.
- `supportedAnalyticShapes: new Set<string>()`: incorrect — all 5 shapes are implemented. See M-4.
- `supportedEmitterKinds`: advertises `['directional', 'point', 'spot', 'rect-area', 'mesh-area']`. All five are wired CPU-side and shader-side. `disc-area` correctly absent. Clean.
- `maxBounces`: clamped to `PROTOTYPE_MAX_BOUNCES = 8` and shader loop caps at `min(params.maxBounces, 8u)`. Consistent.

### H. TypeScript
- No `as any` or `as unknown as` casts found in pt-webgpu source files.
- `paramsU32[N]` and `paramsF32[N]` typed array accesses are not flagged by `noUncheckedIndexedAccess` because they write (not read) — only reads produce `T | undefined`. All reads from scene buffers use `?? 0` guards. Clean.
- 3 pre-existing tsc errors in `pt-webgl/src/index.ts` (duplicate @types/three, `isScene: boolean` vs `true`, `isCamera: boolean` vs `true`). Pre-existing, not introduced by pt-webgpu work.

### I. Engine interface contract
- All 7 required `Engine` methods present with correct signatures.
- State transitions: `'initializing'` → `'ready'` in factory (line 543), `'paused'` on pause(), back to `'ready'` on resume(), `'disposed'` on dispose(). Correct.
- After `dispose()`, `renderFrame` hits `#assertLive` which checks `state === 'disposed'` first (line 106). Throws correctly.
- `setScene` does NOT call `#assertLive` — it only checks `state === 'disposed'`. This means `setScene` works even in `'paused'` state, which is correct (scene can be updated while paused).

### J. Subtle issues
- `multiplyMat4(input.projMatrix, input.viewMatrix)`: column-major convention. `VP = P * V` in column-major is the standard Three.js convention; `vp = projMatrix * viewMatrix` gives the combined VP matrix where V is applied first. Correct.
- `invertMat4` returning `null` for near-singular matrices: the fallback at line 312 uses an identity matrix, which would make ray generation use NDC-space coordinates as world-space. This is visually wrong but safe (no crash, no NaN). An assertion or warning would improve debuggability.
- No race conditions: all GPU submission is synchronous from the JS side (submit/writeBuffer are fire-and-forget). No async awaits in the render path.
