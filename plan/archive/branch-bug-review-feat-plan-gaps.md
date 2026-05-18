# feat/plan-gaps Bug Review — 2026-05-09

## Summary

- Findings: HIGH=3, MEDIUM=3, LOW=2

---

## HIGH severity

### H-1 — ppgUpdate leaf stride is 32u (128 bytes) but allocation is 256 bytes per leaf

**File:** `packages/walkaround-hybrid/src/ppg/wgsl/ppgUpdate.wgsl.ts:164`

**Issue:** `ppgLeafSlot` computes `leafIdx * 32u + binIdx * 2u + field`. The PPG leaf buffer is allocated with `PPG_LEAF_BYTE_STRIDE = 256` bytes per leaf (verified: `packages/walkaround-hybrid/src/ppg/types.ts:129`). 256 bytes / 4 bytes per u32 = 64 u32 slots per leaf. The shader uses stride 32u (128 bytes). For any scene with more than one occupied PPG cell, every write to cell N lands in cell N/2's memory — corrupting both cells' directional bins silently.

The PPG leaf struct in `ppgSample.wgsl.ts` is read via struct-field access (`ppgLeaves[leafIdx].bins[b].x`), so the GPU derives the correct 256-byte offset there. The index arithmetic in `ppgUpdate.wgsl.ts` is half that — a read/write aliasing mismatch that produces garbage results for all cells beyond index 0.

**Why HIGH:** Data corruption on every frame that runs the PPG update pass. Silent — no GPU error, wrong radiance bins trained. The fix was committed in Branch A (`leafIdx * 64u`) and is confirmed reverted here by the diff.

**Fix:** Change `leafIdx * 32u` → `leafIdx * 64u` at line 164 of `ppgUpdate.wgsl.ts`. The comment on lines 107-108 already states the correct formula (`leafIdx * PPG_DIRECTIONS * 2 = 16 * 2 = 32`), but `PPG_DIRECTIONS * 2 = 32` u32 pairs × 4 bytes = 128 bytes, not 256. The comment arithmetic itself needs correction: the correct formula in u32 slots is `leafIdx * (PPG_LEAF_BYTE_STRIDE / 4)` = `leafIdx * 64u`.

---

### H-2 — lumFixed clamp uses 0xFFFFFF (16M) instead of 0xFFFFFFFF (4G)

**File:** `packages/walkaround-hybrid/src/ppg/wgsl/ppgUpdate.wgsl.ts:195`

**Issue:** `let lumFixed = u32(clamp(lum * PPG_RADIANCE_SCALE, 0.0, f32(0xFFFFFFu)))`. With `PPG_RADIANCE_SCALE = 65536.0`, the max representable lum before clamping is `0xFFFFFF / 65536 ≈ 256 nits`. Sun-through-glass primary hits routinely exceed 1000 nits. Any lum > 256 is clamped to 65535 in fixed-point, skewing the directional histogram toward the over-clamped bins and misrepresenting high-energy directions in the learned guiding PDF.

The correct upper bound for a u32 fixed-point accumulation is `f32(0xFFFFFFFFu)` (≈ 4.29G → max lum ≈ 65536 nits, adequate for any HDR scene). Branch A fixed this; Branch B reverted it.

**Why HIGH:** Silently biases the PPG guiding PDF for any emitter above ~256 nits — covers sun/spot fixtures in the target scene. Wrong sampling directions increase rendering variance systematically.

**Fix:** Change `f32(0xFFFFFFu)` → `f32(0xFFFFFFFFu)` at `ppgUpdate.wgsl.ts:195`.

---

### H-3 — Welford varianceBuffer is never written; Sprint 9 + SVGF both read stale zeros

**Files:**

- `packages/walkaround-hybrid/src/pipeline/resourceManager.ts:441` — buffer allocated
- `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:415,488` — buffer bound as input
- No shader in the pipeline writes to it

**Issue:** `varianceBuffer` (RG32Float, r=mean, g=M2) is allocated in `createFrameResources` and documented as Sprint 9's Welford running-variance store. Two passes read it:

1. **SVGF variance pass** (`WalkaroundGPUPipeline.ts:415`): `varianceIn = this.res.varianceBuffer.createView()`. For the first 4 frames (`frameCount < 4`) the SVGF variance pass uses a spatial 3×3 estimate and ignores `varianceIn` — so filtering is correct those frames. After frame 4, the shader branches to `welfordVariance(state, frameCount)` where `state` is always `{mean=0, m2=0}` (never written). This returns 0 for all pixels. With variance=0, the SVGF à-trous color weight becomes `exp(-dLum² / (sigmaColor² × 0.001 + 1e-6))` ≈ a near-zero tolerance — SVGF effectively freezes into a minimum-blur mode after 4 frames, defeating variance-guided denoising.

2. **sampleBudget pass** (`WalkaroundGPUPipeline.ts:488`): `varianceView = this.res.varianceBuffer.createView()`. Since the buffer is always zero, `welfordVariance(state, n) = 0 < threshold_low (0.01)` for every pixel on every frame. The tier texture is written but always contains tier=1 (converged). The entire sampleBudget pass runs every frame and produces output that will never influence ray counts — it is a no-op that wastes GPU time.

No shader in the pipeline has a binding that writes to `varianceBuffer`. The `welfordUpdate` function exists in `common.wgsl.ts` but is never called from any pass that has a write binding on `varianceBuffer`.

**Why HIGH:** SVGF denoiser quality regresses from good (frame 0-3) to broken (frame 4+) every time the accumulator resets. sampleBudget is a dead compute pass for the lifetime of the engine. Both bugs are invisible from CPU-side telemetry.

**Fix:** Wire `varianceBuffer` as a write-capable binding on either the shade pass or the temporal-accum pass, and call `welfordUpdate` per pixel with the new luminance sample. The most natural home is the temporal-accum pass (already has the per-pixel current radiance). Alternatively, if Welford is deferred, sampleBudget should read from `svgfVarianceTexture` (which IS written by the SVGF variance pass) to get a useful variance estimate.

---

## MEDIUM severity

### M-1 — `buildLightTree` return field renamed from `nodePowerPrefixSum` to `cdf` — consumers may break

**File:** `packages/shared-samplers/src/lightTree.ts:238`

**Issue:** Branch B renames the return field from `nodePowerPrefixSum: Float32Array` to `cdf: Float32Array` and changes the semantics description. The diff also changes the field's described layout from "unnormalised prefix-sum (values can exceed 1.0 for trees with > 1 leaf; NOT a true CDF)" to "flat CDF for stochastic root-to-leaf descent." The mathematical content appears identical (same running-sum computation), but the name change and contradictory description create a semantic ambiguity: the old JSDoc explicitly warned the array is not a true CDF; the new JSDoc calls it one.

More critically: any consumer on `main` that destructures `{ nodes, nodePowerPrefixSum }` from `buildLightTree` silently gets `undefined` for `nodePowerPrefixSum` (TypeScript catches this at compile time if the type is imported, but runtime consumers in test fixtures or non-typed JS would silently break).

**Why MEDIUM:** Compile-time breakage is recoverable if tsc is run. Runtime breakage in any non-typed caller or test that holds a cached reference is silent.

**Fix:** Verify all callers have been updated to `cdf`; also clarify whether this is a true probability CDF or an unnormalised prefix-sum (the current documentation contradicts the implementation — internal node powers are counted before their children, so values can exceed 1.0 for multi-leaf trees).

---

### M-2 — L-2 degenerate centroid handling removed from `buildLightTree`

**File:** `packages/shared-samplers/src/lightTree.ts:142` (region removed)

**Issue:** Branch A's `buildSubtree` included an explicit guard: when the centroid AABB span was < 1e-4 (all emitters co-located), it fell back to a power-sorted median split rather than axis-sorted. Branch B removes this guard. When all centroids are co-located (`spanX = spanY = spanZ = 0`), the axis selection computes `axis = 0` (x wins by `>= >= >= >=` tiebreak), sorting produces an arbitrary stable ordering, and the median split proceeds. The tree is built without crashing, but all nodes have AABB span 0 — the spatial proximity correction at GPU traversal leaves is meaningless, and sampling degrades to power-only weighting with a useless AABB. This covers the cathedral glass panel case (one panel = all triangles at the same world position).

**Why MEDIUM:** Degrades sampling quality silently for co-located emitter sets — a known target scene pattern. No crash, but MIS weights are wrong for those emitters.

**Fix:** Re-introduce the degenerate case guard with a power-median fallback. Alternatively, document the degraded behavior with a clear comment.

---

### M-3 — `mixturePdf` zero-probability guard removed

**File:** `packages/shared-samplers/src/mixturePdf.ts:92` (guard deleted)

**Issue:** Branch A guarded against `sum(probabilities) == 0` with an explicit throw. Branch B removes the guard and the corresponding JSDoc. Callers that pass all-zero probabilities now silently return 0 from `mixturePdf` without a diagnostic error. This makes zero-denominator MIS weight bugs harder to detect — the shader or CPU caller proceeds with `pdf = 0`, which produces division-by-zero or NaN in MIS weight computation downstream.

**Why MEDIUM:** Removes a defensive programming layer. The bug it was guarding against (all-zero strategy weights) is a real caller logic error that occurs during incorrect multi-strategy wiring. Without the guard, it manifests as NaN or black pixels with no stacktrace.

**Fix:** Restore the throw guard. The zero-probability case is always a caller logic error per the MIS contract.

---

## LOW severity

### L-1 — `WalkaroundGPUPipeline` header comment says "3 iterations" for à-trous; actual dispatch is 5

**File:** `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:15`

**Issue:** The class-level doc comment reads: `5. À-trous denoiser (3 iterations, stepWidths 1, 2, 4)`. The actual loop at line 428 runs `for (let iter = 0; iter < 5; iter++)` — 5 SVGF à-trous passes with step widths 1, 2, 4, 8, 16. The comment predates the SVGF integration and was never updated.

**Why LOW:** Documentation-only. No runtime impact. Confuses readers tracing pass counts.

**Fix:** Update line 15 to: `5. SVGF: variance estimation + 5 à-trous iterations (step widths 1, 2, 4, 8, 16)`.

---

### L-2 — `vitrumSceneToThree` silently drops `InstancedMeshPrimitive` primitives

**File:** `packages/three-bindings/src/vitrumSceneToThree.ts:244`

**Issue:** `vitrumSceneToThree` iterates `vitrumScene.primitives` and handles only `p.kind === 'mesh'`. The core `Scene` type (`packages/core/src/scene.ts:139,177`) also defines `InstancedMeshPrimitive` (kind `'instanced-mesh'`). Any host that passes instanced mesh primitives to `setScene` will see them silently dropped with only a `console.warn`. Since `HybridEngine._coreSceneSuppliesMeshes()` checks for `kind === 'mesh'` only, a scene with exclusively instanced primitives returns false from that predicate — the engine falls back to the host's `threeScene` for BVH construction rather than using the core Scene, which may produce an incorrect BVH.

**Why LOW:** Currently no known caller passes instanced primitives through this path, but the silent fallback violates the "core contract is the thing that's fixed" principle. Will cause a hard-to-diagnose BVH mismatch when a host later uses instanced geometry.

**Fix:** Either implement `InstancedMeshPrimitive → THREE.InstancedMesh` conversion, or promote the `console.warn` to a thrown error so the caller knows the scene was not fully translated.

---

## Audit findings carryover status (A's fixes vs B's state)

| Finding                                     | A (main) fix                                    | B (feat/plan-gaps) state                                                                              | Diff                                                                                                  |
| ------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| H-1 ppgLeafSlot stride (32u→64u)            | Fixed: `leafIdx * 64u`                          | **REVERTED** to `leafIdx * 32u`                                                                       | Regression                                                                                            |
| H-2 lumFixed clamp (0xFFFFFFu→0xFFFFFFFFu)  | Fixed: full u32 range                           | **REVERTED** to `f32(0xFFFFFFu)`                                                                      | Regression                                                                                            |
| M-1 `nodePowerPrefixSum` naming             | Renamed to `cdf` with clarified JSDoc           | Renamed to `cdf` with contradictory JSDoc (calls it a "true CDF" but values exceed 1.0)               | Partial; semantic ambiguity introduced                                                                |
| M-2 sigmaColor comment in svgfBindings      | Comment noted scene-specific tuning             | Comment deleted; replaced with generic Schied citation                                                | Regression (context lost)                                                                             |
| M-4 BLOCKING CONDITION comment in ppgSample | Prominent box warning present                   | **REMOVED** — replaced with single-sentence note that incorrectly calls brute-force O(N) "acceptable" | Regression: safety signal gone                                                                        |
| M-5 `mixturePdf` throw guard                | Guard present: throws on all-zero probabilities | **REMOVED**                                                                                           | Regression                                                                                            |
| M-6 svgfBindings iteration JSDoc            | States "N times, host controls count"           | Hardcoded to "5 times"                                                                                | Partial: now mismatches the `SVGFUniforms.iteration` field comment which still says "0-4 (unbounded)" |
| L-2 centroid degeneracy fallback            | Power-median fallback present                   | **REMOVED**                                                                                           | Regression (see M-2 above)                                                                            |

---

## Runtime integration risk assessment per sprint

**Sprint 9 — sampleBudget pass + resolve pass**
Risk: **HIGH**. sampleBudget reads `varianceBuffer` which is never written (H-3). The tier texture is allocated and dispatched but always contains tier=1. The resolve pass itself is structurally correct (bind group layout, shader bindings, dispatch dimensions all verified consistent), but the sampleBudget output it conceptually depends on is a no-op. Net effect: the resolve pass runs correctly as a checkerboard fill, but the adaptive ray budget mechanism is entirely inert.

**Sprint 10a — SVGF replacing à-trous**
Risk: **MEDIUM**. The SVGF pipeline replacement is structurally sound: shader module compiles separately from COMMON_WGSL (no duplicate struct risk), bind group layout and builder entries align across all 8 bindings for both entry points, workgroup sizes match dispatch dimensions (16×16 with wgX16/wgY16). The bug is functional: after frame 4 the variance input to the à-trous edge-stop is zero (H-3), causing over-tightening of color weights. The output texture path is correct — `svgfVarianceTexture` is produced by the variance pass and consumed by the atrous pass, and `atrousFinalTex` feeds the temporal accumulator as expected. The composite pass reads the resolved texture, not the raw atrous output, which is also correct.

**Sprint 11 — PPG buffer allocation**
Risk: **HIGH**. Buffers are allocated and disposed correctly. However, H-1 and H-2 mean the update shader corrupts all leaf data beyond cell 0 and saturates high-radiance samples. The M-4 BLOCKING CONDITION removal hides the O(N) brute-force scan warning — if the dispatch is ever wired, it will be unsuitable for live use at typical resolutions. PPG buffers are not yet dispatched on this branch, so there is no runtime GPU impact today. The buffers accumulate ~3.36 MB per engine instance.

**Sprint 3 — Back-face NEE (commit c7131a1)**
Risk: **LOW**. Verified: commit c7131a1 is a docs-only mark ("Sprint 3 back-face NEE resample complete"). No WGSL or JS runtime code changes are associated with it. The NEE implementation lives in the upstream `three-gpu-pathtracer` fork, not in this repo. No new runtime code was introduced.

**baea01c — vitrumSceneToThree + HybridEngine BVH integration**
Risk: **MEDIUM**. The scene-to-Three translation is structurally correct for mesh primitives and all four emitter kinds. The DDGI traversal scene is properly owned and disposed on teardown. The race condition introduced by the factory boot scene (H-3 class: new async init started by factory, then host immediately calls `setScene` with real scene — two concurrent `_initPipeline` chains with no per-init cancellation token, only a global `_disposed` flag) can cause the stale init to complete and overwrite `_pipeline`/`_bvhBuffers` after the real scene's init has already settled. GPU buffers from the losing init are never destroyed. This is a new pattern introduced in this branch (confirmed by diff).
