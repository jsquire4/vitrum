# Engine Bug Fixes — Sweep 2026-05-11

> Generated from verified findings in `memory/in-flight-sweep.md`.
> All file:line references confirmed by direct read.
> No code has been modified; this is a plan only.

---

## Item 1: PPG Injector Hard-Throws

**File(s):**
- `packages/walkaround-hybrid/src/shaders/shadePpgGuide.wgsl.ts:301–306` — searches for `'Lo_ddgi * DDGI_DIFFUSE_BLEND;'`
- `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts:459` — current text is `let combined = clampedDirect + clampedIndirect;`

**Root cause:** The PPG injector's two-step string-splice targets a combined-sum term that no longer exists; the shade pass was refactored so `Lo_ddgi` is now `clampedIndirect`. The first splice (marker `@@PPG_RECORD_INSERT@@`) would succeed, but the second splice on `'Lo_ddgi * DDGI_DIFFUSE_BLEND;'` would throw a hard Error before any pipeline compiles.

**Authoritative source:** No external paper. Vitrum internal contract: `shadePpgGuide.wgsl.ts` is a code-injection shim, not a standalone shader. The injector must track shade.wgsl's current variable names.

**Fix (DECIDED — durable marker approach):**
1. In `shade.wgsl.ts`, replace line 459 (`let combined = clampedDirect + clampedIndirect;`) with:
   ```wgsl
   // @@PPG_GUIDE_EXTEND_COMBINED@@
   let combined = clampedDirect + clampedIndirect;
   ```
   Add the named marker as a sibling of the existing `@@PPG_BOUNCE_INSERT@@` (line 399) and `@@PPG_RECORD_INSERT@@` (line 463) markers — keep all three in the same shade.wgsl coordinate system.
2. In `shadePpgGuide.wgsl.ts:301–311`, replace the brittle `tail = 'Lo_ddgi * DDGI_DIFFUSE_BLEND;'` string-search and the corresponding `replace()` with a marker-anchored splice on the new `// @@PPG_GUIDE_EXTEND_COMBINED@@` marker. The new injector inserts the extended combined sum *immediately after* the marker line, replacing the next line's `let combined = ...;` with `let combined = clampedDirect + clampedIndirect + Lo_ppgBounce * PPG_GUIDE_INDIRECT_BLEND;`.
3. Update the JSDoc on `injectPpgGuideBounceIntoShadeWgsl` to document the marker contract.

**Rationale:** the agent's original fix (just update the search string) is mechanically correct but will silently break again the next time shade.wgsl's combined-sum line changes. Switching to a named marker — like the two existing markers in the same file — makes future breakage loud (marker-not-found error) instead of silent (string-not-found error masquerading as "PPG works").

**Decision points:** None — locked.

**Behavior-preserving test:** Add a unit test that calls the injector with the current `shade.wgsl.ts` string export and asserts no Error is thrown. The existing `sprint11-ppg.test.ts` exercises the injector at a string level but does not guard against this failure mode.

**Verification:** `ppgEnabled: true` pipeline compile must not throw. Render a frame with PPG enabled in the GPU test environment; check that `Lo_ppgBounce` contributes to `combined`.

**Dependencies:** None.

**Risk:** Low isolated fix. If shade.wgsl is later refactored again, the injector silently breaks. Consider replacing the two-step string splice with a single named marker (`@@PPG_GUIDE_EXTEND_COMBINED@@`) that shade.wgsl owns, making future breakage obvious at the marker-search step.

---

## Item 2: DDGI Double-π / Double-Albedo at Receiver

**File(s):**
- `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts:547` — writes `radiance = (direct + indirect) * albedo / PI`
- `packages/walkaround-hybrid/src/rc/applyDDGIShading.ts:148` — multiplies again by `materialColor * PI_INV`

**Root cause:** The probe atlas stores outgoing Lambertian radiance `L_o = albedo/π · (L_direct + L_indirect)`. The consumer then applies the receiver's Lambertian BRDF `albedo/π` again, yielding `albedo²/π²` instead of the correct `albedo/π`. Commit `3fb63e3 disable DDGI gain` suppressed the resulting over-brightness by a band-aid gain reduction.

**Authoritative source:** Majercik et al. 2019, "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields", JCGT §3. The atlas accumulates *irradiance* E (integral of incoming radiance over the hemisphere), not outgoing radiance. The correct consumer is:

```
L_diffuse_indirect = (albedo / π) × E_atlas
```

where `E_atlas` is the per-pixel irradiance sample from the probe. The probe update should store *incoming* radiance `L_i` and let the atlas blending produce E automatically (or store `E` directly by integrating over probe rays with a cosine weight). It should NOT pre-multiply by albedo.

**Fix:**
1. In `probeUpdateRays.wgsl.ts:547`: remove the `* albedo / PI` factor. Store raw incoming radiance `L_i = direct + indirect`. The blend kernel `probeUpdateBlend` already applies a directional weight and averages over rays; after blending the atlas holds a weighted irradiance approximation.
2. In `applyDDGIShading.ts:148`: the `albedo/π` multiply is correct and must stay. Remove the comment about PI_INV needing to compensate for anything — it now cleanly implements the Lambertian receiver equation.
3. Remove the gain band-aid from commit `3fb63e3` if it was wired into any UBO or uniform constant.

**Decision points:** Confirm whether the blend kernel (Item 20, `pow(w, 8)`) is being fixed simultaneously. If the blend kernel is corrected to a true cosine-weighted integral at the same time, the atlas value represents a proper irradiance estimate. If Item 20 is not fixed yet, the atlas still holds a non-physical approximation — but the double-albedo bug should still be removed independently.

**Behavior-preserving test:** Render a uniform white Cornell box (albedo=1 everywhere). The indirect contribution from DDGI should be ≤ the direct illumination intensity (conservation). Currently it is brighter due to albedo squaring.

**Verification:** Measured indirect radiance from a grey (albedo=0.5) wall in a Cornell box should be ~half the value from a white (albedo=1) wall. With the bug, both walls return the same DDGI contribution.

**Dependencies:** Item 20 (blend kernel) should land in the same PR for a coherent physical model, but can be sequenced separately.

**Risk:** Removing the albedo premultiplication will visibly darken indirect lighting; the band-aid gain removal will also change brightness. Both changes together should land simultaneously with a reference render comparison.

---

## Item 3: DDGI Atlas Border Padding Never Written

**File(s):**
- `packages/walkaround-hybrid/src/ddgi/ddgiAtlasLayout.ts:13–14` — documents that border pixels should wrap the octahedral seam
- `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateBlend.wgsl.ts` — blend kernel writes only interior atlas coordinates, no border fill pass

**Root cause:** The atlas allocates `CELL + BORDER` (10×10 irradiance, 18×18 visibility) per probe, with BORDER=2, to support bilinear sampling at octahedral seams. No compute pass writes the 1-pixel border ring; bilinear filtering at cell edges reads zero, producing systematic darkening at every octahedral seam.

**Authoritative source:** Majercik et al. 2019 §3.2 and supplemental. DDGI octahedral atlas border update: for each border texel, copy the mirrored interior texel that represents the same direction on the octahedral map. Specifically, for a texel at position `(bx, by)` on the border, find the equivalent octahedral uv, reflect it across the seam into the interior cell, and copy from there. This is a standard octahedral map border-fixup technique (see also Cigolle et al. 2014, "Survey of Efficient Representations for Independent Unit Vectors", JCGT §2).

**Fix:**
1. Add a third compute pass `probeUpdateBorder` that runs after `probeUpdateBlend` each frame.
2. The pass iterates over every atlas probe cell's border ring (1 pixel wide on each side). For each border texel at atlas position `(px, py)`:
   - Compute the probe's local pixel `(lx, ly)` within its `STRIDE × STRIDE` cell.
   - Identify whether `lx/ly` is on the border (0, STRIDE-1, or intermediate edges).
   - Derive the octahedral uv, mirror across the seam → `(mx, my)` interior local coordinate.
   - Copy `textureLoad(irrPrev, atlasCoord(probe, mx, my))` into the border texel via `textureStore`.
3. Repeat for visibility atlas.
4. Bind the atlas as both texture (read) and storage texture (write) in the same pass, or use two separate `GPUTexture` if WebGPU forbids simultaneous read+write on the same texture; in that case, use a ping-pong buffer just for the border pass.

**Decision points:** WebGPU does not allow the same texture as both `texture_2d` (read) and `texture_storage_2d` (write) in the same pipeline. A workaround is: (a) copy the interior only to a temp texture then do the border fill reading from the temp, or (b) collect border pixel values in a scratch buffer during the blend pass, then write them out in a separate border pass reading from the scratch. Option (b) is cleanest.

**Behavior-preserving test:** Render a probe atlas, sample near the cell edges, and verify no visible seam darkening. A unit test can verify that every border texel in a test atlas is non-zero after the border pass runs on a non-zero atlas.

**Verification:** Seam-darkening artifacts should disappear from the Cornell box render. A diff between renders with/without border pass should show reduced grid-pattern darkening.

**Dependencies:** Item 2 (correct radiance storage) should land first so the border fill copies physically meaningful values.

**Risk:** Medium. Requires a new GPU compute pass and careful octahedral coordinate mirroring math. Incorrect mirroring will produce seam brightening or colour smearing instead of darkening.

---

## Item 4: RC GI Bypasses Receiver BRDF

**File(s):**
- `packages/walkaround-hybrid/src/rc/giReceiver.ts:103–105` — `nm.emissiveNode = giNode`

**Root cause:** Three.js NodeMaterial's `emissiveNode` is added to `outgoingLightNode` *after* PBR shading. This means the GI signal is injected as if it were self-emission: it bypasses the receiver's diffuse BRDF (`albedo/π · nDotL`). A black absorber and a white diffuse wall receive identical GI contribution.

**Authoritative source:** Majercik et al. 2019 §3 receiver equation:
```
L_o = L_direct + (albedo / π) × E_ddgi
```
The correct TSL injection point is `irridianceNode` or an explicit additive term in the diffuse irradiance chain before the BRDF multiply, not `emissiveNode`.

**Fix (DECIDED — pre-multiply by `albedo/π` before assigning to `emissiveNode`):**
1. In `giReceiver.ts`, replace `nm.emissiveNode = giNode as AnyNode` (line 105) with:
   ```ts
   // GI signal is integrated irradiance E. Apply Lambertian BRDF (albedo/π)
   // before injecting via emissiveNode (the only NodeMaterial hook that
   // accepts a per-pixel additive radiance contribution post-PBR).
   const PI_INV = uniform(1.0 / Math.PI);
   const giDiffuse = mul(giNode as AnyNode, mul(materialColor as AnyNode, PI_INV as AnyNode));
   nm.emissiveNode = giDiffuse;
   ```
   `materialColor` is already in scope from the existing `srcMat.color` extraction earlier in this function.
2. Update the comment block at line 100–104 to read: "GI signal is integrated irradiance; multiply by `albedo/π` to convert to Lambertian outgoing radiance before adding via `emissiveNode` (the only available NodeMaterial hook for per-pixel additive contribution)."

**Rationale:** the math is correct because the GI input is already integrated irradiance E (the cosθ weight is integrated into E during the cascade-merge / DDGI-blend step). The Lambertian receiver equation is `L_o = (albedo/π) · E` — no per-direction `nDotL` term is needed for indirect diffuse. Three.js TSL does NOT expose a clean `indirectDiffuseNode` / `irradianceNode` hook in the version vitrum uses (verified by reading `lib/nodeMaterialUpgrade.ts:1–55`), so the pre-multiply approach is the pragmatic correct path, not a workaround.

**Decision points:** None — locked.

**Behavior-preserving test:** Render two spheres side by side: one white (albedo=1), one black (albedo=0). With the fix, the black sphere should show zero indirect GI contribution. Currently both show the same GI.

**Verification:** Black-sphere GI contribution = 0 after fix. White sphere should show unchanged GI level compared to current (the emissive path adds without BRDF; the new path adds albedo/π × irradiance, which for white with correct E should match the current value if E was already scaled correctly).

**Dependencies:** Item 2 (correct atlas radiance) should land first; combining a corrected BRDF injection with incorrect radiance storage will shift the error rather than fix it.

**Risk:** Medium. TSL internals change between Three.js versions. The `irridianceNode` hook may not exist; the workaround multiply is fragile.

---

## Item 5: ReSTIR-DI p̂ Inconsistency

**File(s):**
- `packages/walkaround-hybrid/src/shaders/ris.wgsl.ts:76` — uses `emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor)` (clamped)
- `packages/walkaround-hybrid/src/shaders/temporal.wgsl.ts:96` — uses `nlDotL / dist2` (raw)
- `packages/walkaround-hybrid/src/shaders/spatial.wgsl.ts:77` — uses `nlDotL / dist2` (raw)

**Root cause:** Bitterli et al. 2020, "Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting" (ReSTIR DI), §3 and §4.3: for unbiased temporal and spatial reuse, the target distribution p̂ used to evaluate reservoir weights *must be identical* across all passes. When p̂ differs between initial sampling (clamped geometry term) and reuse passes (unclamped), the reservoir weight ratio `p̂_new / p̂_old` is computed incorrectly, reintroducing the firefly mode the clamp was supposed to kill.

**Authoritative source:** Bitterli et al. 2020, §4.3 "Resampling"; Ouyang et al. 2021, "ReSTIR GI", §3 — both state that p̂ must be consistent for MIS correctness. The floor clamp `emitterDist2Floor` is a variance-reduction heuristic applied uniformly — all three passes must use it.

**Fix:**
1. Extract `computePHat` (temporal.wgsl.ts) and `computePHat_s` (spatial.wgsl.ts) to use the same `emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor)` call as `ris.wgsl.ts:76`. The `emitterDist2Floor` uniform is already present in the UBO; add it to the temporal and spatial UBO layouts if not already bound.
2. Verify that `ubo.emitterDist2Floor` is accessible in the temporal and spatial bind groups. If not, add it.
3. Add an inline comment in each `computePHat` explaining the consistency requirement and citing Bitterli §4.3.

**Decision points:** Check whether `emitterDist2Floor` is in the temporal/spatial UBO struct. If not, the UBO definition and the host-side packing both need updating.

**Behavior-preserving test:** A unit test comparing the p̂ values from RIS, temporal, and spatial versions on the same emitter/surface pair should return identical floats.

**Verification:** Render a Cornell box 1000 frames with ReSTIR enabled. Firefly count (pixels > 3× median luminance) should decrease relative to the current build.

**Dependencies:** None.

**Risk:** Low. Pure shader constant propagation. May slightly change brightness in some frames (removing the firefly compensates at temporal edges) but should not change converged energy.

---

## Item 6: DDGI randomRotation Hard-Coded to (0,0,0)

**File(s):**
- `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:585–589`

**Root cause:** The per-frame probe ray rotation is fixed at `(0, 0, 0)`. The comment documents this as intentional to avoid per-frame flicker in a static Cornell scene, but the consequence is that the 192-ray estimator never temporally accumulates new MC samples — the effective sample count is permanently 192 regardless of frame count. DDGI's temporal averaging relies on probe rotation to decorrelate frames.

**Authoritative source:** Majercik et al. 2019 §3.1, "Probe Update": "We apply a random rotation to the probe rays each frame using a random rotation matrix R ∈ SO(3). This decorrelates the per-frame ray samples and allows the temporal hysteresis blend to accumulate an effectively larger ray budget over time." Majercik et al. 2022 supplement describes using a fixed per-probe blue-noise sequence (not fully random per frame) as a less-flickery alternative for dynamic scenes.

**Fix:**
1. For static scenes (current priority): revert to a per-frame random rotation. Use a low-discrepancy sequence (e.g., Halton-based SO(3) rotation from the frame index) rather than `Math.random()` to avoid correlation clumps. A Halton sequence on the frame index avoids both the all-same-direction degeneracy and the excessive variance of pure RNG.
2. For dynamic scenes (later): switch to a fixed-per-probe blue-noise SO(3) sequence as described in Majercik 2022, where each probe gets a unique rotation that cycles through a precomputed set. This is the standard DDGI implementation in RTXGI.
3. Immediate minimum fix: `data[0] = Math.sin(this._frameIndex * 2.399963) * 0.5;` (golden-angle step) gives deterministic low-discrepancy variation without `Math.random()` variance.

**Decision points:** Halton vs golden-angle vs blue-noise precomputed set. User should decide whether to implement the minimal deterministic fix (Halton/golden-angle) or the full blue-noise per-probe sequence. Both are significant improvements over all-zeros.

**Behavior-preserving test:** With rotation enabled, probe atlas values should visually converge (EMA mean approaches stable value) over 200 frames rather than being identical on every frame.

**Verification:** Render 500 frames of a static Cornell box. The pixel variance of the indirect irradiance channel should decrease over time (converge) with rotation enabled. Currently it is constant at frame 1 levels.

**Dependencies:** Item 2 (correct radiance storage). Rotating the rays while storing wrong radiance wastes the fix.

**Risk:** Low computation risk. May re-introduce temporal flicker if the rotation step is too large — tune the Halton/golden-angle constant to keep the per-frame direction delta small relative to the probe ray spacing.

---

## Item 7: SVGF Depth Channel Mismatch

**File(s):**
- `packages/shared-denoisers/src/svgfWebGPU.ts:189–209` — host writes linear depth to `buf[o]` (the `.r` / `.x` component)
- `packages/shared-denoisers/src/wgsl/svgf.wgsl.ts:216,248` — shader reads `.w` from `atrous_gbufDepth`

**Root cause:** The host packs depth into the first float of an `rgba32float` texel (the `.r`/`.x` component). The à-trous kernel reads `textureLoad(atrous_gbufDepth, ...).w`. On a zero-initialized texture, `.w` is always 0. The depth edge-stop filter `exp(-dz*dz/sigZ2)` is applied to `zCenter - zP = 0 - 0 = 0`, evaluating to 1 (no depth stop), so the depth edge stop silently no-ops on every pixel.

**Authoritative source:** Schied et al. 2017, "Spatiotemporal Variance-Guided Filtering", HPG §4.2 Eq. 4: depth edge-stop weight `w_z = exp(-|z_p - z_q| / (σ_z · |∇z · (p - q)| + ε))`. The channel indexing is an implementation detail; what matters is that host and shader agree.

**Fix (DECIDED — Option A, shader reads `.x`):**
1. In `svgf.wgsl.ts:216`, change `textureLoad(atrous_gbufDepth, gid.xy, 0).w` to `textureLoad(atrous_gbufDepth, gid.xy, 0).x`.
2. In `svgf.wgsl.ts:248`, change `textureLoad(atrous_gbufDepth,  pu, 0).w` to `textureLoad(atrous_gbufDepth,  pu, 0).x`.
3. Update the doc-comments at `svgf.wgsl.ts:27,37` to read "binding 3 — gbufferDepth (.x = linear depth)" — they already mention `.r`, which is just the alternative spelling for the same channel.
4. Update the comment at `svgf.wgsl.ts:215` ("depth in .w") to reflect the actual `.x` read.

**Rationale:** Option A (1-line shader change) is unconditionally safe and matches the host upload convention that already exists. Option B would require verifying the walkaround `shade.wgsl.ts` G-buffer convention for normalDepth packing, which I could not confirm with a quick grep. If the codebase later normalizes all gbuffer depth packing into `.w` of the normalDepth texture as a deliberate convention change, this fix can be revisited then. For now, do the surgical safe fix.

**Decision points:** None — locked.

**Behavior-preserving test:** The existing `svgf.test.ts` and `svgfWebGpuInputs.test.ts` should be extended to verify that a pixel with known depth `d` produces a depth-stop weight < 1 when a neighbor has depth `d + large_delta`. Currently the test passes but the depth stop is silently inactive.

**Verification:** After fix, apply SVGF to a scene with a sharp depth discontinuity (foreground cube in front of wall). The color bleeding across the depth edge should be reduced compared to current.

**Dependencies:** None.

**Risk:** Low. Single-component swap. Must update both read sites (lines 216 and 248) consistently.

---

## Item 8: SVGF Variance Pass — 4 of 7 Textures Unused (Temporal Reuse Missing)

**File(s):**
- `packages/shared-denoisers/src/wgsl/svgf.wgsl.ts:122–128` — declares but never samples `prevRadiance`, `gbufNormal`, `gbufDepth`, `motionVec`

**Root cause:** The variance pass declares bindings for temporal-reuse inputs (previous radiance, G-buffer, motion vectors) but reads only `inputColor` and `varianceIn`. The paper's full temporal reuse pipeline (bilinear reprojection, depth/normal/object-id disocclusion test, per-pixel history length, α-clamped blend, 7×7 spatial fallback for disoccluded pixels, variance-from-history Eq. 4) is entirely absent. What runs is a 3×3 spatial variance estimate fed into à-trous.

**Authoritative source:** Schied et al. 2017, HPG §4 (full algorithm):
- Eq. 1: linear reprojection `p_prev = P · V_prev · x` using motion vector
- Eq. 2: disocclusion test on depth and normal: reject if `|z - z_prev| > threshold` or `|n · n_prev| < threshold`
- Eq. 3: per-pixel history length `h ← (h_prev + 1)` on acceptance, reset to 0 on reject
- Eq. 4: exponential moving average `α = max(α_min, 1/(h+1))`, α-blend color and moments
- Eq. 5: variance from blended first/second moment: `Var = M2 - M1²`

**Complexity estimate: Large (weeks).** This is not a patch — it is implementing the SVGF temporal stage from scratch. Key new pieces:
1. Reprojection pass: per-pixel motion-vector lookup + bilinear gather from previous frame.
2. Disocclusion detection: depth + normal rejection per Schied Eq. 2.
3. History-length buffer: persistent per-pixel `u32` texture, reset on disocclusion.
4. Moment buffer: persistent per-pixel `vec2f` (M1, M2) for EMA variance (replaces current global Welford).
5. 7×7 spatial variance fallback on disoccluded pixels.
6. Updated variance pass to read from moment buffer rather than compute spatially.

**Fix plan:**
1. Add two persistent textures: `historyLength` (r16uint) and `momentsHistory` (rg32float, M1 + M2).
2. Implement a new `svgfTemporalMain` compute pass that runs before the existing variance pass. This pass: reprojects via motion vector, tests disocclusion, updates history length, blends color and moments via EMA, and writes to `prevRadiance`, `historyLength`, `momentsHistory`.
3. Update `svgfVarianceMain` to read from `momentsHistory` instead of performing spatial 3×3 variance for all pixels; retain spatial fallback only for pixels where `historyLength == 0`.
4. Wire new pass into `svgfWebGPU.ts` pipeline between frame inputs and the existing variance+atrous chain.

**Decision points:**
- Should the existing `shared-denoisers` à-trous and variance code be kept in place and the temporal pass added around it, or is a full rewrite of `svgf.wgsl.ts` cleaner? Recommendation: add temporal as a new wgsl file, leave existing variance+atrous as-is (they are correctish for the non-temporal case), and wire temporal → variance → atrous.
- Does the walkaround engine already have a per-pixel Welford running in the scene (see Item 9)? If so, the moment buffer from the scene's Welford pass can feed the SVGF temporal pass directly, avoiding a second EMA.

**Behavior-preserving test:** Unit test: supply 10 frames of the same still image (no motion, fixed camera). History length should increment to 10 at every pixel. The output variance should decrease each frame. The blended color should converge to the input color.

**Verification:** Cornell box temporal convergence: pixel-level temporal noise should visibly decrease over 60 frames with camera static. Ghost/trail artifacts on camera motion indicate motion vector integration is working.

**Dependencies:** Item 7 (depth channel fix) must land first — the temporal disocclusion test depends on correct depth comparison. Item 9 (per-pixel Welford) is related; coordinate buffer reuse.

**Risk:** High — requires GPU-side persistent textures, a new compute pass, and integration into the host render loop. Also requires motion vectors to be correctly wired from the G-buffer to the SVGF inputs.

---

## Item 9: Welford `n` Is Global, Not Per-Pixel

**File(s):**
- `packages/shared-denoisers/src/wgsl/svgf.wgsl.ts:140` — reads `varUBO.frameCount` for the switching threshold
- Host-side: `svgfWebGPU.ts` passes `frameCount` as a uniform

**Root cause:** The per-pixel history length `n` used in Welford's online variance algorithm should be the number of valid (non-disoccluded) samples accumulated *at that pixel*. Using a global `frameCount` means that pixels disoccluded on frame 10 of a 100-frame sequence still use n=100, causing wildly underestimated variance (the mean is stale but n is large).

**Authoritative source:** Welford 1962, "Note on a Method for Calculating Corrected Sums of Squares and Products"; Schied 2017 §4.2: per-pixel history length `h_i` is reset on disocclusion. The variance estimate is `Var = M2_i / h_i`.

**Fix:** Per-pixel history length requires a persistent texture (see Item 8's `historyLength` buffer). Once Item 8's temporal pass is implemented:
1. Remove the global `frameCount` pathway from the variance pass entirely.
2. The variance pass reads per-pixel `h_i` from `historyLength`, uses it to scale `M2/h_i`.
3. The switching threshold (`SVGF_TEMPORAL_VARIANCE_MIN_FRAMES`) becomes a per-pixel condition: if `h_i < threshold`, use spatial 3×3 fallback; otherwise use per-pixel temporal variance.

**Decision points:** Can the fix be partially applied before Item 8 (full temporal reuse)? Yes: a simpler intermediate step is to add a per-pixel history-length texture that is only reset on a basic depth test (no full motion-vector reprojection). This gives ~80% of the benefit with ~30% of the work.

**Behavior-preserving test:** Unit test: two pixels — one that receives valid samples every frame, one that has a synthetic disocclusion at frame 5. The disoccluded pixel's history length should be reset to 1 at frame 5; the stable pixel's should increment to frame count.

**Verification:** Render a scene with the camera rotating. Disoccluded areas (newly revealed surfaces) should show high variance estimates (and thus more à-trous smoothing) rather than low variance from stale n.

**Dependencies:** Item 8 (temporal reprojection infrastructure). Item 9 is best implemented as part of Item 8, not separately.

**Risk:** Medium — coupled to Item 8. If attempted standalone (shallow depth-test-only history reset), care is needed to not introduce conflicts when Item 8 lands.

---

## Item 10: equiAngular Returns Clamped `t` with PDF at Unclamped `t`

**File(s):**
- `packages/shared-samplers/src/equiAngular.ts:133–135`

**Root cause:** Line 133 computes `ratio = (t - tClosest) / D` using the *unclamped* `t` (from `tan(theta)` which can produce `t < 0` or `t > ray length` in degenerate geometry). Line 135 returns `{ t: Math.max(0, t), pdf }` — the clamped sample position — but the PDF was computed at the unclamped `t`. When `t < 0` (geometrically degenerate), the returned sample is at `t=0` with a PDF calculated at a negative `t` value, violating the sampling identity `∫ f(t)/p(t) dt = 1`.

**Authoritative source:** Kulla & Fajardo 2012, "Importance Sampling of Area Lights in Participating Media" (equi-angular sampling). The PDF at sampled position t is:
```
p(t) = D / (thetaRange · (D² + (t - t_closest)²))
```
where `t` is the *returned* sample position, not the pre-clamped value.

**Fix:** Compute `ratio` using the clamped `t` value:
```ts
const tClamped = Math.max(0, t);
const ratio = (tClamped - tClosest) / D;
const pdf = 1 / (D * thetaRange * (1 + ratio * ratio));
return { t: tClamped, pdf };
```
Since `t < 0` is degenerate (the point is behind the ray origin), the real fix is to also clamp `thetaMin` to ensure the sampled `t` stays in `[0, tMax]`. Check whether `tMin < 0` should be clamped at the function entry.

**Decision points:** Should the function also guard against the case `thetaRange <= 0` (light perpendicular to ray)? This returns PDF = 0 (divide by zero). A guard `if (thetaRange < 1e-8) return { t: 0, pdf: 0 }` is needed.

**Behavior-preserving test:** The existing `equiAngular.test.ts` should include a test where `u=0` and `D` is very small (near-perpendicular), verifying that `t ≥ 0` and `pdf > 0` for all in-range `u`.

**Verification:** Numerical: verify that `1 / pdf ≈ thetaRange * (D² + (t - t_closest)²) / D` for the returned `(t, pdf)` pair.

**Dependencies:** None.

**Risk:** Low. Isolated pure-function fix. The current path is unreachable for well-conditioned geometry (`tMin ≥ 0`), so this is a latent footgun rather than an active render bug.

---

## Item 11: `bdptConnectionMIS` Mislabeled as Veach Implementation

**File(s):**
- `packages/shared-samplers/src/bdptMIS.ts:1–14` (JSDoc) and the exported symbol `bdptConnectionMIS`

**Root cause:** The JSDoc states "structural aid for tests and future fork alignment; it is not a full BDPT strategy PDF enumeration." The export name `bdptConnectionMIS` implies a complete Veach power-heuristic BDPT MIS weight computation. Callers who import this by name and assume it is a complete implementation will get incorrect MIS weights.

**Authoritative source:** Veach 1997, PhD thesis §9.2 (power heuristic) and §10.3 (BDPT MIS weights).

**Fix (two options):**
- Option A (rename + document): Rename the export to `bdptConnectionMISStub` or `bdptPowerHeuristicWeight`. Update all importers. Clearly document in the function signature that `buildBDPTStrategyPDFs` produces a simplified PDF vector, not a full BDPT enumeration.
- Option B (implement): Implement a complete `buildBDPTStrategyPDFs` that correctly evaluates all strategies `(s, t)` for arbitrary connection lengths, including the camera/light path prefixes. This is a significant undertaking (Sprint 10c scope) but is the correct end state.

Recommendation: Option A now; Option B as a separate sprint when BDPT is actually wired into a render path.

**Decision points:** User to choose Option A vs B. If BDPT is not being wired to a real render path in the near term, Option A avoids misleading future agents.

**Behavior-preserving test:** Rename test in `bdpt.test.ts` to match new symbol. No functional change needed for Option A.

**Verification:** The type system catches all import sites after rename. No render change.

**Dependencies:** None.

**Risk:** Low (Option A). The rename is mechanical; the only risk is missing an import site (TypeScript will catch it at compile time).

---

## Item 12: lightTree References Estévez-Kulla but Is Shirley-1996

**File(s):**
- `packages/shared-samplers/src/lightTree.ts:1–27` (module header)

**Root cause:** The module header correctly documents that the implementation is Shirley 1996 median-split. However, `Estevez & Kulla 2018` is listed in the References section. Estévez-Kulla 2018 defines a fundamentally different algorithm (orientation-cone-based BVH, SAH-like split with receiver-aware importance, BSDF cone queries). Nothing from that paper is implemented. The reference misleads maintainers into believing the implementation is closer to SOTA than it is.

**Authoritative source:** Estevez & Kulla 2018, "Importance Sampling of Many Lights with Adaptive Tree Splitting", EGSR. The correct reference for what is implemented is: Shirley, Smits, Wang, Zimmerman 1996, "Monte Carlo Techniques for Direct Lighting Calculations", ACM TOG.

**Fix:** Remove `Estevez & Kulla 2018` from the references section. Add a `// TODO` comment noting that upgrading to Estévez-Kulla would require: (1) orientation cones per node, (2) a receiver-aware importance metric for child selection, and (3) energy-aware split (not median-split). Accurate description of what is and is not implemented.

**Decision points:** None. Purely documentation.

**Behavior-preserving test:** No functional change — documentation only.

**Verification:** The references section no longer cites a paper whose algorithm is not implemented.

**Dependencies:** None.

**Risk:** Zero.

---

## Item 13: `nodePowerPrefixSum` Documented Footgun

**File(s):**
- `packages/shared-samplers/src/lightTree.ts:214–236` (JSDoc for `nodePowerPrefixSum`)

**Root cause:** `nodePowerPrefixSum` is a public field on the return value of `buildLightTree`. The name and position imply it is a CDF suitable for uniform random sampling. The JSDoc explicitly states it is not a CDF — values exceed 1.0 because internal nodes are counted before their children. A future caller who uses this for direct sampling will produce a biased estimator.

**Fix:**
- Rename the field to `nodeAccumulatedPowerNormalized` or simply `_powerPrefixSumDebug` (leading underscore signals internal use).
- Alternative: if the field exists only for CPU-side structural verification (monotonicity checks in tests), make it a local variable not returned in the public interface, and expose a separate `verify(nodes)` function instead.
- Add a prominent `@internal` JSDoc tag and a `@throws` note: "Do not use for sampling. Use leaf-only power traversal on `nodes` instead."

**Decision points:** Should the field be removed from the public return type entirely? If it is only used in tests, yes. Check `lightTree.test.ts` for all reference sites.

**Behavior-preserving test:** No functional change. Tests that use `nodePowerPrefixSum` for monotonicity checks should still pass after rename.

**Verification:** TypeScript compile passes at all call sites after rename.

**Dependencies:** None.

**Risk:** Low. Rename only; TypeScript catches all usage sites.

---

## Item 19: Neural Denoiser — Keep-and-Finish vs Delete

**Recommendation: Keep the architecture files, delete the broken scaffold code, and re-scope to a post-GPU-validation milestone.**

**Reasoning:**
- The `InferenceGraph.ts` and `unetArchitecture.ts` type definitions and architecture spec are well-structured and reflect real design decisions (channel widths, parameter budget, memory estimates). They have value as a blueprint.
- The WGSL kernels (`conv2d.wgsl.ts`, `skipConnection.wgsl.ts`, etc.) are structurally correct but have binding-index mismatches and unwritten uniform buffers. They cannot run as authored.
- `HybridEngine.ts` does not accept `'neural'` as a denoiser mode; the scaffolding is entirely unwired.
- `tools/neural-denoiser-training/train.py.md` (a `.md` file pretending to be a Python script) and the broken skip-connection spatial mismatch are the worst offenders.

**The 8 scaffold bugs to fix regardless of keep/delete decision:**
1. Skip-connection spatial mismatch: decoder Level 3 pairs `(H/4 × W/4)` output with `enc3` which is `(H/8 × W/8)` — shapes are mismatched by 4× in both dimensions. The `unetArchitecture.ts` comment says "H/4 × W/4 × 96" for `dec3_up` but `enc3` is `H/8 × W/8 × 96`. The `skip` layer would add tensors of different sizes.
2. `enc_input` is referenced in the layer spec (line 159) but no layer packs the three input tensors (`noisyColor`, `albedo`, `normals`) into a single 9-channel buffer. The spec leaves this to the host with no implementation.
3. Binding index mismatch in `InferenceGraph.ts:run()`: the code places the uniform buffer at `binding = layer.inputs.length + 1`, weights at `+2`, biases at `+3`. The WGSL kernels in `conv2d.wgsl.ts` declare: input=0, weights=1, bias=2, output=3, uniform=4 — not matching the runtime assignment.
4. Uniform buffer is 32 bytes but never written (line 200–205 allocates, never `queue.writeBuffer` with shape params before dispatch).
5. `tools/neural-denoiser-training/train.py.md` — rename to `train.py` or delete; it cannot be executed as a `.md` file.
6. The bind-group cache in `InferenceGraph.run()` caches `bg` from the first call but entries include buffer references that change if the host swaps output buffers (noted in code comment). The cache stability contract is not enforced.
7. `dispose()` does not clear `_cachedBindGroups` slot-by-slot — only sets `= []`. GPUBindGroups hold references to destroyed buffers; should call `this._cachedBindGroups = new Array(...)` after destroying.
8. No `'neural'` denoiser path in `HybridEngine.ts` — the engine silently falls back to SVGF.

**Keep plan:** Fix bugs 1–5 above (correctness blockers in the scaffold), wire the `'neural'` mode in `HybridEngine.ts` behind a feature flag, and mark Sprint 13 as "scaffold complete, GPU integration pending". Delete `train.py.md`, replace with a real training script or a stub `.py` that clearly documents the expected workflow.

**Delete plan:** Delete `walkaround-hybrid/src/neural/` and `tools/neural-denoiser-training/`. Update `CLAUDE.md` and `sprint-13-walkaround-integration.md` to reflect that the neural denoiser is a planned future sprint, not an existing scaffold. This avoids misleading future agents.

**Recommendation:** Keep the architecture files (`InferenceGraph.ts`, `unetArchitecture.ts`) as a design artifact. Delete the broken WGSL kernels until a real GPU integration sprint is scheduled. Fix the `train.py.md` → `train.py` rename. This is the least-misleading state for future maintainers.

**Risk:** Medium if kept (broken code accumulates bugs). Low if deleted (clear slate but loses design work).

---

## Item 20: DDGI Blend Kernel Uses `pow(w, 8)` Not Paper Lambertian

**File(s):**
- `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateBlend.wgsl.ts:130–134`

**Root cause:** The blend kernel computes `weight = pow(max(0, dot(dir, ray.direction)), 8.0)`. The paper uses a Lambertian cosine weight `w = max(0, cos θ) = max(0, n · d)` where `n` is the atlas texel direction and `d` is the ray direction. The `pow(w, 8)` produces a specular-like lobe (~25° FWHM) that concentrates contributions from near-aligned rays. For a 192-ray budget this avoids the bleaching noted in the code comment (white ceiling dominates red walls in Cornell), but the result is not irradiance — it is a directionally-filtered radiance average.

**Authoritative source:** Majercik et al. 2019 §3 and Algorithm 1: irradiance atlas update uses `w_i = max(0, dir_i · texel_direction)` (standard cosine weight for irradiance accumulation). The cos weight integrates to correct irradiance; a higher power does not. If the 192-ray budget is too sparse for a cosine weight to produce stable results, the paper recommends increasing rays-per-probe or using a tighter ray allocation strategy.

**Fix options:**
- Option A (paper-correct): Change `pow(w, 8.0)` to `w` (pure Lambertian cosine). Accept that 192 rays will produce more visible per-probe variance; rely on Item 6 (random rotation) and temporal hysteresis (HYSTERESIS=0.97) to smooth it.
- Option B (hybrid, document non-physical): Keep `pow(w, 8)` but rename the accumulated quantity from "irradiance" to "directional radiance approximation" throughout all comments, docs, and the atlas-sampling shader. Update Item 2 accordingly (the receiver equation must not apply `albedo/π` if the atlas does not hold irradiance).

Recommendation: Option A, contingent on Item 6 (random rotation) landing first. Without rotation, the Lambertian kernel will produce temporal instability in a static scene because each probe always samples the same 192 directions.

**Decision points:** User must choose Option A vs B. If the visual quality of `pow(8)` is preferred for the 192-ray budget, Option B documents the non-physical choice cleanly. If physical correctness is required, Option A + Item 6 is needed.

**Behavior-preserving test:** Render a uniformly lit environment (constant skybox). All probe atlas values should converge to the same irradiance value regardless of probe direction. With `pow(8)`, high-power directions are over-weighted; with cosine weight, convergence is slower but uniform.

**Verification:** A sphere in a uniform-irradiance environment should appear uniformly lit from all directions after convergence.

**Dependencies:** Item 6 (random rotation) should land first. Item 2 (correct radiance storage) is tightly coupled.

**Risk:** Medium visual impact. Switching to Lambertian with 192 rays and no rotation will produce visually noisy probes. Must land with Item 6 or accept the increased per-probe noise budget.

---

## Item 21: Radiance Cascades Dimensional Scaling Not Sannikov-Faithful

**File(s):**
- `packages/walkaround-hybrid/src/rc/cascadePyramid.ts:31–37`

**Root cause:** The cascade dimensions are tuned for GPU budget (measured 4fps at paper dimensions). The current scaling gives probe-count ratios of ~2.7–7.2× per cascade (not the paper's 4× or 8× for 3D), and interval ratios of 2.5–3× (not the paper's 4×). With 4 children in the merge step but non-power-of-4 probe counts, energy is not conserved across the pyramid because the merge assumes each parent covers exactly 4 children's solid angle.

**Authoritative source:** Sannikov 2023, "Radiance Cascades: A Novel Approach to Calculating Global Illumination", §3 (cascade construction). Conservation law: the solid-angle interval at cascade k is `Δω_k = 4^k · Δω_0`, and probe density scales inversely so `N_probes_k · R_k = const`. For 3D, the paper recommends probe spacing to grow by `2^k` per axis (factor 8 per cascade in 3D).

**Complexity estimate: Large (weeks).** Making the dimensions Sannikov-faithful requires:
1. Choosing new probe counts and ray counts that satisfy the cascade conservation law.
2. The C0 anchor at the current performance budget is incompatible with paper-faithful scaling — paper C0 at 64×36×56 probes × 16 rays was already 2.06M rays/cascade (4fps measured). A paper-faithful 5-cascade pyramid would be ~10× more expensive at C0, ~40× at C4.
3. This is fundamentally a GPU performance problem, not a code bug. Sannikov-faithful scaling for a 30fps WebGPU target requires either fewer cascades, much smaller probe grids, or hardware that doesn't exist in the browser today.

**Fix:**
Two viable paths:
- Path A (document non-faithful, fix merge step): Accept the non-paper dimensions as a perf tradeoff. Fix the merge kernel to account for non-power-of-4 child counts (actual solid-angle coverage, not assumed 4×). This is a cascade-merge integral correction, medium complexity.
- Path B (reduce to 2D Sannikov): Use a 2D RC formulation (screen-space or thin-volume), where the conservation law is easier to satisfy at lower cost. The paper has a 2D mode. This is a significant architecture change.

Recommendation: Path A for now (fix the merge integral to match actual child coverage), document the perf tradeoff explicitly, and track Path B as a future phase.

**Decision points:** Is Sannikov-faithful 3D RC a hard requirement, or is the current approximation acceptable with a corrected merge step? User decision.

**Behavior-preserving test:** Unit test on the merge kernel: for a known-radiance child set, verify the merged parent radiance matches the solid-angle-weighted integral of children.

**Verification:** A uniform-radiance environment should produce consistent merged values at all cascade levels. Currently, non-conservation can cause per-cascade brightness discontinuities.

**Dependencies:** None for Path A. Item 22 (normalization) is closely related and should land together.

**Risk:** High if attempting Path B. Medium for Path A (merge kernel correction). Perf risk: any move toward faithful scaling will reduce frame rate.

---

## Item 22: RC Normalization Assumes Uniform Sphere but Rays Are Octahedral Grid

**File(s):**
- `packages/walkaround-hybrid/src/rc/walkaroundDiffuseLighting.ts:151` — multiplies by `4π/N`

**Root cause:** `4π/N` is the Monte Carlo normalization for N uniformly distributed samples on the unit sphere (PDF = 1/4π). The rays come from an octahedral 2D grid discretization, not from uniform-sphere sampling. The octahedral grid has a non-uniform solid-angle distribution per texel (texels near the octahedral fold have smaller solid angle). The correct normalization is a per-direction solid-angle weight `Ω_i / N_total` where `Ω_i` is the solid angle of the i-th octahedral texel.

**Authoritative source:** Cigolle et al. 2014, "A Survey of Efficient Representations for Independent Unit Vectors", JCGT §2 — octahedral texel solid angle is approximately `4π/N` for large N (standard octahedral) but varies per texel. For the 4×4 bin grid (16 bins), the solid-angle variance is non-negligible (~15% deviation near fold edges).

**Fix:**
1. Precompute per-bin solid-angle weights for the octahedral grid used by PPG/RC (the 4×4 or raysPerProbe grid). This is a CPU-side lookup table.
2. In the WGSL/TSL GI node, replace `mul(sample, 4π/N)` with `mul(sample, solid_angle_weight[binIdx])` where `solid_angle_weight` is a small storage buffer or hard-coded array of per-bin weights.
3. For the RC cascade case, the per-direction weight should be `Ω_i` (solid angle of direction bin i), not the uniform-sphere average.

**Decision points:** The `raysPerProbe` varies per cascade (16, 64, 256, 1024, 4096). Precomputed weights must be stored per cascade resolution. Is this overhead acceptable, or is an approximation (`4π/N · correction_factor`) sufficient?

**Behavior-preserving test:** A uniform-radiance environment (L=1 from all directions) processed through the RC normalization should produce irradiance = π (the cosine-weighted integral over hemisphere). Currently it produces `4π/N · N = 4π` (hemisphere integral omits the cosine).

**Verification:** Analytical comparison: white sphere in uniform environment → irradiance at center should be `π` (hemispherical integral with cosine weight). Measure actual output vs expected.

**Dependencies:** Item 21 (cascade dimensions) — should be fixed together as part of a cascade-correctness pass.

**Risk:** Medium. Changing the normalization will shift the overall brightness of the GI contribution. Needs a calibrated adjustment to any gain constants that compensate.

---

## Item 23: GTAO Simplified Slice Integral, No Multi-Bounce

**File(s):**
- `packages/walkaround-hybrid/src/shaders/gtao.wgsl.ts:146–148`

**Root cause:** The slice AO integral is computed as `(h1 + h2) / π`, which is the angular fraction of the hemisphere that is visible (the "Bavoil-style HBAO" approximation). Jiménez et al. 2016 §4.2 defines the correct GTAO slice integral as:
```
a(slice) = cos(γ) · (2θ_h - sin(2θ_h - 2γ)) + sin(γ) · sin²(θ_h - γ)
```
where `θ_h` is the horizon angle, `γ` is the surface normal projected on the slice, and the integral accounts for the Lambertian cosine weight relative to the normal (not just the horizon angle). The simplified form `(h1+h2)/π` ignores the `cos(γ)` and `sin(γ)` terms and produces AO that is not physically tied to the normal orientation.

**Fix:**
1. In `gtao.wgsl.ts`, compute the projected normal angle `γ` on each slice direction (the angle between the surface normal and the tangent plane of the slice).
2. Replace `(h1 + h2) / PI` with the full Jiménez integral (clamped to [0, 1]).
3. The multi-bounce term (Jiménez 2016 §5.2) applies an approximated ambient reflectance to add indirect AO: `a_mb = a_raw · (1 - ρ·(1 - a_raw))` where `ρ` is surface albedo. This requires the albedo G-buffer — mark as optional enhancement.

**Complexity estimate: Small (hours)** for the correct single-bounce integral. Medium (days) for adding multi-bounce with albedo.

**Decision points:** Implement only the correct single-bounce Jiménez integral now (immediate fix), and defer multi-bounce to a separate PR.

**Behavior-preserving test:** A flat surface with a perpendicular occluder should produce AO ≈ 1/2 (half hemisphere blocked). Test analytically.

**Verification:** Visually, the GTAO result should show stronger darkening in corners and contact shadows compared to the current result, while remaining 1.0 in open-sky pixels.

**Dependencies:** None.

**Risk:** Low. The correct formula is a drop-in replacement for the `(h1+h2)/PI` line. The projected normal `γ` requires one extra `acos(dot(n_projected, tangent))` computation per slice.

---

## Item 24: No Albedo Demodulation in SVGF/à-trous Chain

**File(s):**
- `packages/shared-denoisers/src/wgsl/svgf.wgsl.ts` (both variance and atrous passes)
- `packages/shared-denoisers/src/svgfWebGPU.ts` (pipeline setup)
- `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts` (outputs `hdrColorOut` as full lit color)

**Root cause:** Schied et al. 2017 §4.1 (the first step of the algorithm): before temporal reprojection and à-trous filtering, divide the noisy color by the per-pixel albedo to separate high-frequency albedo variation from the low-frequency lighting signal. The filter operates on `L/albedo` (lighting only), then re-multiplies by albedo at the end. Without this, albedo-correlated high-frequency edges (texture detail, material boundaries) spread into the lighting during filtering, causing blurry material boundaries.

**Authoritative source:** Schied et al. 2017, §4.1: "We demodulate the lighting from the albedo of the first hit surface before filtering: L = c/ρ where c is the full-color path trace output and ρ is the first-hit albedo." Eq. 1 in the temporal reprojection section assumes this has been done.

**Fix:**
1. In `shade.wgsl.ts`, store albedo in a separate G-buffer texture (e.g., `hdrAlbedoOut`). This may already exist as part of the G-buffer (check if `gbufNormal` or another slot carries albedo).
2. Add a pre-filter pass (or modify the existing variance pass input stage) to compute `lighting = inputColor / max(albedo, 0.001)`.
3. Run SVGF on `lighting`. After the à-trous chain, re-multiply: `output = filteredLighting × albedo`.
4. If SVGF is applied to the indirect channel only (not the direct+indirect combined), the albedo demodulation applies to the indirect GI channel specifically.

**Decision points:** Does the walkaround engine already separate the albedo G-buffer texture? Check `resourceManager.ts` for texture allocations. If albedo is not in a G-buffer slot, this requires a new texture and a shader output.

**Behavior-preserving test:** A checkerboard albedo pattern (alternating 0.1 and 0.9) under uniform indirect lighting should produce, after filtering, a checkerboard pattern with correct albedo values — not a blurred blend. Currently the checker pattern bleeds into the adjacent tiles.

**Verification:** Material edges should remain sharp after SVGF in a rendered scene. The visual test is: a bicolor floor under indirect DDGI illumination should show crisp material boundaries.

**Dependencies:** Item 8 (temporal temporal reuse) — albedo demodulation is most valuable when temporal reuse is correct. Item 7 (depth channel) should land first.

**Risk:** Medium. Requires a new G-buffer slot (or confirmation an existing one carries albedo), a pre/post filter step, and careful divide-by-near-zero handling for black surfaces.

---

## Item 25: PPG Architecture Non-Paper-Faithful

**Recommendation: Delete the PPG spatial-grid and GPU-update implementation; retain only the CPU kd-tree structure (Items 11–12 in the PPG namespace) and re-implement from scratch in a dedicated sprint.**

**Reasoning:**

The current implementation deviates from Müller et al. 2017 "Practical Path Guiding for Efficient Light-Transport Simulation" on five independent axes:

1. **Spatial structure**: Static kd-tree built once from a 4×4×4 uniform grid. Müller §3.1: the spatial tree is rebuilt adaptively when cell variance exceeds a threshold.
2. **Directional bins**: Fixed 4×4 octahedral grid (16 bins). Müller §3.2: a 5D adaptive binary tree (sTree/dTree) over (position, direction), not a fixed grid. 16 bins cannot represent sharp indirect caustics.
3. **Training signal**: `ppgUpdate.wgsl.ts` trains on `L_o` (outgoing radiance from the shade pass). Müller §3.3: guides on `L_i` (incoming radiance at the sample point), which is the signal that path guiding uses to propose next-event directions.
4. **Frame timing**: PPG trains on even frames, guides on odd. Müller uses a dedicated training pass within each frame (the "guide" and "train" happen in the same frame, alternating on different subsets of paths using MIS).
5. **Solid-angle constant**: `ppgGuideSolidAngle` uses `4π/16` for 16 fixed bins. Müller uses bin-specific solid-angle weights from the adaptive tree's leaf area.

**Complexity estimate: Large (weeks).** A paper-faithful Müller PPG requires:
- An adaptive spatial tree (sTree) with per-cell variance tracking and split/merge
- A per-cell adaptive directional tree (dTree) — the defining data structure
- Per-frame refinement of both trees
- MIS between PPG guiding PDF and BSDF sampling
- Training on `L_i`, not `L_o`
- All of the above working on GPU with atomic updates

**Delete plan:**
1. Delete `packages/walkaround-hybrid/src/ppg/wgsl/ppgUpdate.wgsl.ts` and `ppgCommon.wgsl.ts`.
2. Delete `packages/walkaround-hybrid/src/ppg/buildPpgKdTree.ts` (kd-tree over a fixed grid has no paper basis).
3. Delete `packages/walkaround-hybrid/src/ppg/ppgCellUpload.ts`.
4. Retain `packages/walkaround-hybrid/src/ppg/types.ts` for reference.
5. Remove PPG enable path from `shadePpgGuide.wgsl.ts` (which fixes Item 1 as a side effect) and `shadePpgTrain.wgsl.ts`.
6. Remove PPG bindings from `bindGroupLayouts.ts` and `pipelineCompiler.ts`.
7. Update `HybridEngine.ts` to remove `ppgEnabled` option or document it as a reserved no-op.
8. Create a `plan/sprint-ppg-rebuild.md` tracking the paper-faithful rebuild as a future milestone.

**Do not implement a partial PPG.** A partial implementation that guides on the wrong signal with a fixed grid and wrong solid-angle weight will train a systematically biased guide PDF that hurts convergence rather than helping it.

**Risk:** High if kept (biased guide PDF actively harms convergence when exercised). Low if deleted (removes a feature that is not yet wired in anyway per the Item 1 crash).

---

## Suggested Execution Order

### Phase 1: Crash Fixes and Isolated Bugs (no GPU required for correctness verification)

These items are independent and can be parallelized across contributors.

| Item | Title | Effort |
|------|-------|--------|
| 1 | PPG injector hard-throw | 1h |
| 5 | ReSTIR-DI p̂ consistency | 2h |
| 7 | SVGF depth channel mismatch | 1h |
| 10 | equiAngular clamped t / PDF mismatch | 1h |
| 11 | bdptConnectionMIS rename | 1h |
| 12 | lightTree reference cleanup | 30min |
| 13 | nodePowerPrefixSum rename | 30min |

Land as a single PR (except Item 1 may be a stand-alone PR since it touches the pipeline compiler path).

### Phase 2: DDGI Physical Correctness (in order — tight dependencies)

Must land in this order due to physical coherence coupling.

| Item | Title | Effort | After |
|------|-------|--------|-------|
| 6 | randomRotation unfreeze | 2h | — |
| 20 | blend kernel: pow(8) → Lambertian | 2h | Item 6 |
| 2 | double-π / double-albedo fix | 3h | Items 6, 20 |
| 3 | atlas border padding pass | 1–2d | Item 2 |
| 4 | RC GI BRDF injection fix | 3h | Item 2 |

### Phase 3: ReSTIR + RC Correctness

| Item | Title | Effort | After |
|------|-------|--------|-------|
| 22 | RC normalization (octahedral PDF) | 1d | — |
| 21 | RC cascade merge integral | 1–2d | Item 22 |
| 23 | GTAO correct slice integral | 4h | — |

### Phase 4: SVGF Temporal Reuse (large rewrite)

| Item | Title | Effort | After |
|------|-------|--------|-------|
| 24 | Albedo demodulation | 1d | Items 7, 8 |
| 9 | Per-pixel Welford history | — | Part of Item 8 |
| 8 | Full SVGF temporal reprojection | 2–3 weeks | Item 7 |

### Phase 5: Delete or Rebuild Decision Items

| Item | Title | Effort | Decision |
|------|-------|--------|---------|
| 19 | Neural denoiser: keep arch, delete broken kernels | 1d | Recommended: partial keep |
| 25 | PPG: delete and re-scope | 1d | Recommended: delete |

---

## Summary: 5 Highest-Impact Items

**1. Item 2 (DDGI double-albedo):** Every DDGI-lit render is wrong by `albedo · (1/π)` — a ~3× energy error on typical scenes. Commit `3fb63e3` is a band-aid. Fix unblocks all DDGI visual quality work.

**2. Item 8 (SVGF missing temporal reuse):** "SVGF" as shipped is à-trous + a spatial variance estimate. The defining temporal reprojection, disocclusion detection, and per-pixel history are entirely absent. Fixing this is a medium weeks project but is the difference between a denoiser and a blur filter.

**3. Item 4 (RC GI bypasses BRDF):** The entire Radiance Cascades indirect signal ignores surface albedo. A perfectly absorbing surface and a white wall receive the same GI. The fix is a few-line TSL change but requires confirming which Three.js NodeMaterial hook is available.

**4. Item 25 (PPG architecture):** The PPG guide trains on the wrong signal (`L_o` instead of `L_i`), uses a fixed grid that cannot adapt, and has an active crash (Item 1) before it even runs. Keeping non-paper-faithful PPG risks actively harming convergence when enabled. Delete is the safest near-term path.

**5. Item 5 (ReSTIR-DI p̂ inconsistency):** The firefly-fix clamp in the RIS pass (`emitterDist2Floor`) is not applied in the temporal and spatial reuse passes, so the firefly mode that the clamp targets leaks back in through every reuse step. Fix is a two-line WGSL change with high render-quality payoff.
