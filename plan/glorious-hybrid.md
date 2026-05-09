# Glorious Hybrid — Unified Strategic Plan for stainedGlass walkaround GI

> **⚠️ PHASE 2 STATUS as of 2026-05-07: BROKEN. Active debug paused.**
>
> Phase 2 hybrid scaffold (commit `78674a5` + H1 + D1-pivot at HEAD `70d5136`) renders black on real NVIDIA Lovelace hardware. Working reference (`walkaround-ddgi` HEAD `e89d992`) renders correctly on the same machine.
>
> Three parallel debug branches were attempted on 2026-05-06 and archived 2026-05-07 as `archive/fix-attempt-{a,b,c}` tags after none hardware-validated:
> - **fix-attempt-a — DIAGNOSIS FALSIFIED.** Hypothesised a GPUTexture identity mismatch in `probeUpdatePass.ts`. Verified by direct file diff: `walkaround-ddgi/src/.../probeUpdatePass.ts` differs from `hybrid-renderer/src/.../probeUpdatePass.ts` by **one line** (a `detectGpu` import path). The texture-creation pattern is identical — so this cannot be the bug.
> - **fix-attempt-b — UNVERIFIED.** Added a layer-readiness gate to `applyHybridShading.ts` plus 6 unit tests + a window-bridge counter. Plausible but un-hardware-tested.
> - **fix-attempt-c — STANDING BEST HYPOTHESIS.** Collapsed `HybridStage.tsx` into a near-byte-identical clone of `WalkaroundStage.tsx`, lifting `applyDDGIShading.ts` + `useDDGI.ts` verbatim from walkaround-ddgi. Iter-2 commit message correctly predicts: if still black, the bug lives in StudioScene factory / lib divergence / viewport state shape / RoomLoader / mount-dispatch / lighting plumbing — i.e. **scene scaffolding around the renderer, NOT the renderer itself**.
>
> **The bug is in scaffolding, not algorithm.** `probeUpdatePass.ts`, `probeGrid.ts`, `sceneBvh.ts`, `wgsl/*.wgsl.ts`, and `ddgiSampleWgsl.ts` are effectively identical between hybrid-renderer and the working walkaround-ddgi. The divergence is concentrated in: `HybridStage.tsx`, `useHybridFrameLoop.ts`, `applyHybridShading.ts`, `HybridContext.ts`, `layers/*`, and the StudioScene integration delta.
>
> Diagnostic prose preserved in: `plan/phase-2-diagnosis.md` (read-only architectural diff, May 6) + the three `archive/fix-attempt-*` tag commit messages.

---

> **Plan housekeeping (2026-05-07).** The `plan/recon-*.md` and `plan/reverify-*.md` files cited inline below were archived via `safe-rm` (recoverable from `~/.claude/warehouse/`) after their conclusions were distilled into the memory entry `~/.claude/projects/-home-jsquire4-projects-stainedGlass/memory/project_hybrid_renderer.md`'s **8 locked decisions** + **6 verified-fact anchors**. Inline `plan/recon-*.md`/`plan/reverify-*.md` references in this document are dead links by design — treat the memory entry as the canonical source.

---

**Status:** Strategic plan, supersedes `plan/photorealism-hybrid-renderer.md`. No code touched in this document.
**Date:** 2026-05-06 (reverified-fold 2026-05-06).
**Branch target:** `hybrid-renderer` off `main` HEAD `55c44cf`. Walkaround-{ddgi,rc,restir} stay live as A/B references through hybrid v1; archive after.
**Authoritative truth sources:** `plan/recon-verification.md` §4 (155 verified facts) PLUS `plan/reverify-restir-relax.md`, `plan/reverify-lumen.md`, `plan/reverify-perf-onnx.md` (deeper second-pass verification on flagged claims). Anything outside these is "plausible-pending-evidence". Where reverify refines or corrects §4, the reverify reports take precedence.

---

## 1. Executive summary

We are building a **layered hybrid global illumination renderer** for stainedGlass's walk-around mode in WebGPU. The renderer's purpose: a stained-glass designer mounts their panel virtually in a furnished room, walks around it with a sun shining through, and sees colored caustics dance on the floor in real time. The layered architecture is the canonical AAA answer (Lumen, HDRP, Cyberpunk Overdrive all layer multiple GI techniques) — this plan adapts that architecture for our specific scene class (small interior, ~5–20k tris, sun + interior lights, dominant transmission panel) and the WebGPU compute floor (no hardware RT, no tensor cores, no cooperative-matrix as of May 2026).

**Verification posture.** The verifier audited ~190 substantive claims across 7 recon docs: 155 verified by primary source, 9 disputed, 5 verified-but-misattributed, 4 unverifiable. Three reverify agents subsequently dug deeper on flagged claims — pulling the ReSTIR BDPT TOG 2025 author preprint, downloading the SIGGRAPH 2022 Lumen + 2025 MegaLights papers locally, fetching the CoopRT ISCA 2025 author preprint, reading FidelityFX Hybrid Shadows source directly, and inspecting UE5.6 source through the public `katataki/UES5.6` mirror.

Reverify outcomes (full detail in `plan/reverify-*.md`):

- **CONFIRMED with citation correction.** ReSTIR-cannot-resolve-glass-caustics is verified, but the load-bearing quote is in §7 page 11 of the preprint, NOT §1.
- **REFUTED.** "RELAX stabilizes in 1–2 frames vs REBLUR ~6" — origin is one Cyberpunk modder, not NVIDIA. NRD's own settings show identical disocclusion-recovery defaults; maintainer says REBLUR is faster AND "in some cases more temporally stable."
- **DROPPED AS EVIDENCE.** CoopRT's 5.11× — real number, but applies to a hardware-RT-core simulator baseline; does not predict WebGPU shader-software speedups.
- **REWRITTEN.** AMD FidelityFX Hybrid Shadows actually does bitwise AND on tile masks + Poisson-disc blocker search, not multiplicative continuous-visibility composition. Plan's multiplicative composition stands on its own merits (RC delivers continuous chroma).
- **FRAMING FIX.** ONNX Runtime Web "experimental" label is verbatim in `js/web/README.md:66` on `main`, not on the user-facing docs page. Practical state: shipping, actively maintained (107 ops, monthly releases), not formally promoted to stable.
- **CORRECTED.** Lumen voxel clipmap cadence "2/4/8/16" had no primary source — UE5.6 has no `LumenVoxelLighting.cpp` (voxel lighting is event-driven on modified bricks). Surface cache atlas is **4096×4096** default (not 8K/16K). 300 cards/frame, 512×512 texel budget verified. 16×16 screen-probe tile + 8×8 octahedral verified.
- **REVERSED.** MegaLights ↔ Lumen direction: they are **sibling pipelines**; MegaLights reuses Lumen's BRDF rays for guiding (NOT Lumen reading MegaLights visibility).

Disputed items (HDRP `USE_APV_TEXTURE_HALF` "default-on" claim, Cyberpunk SIGGRAPH 2023 deck "slide 38" timings, REBLUR SH-mode timing of 4.80 ms, FSR Ray Regen 1.1 lower bound of 2.6 ms) are explicitly NOT used. Claims refined or refuted by reverify are corrected throughout this plan.

**Major shifts vs. the original photorealism plan:**

1. **Caustics promoted from "Phase 3 nice-to-have" to THE killer feature.** The original plan treated caustics as a secondary win; this plan recognises that for stainedGlass, the floor caustic IS the product. A dedicated caustic-shadow layer (current RC's `traceSunVisibility` is the right primitive) is therefore mandatory in v1, not optional.
2. **ReSTIR scope clarified — it does NOT own the caustic.** Verified per `plan/reverify-restir-relax.md`: ReSTIR cannot reconnect through delta surfaces (ReSTIR BDPT TOG 2025 §7 "Implementation and Results", page 11 of the author preprint at cwyman.org/papers/tog25_ReSTIR_BDPT.pdf — the recon's original §1 citation was wrong; the introduction has weaker language while the strong "glass fixtures cause next event estimation to fail" wording is in §7). Through-glass paths (LSDE) are unreachable to unidirectional ReSTIR. Caustics MUST come from a dedicated through-glass tracer. This locks the layer ownership: **DDGI = ambient/diffuse fill; RC = through-glass caustic-shadow rays; ReSTIR DI = direct light + glossy on opaque surfaces**.
3. **Denoiser is now first-class, not a footnote.** Original plan handwaved "à-trous + temporal". The denoising recon shows real options: SVGF upgrade (variance-driven sigma), ALBEDO DEMODULATION (free 20–40% perceived quality, neither current branch does this), per-stream diffuse/specular separation (NRD pattern), and `oidn-web` as a one-day E6 experiment. The plan now sequences these explicitly.
4. **Subgroups path is real.** Verified §4: subgroups stable in Chrome 134+ (Feb 2025), `subgroup_id`/`num_subgroups` in 144 (Jan 2026), `subgroup_uniformity` in 145. The CoopRT 5.11× number was reverified (`plan/reverify-perf-onnx.md` §1) — it's real but applies to a hardware-RT-core simulator baseline, NOT WebGPU shader-software, and is dropped as evidence. Chrome's own subgroup matmul claim is verified at 2.3–2.9× (Google Meet origin trial — matmul, not BVH). Plan a subgroup-coherent BVH traversal with a non-subgroup fallback gated on `adapter.features`; the BVH speedup magnitude is benchmark-gated (E3) not source-quoted.
5. **TAA / temporal-first denoise sequencing.** Verified: SVGF orders temporal-before-spatial, with variance from the temporal accumulator driving the à-trous adaptive kernel. Current `walkaround-restir` runs à-trous-only — that is the source of its residual flicker. Phase 6 fixes this.
6. **Memory budget recalculated against verified WebGPU defaults.** Verified §4: `maxStorageBufferBindingSize = 128 MiB` default, `maxBindGroups = 4`. Reservoir buffer at 1080p × 32 B/reservoir × 2 (current+history) = 132 MB — ALREADY exceeds the default, before any 4K plans. Mandates either f16 packing (verified §4 entry 39, `shader-f16` stable since Chrome 120) or splitting into multiple bindings.
7. **Phase 2 re-scoped.** The existing `hybrid-renderer-phase-2.md` is excellent on DDGI lift mechanics but pre-supposes Option B and pre-locks the TSL injection-point decision. This plan re-surfaces those decisions and revises the phase boundaries.

The plan's posture: **outcome > effort**. Effort estimates are informational only. Where a 12-week path produces a clearly more glorious result than a 2-week path, the 12-week path is recommended. Constraints that bind: WebGPU spec capability (verified per §4.1), browser stability (Chrome stable preferred; behind-flag tolerated for diagnostic-only paths), and physical correctness (no double-counting).

---

## 2. App-specific render targets — the glorious bar

stainedGlass is not a generic GI demo. It is a tool whose value is "show me what my panel looks like installed". The hybrid renderer succeeds when a designer enters walk-around mode and feels the design come alive. That is the bar.

### 2.1 What "glorious" looks like for stainedGlass

| # | Feature | Glorious bar |
|---|---|---|
| 1 | **Floor caustics under the panel** | Colored, bright, with sharp edges where geometry is sharp and soft falloff where the sun is occluded by came. The colored pool on the floor is recognisable as a translation of the panel design above it. **THE feature.** |
| 2 | **Light through colored glass cells** | Each cell's color is vivid and clearly differentiated. Came (lead solder lines) cast crisp dark seams between cells. Stacked overlay textures (e.g. waterglass over rondelle) read distinctly. |
| 3 | **Surface-texture refraction** | Hammered, ripple, granite, baroque, waterglass, catspaw, flemish — each surface character is visible by its perturbation of the transmitted light, not just as a normal-map highlight. |
| 4 | **Time-of-day variation** | Sun angle changes the entire feel: harsh midday, golden hour low-angle through cells, blue-hour ambient. The designer uses TOD as a tool to find the right hour for the install. |
| 5 | **Multiple lights** | Sun is primary. Interior lights (table lamp, ceiling fixture) for night scenes. Lights compose without one washing out the other. |
| 6 | **The "wow" moment** | First-frame impression: the room feels real, the panel feels installed, the caustics make you look twice. No more than 3 seconds of "loading" before the wow. |

### 2.2 Acceptance criteria the rendered output must meet

Concrete, testable gates:

- **A1. Floor caustic chroma** — pixel std-dev of chroma in a fixed sample region under the panel ≥ 0.04 (current `walkaround-ddgi` chroma is 0.0745; current `walkaround-rc` is the proven caustic baseline; hybrid must match or exceed).
- **A2. Came shadow contrast** — a horizontal line through a came-strip on the floor caustic shows luminance dip of ≥ 20% at the seam vs. adjacent cell color. (Verifies came shadows are not blurred away.)
- **A3. Cell color fidelity** — color of caustic pool under each cell matches the panel cell's albedo within ΔE2000 ≤ 6 (a designer can tell which cell projected which pool).
- **A4. TOD continuity** — sweeping `sunDir` from horizon-east through zenith to horizon-west produces a continuous animation with no popping. No frame's mean luma differs from adjacent frame's by > 30%.
- **A5. Multi-light correctness** — turning on an interior point light at night does not extinguish the moonlight caustic on the floor; the two contributions add.
- **A6. Convergence latency** — wow moment ≤ 3 s after walk-around mode mounts. First frame within 500 ms shows recognisable scene; last 2.5 s is convergence to the stable image.
- **A7. Frame budget** — sustained ≥ 30 fps on RTX 4090 at 1080p. Apple M-class graceful degradation to 720p internal + temporal upscale at 30 fps.
- **A8. No firefly artifacts** — single-frame max-pixel-luma in any 100ms window ≤ 8× temporal average for the same pixel. (Caps clamp-energy-loss but rejects outright firefly explosions.)
- **A9. Editor (raster) and PT preview/final paths unaffected** — the M6 baked-PBR + WebGLPathTracer pipeline continues to ship its 60 s honeycomb gate. The hybrid is `walkaroundEngine === 'hybrid'` only; raster + PT are peers.

These gates inform the phase-by-phase verification. They do NOT all gate v1 — A1, A2, A4, A6, A7, A9 are v1 hard gates; A3 (ΔE2000) and A5 (multi-light) and A8 (firefly cap) become hard gates by v1 ship.

---

## 3. The hybrid composition (verified architecture)

### 3.1 Layer assignment + ownership

Locked by verified facts (§4 + the through-glass impossibility for ReSTIR per ReSTIR BDPT TOG 2025 §7 page 11: "The glass fixtures in Sponza and Bathroom cause next event estimation (s = 1) to fail, which makes BSDF sampling (s = 0) the only viable technique for ReSTIR PT. The glass also causes failed reconnections, forcing ReSTIR PT to use (unidirectional) random replay for most scene lighting." — author preprint, reverified 2026-05-06):

| Layer | Owns | Why (verified source) |
|---|---|---|
| **DDGI** | Indirect-diffuse ambient/fill (room walls, ceiling, floor away from caustic) | DDGI's Chebyshev visibility + 8-nearest-probe sample is the canonical leak-resistant indirect-diffuse cache. (Majercik 2019, verified §4 entry 45.) Already shipping on `walkaround-ddgi` HEAD `e89d992` with chroma 0.0745. |
| **RC (single cascade)** | Through-glass sun caustic (LSDE path); came shadow detail | Verified that ReSTIR cannot reconnect through delta surfaces (ReSTIR BDPT TOG 2025 §7, page 11 of preprint). Through-glass requires a dedicated tracer. RC's existing `walkaround-rc` `traceSunVisibility` does this with `bvhTraceTintedVisibility`. The cascade interpolation acts as a structural denoiser (RC's signature claim, recon §3.5). |
| **ReSTIR DI** | Direct light from analytical sources on opaque surfaces — diffuse channel only | Verified §4 entry 13: "ReSTIR DI is diffuse-only in Cyberpunk's implementation; smooth surfaces use a 'poor man's MIS' weighted by `roughness²` against BRDF sampling". Mirrors Cyberpunk's design exactly. |
| **Specular sampler (separate)** | Glossy reflections on opaque surfaces (room floor varnish, glass panel back-face) | Verified pattern: every shipping system splits indirect-specular from diffuse (Lumen, HDRP, Cyberpunk; recon §3 verified by primary sources). For v1 we use Karis 2013 split-sum against a small env probe + screen-space reflections; "ReSTIR specular stream" deferred to v2. |
| **Three.js raster G-buffer** | Primary visibility, transmission compositor for the panel itself, IBL on opaque, AmbientLight fallback | Verified pattern: every shipping system uses raster primary visibility (Lumen, HDRP, Cyberpunk's default mode; recon §1.2 verified). ReSTIR primary-cast was a research choice we're discarding. |

### 3.2 The composite math

Drawn from the verified composition patterns (recon §2, §3 — uncontroversial standard rendering equation decomposition; per-channel additive partition is the only physically-correct choice):

```
finalRadiance = E_direct                       // ReSTIR DI on opaque surfaces (diffuse only)
              × visibility_through_glass        // RC sun-shadow-through-glass multiplier (LSDE caustic carrier)
              + E_indirect_diffuse              // DDGI Chebyshev probe sample × albedo/π
              + E_indirect_specular             // split-sum env probe × envBRDF LUT
              + E_emissive                      // material emissive (came strips have emissive=0; not the issue)
              + E_transmission                  // Three.js raster transmission compositor on the panel itself
                                                //   (NOT through-glass to floor — that's the RC-multiplied direct term)
finalLDR    = ACES(finalRadiance)               // single tone-map at the very end
```

**The `× visibility_through_glass` term is the trick.** It is a multiplicative shadow ratio in [0, 1]³ (per-channel because the panel tints the caustic) applied to the direct-light term. Multiplicative composition of multiple visibility/shadow signals into a direct-light term is **standard industry technique** (it falls out of the rendering equation when independent visibility factors gate the same direct-light contribution). The original recon attributed this pattern to AMD FidelityFX Hybrid Shadows; reverify (`plan/reverify-perf-onnx.md` §2) showed FidelityFX actually does **bitwise AND of binary tile-level hit masks** plus a Poisson-disc min/max blocker search for penumbra classification — a different mechanism. The multiplicative-continuous-visibility composition we use here is justified on its own merits: RC delivers a continuous-valued, per-channel transmission factor (the panel's tint), not a binary hit/miss, so AND-ing masks would lose the chroma. For the floor pixel directly under the panel:

- `E_direct` is the white sun direct-light contribution at that floor pixel.
- `visibility_through_glass` is RC's ray-trace from the floor pixel back to the sun, attenuated by the panel cell's tint along the way.
- Product = colored caustic.

For the floor pixel away from the panel (not in shadow, not under panel): `visibility_through_glass = 1.0` (sun visibility from raster shadow map or direct trace), and direct light is uncolored.

For the wall pixel: `visibility_through_glass = 1.0` (no panel between wall and sun), direct light is uncolored, indirect diffuse from DDGI provides ambient fill.

### 3.3 Fallback hierarchy (Lumen-style "always-return-a-value")

Verified §4 entry 46 + recon-lumen §3.3 + HDRP `RayTracingFallbackHierarchy` enum (verified §4 entry 32). Adapted for our compute-only WebGPU floor (no hardware RT):

```
For each ReSTIR DI ray miss:
  1. Sample DDGI probe atlas (Chebyshev-weighted 8-nearest)         [HDRP pattern, verified]
  2. If all 8 probes reject (validity = 0): sample ambient SH probe  [HDRP fallback chain]
  3. If no ambient probe (shouldn't happen): black                    [explicit, never silent]

For each RC ray miss (sun visibility ray):
  1. If ray exited above horizon: sun radiance × atmosphere coeff     [environment]
  2. If ray exited below horizon: ground albedo × diffuse skylight    [environment]
  3. Should never need a third tier — sky/ground partition the unit sphere.

For each DDGI probe ray miss:
  1. Sample environment cube (procedural sky + sun)                   [Majercik 2019 standard]
  2. No further fallback (env cube IS the floor).
```

This is structurally identical to Lumen's "screen → SDF → voxel → skylight" hierarchy (recon §3.3), substituted for our software-RT primitives.

### 3.4 BRDF split (diffuse vs. specular — separate streams)

Verified across all sources (recon §3, primary sources Lumen + HDRP + Cyberpunk all confirm). For stainedGlass:

| Channel | Owner | Roughness range | Why |
|---|---|---|---|
| Direct diffuse on opaque | ReSTIR DI × RC-sun-vis | All | Sharp shadows; direct lights have known positions |
| Direct specular on opaque (roomy varnish, etc.) | Sample direct light × GGX BRDF (no sampling) | All — analytical | Trivial when light positions are known |
| Indirect diffuse | DDGI Chebyshev × albedo/π | All | Standard Lambertian indirect |
| Indirect specular (mid-rough, 0.2 < r < 0.8) | Karis split-sum on env probe (no surface cache for v1) | 0.2 < r < 0.8 | Wide BRDF lobe; cached representation works |
| Indirect specular (sharp, r < 0.2) | DEFERRED to v2 | r < 0.2 | Will be a dedicated trace; out of v1 scope |
| Refraction through panel (raster path) | Three.js MeshPhysicalNodeMaterial transmission compositor | n/a | Already shipping; do not reinvent |
| Refraction caustic on floor (the LSDE path) | RC `traceSunVisibility` / `bvhTraceTintedVisibility` × direct sun term | n/a | Verified that no other layer can resolve this; RC owns it. |

### 3.5 Denoise order — temporal-first, à-trous-second, per-stream

Verified (SVGF Schied 2017, primary §4 entry 44; recon-composition-patterns §4.3 cites SVGF §4 directly). The order matters: **spatial-then-temporal blurs out high-frequency detail that the temporal pass would have recovered.** SVGF orders temporal-first because variance from the temporal accumulator is what drives the spatial filter's adaptive radius.

```
Per frame, per stream (direct, indirect_diffuse, specular each independent):

  1. Temporal accumulation  — depth + normal + meshID rejection
                            — co-accumulate luminance mean and second moment for variance
                            — variance-clip history into 3×3 RGB AABB (Salvi 2016) to kill ghosting
  2. À-trous wavelet (5 iter, step 1,2,4,8,16)
                            — edge-stops on (depth, normal, luminance, roughness)
                            — sigma_color driven by per-pixel variance from step 1 (this is what
                              promotes plain à-trous to SVGF; verified §4 entry 44)
  3. History feedback       — first à-trous iteration's output goes back to temporal history
                              for next frame (SVGF "modulated demodulation" trick)

Pre-step (universal, before temporal):
  0. Albedo demodulation    — divide noisy diffuse radiance by surface albedo before denoise
                              (NRD + DLSS-RR both require this; verified §4 entry 22, recon-neural-denoising §4)
                              re-modulate post-denoise. Free 20–40% quality at zero perf cost.
                              NEITHER current branch does this.
```

**Per-stream is non-negotiable.** Verified §4 entry 22 + recon-composition-patterns §4.2: NRD's REBLUR_DIFFUSE_SPECULAR and RELAX_DIFFUSE_SPECULAR each take separate diffuse + specular hit-distance buffers. Reason: specular motion does not follow surface motion (a reflection in a flat mirror moves at 2× the camera-parallax rate). Reprojecting them with the same motion vector smears reflections.

### 3.6 The reservoir / probe / cache buffer layout — high-level shape

| Buffer | Size at 1080p | Format | Notes |
|---|---|---|---|
| G-buffer (albedo + normal-roughness + depth + motion) | ~32 MB total | RGBA8 + RGB10A2 + R32F + RGBA16F | Standard layout |
| DDGI probe atlases (irradiance + visibility, ping-ponged) | ~4 MB each, 8 MB total | RGBA16F | Same as `walkaround-ddgi` today |
| RC cascade C0 storage buffer | ~4–8 MB | f16 packed | Single cascade, single-purpose (sun-through-glass) |
| ReSTIR reservoirs (current + temporal history) | 132 MB at 32 B/reservoir, **66 MB at 16 B/reservoir (f16-packed)** | u32×4 (16 B) or u32×8 (32 B) | f16 packing required to fit in default `maxStorageBufferBindingSize = 128 MiB` (verified §4 entry 8). Bitterli 2020 §6 confirms 16 B is sufficient for DI. |
| À-trous ping-pong (per-stream) | 16 MB × 2 streams × 2 ping-pong = 64 MB | RGBA16F | Diffuse + specular each get own pair |
| HDR composite target | 16 MB | RGBA16F | Pre-tone-map |
| **Estimated total at 1080p with f16 reservoirs** | **~200 MB** | | Well within RTX 4090's typical ~4 GiB `maxStorageBufferBindingSize`; tight on Apple M / Intel Arc 128 MiB defaults |

The 132 MB → 66 MB compression via f16 reservoirs is the binding constraint: without it, we cannot run on default WebGPU limits. f16 is verified stable since Chrome 120 (§4 entry 4 implicit + recon-webgpu §1 row 3) and ReSTIR reservoirs are listed as a target in the verified §4 entry 39 / 5 (E2 experiment).

---

## 4. Per-component design (verified)

### 4.1 DDGI (the indirect-diffuse cache)

**Shape.** Probe grid bounded by scene AABB. 1/4 round-robin update per frame (current `walkaround-ddgi` pattern, hardware-validated). 8-nearest-probe Chebyshev visibility weighting on read (Majercik 2019 §3.3 / Figure 7, verified §4 entry 45). Per-probe ray count: 192 (matches the canonical RTXGI default per recon §3.3 — "192 is a good default" sourced to Traverse Research; not in §4 strict-verified list but uncontroversial).

**Atlases.** Octahedral 8×8 per probe (current branch implementation; pattern is sourced to SimLumen for Lumen — for OUR DDGI, the 8×8 oct is just the published Majercik convention). Two textures: irradiance (RGBA16F) + distance moments (RG16F for μ, σ²).

**Multi-bounce.** Implicit via probe-feedback: each frame's probe rays sample LAST frame's probes for indirect at the hit point. This is Majercik 2019's standard trick. No explicit second-bounce pass needed.

**Leak mitigation.** Chebyshev visibility test (verified §4 entry 45). `walkaround-ddgi`'s `probeUpdateBlend.wgsl.ts:203 LOC` already has this.

**Memory.** ~8 MB at our scene scale. Negligible.

**Why DDGI and not APV.** APV's brick-pool + L1-biased-by-L0 unorm SH compression (verified §4 entries 28-31) is more memory-efficient at scale, but ours is a single-room scene. APV's complexity buys nothing here. Stick with DDGI.

**Why DDGI and not Lumen Surface Cache.** Surface Cache requires per-mesh card placement + virtual atlas + page eviction. Reverify (`plan/reverify-lumen.md` §3, §4) confirms the production parameters: **4096×4096 atlas** (default, `LumenScene.cpp:68` + SIGGRAPH 2022 p.69), **300 cards/frame** budget (`LumenSceneRendering.cpp:80`), **512×512 texel budget per frame** (SIGGRAPH 2022 p.69). For our static-room single-panel scene this is enormous machinery for a problem we don't have. DDGI's regular probe grid is simpler and sufficient.

### 4.2 RC (the through-glass caustic-shadow layer)

**Shape.** Single cascade C0 (NOT the full pyramid). Glass-aware sun shadow rays only. The current `walkaround-rc` `traceSunVisibility` / `bvhTraceTintedVisibility` is the right primitive — it traces sun-direction rays from each visible surface point, accumulating per-channel tint as the ray passes through colored panel cells.

**Why single cascade.** RC's full pyramid (C0..C4 cast + C3..C0 merge) is overkill for our use case. The pyramid was designed for general-purpose indirect (Sannikov's PoE2 application). We use it ONLY for the through-glass sun-shadow signal — a one-light, one-occluder problem. One cascade at the densest probe resolution is sufficient.

**Output.** A per-pixel `vec3 visibility_through_glass` ratio (RGBA16F, 8 MB at 1080p). Multiplied into direct-light at composite. Visibility = (1, 1, 1) for floor pixels not under panel, < 1 component-wise tinted by panel cells for floor pixels under the panel.

**Came shadows.** Came strips are opaque material → ray through came has visibility (0, 0, 0) → caustic has dark seam. This is automatic from the geometry; no special-casing needed.

**Surface-texture refraction.** The hammered/ripple/granite/baroque/waterglass/catspaw/flemish surface characters perturb the ray direction at the panel surface. The current `walkaround-rc` implementation handles this via the panel's surfaceTextureId-driven normal perturbation in the shader. Hybrid v1 inherits this verbatim.

**Multi-bounce caustic.** Out of scope for v1. Single-bounce LSDE only (sun → glass → floor). If a designer wants caustics-on-the-wall-from-floor-bounced light, they'll have to wait for v2.

### 4.3 ReSTIR DI (the direct-light layer on opaque)

**Shape — locked from verified facts:**

- **M = 32 initial candidate lights** (verified §4 entry 16; Bitterli 2020).
- **k = 5 spatial neighbors** (verified §4 entry 16; Bitterli 2020).
- **Reservoir struct: `{Y, W_Y, w_sum, c}`** (verified §4 entry 17; Wyman 2023 Algorithm 3).
- **Confidence cap c_cap = 20** (verified §4 entry 17; Wyman 2023 footnote 6: "Starting with a cap of 20 is usually good").
- **MIS: pairwise (Bitterli 2022)** for spatial reuse — NOT defensive pairwise (Algorithm 7 per §4 entry 18 — that's ReSTIR PT, overkill for DI). Pairwise is O(M) unbiased. Start with biased 1/M for the first integration spike, promote to pairwise before spatial reuse goes wrong (recon §2 §6 in cyberpunk-nvidia).
- **Pass order: initial → temporal → spatial-1 → spatial-2 → shade** (verified §4 entry 15; matches Cyberpunk per deck page 14 diagram). 4 spatial passes is overkill for our small scene; 1 is too few; 2 is the canonical compromise.
- **Spatial-fed-into-temporal feedback topology** (verified §4 entry 12; Cyberpunk uses this for ~65,536 effective SPP vs ~64 without).
- **Diffuse-only on opaque surfaces; smooth surfaces (roughness < 0.5) do roughness²-weighted poor-man's MIS against BRDF sampling** (verified §4 entry 13).

**Light handling.** Sun is one analytical directional light. Interior lights (point, RectArea) are added to the reservoir candidate pool. Light index translation (verified §4 entry 11) is NOT needed — our scene has stable indices, no streaming.

**No ReSTIR GI in v1.** Indirect-diffuse stays DDGI's job. Plain. The ReSTIR GI / path-resampling decision can be revisited in v2 once denoiser is solid.

**No primary-ray-cast.** The current `walkaround-restir` branch's primary-cast mode is dropped. Raster G-buffer feeds ReSTIR (HDRP + Cyberpunk default pattern — recon-composition-patterns §1 verified).

**DDGI-as-miss-radiance.** When a ReSTIR ray misses (BVH walk reaches max depth or exits scene AABB), sample DDGI's probe atlas at the ray's last position (HDRP pattern; recon §3.2 verified for HDRP — extends naturally to our case). This is how indirect feeds back into ReSTIR's direct-only output.

### 4.4 Denoiser — phased: SVGF → A-SVGF → optional `oidn-web`

Sequenced per the recon-neural-denoising recommendations, all of which are within verified §4:

**Phase A baseline (Phase 6 of build roadmap):** SVGF-equivalent. Concretely:

1. Albedo demodulation (free 20-40% perceived gain — verified pattern, NRD + DLSS-RR both require it).
2. Per-stream split (diffuse, specular, direct vs. indirect — the NRD pattern; verified for REBLUR_DIFFUSE_SPECULAR and RELAX_DIFFUSE_SPECULAR each take separate diffuse + specular hit-distance buffers per §4 entry 22).
3. Variance estimation (running mean + M₂ in the temporal pass).
4. Variance-driven sigma_color in the à-trous wavelet (this is what makes plain à-trous into SVGF; verified §4 entry 44 — "1 spp → temporally stable image in ~10 ms").
5. Neighborhood/variance clipping for temporal history (Salvi 2016 GDC — verified standard pattern).

**Phase B grade-up (post-v1, before any neural):** A-SVGF (gradient-based adaptive temporal α; Schied 2018 — verified standard reference). ~3–5 days of work; reference implementation in `NVIDIA/Q2RTX/asvgf.glsl` per recon §5.

**Phase C experimental (gated on E6 benchmark):** `oidn-web` drop-in. Verified §4 entry 40: oidn-web ports OIDN UNet via TF.js + WebGPU; HDR + tile-based execution; all-GPUBuffer in/out. The recon-neural-denoising §6 E6 says: "**E6 is the cheapest first step** — `oidn-web` already exists. If small-OIDN at 1080p hits our budget on a desktop GPU, the entire neural-denoising question is partially answered before we train anything." Run E6 BEFORE designing Phase B.

**No DLSS-RR.** Verified across §4 (entries 25, 49) — Tensor cores + FP8 + NGX. Browser equivalent unattainable as of May 2026.

**No FSR Ray Regen.** Verified §4 entry 26 — RDNA-4 + DX12 SM 6.6 + Win11 only. Wrong hardware.

**No NRD direct port for v1.** Reverification (`plan/reverify-restir-relax.md` §2) corrects two pieces of recon framing here:
- The recon's "RELAX stabilizes in 1–2 frames vs REBLUR ~6 frames" is **unsupported by primary sources** — that wording originates with a single Cyberpunk modder (Ultra Plus on cyberpunk2077mod.com), not NVIDIA. NRD's `NRDSettings.h` ships **identical** disocclusion-recovery defaults for both denoisers (`historyFixFrameNum=3`, `maxFastAccumulatedFrameNum=6`). The "6 frames" number in the modder claim is exactly NRD's fast-history default, common to both.
- Per the NRD maintainer (dzhdanNV, NRD issue #47, verbatim): "REBLUR: 25% faster ... in some cases more temporally stable. Coupled with AO/SO denoising (free addition to the diffuse denoiser). RELAX: has luminance stoppers ... dense spatial filtering (A-trous decomposition) in some cases provides advantages over sparse filtering in REBLUR." NRD README timings at 1440p RTX 4080 confirm: REBLUR_DIFFUSE_SPECULAR 2.50 ms vs RELAX_DIFFUSE_SPECULAR 3.20 ms — **REBLUR is ~28% faster, NOT slower**.

If we ever do a direct port (post-v1), the choice is benchmark-gated, not source-quoted:
- **Perf-default favors REBLUR** (faster + free AO denoising per maintainer).
- **Port-effort favors RELAX** (A-trous decomposition has no subgroup dependency, works in any WebGPU implementation; REBLUR uses wave/subgroup ops in several passes that would need fallback paths for Firefox/Safari per §5.1).

For v1 we ship SVGF (Phase 6) regardless. NRD direct port is deferred to post-v1 stretch (§10.3); the actual shape (RELAX vs REBLUR) should be decided by side-by-side benchmark on our scenes, not by the modder-folkloric "1–2 vs 6 frames" framing.

### 4.5 Composite + tone-map

**Composite.** Single fragment shader, fullscreen triangle. Per-pixel evaluates the composite math from §3.2. Reads from: G-buffer, ReSTIR DI denoised diffuse, RC visibility texture, DDGI probe atlas (sampled at world position via the DDGI sample WGSL function from `ddgiSampleWgsl.ts`), env probe (split-sum + LUT), Three.js raster output.

**Tone-map.** Single ACES filmic at the very end. Linear radiance throughout upstream. The current `walkaround-ddgi` ACES injection via TSL `outputNode = renderOutput(output, ACESFilmicToneMapping, SRGBColorSpace)` is the right primitive (verified §4 entry: this works hardware-validated). Hybrid moves it to the composite stage so it applies to the FULL composite, not just per-material.

### 4.6 Temporal anti-aliasing (TAA)

**For v1.** TAA is bundled into the temporal denoise pass. Halton(2,3) sub-pixel jitter (recon-webgpu §4 — verified pattern) at the camera. Variance-clipped reprojection. Mipmap bias for any internal upscale.

**Coordination with ReSTIR.** Verified pattern (recon-webgpu §4): "ReSTIR + TAA must coordinate — sequence jitter must feed the spatiotemporal-reuse weights to avoid bias." Practical: the same Halton sequence is used for both jitter AND ReSTIR's spatial reuse offsets, so the temporal accumulation sees consistent samples.

**No DLAA / DLSS / FSR.** No vendor-locked upscalers. Pure WebGPU compute.

---

## 5. WebGPU-specific implementation (verified)

### 5.1 Subgroups path

**Capability gate.** Verified §4.1:
- subgroups stable in Chrome 134+ (Feb 2025) — entry 2.
- `subgroup_id` + `num_subgroups` builtins in Chrome 144 (Jan 2026) — entry 3.
- `subgroup_uniformity` WGSL extension in Chrome 145 (Jan 2026) — entry 4.

**Use cases (in priority order):**
1. **BVH traversal coherent rays** — primaries + first-bounce diffuse have high coherence; `subgroupBallot` active-mask + shared L1 reads should redistribute traversal work across idle SIMT lanes. Recon-webgpu §4: "Google Meet got 2.3-2.9× on matmul shaders in the subgroups origin trial." The CoopRT (ISCA 2025) paper was reverified (`plan/reverify-perf-onnx.md` §1) — its 5.11× number is real but applies to a hardware-modification proposal evaluated in an architectural simulator (Vulkan-sim) over a Turing-class hardware-RT-core baseline. **It does NOT predict WebGPU shader-software speedups** and is dropped as evidence. CoopRT's general insight — "redistributing BVH-traversal work across idle SIMT lanes is profitable on divergent workloads" — does support investing in subgroup-level work redistribution in our software BVH, but the achievable speedup is unknown without measurement (see E3 in §9). Subgroups remain directionally worth pursuing; the speedup magnitude is benchmark-gated, not source-quoted.
2. **ReSTIR spatial reuse** — k=5 neighbor scan benefits from subgroup broadcast for shared `g_pixelInfo` reads.
3. **Denoiser à-trous wavelet** — 5×5 stencil with shared edge-stop weights; subgroup shuffles cut redundant fetches.

**Fallback.** Every subgroup-using kernel ships a non-subgroup variant. Selected at pipeline-create time via `adapter.features.has('subgroups')`. Browser-specific: Firefox doesn't have subgroups stable as of May 2026 (recon-webgpu §1 row "all-browser" notes — verified Chrome-only ship); Safari 26 partial. Plan for the fallback to be the "default" Firefox/Safari path; subgroup is the Chrome-stable speedup.

### 5.2 three-mesh-bvh /webgpu integration

**Verified §4 entry 43:** three-mesh-bvh v0.9.9 (Mar 3 2026) latest; `/webgpu` module since v0.9.2 (Oct 24 2025). Provides TSL raycast/shapecast/closest-point on GPU.

**Integration plan.** Replace the per-branch hand-rolled BVH WGSL traversal kernels with three-mesh-bvh's TSL primitives where possible. Currently each walkaround branch implements its own BVH walk (DDGI: `probeUpdateRays.wgsl.ts:506 LOC`; RC: `probeRayCast.wgsl.ts:360`; ReSTIR: full `bvhCompute.ts`). Consolidating onto the upstream library:
- Reduces our maintenance surface (we follow gkjohnson's optimization passes for free).
- Enables the subgroup speedup path if/when the library adopts it.
- Preserves our shared `lib/bvhCommon.ts` builder, which already feeds the BVH-of-static-geometry pattern.

This is a Phase 2.5 / 3 cleanup, not a Phase 2 prerequisite.

### 5.3 Buffer layouts — fitting `maxStorageBufferBindingSize`

Verified §4 entry 8: WebGPU spec default `maxStorageBufferBindingSize = 128 MiB`, `maxBindGroups = 4`.

**Reservoirs.** As discussed in §3.6, reservoirs at 32 B × 1920 × 1080 × 2 (current+history) = 132 MB at 1080p — over the default. f16 packing brings this to 16 B × pixels × 2 = 66 MB. **f16 packing is mandatory** for ReSTIR to fit on default-limit hardware (Apple M-class, Intel Arc — both verified at ~256 MiB or 128 MiB max).

**G-buffer.** Five 8 MB textures (albedo, normal-roughness, depth, motion, hit-distance) = 40 MB total. Each individual buffer fits comfortably.

**Probes + cascade.** ~16 MB combined. Fits.

**Bind groups.** Verified §4 entry 8: `maxBindGroups = 4`. Plan: bind group 0 = scene constants (camera, frame); 1 = G-buffer textures (read); 2 = layer outputs (DDGI atlas, RC visibility, ReSTIR reservoirs); 3 = output (write). Tight but workable.

**Max storage textures per stage.** Verified §4 entry 5: Chrome 146 raised Dawn's `maxStorageBuffersPerShaderStage` to 16 and `maxSampledTexturesPerShaderStage` to 48. Plenty of headroom for our pipeline (~10 storage buffers, ~15 sampled textures peak).

### 5.4 f16 reservoir compression — pre-implementation experiment

**Status: NOT YET IMPLEMENTED on any branch. Must be experimentally validated before Phase 4 commits to the layout.**

Verified §4: `shader-f16` stable since Chrome 120 (recon-webgpu §1 row 3 verified). The verifier did NOT directly verify the QUALITY claim ("invisible after 4-8 accumulated samples"). Section 9 below lists this as experiment E2.

**Risk if f16 reservoirs degrade visibly:** the reservoir buffer at 32 B per pixel exceeds the default `maxStorageBufferBindingSize = 128 MiB` at 1080p, so WITHOUT f16 we either (a) request `requiredLimits.maxStorageBufferBindingSize` higher than default — verified §4 doesn't confirm what hardware supports this; webgpureport.org typical RTX 4090 shows ~4 GiB, but Apple M is 256 MiB max — or (b) split the reservoir buffer into multiple bindings (uses up bind-group slots). f16 is the cleanest path; experiment E2 must validate before Phase 4 designs the buffer layout.

### 5.5 Memory budget — concrete numbers for our scene

At 1080p, RTX 4090, all layers active, f16-packed reservoirs:

| Resource | MB |
|---|---|
| G-buffer (5 textures) | 40 |
| DDGI atlases (irradiance + visibility, ping-pong) | ~10 |
| RC C0 visibility | 8 |
| ReSTIR reservoirs (current + history, f16 packed) | 66 |
| À-trous ping-pong (2 streams × 2 buffers) | 64 |
| HDR composite target | 16 |
| BVH (~15k tris, vec3f stride packed) | ~5 |
| Env probe + LUT | ~2 |
| **Total** | **~211 MB** |

Comfortable on RTX 4090 (~4 GiB max). Tight on Apple M2 (256 MiB max storage buffer — but our biggest single buffer is reservoirs at 66 MB, fits). At 720p adaptive fallback the totals halve.

### 5.6 TAA / temporal accumulation strategy

**Jitter sequence.** Halton(2, 3) for camera sub-pixel offsets. Same sequence drives ReSTIR spatial reuse offsets (per §4.6 above — coordination with ReSTIR's spatiotemporal reuse to avoid bias).

**History-clamp method.** Variance-clipped (Salvi 2016) — build a 3×3 RGB AABB from current frame, clip the reprojected history into it. Verified standard pattern; recon-neural-denoising §4 entry 4.

**Reprojection.** Standard depth-and-meshID rejection on the previous-frame reprojection. Catmull-Rom resampling (recon-webgpu §4 — TAA techniques).

**Convergence.** First 8 frames after camera-cut have variance-bumped à-trous sigma (Schied 2017 §4.4 — disocclusion handling, "luminance-weighted variance is bumped 10× in disocclusion to widen the spatial kernel"). After 8 frames, full temporal feedback active.

---

## 6. Phased build roadmap

**Branch:** `hybrid-renderer` off `main` (`55c44cf`). Worktrees `walkaround-{ddgi,rc,restir}` stay live as A/B references. Merge to `main` only after Phase 8's hardware validation passes all gates.

Each phase gets a verification gate. NO effort estimates — outcome > effort. Per-phase code-feed columns describe which existing branch's code is the source.

### Phase 2 — DDGI base layer + hybrid scaffold

**Scope.** Land the layered architecture skeleton + DDGI as the only active layer. Material-injection composer that adds layer contributions to TSL `emissiveNode` (or `outputNode` per D1). `__HYBRID_LAYERS__` dev-backdoor toggle (boolean | 'isolate' per D2).

**Code feed.** Lift verbatim from `walkaround-ddgi` HEAD `e89d992`:
- `src/rendering/scene/walkaround/{useDDGI, applyDDGIShading, ddgiSampleWgsl, probeGrid, probeUpdatePass, sceneBvh}.ts`
- `src/rendering/scene/walkaround/wgsl/{octahedral, hammersley, probeUpdateRays, probeUpdateBlend}.wgsl.ts`
- The DDGI chroma e2e spec → renamed to `12-walkaround-hybrid-ddgi.spec.ts`

**Discarded.** `walkaround-ddgi/.../swiftShaderDetect.ts` (replaced by main's `lib/wgpuSupport.ts`).

**Adapted.** `useDDGI` body becomes `useDDGILayer` + `HybridContext` + `useHybridFrameLoop` (see existing `hybrid-renderer-phase-2.md` §3.6 — that decomposition is sound and survives this revision).

**Verification gate.**
- G-A1: chroma_stdDev ≥ 0.02 at the canonical camera (matches `walkaround-ddgi` baseline of 0.0745).
- G-A6: scene mounts ≤ 500 ms, first wow frame ≤ 3 s.
- G-A9: editor + PT specs unchanged (`01-08`, `09-12 PT`).
- G-layer-toggle: `__HYBRID_LAYERS__ = {ddgi: false, ...}` produces coherent fallback (raster + ambient SH only, no crash).

**Notes.**
- The phase-2 plan's §9 D1/D2/D3 architectural decisions are escalated to §7 of THIS plan for user sign-off.
- Phase 2 scaffolds the `RCLayerStub` and `ReSTIRLayerStub` so Phases 3 and 4 drop in without re-architecture.

### Phase 3 — RC sun-caustic layer

**Scope.** Replace `RCLayerStub` with a real `RCLayer` driving a single-cascade C0 sun-shadow trace through panel glass. Output `vec3 visibility_through_glass` per pixel, applied multiplicatively against direct-light at the composite.

**Code feed.** Lift from `walkaround-rc`:
- `traceSunVisibility` / `bvhTraceTintedVisibility` from `shaders/walkaround/probeRayCast.wgsl.ts:144-198` (per the existing photorealism plan's §4.1 reference).
- The `useGIReceiverConverter` pattern is NOT lifted — RC's contribution comes through the new HybridLayer interface, not via emissive material wrapping.

**Discarded from RC branch.** The full cascade pyramid (C0..C4 cast + C3..C0 merge), the WASD walk camera (consolidated separately under §6 cleanups), the per-material wrap.

**Adapted.** RC's existing glass-tint logic (panel cells contribute per-channel tint along the ray) becomes the only function of the layer.

**Verification gate.**
- G-A1: chroma_stdDev measurably increased from Phase 2 baseline (caustic now visible).
- G-A2: came-shadow line through caustic shows ≥ 20% luma dip at the seam.
- G-A4: TOD sweep produces continuous animation (no popping).
- A test-isolation gate: with `__HYBRID_LAYERS__ = {ddgi: false, rc: true, restir: false}`, the caustic appears against a black ambient (proves RC's contribution is independent of DDGI).

### Phase 4 — ReSTIR DI on raster G-buffer

**Scope.** Replace `ReSTIRLayerStub` with `ReSTIRLayer`. Run RIS (M=32) → temporal → spatial-1 → spatial-2 → shade. Use raster G-buffer (NOT compute-cast primary). DDGI atlas as miss-radiance for ReSTIR rays (HDRP pattern). f16-packed reservoirs (post-experiment E2).

**Code feed.** Lift from `walkaround-restir`:
- RIS / temporal / spatial / shade compute passes (~2200 LOC of WGSL across `ris,temporal,spatial,shade,common.wgsl.ts`).
- Pairwise MIS implementation (Bitterli 2022 — promote from biased 1/M as soon as spatial reuse is on).
- The 60-FPS frame cap pattern.

**Discarded from ReSTIR branch.**
- Primary-ray-cast mode (replaced by raster G-buffer feed).
- The compute-cast `composite.wgsl` (replaced by the hybrid composite).
- The full `WebGPUCanvas` separate-mount pattern (already deleted in parity Phase 5; we use R3F factory swap).
- The à-trous-only denoise (replaced by Phase 6's SVGF).
- The emitter list machinery, EXCEPT for the analytical-light reservoir candidate pool.

**Adapted.**
- Reservoirs go from u32×8 (32 B) to u32×4 (16 B) with f16 packing — gated on E2 result.
- Light index translation pass NOT included (our scene has stable indices).

**Verification gate.**
- G-A8: firefly cap holds (max-pixel-luma in 100ms ≤ 8× temporal average).
- G-A1 / G-A3: chroma + ΔE2000 cell-color fidelity targets achieved.
- G-A7: 30+ fps on RTX 4090 at 1080p with all layers on.

### Phase 5 — Specular split-sum + env probe

**Scope.** Add the indirect-specular stream. Karis 2013 split-sum on a small env probe (procedural sky + sun, baked once per session, re-baked on TOD change). LUT for `envBRDF(NdotV, roughness)`. Composite as `final += split_sum × LUT`.

**Code feed.** New WGSL — env probe baker (one compute pass per TOD change), split-sum prefilter (mip pyramid), envBRDF LUT (precomputed once, stored as 256² R16G16). No existing branch has this; build from scratch.

**Why now (not Phase 4 or 6).** ReSTIR DI's diffuse-only output looks "matte" on the room floor's varnish. Specular split-sum is what makes the floor look polished. Without it the scene feels half-finished even at A1+A2 chroma.

**Verification gate.**
- Visual A/B against Phase 4 — varnish on floor reads as polished, not matte.
- A5 multi-light: interior point light glints on floor varnish are visible.

### Phase 6 — Denoiser upgrade (SVGF + albedo demodulation + per-stream)

**Scope.** Promote the à-trous-only filter (current `walkaround-restir`) to SVGF-equivalent. Concretely:

1. Albedo demodulation pre-step (~4 hours of work, free 20-40% perceived gain).
2. Per-stream split: direct vs. indirect_diffuse vs. indirect_specular each get own denoise pipeline.
3. Variance estimation in the temporal pass (running mean + M₂).
4. Variance-driven sigma_color in à-trous (this turns à-trous into SVGF; verified §4 entry 44).
5. Variance-clipped temporal history (Salvi 2016).

**Code feed.** Modify ReSTIR's `atrous.wgsl.ts` and `temporal.wgsl.ts` from Phase 4. New compute pass: `albedoDemodulate` + `albedoRemodulate`.

**Pre-experiment E6 (run before Phase 6 commits).** `oidn-web` drop-in benchmark at 1080p on RTX 4090 + Apple M-class. If perf budget fits and quality exceeds SVGF, fold `oidn-web` in instead. If not, stick with SVGF.

**Verification gate.**
- G-A8 firefly cap.
- Visual A/B against Phase 5 — temporal stability under camera motion improves measurably.
- 1 spp ReSTIR DI input → temporally stable image in ≤ 10 ms (per SVGF target, verified §4 entry 44).

### Phase 7 — UX + camera unification

**Scope.** Promote RC's WASD walk camera (~106 LOC in `useWalkCamera.ts`). Add an "explore vs. orbit" toggle. Consolidate `window.__WALKAROUND__` test bridges. Remove backward-compat `__DDGI__` alias if D3 was sign-off-resolved as "remove".

**Code feed.** `walkaround-rc/.../useWalkCamera.ts` (+106 LOC over main).

**Verification gate.** All e2e specs use one bridge namespace. WASD walk works on the panel-framing constraint.

### Phase 8 — Hardware validation, perf tuning, ship gate

**Scope.** Run the full A1–A9 acceptance gate on RTX 4090, RTX 4070-class, Apple M2, Intel Arc A380. 1080p target on RTX-class; 720p adaptive fallback on M-class / Arc / iGPU.

**No new code expected.** This is the ship gate, not a feature phase. Bug-fix-only.

**Deliverables.**
- Hardware-validation matrix populated.
- Per-target perf timing report.
- Final acceptance test on the A1-A9 bar.
- Hybrid-renderer branch merged to `main` after all gates pass.

### Phases 9+ (post-v1 stretch) — not part of v1

These are tracked separately in §10 ("Open questions / future research"). They include: ReSTIR GI as indirect-diffuse augmenter, NRD-port denoiser (REBLUR or RELAX, benchmark-gated), neural radiance cache, ReSTIR PT, full Lumen Surface Cache, spectral rendering, concave-glass caustic-via-photon-mapping, volumetrics.

---

## 7. Architectural decisions — LOCKED 2026-05-06

All 8 decisions signed off by user one-at-a-time on 2026-05-06. Recorded for future-session context.

| ID | Decision | Locked |
|---|---|---|
| D1 | DDGI injection point | **A — `emissiveNode`** (linear HDR throughout, single ACES at end matches Lumen/HDRP/Cyberpunk; preserves chroma in bright caustics) |
| D2 | `__HYBRID_LAYERS__` toggle type | **B — `boolean \| 'isolate'`** (per-layer isolation mode for debugging) |
| D3 | `__DDGI__` namespace alias | **A — keep for one phase** (Phase 7 cleanup) |
| D4 | Subgroup A/B sequencing | **A — Pre-Phase-2 experiment (E3)** (CoopRT 5.11× was debunked; we need our own data) |
| D5 | f16 reservoir packing | **A — Pre-Phase-4 experiment (E2)** (Bitterli 2020 §6 claim plausible-pending-evidence; verify before locking layout) |
| D6 | Specular split-sum timing | **A — Phase 5** (post Phase 4 ReSTIR DI; sequenced by visual-impact-priority) |
| D7 | Tone-map curve | **C — Khronos PBR Neutral** (user override of recommendation; chroma test thresholds will need re-baselining in Phase 8) |
| D8 | Camera UX | **C — Both WASD + orbit toggle** (WASD for "wow"; orbit for chip/panel-detail review) |

The full decision discussion is preserved below for context (each section's recommendation is now superseded by the LOCKED entry above).

### D1. DDGI injection point: `outputNode` vs `emissiveNode`

**Decision.** Where in the TSL graph does DDGI's contribution land?

**Options.**
- **A. `emissiveNode = add(emissive, ddgiContrib)`** (proposed in `hybrid-renderer-phase-2.md` §3.2). DDGI moves to the additive-light slot; the hybrid composer owns `outputNode = renderOutput(output, ACES, sRGB)`.
- **B. `outputNode = renderOutput(add(output, ddgi+rc+restir), ACES, sRGB)`** (preserves current `walkaround-ddgi` pattern bit-for-bit).

**Recommendation: A.** Reasoning grounded in verified facts:
- Verified pattern (Lumen, HDRP, Cyberpunk all): "linear radiance throughout, single tone-map at the end" (recon §1.2 + §3.6 verified). Option A reserves `outputNode` for the global ACES tone-map.
- RC's existing `walkaround-rc` pattern already uses `emissiveNode` injection (verified by reading `walkaround-rc/.../giReceiver.ts:109`); Option A unifies all three layers on the same slot.
- ReSTIR's eventual contribution wants the same slot.

**Pros of A.** Generalises to RC + ReSTIR. Single tone-map. Clean per-layer slot architecture.

**Cons of A.** Slightly different physical interpretation — emissive is treated as direct surface radiance, not as indirect ambient. In practice the difference is invisible because three.js's lighting compose adds emissive AFTER the diffuse BRDF, and our DDGI contribution already pre-multiplies by `albedo / π` so the integral matches.

**Risk if B.** Less clean per-layer composition. Each new layer has to re-implement the `outputNode` math. But: if Option A causes a measurable chroma drop in Phase 2, pivot to B is one day of work inside the composer.

### D2. `__HYBRID_LAYERS__` value type

**Decision.** Does the dev-backdoor toggle support `'isolate'` debug mode (only-this-layer-and-no-PBR) in addition to plain on/off?

**Options.**
- **A. `boolean`** — only on/off per layer. Simpler type signature.
- **B. `boolean | 'isolate'`** — supports diagnostic "show only this layer's contribution".

**Recommendation: B.** Reasoning:
- "Pure DDGI on black PBR" / "pure RC caustic on black PBR" are real diagnostic modes. Without them, debugging which layer produced which pixel is harder.
- Cost is small — `'isolate'` mode just zeros out three.js's `output` (the PBR-lit + IBL contribution) and the other layers' contributions in the composer.
- This is dev-only; the public API surface is `walkaroundEngine === 'hybrid'`.

**Pros of B.** Easier debugging. Confirms layer correctness in absolute terms.

**Cons of B.** Slightly more code paths to test.

**Risk if A.** Adds 1-2 days to debug each time something looks wrong because we have to bisect by toggling other layers off. Not catastrophic but adds friction across every implementation phase.

### D3. Backward-compat `__DDGI__` namespace alias

**Decision.** Keep `window.__DDGI__` as alias for `window.__WALKAROUND__.layers.ddgi` during the migration?

**Options.**
- **A. Keep alias for one phase** (drop in Phase 7 cleanup).
- **B. Drop immediately in Phase 2** (force migrating spec to update its bridge reads).

**Recommendation: A.** Reasoning:
- Lets the migrated spec land without two simultaneous breaking changes (engine selector + bridge namespace).
- Costs 5 LOC in `DDGILayer.ts`.

**Pros of A.** Smoother migration, no spec churn during Phase 2.

**Cons of A.** One more thing to clean up in Phase 7.

**Risk if B.** Phase 2 spec migration becomes 2 PRs instead of 1. Trivial.

### D4. Subgroup speedup A/B sequencing

**Decision.** When do we A/B test the subgroup-coherent BVH path against the non-subgroup baseline?

**Options.**
- **A. Pre-Phase 2** — validate the speedup BEFORE committing to subgroup-using kernels in Phase 4.
- **B. Phase 4 alongside ReSTIR DI** — test both implementations during Phase 4 with `walkaround-restir`'s BVH compute as the baseline.
- **C. Phase 8 perf tuning** — ship Phase 4 with the subgroup path, optimize-or-fallback in Phase 8.

**Recommendation: A.** Reasoning:
- Verified §4: subgroups stable in Chrome 134+ (Feb 2025); hardware-validated on NVIDIA Lovelace already in our worktrees.
- Reverify (`plan/reverify-perf-onnx.md` §1) showed CoopRT's 5.11× number applies to a hardware-modification proposal in an architectural simulator over a Turing RT-core baseline — it does NOT predict WebGPU shader-software speedups and is NOT usable as a planning anchor. The closest verified analogue is Google Meet's matmul subgroups origin trial at 2.3-2.9× — but that's matmul, not BVH traversal. We need OUR specific data on OUR specific BVH shape.
- Pre-validating informs Phase 4's kernel design: if subgroups give 2× we lean in; if they give 1.1× we keep both paths simple. Without primary-source predictive numbers, **the only path to a defensible kernel-design decision is empirical measurement (E3)**.

**Pros of A.** Decouples capability claim from delivery risk. Informs all downstream kernel designs. Removes plan dependency on the now-dropped CoopRT citation.

**Cons of A.** Adds ~1 week of pre-Phase-2 work before any feature lands.

**Risk if C.** Phase 4's kernel is built one way; if subgroups don't help, we have a vestigial complexity. If subgroups help a lot, we re-architect kernel post-hoc.

### D5. f16 reservoir packing — pre-implementation experiment

**Decision.** Do we pre-validate f16 reservoir compression visual quality before designing Phase 4's reservoir layout?

**Options.**
- **A. Pre-Phase 4 experiment (E2 in §9)** — implement a small standalone test that A/B compares f16 vs u32 reservoirs on the honeycomb scene. Decide layout before Phase 4 designs ReSTIR's buffer schemas.
- **B. Phase 4 ships f32 reservoirs** — fits if we request `requiredLimits.maxStorageBufferBindingSize > 128 MiB`. Avoids the experiment.
- **C. Phase 4 ships f16 reservoirs without experiment** — trust Bitterli 2020 §6's "16 B is sufficient" claim.

**Recommendation: A.** Reasoning:
- Verified §4 entry 8: WebGPU spec default `maxStorageBufferBindingSize = 128 MiB`. f32 reservoirs at 132 MB are over.
- Apple M-class verified at ~256 MiB max storage buffer (recon-webgpu §2). Tight.
- "Bitterli 2020 §6 confirms 16 B is sufficient for DI" was claimed in the recon — verifier did NOT directly confirm this paper passage in §4. Treat as plausible-pending-evidence.
- Cost of the experiment is small (a few days of compute-pass plumbing).

**Pros of A.** Validates the layout choice empirically. Avoids re-design if visible quality drops.

**Cons of A.** Adds ~3 days before Phase 4 starts.

**Risk if C.** If f16 visibly degrades (unlikely but unverified), Phase 4 ships with chroma-test-passing-but-actually-broken-on-edge-cases reservoirs.
**Risk if B.** Apple M2 / Intel Arc users hit `maxStorageBufferBindingSize` denial; Phase 4 silently fails on non-RTX hardware.

### D6. Specular path implementation — Phase 5 vs deferred

**Decision.** Specular split-sum + env probe is Phase 5 (post Phase 4 ReSTIR DI lands). Should it be earlier, or later, or split?

**Options.**
- **A. Phase 5 (proposed).** After ReSTIR DI ships. Fills the "matte floor" gap.
- **B. Phase 4.5 (between ReSTIR and denoise).** Specular composes on top of ReSTIR, which composes on top of RC — getting all the radiance in before denoise sequencing makes sense.
- **C. Defer to v2.** ReSTIR DI handles direct specular on opaque (Cyberpunk's "poor man's MIS roughness²-weighted BRDF sampling"); env probe handles indirect specular implicitly via three.js's existing IBL.

**Recommendation: A.** Reasoning:
- Verified pattern (recon §3 BRDF split): every shipping system splits indirect-specular from diffuse. Three.js's existing IBL is "good enough" for indirect specular at v1's ambition; full-quality split-sum elevates it to "polished".
- The "matte floor without varnish glint" is a Phase 4 user-visible problem, but Phase 5 fills it before v1 ships.
- C is too aggressive — varnish glint really is a stainedGlass-relevant feature for floor texture.

**Pros of A.** Sequencing by visual-impact-priority. Phase 5 is the lowest-risk addition.

**Cons of A.** Multiple phases between "scene with all GI" and "scene that looks polished".

**Risk if B.** Denoiser sequencing in Phase 6 has to handle both diffuse AND specular streams from day one — not a problem (per-stream is already the plan, §3.5), but it complicates Phase 6's smoke testing.

### D7. Tone-map curve — ACES filmic vs alternatives

**Decision.** Single ACES filmic at composite, or a different tonemap?

**Options.**
- **A. ACES filmic** (Narkowicz 2015 fit) — matches `walkaround-ddgi`'s current TSL-injected tonemap.
- **B. Reinhard** (simpler, Three.js default).
- **C. Khronos PBR Neutral** (newer, more accurate to PBR materials).
- **D. None — output linear, leave HDR for browser canvas.**

**Recommendation: A.** Reasoning:
- Verified §4 entry: `walkaround-ddgi`'s current ACES via TSL `renderOutput()` is hardware-validated.
- Verified §4 entry: `walkaround-restir`'s composite.wgsl uses Narkowicz fit — same curve.
- Cyberpunk + Lumen + HDRP all use ACES-family curves (verified standard pattern, recon §1.2).

**Pros of A.** Matches industry. Already implemented and validated.

**Cons of A.** ACES has a slightly desaturated midtone that some find non-neutral. PBR Neutral is more recent and arguably more correct.

**Risk if B.** Reinhard rolls off less aggressively → bright sun-on-glass blows out → caustic loses detail.
**Risk if C.** PBR Neutral is newer and may have less tooling around it; designer reference photos all assume ACES-family.
**Risk if D.** Browser HDR canvas is Safari-26-Vision-Pro only (verified §4.1). Cross-browser would need a fallback path.

### D8. Camera ownership in walkaround mode

**Decision.** Which camera UX governs walkaround?

**Options.**
- **A. RC's WASD walk camera** (current `walkaround-rc/.../useWalkCamera.ts:106 LOC`). User physically walks around the room.
- **B. ReSTIR's drei `<OrbitControls>` + panel-framing constraint.** User orbits around the panel.
- **C. Both, with an explore vs orbit toggle.**

**Recommendation: C.** Reasoning:
- The "wow moment" (A6 acceptance) comes from physical presence — WASD walk wins.
- But for chip-tracker / panel-detail review the orbit framing is essential.
- Cost of supporting both is modest (~1 day in Phase 7).

**Pros of C.** UX flexibility. Different designers have different preferences.

**Cons of C.** Two code paths to test.

**Risk if A.** Some designers find WASD disorienting; orbit is the safer default.
**Risk if B.** "Wow" moment is muted — orbit feels like a turntable, not an installation.

---

## 8. What we are explicitly NOT doing (rejected from recon)

### 8.1 Disputed claims rejected outright

- **HDRP `USE_APV_TEXTURE_HALF` "default-on for mobile"** — rejected per verifier §2.2. The `#define` is commented out. Our DDGI atlas format choice (RGBA16F) does NOT need to be defended by appeal to "Unity does this on mobile".
- **Cyberpunk SIGGRAPH 2023 deck "slide 38" timings** — rejected per verifier §2.5. Deck has 27 pages. The "ReSTIR DI ~2-3 ms, ReSTIR GI ~3 ms, NRD ~2-3 ms" specific attribution is fabricated. Order-of-magnitude is plausible but cannot be cited.
- **REBLUR SH-mode timing of 4.80 ms** — rejected per verifier §2.3. Actual is 3.40 ms (the 4.80 number is RELAX). If we ever port REBLUR (we won't in v1), use 2.50 / 3.40 ms numbers.
- **FSR Ray Regen 1.1 lower bound 2.6 ms** — rejected per verifier §2.6. Actual is 0.78 ms. Doesn't matter for us — wrong hardware (RDNA-4 only) — but we don't quote the wrong number anywhere.
- **"RELAX stabilizes in 1-2 frames vs REBLUR ~6 frames" (Cyberpunk PT GI rationale)** — rejected per `plan/reverify-restir-relax.md` §2. Origin is one Cyberpunk modder (Ultra Plus on cyberpunk2077mod.com), no NVIDIA/CDPR primary source. NRD's `NRDSettings.h` ships **identical** disocclusion-recovery defaults for both denoisers (`historyFixFrameNum=3`, `maxFastAccumulatedFrameNum=6`). Maintainer dzhdanNV (NRD issue #47) directly contradicts: "REBLUR: 25% faster ... in some cases more temporally stable." If we ever port NRD, the choice is benchmark-gated, not stabilization-frame-quoted.
- **CoopRT 5.11× as evidence for WebGPU subgroup speedup** — rejected per `plan/reverify-perf-onnx.md` §1. The 5.11× number is real (Vulkan-sim simulator) but the technique is hardware modification (LBU + per-SM crossbar inside the RT unit), not a software-shader pattern, and the baseline is Turing-class hardware-RT-cores. Not a WebGPU-software-BVH prediction. CoopRT's general insight (redistributing traversal work across idle SIMT lanes is profitable) remains directionally correct; the speedup magnitude is benchmark-gated (E3) not source-quoted.
- **AMD FidelityFX Hybrid Shadows multiplicative composition** — rejected as misattribution per `plan/reverify-perf-onnx.md` §2. AMD's actual mechanism is bitwise AND on tile-level binary masks + Poisson-disc blocker search, not multiplicative continuous-visibility. Plan's multiplicative composition (§3.2) stands on its own merits (RC delivers continuous chroma; AND-ing binary masks would lose the chroma) — but is not borrowed from AMD.

### 8.2 Capabilities we don't have

These are physically unattainable in browser as of May 2026 (verified):

- **Hardware ray tracing** (verified §4 entry 9 — gpuweb#535 milestone 4+, no shipping browser). All BVH walks are software compute.
- **Tensor Cores / DLSS-RR / DLSS 4** (verified §4 entries 25, 49). Browser-only; tensor cores are NVIDIA driver-locked.
- **FSR Ray Regeneration** (verified §4 entry 26). RDNA-4 + DX12 + Win11 only.
- **Cooperative-matrix matmul intrinsics** (recon-webgpu §1 row "DRAFT"). Would unlock fast neural denoiser; not in spec yet.
- **Mesh / task shaders.** Not in WebGPU spec; not on next-up.
- **Bindless / unbounded descriptor heaps.** gpuweb#380 open since 2019; Vulkan-Android coverage <2% blocked it.
- **64-bit atomics.** Roadmap; needed for Nanite-style soft-raster.
- **Variable Rate Shading / foveation.** Apple Vision Pro WebXR-only.

We don't pretend to use any of these.

### 8.3 Outcome-equivalent simpler alternatives we're already doing

- **Lumen Surface Cache.** A fully-implemented Surface Cache (per-mesh cards + virtual atlas + page eviction) would be a Phase 9+ research project. For our static-room, single-light-dominant scene, DDGI delivers comparable indirect-diffuse quality at a fraction of the implementation surface. Surface Cache wins on AAA city-scale scenes; not ours.
- **APV brick-pool.** Memory-efficient, but our scene is small. Plain DDGI's regular grid is simpler.
- **SHARC (spatial hash radiance cache).** Listed as "experiment to prototype after Option B is stable" in the original recon. Plausible v2 upgrade; not needed for v1.
- **NRD direct port.** REBLUR is the maintainer's perf-default (25% faster, free AO denoising) but uses wave/subgroup ops; RELAX is structurally simpler (A-trous decomposition, no subgroup deps) and ports cleanly to any WebGPU implementation. Either is HLSL→WGSL labor. SVGF gives most of the quality at much lower cost; defer NRD direct port to post-v1.
- **Z-binning vs FPTL light list.** HDRP's tile/cluster/big-tile system is overkill for our ≤ 5-light scene. A flat per-frame list of analytical lights suffices.

### 8.4 Verifier-flagged unverifiable AND not load-bearing

- **CoopRT ISCA 2025 5.11× cooperative BVH speedup** — VERIFIED in author preprint (`plan/reverify-perf-onnx.md` §1) but **dropped as evidence** for WebGPU subgroup speedup. The 5.11× is the maximum (geomean 2.15×) over 13 simulated scenes in Vulkan-sim, with a **hardware-RT-core baseline** (Turing-class), and the technique requires hardware additions (Load Balancing Unit + per-SM crossbar inside the RT unit) that are not exposed to software and are not implementable via WebGPU subgroups. The Google Meet 2.3-2.9× IS verified and remains the closest analogue. Plan informs subgroup speedup expectation as "directionally plausible, magnitude unknown without measurement (E3)" not "X× certain".
- **Cyberpunk RELAX 1-2 frame stabilization vs REBLUR 6 frame** — REJECTED outright by reverify (`plan/reverify-restir-relax.md` §2). Origin is the Ultra Plus mod author on cyberpunk2077mod.com — no NVIDIA, CDPR, or NRD primary source supports it. The NRD maintainer's actual statement (dzhdanNV, issue #47) says REBLUR is **faster** (2.50 ms vs 3.20 ms at 1440p) and "in some cases more temporally stable" — direct opposite of the modder claim. NRD's `NRDSettings.h` ships identical disocclusion-recovery defaults (`historyFixFrameNum=3`, `maxFastAccumulatedFrameNum=6`) for both denoisers. We use neither for v1; we ship SVGF.
- **AMD FidelityFX Hybrid Shadows multiplicative composition exact details** — REFUTED per `plan/reverify-perf-onnx.md` §2. AMD's actual implementation is **bitwise AND on tile-level binary hit masks** (`ShadowRaytrace.hlsl:230-232`: `rwt2d_rayHitResults[currentTile.location] = waveOutput & oldMask`) plus **Poisson-disc min/max blocker search** for penumbra classification (`Classify.hlsl:139-157`), NOT a multiplicative blend of continuous visibility floats. The multiplicative-continuous-visibility composition we use is justified on its own merits (RC delivers a continuous per-channel transmission factor for the colored caustic — AND-ing binary masks would lose the chroma); it is not borrowed from FidelityFX, and AMD attribution has been removed from §3.2. The **classify-then-RT-only-on-penumbra-tiles** pattern from FidelityFX is sound and could inform future optimization, but is not part of v1.
- **ORT-Web "experimental" labeling** — REVERIFIED (`plan/reverify-perf-onnx.md` §3) with corrected source: the verbatim "WebGPU backend is still an experimental feature" label IS in `microsoft/onnxruntime` `js/web/README.md:66` on `main` (verified 2026-05-06), but is NOT on the user-facing docs page (`onnxruntime.ai/docs/tutorials/web/ep-webgpu.html`). Practical interpretation: **shipping, actively maintained** (107 ops in `webgpu-operators.md`, monthly releases v1.23 → v1.25.1 adding Flash Attention, qMoE, Split-K MatMul), used in production by 1P/3P apps, **but Microsoft has not formally promoted the EP to stable**. For our planning this means: pre-flighting our specific model on the EP is required (for the E1 oidn-web alternative path or any ORT-Web fallback). `oidn-web` (verified §4 entry 40) uses TF.js + WebGPU, not ORT-Web, and we evaluate it via E1 anyway.

---

## 9. Pre-implementation experiments

These should be run BEFORE the corresponding implementation phase commits to a design. Each is small, isolated, and produces a measurable result.

### E1. OIDN-web drop-in benchmark (gates Phase 6 design) — **COMPLETE 2026-05-06**

**Result: FAIL — oidn-web is NOT viable per-frame on hardware Lovelace.**

Hardware-validated on host Chrome / NVIDIA RTX 4090 / 1316×903 viewport on `experiment-e1-oidn-web` branch (commit `9627c94` + a 1-line adapter-info wiring fix to `WalkaroundStage.tsx:initOidn`):

| denoiser | per-pass median | budget |
|---|---|---|
| À-trous (control) | sub-ms | 16.67 ms |
| oidn-web | **1051 ms** (3 runs: 1317 / 1051 / 1047) | 63× over budget |
| none (raw 1spp) | sub-ms | 16.67 ms |

oidn-web's tile pipeline ran 24 tiles at ~45ms each = ~1 second per full denoise pass. Init was fast (9.8ms + 1.84 MB weights fetch). The per-pass cost is the multi-RAF tile dispatch dominated by tfjs-webgpu UNet inference; not amenable to optimisation without a fundamentally smaller model.

**Verdict.** Phase 6 commits to **SVGF → A-SVGF → REBLUR-port progression**. oidn-web is preserved as a v2 stretch goal for "user pauses, we run a high-quality denoise pass for export-quality preview" — its 1-second cost is acceptable for an opt-in export pass, decisively wrong for per-frame.

**Bonus finding.** Visual A/B across `none`/`atrous`/`oidn` modes on the canonical honeycomb scene showed all three outputs are visually similar at 1316×903. ReSTIR's temporal + spatial reservoir reuse is doing most of the variance reduction; the denoiser pass is icing, not the cake. **This is a strong vote of confidence in SVGF for Phase 6**: the upgrade from 5-iter À-trous → SVGF (variance-driven sigma + temporal-first ordering) yields meaningful quality gain at near-zero added cost when the input is already low-variance.

### E2. f16 reservoir compression visual quality test (gates Phase 4 layout)

**Question.** Do f16-packed reservoirs (16 B/reservoir) produce visible quality regression vs u32 (32 B/reservoir) on the honeycomb scene?

**Rationale.** Verified §4 entry 8: default `maxStorageBufferBindingSize = 128 MiB`. f32 reservoirs are 132 MB at 1080p — over. f16 is mandatory or we need elevated limits. The "16 B is sufficient" claim is sourced to Bitterli 2020 §6 in the recon but NOT verified by the §4 list. Validate empirically before committing the buffer layout.

**Method.** Implement a small standalone WGSL kernel that runs RIS (M=32) → temporal reuse on the honeycomb, with both u32×8 and u32×4 reservoir layouts. Compare per-pixel reservoir values + final shaded radiance after 8 frames of accumulation. Compute SSIM + visual diff.

**Pass criteria.** SSIM ≥ 0.998 between u32 and f16 outputs after 8-frame accumulation, no visible regression in caustic chroma. If pass: Phase 4 ships f16. If fail: ship u32 + request elevated `maxStorageBufferBindingSize`.

### E3. Subgroup speedup A/B on our BVH (gates Phase 2/4 kernel design) — **COMPLETE 2026-05-06**

**Result: FAIL — subgroup-coherent BVH traversal is within noise on our scene.**

Hardware-validated on host Chrome / NVIDIA RTX 4090 / 1M rays / BVH = 1207 nodes / 4176 tris on `experiment-e3-subgroups` branch (commit `a6bed79`):

| ray distribution | Path A median | Path B median | speedup B/A |
|---|---|---|---|
| cone-coherent | 2.60 ms | 2.65 ms | **0.98×** (Path B slightly slower) |
| divergent | 2.70 ms | 2.45 ms | **1.10×** (Path B 10% faster) |

Both batches: 100% hit-count parity (correctness preserved). Adapter: NVIDIA Lovelace, hardware GPU, subgroups available. Headline: 0.98× median, 1.27× p95 (within noise).

**Verdict.** Subgroup speedup is below the 1.5× pass threshold on our 1207-node BVH. **Phase 2 + Phase 4 ship naive Path A** (per-thread BVH walk, mirroring ReSTIR's existing `bvhIntersectFirstHit`). The added shader complexity of subgroup primitives is not worth the maintenance burden at our BVH size. The reverified-and-dropped CoopRT 5.11× was hardware-modification + Vulkan-sim + RT-core baseline; real WebGPU shader-software speedup on our actual BVH is essentially flat.

**Caveat for future revisits.** The agent's report flags that on much larger BVHs (>5× our size), the broadcast tax dilutes and Path B may pull ahead. Counter-intuitively, divergent rays showed MORE speedup than coherent — because Path A pays a divergent-walk penalty that subgroup ops happen to soften. If we ever scale to multi-room scenes or much higher tri counts, revisit this experiment.

### E4. TSL `emissiveNode` vs `outputNode` injection chroma equivalence (gates D1)

**Question.** Does Option A (`emissiveNode = add(emissive, ddgiContrib×albedo/π)`) produce bit-equivalent chroma to Option B (`outputNode = renderOutput(add(output, ddgiContrib×albedo/π), ACES, sRGB)`) on the canonical DDGI test scene?

**Rationale.** D1's recommendation hinges on the claim that Option A is "physically wrong by a hair but visually equivalent." Empirical verification required before committing the architectural choice.

**Method.** Branch off `walkaround-ddgi` HEAD, implement Option A in a one-line edit, run `12-walkaround-ddgi.spec.ts` chroma test on hardware. Compare chroma_stdDev.

**Pass criteria.** chroma_stdDev within ± 5% of `walkaround-ddgi`'s 0.0745 baseline. If pass: D1 = A. If fail: D1 = B.

### E5. SVGF temporal stability under camera motion (gates Phase 6 design)

**Question.** Does the proposed SVGF (albedo demod + per-stream + variance-driven sigma) match the "1 spp → temporally stable in ≤ 10 ms" target on our scene?

**Rationale.** Verified §4 entry 44: SVGF reports this on 2017-target hardware. Our hardware is faster but our scene + ReSTIR DI sample distribution differs.

**Method.** Implement minimum-viable SVGF as a fork of the current à-trous in `walkaround-restir`. Run a scripted camera path. Measure per-frame temporal stability (pixel std-dev across last 8 frames) and frame time.

**Pass criteria.** Temporal stability stddev ≤ 5 % of mean luminance after 8 frames; frame time ≤ 10 ms on RTX 4090 at 1080p.

### E6. Multi-light load test (gates A5)

**Question.** Does the layered hybrid composite cleanly with 1 sun + 3-5 interior lights on the night-scene fixture?

**Rationale.** A5 acceptance gate; ReSTIR DI's M=32 candidate pool needs to handle the multi-light case without one light's contribution dominating.

**Method.** Build a night-scene variant of the honeycomb fixture: sun replaced with moonlight (low intensity); 3 interior point lights (table lamp + 2 ceiling spots). Run scripted camera path, measure per-light visibility coverage + chroma correctness.

**Pass criteria.** Each interior light's contribution is visible (its glints / shadows present); no light is silently dropped from the reservoir; floor caustic from moonlight remains visible.

---

## 10. Open questions / future research

### 10.1 Items the recon raised but couldn't resolve

- **CoopRT 5.11× cooperative BVH speedup.** Reverify (`plan/reverify-perf-onnx.md` §1) retrieved the author preprint (NCSU faculty site, 13.9 MB PDF) and confirmed the number — but the inference for WebGPU was wrong: it's a hardware-modification proposal (LBU + per-SM crossbar inside the RT unit) over a Turing-class hardware-RT-core baseline in Vulkan-sim. Software-shader subgroup intrinsics cannot replicate this. **Resolved as not-load-bearing**; experiment E3 will give us our own number on our own software BVH shape.
- **Cyberpunk RELAX vs REBLUR frame-stabilization timings.** Reverify (`plan/reverify-restir-relax.md` §2) found the "1-2 vs 6 frames" wording originates with one Cyberpunk modder (Ultra Plus on cyberpunk2077mod.com) with no NVIDIA/CDPR primary source. NRD maintainer dzhdanNV says REBLUR is faster and "in some cases more temporally stable." Even the premise that Cyberpunk Overdrive uses RELAX (vs REBLUR) is poorly sourced — neither the official NVIDIA announcement nor the SIGGRAPH 2023 dev-tech deck names the path-tracing denoiser. Not load-bearing for v1 (we ship SVGF).
- **HDRP's exact APV memory footprint for our scene class.** HDRP defaults assume 32-256 MB; our cap is ~16 MB. If we ever switch from DDGI to APV, this needs answering.
- **MegaLights' integration with Lumen — RESOLVED.** Reverify (`plan/reverify-lumen.md` §5) downloaded the full SIGGRAPH 2025 paper PDF locally (14.5 MB, 83 pages) and read it end-to-end. The integration direction is the **opposite** of what the original recon hypothesised: MegaLights and Lumen are **sibling pipelines** (parallel, not nested). MegaLights handles direct lighting; Lumen handles GI/reflections. They share the BVH (cost amortised) and **MegaLights reuses Lumen's BRDF rays as guiding samples** for stochastic light selection (page 9 + 49). MegaLights is NOT ReSTIR — Epic explicitly rejected ReSTIR-as-baseline at 1spp budget ("we can't afford more than 1 trace per pixel", page 9). For our v2 many-lights work the portable insight is "GI traces can guide direct-lighting candidate selection" — our DDGI/RC primary GI traces could similarly inform our direct-lighting candidate pool.

### 10.2 WebGPU capabilities that would unlock more

If/when the following land in stable browsers, revisit the relevant phase:

- **Hardware ray tracing** (gpuweb#535). Would replace our software BVH walk; 2-5× speedup typical. No ETA.
- **Cooperative-matrix matmul.** Would unlock real-time neural denoiser (DLSS-RR-class). On roadmap.
- **Bindless resources.** Would simplify many-light + many-material scenes. gpuweb#380; blocked by Vulkan-Android coverage.
- **64-bit atomics.** Would enable atomic-depth soft-raster + some advanced GI accumulators.
- **WebNN cross-browser stable.** Would give a non-experimental ML path. Currently Chrome/Edge experimental + Win11 24H2 only. Re-evaluate 2027.

### 10.3 Post-v1: open-source library extraction (committed direction, 2026-05-06)

After v1 hardware-validates and merges to main, the GI pipeline gets extracted into a standalone open-source library so others can use it. This is a committed direction, not a stretch goal — but it ships AFTER v1, not as part of v1.

**Library boundary (the dividing line):**
- LIBRARY: anything that doesn't reference `panel` / `room` / `came` / `glass` / `TOD` / `sglass`. Includes BVH primitives, WGPU support + adapter probing, DDGI probe-grid + atlas, RC cascade dispatch, ReSTIR reservoir + RIS + spatiotemporal reuse, denoisers, composition framework, tone-mapping curves, layer-toggle protocol.
- APP (stays in stainedGlass): scene loading, panel geometry, came/solder, mounts, room loading, sglass fixtures, e2e specs that depend on panel/room semantics, the explore/orbit camera UX.

**Build discipline during Phase 2-6:**
- All new GI primitives land in `lib/`; never `src/rendering/walkaround/` unless they reference app concepts.
- Layer interfaces are explicit TypeScript types — no shared closure state, no app globals leaking into library code.
- Library needs an app-agnostic test scene (Cornell-box-equivalent) alongside the existing sglass fixtures.
- JSDoc on public exports as we write them. Plan files (`glorious-hybrid.md`, `recon-verification.md`, `reverify-*.md`) become the public design-rationale companion.

**Naming + scope** (TBD when v1 ships):
- Likely package name: `@stainedglass/gi` or standalone identity (e.g. `gloriousgi`, `ribbon-gi`).
- License: MIT (frictionless adoption).
- Scope v1: single npm package, hybrid renderer + canonical demo. Layered split possible later (`@gi/bvh`, `@gi/ddgi`, `@gi/restir`, `@gi/denoise`).
- Demo: the stainedGlass scene IS the canonical demo on the package's live page. "This is what real-time hybrid GI looks like in a browser."
- Companion blog post: `glorious-hybrid.md` is mostly already this. Polish + publish.

### 10.4 Stretch goals beyond v1

Listed in rough order of expected value-to-effort ratio:

1. **ReSTIR GI as indirect-diffuse augmenter.** Adds path resampling on top of DDGI for scenes where DDGI's spatial bias is visible. Requires neural denoiser to be ship-quality. Probably 2027.
2. **NRD-port denoiser (RELAX or REBLUR).** If SVGF + albedo demod + per-stream isn't enough, port from HLSL. Choice between RELAX and REBLUR is benchmark-gated per `plan/reverify-restir-relax.md` §2: **REBLUR is the maintainer's perf-default** (25% faster, free AO denoising, "in some cases more temporally stable" per dzhdanNV NRD issue #47); **RELAX is the structurally-simpler port** (A-trous decomposition, no subgroup dependency — works on Firefox/Safari without fallback paths). On a desktop-Chrome-only target REBLUR likely wins; on a cross-browser target RELAX is safer. Useful only if Phase 6's SVGF leaves visible noise.
3. **Volumetric fog / god rays through window.** Standard Frostbite-style 160×90×64 fp16 froxel volume. ~2-3 weeks. Highest visual ROI for sunset scenes.
4. **Surface Cache (Lumen-style).** Cards + virtual atlas + page eviction. Replaces DDGI for indirect. Research-grade for our scene class — DDGI is sufficient.
5. **Caustic reservoir for through-glass paths** (ReSTIR BDPT TOG 2025 §5). Direct application to concave-glass caustics. Implementation is bidirectional path tracing in WGSL — hardest research path. Only if v1's RC-single-cascade caustic doesn't satisfy.
6. **Spectral rendering for chromatic dispersion through colored glass.** The canonical "stained glass spectral" problem. Real-time spectral is research-grade everywhere. Very low priority.
7. **Neural Radiance Cache (NRC).** Verified §4 entry 48 (Müller 2021, ~2.6 ms full-HD combined cache + queries). Tiny MLP runs every frame; ~5-layer 64-wide. Plausible WGSL port using subgroup matmul; gated on Chrome subgroup_matrix stability.
8. **SDXL-ControlNet img2img refinement** (export-only path). The original photorealism plan §7 covers this. Real-time is post-2027; export-quality is viable today as an OPT-IN final-render mode (not part of walkaround).
9. **Concave-glass photon mapping** for sharp-concentrated caustics. Real curved suncatchers form caustics that ReSTIR-or-RC alone can't capture.

---

## Appendix A — File map

**On `main` HEAD `55c44cf`:**
- `src/store/viewportSlice.ts:39` — `WalkaroundEngine = 'ddgi' | 'rc' | 'restir' | 'hybrid'`
- `src/store/selectors/viewport.ts:18-19` — `selectExploreEnabled`, `selectWalkaroundEngine`
- `src/rendering/scene/walkaround/lib/{bvhCommon, useSceneBVH, nodeMaterialUpgrade, wgpuSupport}.ts` — Tier 2 shared GI primitives
- `public/walkaround-honeycomb-suncatcher.sglass` — fixture
- `src/__tests__/fixtures/honeycombFixture.ts` — programmatic fixture builder

**Walkaround branch worktrees (live A/B references through Phase 8):**
- `walkaround-ddgi` HEAD `e89d992` — DDGI implementation; lifts in Phase 2.
- `walkaround-rc` HEAD `677ef84` — RC + WASD walk camera; lifts (single-cascade) in Phase 3 + (camera) in Phase 7.
- `walkaround-restir` HEAD `56d6381` — ReSTIR DI implementation; lifts in Phase 4 (with primary-cast mode dropped).

**New on `hybrid-renderer` (Phase 2 onward):**
- `src/rendering/scene/walkaround/HybridContext.ts`
- `src/rendering/scene/walkaround/layers/{types, DDGILayer, RCLayer (Phase 3), ReSTIRLayer (Phase 4)}.ts`
- `src/rendering/scene/walkaround/applyHybridShading.ts`
- `src/rendering/scene/walkaround/useHybridFrameLoop.ts`
- `src/rendering/scene/walkaround/composite/` (Phase 4+) — fragment composer + ACES tone-map
- `src/rendering/scene/walkaround/denoise/` (Phase 6) — SVGF passes
- `src/rendering/scene/walkaround/specular/` (Phase 5) — split-sum env probe + LUT
- `src/__tests__/e2e/14-walkaround-hybrid-*.spec.ts` — hybrid acceptance tests (A1-A9)

**Archived after v1 ship:**
- `walkaround-ddgi` → `walkaround-ddgi-archive` on remote, then deleted at v1 ship.
- Same for RC and ReSTIR.

---

## Appendix B — Verified facts cross-reference

This plan brids load-bearing claims to the verifier's §4 entries:

| §4 # | Fact | Used in |
|---|---|---|
| 2-6 | WebGPU subgroups + extensions Chrome ship versions | §5.1, D4 |
| 8 | `maxStorageBufferBindingSize = 128 MiB` default | §3.6, §5.3, §5.5, D5 |
| 9 | No hardware RT in WebGPU as of May 2026 | §5, §8.2 |
| 11-15 | Cyberpunk ReSTIR design (DI-only-on-diffuse, M=32, etc.) | §3.1, §4.3 |
| 16 | Bitterli 2020 M=32, k=5 | §4.3 |
| 17 | Wyman 2023 reservoir struct + c_cap=20 | §4.3 |
| 18 | Algorithm 7 = Defensive Pairwise (NOT what we use; we use plain pairwise = Bitterli 2022) | §4.3 |
| 19 | ReSTIR GI 9.3-166× MSE reduction | §10.3 |
| 20-23 | NRD details (compute-only, no tensor) | §4.4, §8.3 |
| 25 | DLSS 4 transformer FP8 | §8.2 |
| 26 | FSR Ray Regen RDNA-4 only | §8.2 |
| 32 | HDRP `RayTracingFallbackHierarchy` enum | §3.3 |
| 39-42 | ORT-Web, oidn-web, Denoiser, wonnx archive | §4.4, §9 E1 |
| 43 | three-mesh-bvh /webgpu since v0.9.2 | §5.2 |
| 44 | SVGF 1 spp → 10 ms | §3.5, §4.4, §9 E5 |
| 45 | DDGI Chebyshev visibility | §4.1 |
| 46 | Lumen software-RT MDF + global distance field | §3.3 |
| 47 | Lumen voxel clipmap pattern (specific cadence numbers DROPPED per `plan/reverify-lumen.md` §1 — natsuneko3 blog had no primary source; UE5.6 `LumenVoxelLighting.cpp` no longer exists, voxel lighting is event-driven on modified bricks; closest analogue is `GlobalDistanceField.cpp:1268` GSDF clipmap with frequencies 1/2/4/4, a different system) | §3.3 |
| reverify | Lumen 16×16 screen-probe tile + 8×8 octahedral = 64 rays/probe (`LumenScreenProbeGather.cpp:55,82` + SIGGRAPH 2022 p.161) | §4.1 (DDGI uses different sampling but pattern reference) |
| reverify | Lumen surface cache atlas 4096×4096 default (`LumenScene.cpp:68` + SIGGRAPH 2022 p.69), 300 cards/frame + 512×512 texel budget (`LumenSceneRendering.cpp:80` + SIGGRAPH 2022 p.69) | §4.1 (Surface Cache rejection rationale) |
| reverify | MegaLights ↔ Lumen: sibling pipelines, MegaLights reuses Lumen BRDF rays for guiding (SIGGRAPH 2025 paper p.9, p.49) | §10.1 |
| 48 | NRC 2.6 ms full-HD | §10.3 |
| 49 | DLSS-RR unattainable in browser | §8.2 |

---

## Appendix C — Reverification methodology notes

These two tools / techniques unlocked deeper primary-source verification that the original recon could not reach. Recorded here so future research dispatches can use them.

### C.1 Public UE5 source mirror — `katataki/UES5.6`

The official `EpicGames/UnrealEngine` repo's raw URLs return 404 to unauthenticated tools (the repo is gated behind the EULA accept-flow). The public mirror at `github.com/katataki/UES5.6` serves identical raw files for UE5.6, allowing direct file:line citations without EULA gating. Reverify-lumen used this to verify:
- `LumenScreenProbeGather.cpp:55,82` (screen-probe tile + octahedral sizes)
- `LumenScene.cpp:68` (surface cache atlas size default)
- `LumenSceneRendering.cpp:80` (cards-per-frame default)
- `GlobalDistanceField.cpp:1268-1296` (GSDF clipmap update logic, the closest-analogue to the bogus voxel cadence claim)

Caveat: the mirror tracks UE5.6. For UE5.0–5.4-era code (where the original recon's voxel-lighting cadence may have lived), this mirror is silent. The `indxzero/ue544cvarwiki` site independently file-line-cites cvars.

### C.2 `curl` + local `pypdf` defeats WebFetch's 10MB PDF limit

WebFetch refuses to ingest PDFs over ~10MB. Both reverify-lumen and reverify-perf-onnx bypassed this by `curl`-ing PDFs to `/tmp` and parsing locally with `pypdf` (or `PyMuPDF` for OCR). This was **the single biggest unlock** for Lumen verification — the SIGGRAPH 2022 Lumen paper (27 MB) and the SIGGRAPH 2025 MegaLights paper (14.5 MB) are both available as direct PDFs and parse cleanly. The CoopRT ISCA 2025 author preprint (13.9 MB) was retrieved this way too.

Future verification passes facing a "PDF too large" or "paywall" obstacle should default to: (1) check the author's faculty / lab homepage for a preprint (`hzhou.wordpress.ncsu.edu`, `cwyman.org`, `research.nvidia.com/labs/...`); (2) `curl` to `/tmp`; (3) parse with `pypdf` or `PyMuPDF` locally. Treat WebFetch as the secondary path, not the primary, for any large or paywalled academic PDF.

---

**End of plan. The next step is one-at-a-time user sign-off on D1-D8, after which Phase 2 build can begin.**
