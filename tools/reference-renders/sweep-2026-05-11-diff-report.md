# Sweep 2026-05-11 — GPU diff report

**Branch pair:** `main` (pre-sweep) vs `feat/sweep-2026-05-11-fixes` (post-sweep)
**Capture date:** TBD
**Operator:** TBD

Fill in Pre-sweep / Post-sweep columns after running Phase A2 and A3 captures.
Visual sign-off requires a human A/B review on the GPU machine.

---

## scenario `m5-glass-fresnel-grazing`

**Verifies:** M5 Item 16 — frDielectric branch (grazing-angle Fresnel on glass sphere)
**Expected change:** Highlight should narrow and brighten at 45° elevation post-fix.
No change expected in the diffuse floor or background.

| Metric | Pre-sweep | Post-sweep | Δ |
|---|---|---|---|
| PSNR vs analytic ref | TBD | TBD | TBD |
| Mean luminance | TBD | TBD | TBD |
| Visual sign-off | TBD ☐ | TBD ☐ | — |

**Notes:**
- TBD

---

## scenario `m5-multi-light-cornell`

**Verifies:** M5 Item 15 — sum-MIS over all rect-area lights (2-light Cornell)
**Expected change:** Floor irradiance should be ≈ 2× single-light reference at convergence.
Both lights should contribute independently; no shadow-darkening artifact between them.

| Metric | Pre-sweep | Post-sweep | Δ |
|---|---|---|---|
| PSNR vs analytic ref | TBD | TBD | TBD |
| Mean luminance | TBD | TBD | TBD |
| Visual sign-off | TBD ☐ | TBD ☐ | — |

**Notes:**
- TBD

---

## scenario `m5-glossy-roughness-sweep`

**Verifies:** M5 Item 14 — Heitz 2018 VNDF sampling (highlight tightness vs roughness)
**Expected change:** At roughness=0.1, a tight specular spike centred on the reflection
direction. At roughness=0.7, a broad lobe matching GGX VNDF shape. The lobe-shape
progression should match the theoretical GGX NDF at each roughness step.
Capture one image per `roughnessVariants` entry: 0.1, 0.3, 0.5, 0.7.

| Metric | Pre-sweep | Post-sweep | Δ |
|---|---|---|---|
| PSNR vs analytic ref | TBD | TBD | TBD |
| Mean luminance | TBD | TBD | TBD |
| Visual sign-off | TBD ☐ | TBD ☐ | — |

**Notes:**
- TBD

---

## scenario `m7-ddgi-grey-vs-white-cornell`

**Verifies:** M7 corrected receiver math — (albedo/π)·E formulation
**Expected change:** Grey-wall variant (albedo=0.5) indirect contribution should be
~half that of the white-wall variant (albedo=1.0) once the EMA converges (frame ≥ 64).
Capture one image per `wallAlbedoVariants` entry: 0.5, 1.0.

| Metric | Pre-sweep | Post-sweep | Δ |
|---|---|---|---|
| PSNR vs analytic ref | TBD | TBD | TBD |
| Mean luminance | TBD | TBD | TBD |
| Visual sign-off | TBD ☐ | TBD ☐ | — |

**Notes:**
- TBD

---

## scenario `m7-ddgi-uniform-environment`

**Verifies:** M7 Item 6 (Halton SO(3) rotation) + M7 Item 20 (Lambertian cosine kernel)
**Expected change:** Sphere in a constant-irradiance environment should appear uniformly lit
post-fix. Pre-fix, the pow(8) kernel and frozen rotation produced visible directional
banding. Any remaining banding in the post-sweep frame is a regression.

| Metric | Pre-sweep | Post-sweep | Δ |
|---|---|---|---|
| PSNR vs analytic ref | TBD | TBD | TBD |
| Mean luminance | TBD | TBD | TBD |
| Visual sign-off | TBD ☐ | TBD ☐ | — |

**Notes:**
- TBD

---

## scenario `m8-ddgi-no-seam-darkening`

**Verifies:** M8 border-fill pass — probe-atlas octahedral seam elimination
**Expected change:** Smooth-normal surface at glancing angle should show no cell-grid
darkening rings post-M8 border-mirror fill. Any per-cell ring pattern in the post-sweep
image is a regression in the border-fill logic.

| Metric | Pre-sweep | Post-sweep | Δ |
|---|---|---|---|
| PSNR vs analytic ref | TBD | TBD | TBD |
| Mean luminance | TBD | TBD | TBD |
| Visual sign-off | TBD ☐ | TBD ☐ | — |

**Notes:**
- TBD

---

## scenario `m9-rc-uniform-environment`

**Verifies:** M9 Item 22 (per-bin Ω solid-angle normalisation) + M9 Item 21 (cascade merge integral)
**Expected change:** Mirrors m7-ddgi-uniform-environment outcome but via the RC cascade
pyramid. All cascade levels should converge to the same uniform irradiance value;
inter-cascade brightness discontinuities indicate a merge-integral regression.

| Metric | Pre-sweep | Post-sweep | Δ |
|---|---|---|---|
| PSNR vs analytic ref | TBD | TBD | TBD |
| Mean luminance | TBD | TBD | TBD |
| Visual sign-off | TBD ☐ | TBD ☐ | — |

**Notes:**
- TBD

---

## scenario `m9-gtao-corner-shadows`

**Verifies:** M9 Item 23 — Jiménez 2016 GTAO slice integral replacing (h1+h2)/π HBAO
**Expected change:** Darker contact shadows in the 90° corner post-fix. Open-sky pixels
outside the corner should be unchanged (AO ≈ 1.0 there). Any brightening in the
corner relative to pre-sweep is a regression; any darkening of the open sky is also
a regression.

| Metric | Pre-sweep | Post-sweep | Δ |
|---|---|---|---|
| PSNR vs analytic ref | TBD | TBD | TBD |
| Mean luminance | TBD | TBD | TBD |
| Visual sign-off | TBD ☐ | TBD ☐ | — |

**Notes:**
- TBD

---

## scenario `m9-albedo-edge-preservation`

**Verifies:** M9 Item 24 — atrous-variance edge-preservation (checkerboard floor Cornell)
**Expected change:** Material edges between black and white tiles should remain crisp
through the atrous spatial filter post-fix. Any colour bleeding across tile boundaries
(grey halo at the tile edge) is a regression in the variance-guided edge-stop function.

| Metric | Pre-sweep | Post-sweep | Δ |
|---|---|---|---|
| PSNR vs analytic ref | TBD | TBD | TBD |
| Mean luminance | TBD | TBD | TBD |
| Visual sign-off | TBD ☐ | TBD ☐ | — |

**Notes:**
- TBD

---

## scenario `m17-stretched-sphere-shading`

**Verifies:** M4 Item 17 — transformNormal inverse-transpose for non-uniform instance scale
**Expected change:** Specular highlight tracks the mathematically correct stretched-surface
normal with scale(2,1,1). Pre-fix, a naive model-matrix normal transform without M^{-T}
produces a shearing artifact. Highlight centre should shift toward the elongated axis
post-fix rather than remaining aligned with the unscaled geometry normal.

| Metric | Pre-sweep | Post-sweep | Δ |
|---|---|---|---|
| PSNR vs analytic ref | TBD | TBD | TBD |
| Mean luminance | TBD | TBD | TBD |
| Visual sign-off | TBD ☐ | TBD ☐ | — |

**Notes:**
- TBD

---

## scenario `m18-thick-glass-attenuation`

**Verifies:** M4 Item 18 — Beer-Lambert path-length clamp removal (glass slab 100 wu)
**Expected change:** Transmittance through the slab should satisfy T ≈ exp(-σ·100).
Pre-fix the path length was clamped at 32 wu, yielding T ≈ exp(-σ·32) and an
erroneously bright exit radiance. Post-fix the slab should appear visibly darker
for any σ > 0.

| Metric | Pre-sweep | Post-sweep | Δ |
|---|---|---|---|
| PSNR vs analytic ref | TBD | TBD | TBD |
| Mean luminance | TBD | TBD | TBD |
| Visual sign-off | TBD ☐ | TBD ☐ | — |

**Notes:**
- TBD
