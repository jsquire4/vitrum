# Real SVGF (Schied 2017) — Future Sprint Placeholder

**Status:** Future / unscheduled
**Created:** 2026-05-11
**Context:** This functionality was deleted/renamed during sweep-2026-05-11 because the
existing implementation was not paper-faithful. This doc preserves the design intent for
when a real implementation sprint is scheduled.

---

## What was renamed and why

The files `svgf.wgsl.ts`, `svgfWebGPU.ts`, `svgfBindings.ts`, `svgfConstants.ts` and
the `denoiser: 'svgf'` mode in `HybridEngine` were renamed to the `atrous-variance`
family (D3, decisions doc). The reason: what shipped was à-trous spatial filtering plus
a 3×3 spatial variance estimate. The defining SVGF temporal stages — bilinear
reprojection, disocclusion detection, per-pixel history length, variance-guided α-clamp
(Schied Eq. 4 edge-stop form) — were entirely absent. Binding slots for `prevRadiance`,
`gbufNormal`, `gbufDepth`, and `motionVec` were declared in the variance pass but never
sampled (verified in sweep: `shared-denoisers/src/wgsl/svgf.wgsl.ts:122–128`).

The renamed `atrous-variance` denoiser is honest about what it does. Schedule this sprint
when a render-quality gap motivates the investment.

---

## Paper reference

Schied, C., Kaplanyan, A., Wyman, C., Patney, A., Chaitanya, C.R.A., Burgess, J.,
Liu, S., Dachsbacher, C., Lefohn, A., Salvi, M.
"Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced
Global Illumination." *High-Performance Graphics*, 2017.
https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering

Key sections: §4 (full algorithm), Eq. 1–5.

---

## Algorithm overview (full paper, not the renamed stub)

The paper has two stages: **temporal reuse** and **spatial (à-trous) filtering**.
The renamed `atrous-variance` pipeline implements only the spatial stage.

### Stage 1 — Temporal reprojection and variance accumulation

**Eq. 1 — Linear reprojection:**
```
p_prev = P · V_prev · x
```
Use the per-pixel motion vector (from G-buffer) to find the previous-frame screen-space
position of the current pixel. Sample previous radiance and moments at that position
using bilinear gather.

**Eq. 2 — Disocclusion test:**
```
reject if |z - z_prev| > σ_z · max(z, z_prev)
reject if dot(n, n_prev) < σ_n
```
Additionally reject if `objId != objId_prev` (object identity, prevents blending across
independently-moving objects). Both depth and normal thresholds (`σ_z`, `σ_n`) should be
UBO-plumbed tunables.

**Eq. 3 — Per-pixel history length:**
```
h_i ← h_prev + 1  (if accepted)
h_i ← 0           (if rejected / disoccluded)
```
Requires a persistent `historyLength` texture (`r16uint`, one value per pixel). Reset on
disocclusion; increments otherwise.

**Eq. 4 — Exponential moving average (EMA) with α-clamp:**
```
α = max(α_min, 1/(h_i + 1))
color_out = α · color_in + (1 − α) · color_prev
M1_out    = α · color_in + (1 − α) · M1_prev      (first moment)
M2_out    = α · color_in² + (1 − α) · M2_prev     (second moment)
```
`α_min` prevents the EMA weight from becoming arbitrarily small at large history counts.
Schied uses `α_min = 0.05`; should be a UBO tunable.

**Eq. 5 — Per-pixel variance from moments:**
```
Var_i = M2_i − M1_i²
```
Clamp to `[0, ∞)`. Requires a persistent `momentsHistory` texture (`rg32float`: M1, M2).

### Stage 2 — Spatial à-trous filtering with variance-guided edge stops

The current `atrous-variance` pipeline already does this part. The edge-stop kernel
(Schied Eq. 4 in the spatial section, not to be confused with the EMA Eq. 4) is:

```
w(p, q) = w_z(p,q) · w_n(p,q) · w_l(p,q)

w_z(p,q) = exp(−|z_p − z_q| / (σ_z · |∇z · (p−q)| + ε))
w_n(p,q) = max(0, dot(n_p, n_q))^σ_n
w_l(p,q) = exp(−|l_p − l_q| / (σ_l · √max(Var_p, ε) + ε))
```

The variance term `√Var_p` in `w_l` is what makes the filter "variance-guided":
high-variance regions get weaker luminance edge stops (allowing more blur to reduce
noise); low-variance regions get stronger stops (preserving detail). Without temporal
variance, the current `atrous-variance` pipeline uses a fixed global `frameCount`
scalar for `Var_p`, which underestimates variance in disoccluded regions and
overestimates it in converged regions.

The 7×7 spatial fallback on disoccluded pixels (`h_i == 0`) replaces the standard
à-trous pass for those pixels to prevent them from bleeding undefined temporal history
into the filtered output.

---

## New GPU resources required

| Resource | Format | Lifetime | Notes |
|---|---|---|---|
| `historyLength` | `r16uint`, full res | Persistent (frame-to-frame) | Reset on disocclusion |
| `momentsHistory` | `rg32float`, full res | Persistent | M1 and M2 per pixel |
| `prevRadiance` | `rgba16float`, full res | Persistent | Previous frame color post-EMA |
| Motion vector input | `rg32float`, full res | Per-frame | Must be wired from G-buffer |

The current `atrous-variance` bind group declares `prevRadiance`, `motionVec`,
`gbufNormal`, `gbufDepth` as bindings but never samples them. Real SVGF requires these
to be populated by the walkaround engine before the denoiser runs.

---

## New compute passes required

1. **`svgfTemporalMain`** — runs before the existing variance pass. Reprojects,
   tests disocclusion, updates history length, EMA-blends color and moments, writes
   `prevRadiance`, `historyLength`, `momentsHistory`.

2. **`svgfVarianceMain`** (update existing) — reads from `momentsHistory` instead of
   performing the 3×3 spatial variance; retains spatial fallback only for
   `historyLength == 0` pixels.

3. **à-trous passes** (existing) — unchanged structurally; the variance texture they
   read must now come from the updated `svgfVarianceMain`.

---

## Relationship to `atrous-variance` (the renamed stub)

The renamed pipeline survives as-is. Real SVGF adds the temporal stage around it.
Implementation strategy: add the temporal pass as a new WGSL file, keep existing
variance+atrous files unchanged (they are correctish for the non-temporal case), and
wire: temporal → variance → atrous.

---

## Albedo demodulation (Schied §4.1, independent but closely related)

Before the temporal pass: divide noisy color by per-pixel albedo:
```
lighting = color / max(albedo, 0.001)
```
Filter `lighting` through the SVGF pipeline. After à-trous: re-multiply by albedo.
This prevents albedo-correlated high-frequency texture edges from spreading into the
lighting estimate during filtering. Requires an albedo G-buffer texture. See
archive/sweep-2026-05-11-fixes-engines.md Item 24 for the full spec.

---

## Complexity estimate

Large — 2–3 weeks of GPU-side work. The temporal pass requires two new persistent
textures, a reprojection step, bilinear gather from the previous frame, and correct
disocclusion logic. The temporal pass must be debugged before the variance improvement
is visible. Not worth scheduling unless the `atrous-variance` pipeline's noise floor
(which is already reasonable for the walkaround use case) is clearly insufficient.

---

## Pre-requisites before scheduling

1. `atrous-variance` depth channel fix (D2, Milestone 4) must be landed — the
   disocclusion test depends on a correctly-packed depth buffer.
2. Motion vectors must be wired from the walkaround G-buffer to the denoiser.
   Currently `motionVec` is bound but always zero.
3. Albedo demodulation (Item 24) should land in the same sprint for the filter to
   perform correctly on textured scenes.
