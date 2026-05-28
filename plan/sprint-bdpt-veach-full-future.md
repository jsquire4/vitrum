# Full Veach §10.3 BDPT MIS — Future Sprint Placeholder

**Status:** Partially landed — BDPT is dispatching in pt-webgpu (opt-in via extensions['vitrum.ptWebgpu.bdpt']) and pt-webgl (Sprint 10c). What remains unbuilt is full Veach §10.3 multi-strategy MIS weight enumeration.
**Created:** 2026-05-11
**Context:** This functionality was renamed during sweep-2026-05-11 because the existing
implementation was mislabeled. This doc preserves the design intent for when the full Veach §10.3 MIS sprint is scheduled.

---

## What was renamed and why

The exports `bdptConnectionMIS` and `buildBDPTStrategyPDFs` in
`packages/shared-samplers/src/bdptMIS.ts` were renamed to
`bdptConnectionMIS_partial` and `buildBDPTStrategyPDFs_partial` (D8, decisions doc).

The rename reflects reality: the exported function is a single-strategy MIS aid for
fork-side dispatch, not a complete Veach BDPT strategy PDF enumeration. The name
`bdptConnectionMIS` implied full Veach §10.3 coverage, which would lead future
implementers to trust incorrect MIS weights when BDPT is wired into a render path.

BDPT is now dispatching in pt-webgpu (opt-in via extensions['vitrum.ptWebgpu.bdpt']) and in pt-webgl (Sprint 10c). The rename was a few-minute mechanical change; this doc schedules the remaining full Veach §10.3 MIS work.

---

## Paper reference

Veach, E. "Robust Monte Carlo Methods for Light Transport Simulation."
PhD thesis, Stanford University, 1997.
Chapter 9: Power heuristic (§9.2).
Chapter 10: Bidirectional Path Tracing, §10.3 Multiple Importance Sampling.
http://graphics.stanford.edu/papers/veach_thesis/

**Supporting reference:**
Pharr, M., Jakob, W., Humphreys, G. *Physically Based Rendering*, 4th ed.
§16.3 "Bidirectional Path Tracing."
Equation 16.16: the recursive `p_{s+1}/p_s` ratio for strategy PDF enumeration.
https://pbr-book.org/4ed/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing

---

## What BDPT MIS means (Veach §10.3)

A BDPT path of length `k+1` (k bounces) can be constructed by connecting a camera
subpath of length `s` to a light subpath of length `t = k − s`, with `s + t = k`.
There are `k + 2` strategies: `(s=0,t=k+1)`, `(s=1,t=k)`, ..., `(s=k+1,t=0)`.

The MIS weight for the strategy `(s,t)` that was actually used is:

```
w(s,t) = p(s,t)^β / Σ_{all valid (s',t')} p(s',t')^β
```

where β is the power heuristic exponent (typically 2) and `p(s',t')` is the
probability density of generating the same full path using strategy `(s',t')`.

The key challenge is computing all `p(s',t')` efficiently without re-evaluating the
entire path for each strategy.

---

## The recursive ratio trick (PBR4e Eq. 16.16)

The probability densities can be computed recursively. Define:

```
r_i = p_{i+1}(x_0,...,x_k) / p_i(x_0,...,x_k)
```

where `p_i` is the probability of the full path under strategy `i` (which places the
connection at position `i` along the path). Then:

```
p_s(x_0,...,x_k) = p_0 · Π_{i=0}^{s-1} r_i
```

and the ratio `r_i = p_{i+1} / p_i` can be written as:

```
r_i = pL_{t-i-1} / pC_{s-i}  · G(x_i, x_{i+1})
```

where:
- `pL_{j}` = PDF of the light subpath sampling vertex `j` from vertex `j-1`
- `pC_{j}` = PDF of the camera subpath sampling vertex `j` from vertex `j-1`
- `G(xi, xj)` = geometry term: `cos(θi) · cos(θj) / ||xi − xj||²`

This allows computing all `k+2` strategy probabilities in `O(k)` time by sweeping
the ratios from `s=0` to `s=k+1` — no per-strategy full path re-evaluation.

**Implementation of the ratio sweep (pseudocode):**
```ts
// Given: camera subpath vertices[0..s-1], light subpath vertices[0..t-1]
// Connection: camera[s-1] ↔ light[t-1]

// Start from the full-camera strategy p_0 = productOfCameraPDFs
// Sweep ratios outward from the connection point.

let pC = productOfCameraPDFs(vertices, s);
let pL = productOfLightPDFs(vertices, t);
let pFull = pC * pL;  // p_{s,t}

// Enumerate leftward (decrement s, increment t)
let p = pFull;
let weights: number[] = new Array(k+2).fill(0);
weights[s] = 1;  // the chosen strategy; weight = p^β / Σ p_i^β

for (let i = s - 1; i >= 0; i--) {
  // r_i = pL[...] / pC[...] * G
  p *= pL_ratio(i) / pC_ratio(i) * G(connection(i));
  weights[i] = p ** beta;
}

// Reset and enumerate rightward (increment s, decrement t)
p = pFull;
for (let i = s + 1; i <= k + 1; i++) {
  p *= pC_ratio(i) / pL_ratio(i) * G(connection(i));
  weights[i] = p ** beta;
}

const sumWeights = weights.reduce((a, b) => a + b, 0);
const mis_st = weights[s] / sumWeights;
```

---

## Specular vertex zero-weight handling

At a specular vertex (perfect mirror, glass, etc.), the BSDF is a delta function. A
specular vertex on the camera subpath cannot be "re-connected" from the light subpath at
that point — the connection direction would not coincide with the delta function's
support. Therefore:

```
if (vertex[i].isSpecular) {
  weight[s'] = 0  for all s' that connect at or through vertex i
}
```

In practice: any strategy that has the connection point inside a specular vertex chain
gets zero weight. Specular paths only contribute through the strategy where the full path
is sampled as a single unidirectional chain through the specular sequence.

The current `bdptConnectionMIS_partial` does not handle this. The stub returns a
non-zero weight for all strategies, which is incorrect for specular surfaces.

---

## Area-to-solid-angle conversion (the G factor)

The per-vertex PDFs are often specified in **area measure** (probability of sampling a
surface area element). When converting between strategies, you need the geometry term G
to convert between area and projected solid angle:

```
G(xi, xj) = cos(θi) · cos(θj) / ||xi − xj||²
```

where θi is the angle at xi between the connection direction and the surface normal, and
θj likewise at xj.

The ratio `pSolidAngle = pArea / G` converts a surface-area PDF to a solid-angle PDF.
When computing the recursive ratio `r_i`, the conversion from the PDF of one strategy's
vertex to the next must include this G factor to keep units consistent.

**Critical corner case:** at the camera vertex (index 0), there is no surface normal.
The camera PDF is a solid-angle PDF directly (from the camera's response function). At
the light vertex (index k), the light PDF may be in area measure (for area lights) or
solid angle (for environment lights). The recursive ratio computation must account for
these endpoint conventions.

---

## Camera and light endpoint PDFs

**Camera endpoint (s=0, vertex x_0):**
The camera samples an outgoing direction from x_0 (the camera position). For a pinhole
camera, the PDF of sampling direction `ω_0` is:
```
p_camera(ω_0) = 1 / (A_lens · cos(θ_0)³)  [area PDF on film plane]
```
or equivalently in solid angle as the camera's importance function. The importance must
be evaluated for each path that reaches x_0 from any strategy, not just the chosen one.

**Light endpoint (t=0, vertex x_k):**
For area lights, the PDF of sampling a point x_k on the emitter surface is:
```
p_light(x_k) = 1 / A_emitter   [area PDF]
```
For environment lights (infinite sphere), the PDF is a directional PDF from the envmap
sampling distribution.

**Mixed endpoints:** when s=0 (full light path traced from x_k to camera), the
"camera connection" is an importance-weighted hit directly on the image plane. When t=0
(full camera path reaching x_k as an emitter hit), there is no explicit light connection.
Both require special handling in the weight computation.

---

## What `bdptConnectionMIS_partial` (the renamed stub) does today

The current stub (`buildBDPTStrategyPDFs_partial`) accepts a precomputed array of per-vertex
PDFs and returns a power-heuristic weight for a single connection strategy without
enumerating all alternatives. It is useful for a fork-side dispatch where only two
strategies are competing (direct BSDF sample vs. NEE sample), but it is not BDPT MIS.

It is correct for the 2-strategy case (which is what it is used for), but must not be
called for paths with more than 2 competing strategies.

---

## Connection to the rest of the pipeline

BDPT v1 is wired into pt-webgpu (full-tier kernel, CPU light-path fill + evaluateBdptConnection) and pt-webgl (Sprint 10c). What must be added for full Veach §10.3:

1. A **light subpath tracer** — a GPU compute pass that traces paths from emitters
   toward the camera, storing vertices with their associated PDFs and throughputs.
   This is a new GPU pass not yet present in either `pt-webgpu` or `walkaround-hybrid`.

2. A **connection evaluation** step — for each pair of (camera subpath vertex,
   light subpath vertex), test visibility and evaluate the BSDF at both endpoints.
   The connection is accepted if the geometry term G > 0 and neither endpoint is
   specular.

3. A **MIS weight computation** — call the full Veach §10.3 weight function above
   for each accepted connection.

The `pt-webgpu` package (brute-force path tracer) is the natural home for BDPT since
it is already a unidirectional path tracer. Adding BDPT would make it a bidirectional
path tracer, which is a significant scope increase (new GPU buffer for light subpath
storage, new dispatch pass, new MIS code).

---

## Suggested first milestone

Before attempting full BDPT, implement the **two-strategy MIS** case cleanly (the
current stub's intended domain):

1. Direct illumination: BSDF sample vs. area-light NEE sample — this is already done
   in `pt-webgpu` and `walkaround-hybrid` but without the G factor or endpoint PDF
   accounting.
2. One-bounce indirect: camera path to emitter vs. shadow ray connection — this is
   one step above direct illumination and is where BDPT's advantage first appears.

Get numerical tests passing for these cases (matching analytic reference values for
simple scenes like a Lambertian sphere illuminated by a single area light) before
attempting arbitrary-length BDPT.
