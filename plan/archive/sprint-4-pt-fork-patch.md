# Sprint 4 PT fork patch — BSDF cost reduction

> **Status**: APPLIED on 2026-05-12 at fork commit `2640b75`
> Applied as part of H6 BDPT chain (vitrum sweep-2026-05-12-followup).
> Adaptation: `forceFullBSDF` per-material flag deferred; `liteMode` mechanism
> is in place and the override can be added when GPU A/B reveals opal degradation.
> GPU profiling of ms/sample reduction is pending (requires WebGL runtime).
>
> Original planning notes:
> Apply after Sprint 3 fork patches are merged and the Sprint 3 variance
> benchmark baseline is captured (the pre-Sprint-4 ms/sample number becomes
> Sprint 4's baseline).
> Fork branch: `phase4-normalmap-shadow-rays` (now consolidated to `main`)
> Fork path: `~/projects/three-gpu-pathtracer/`

---

## Overview

Sprint 4 reduces BSDF evaluation cost by ~50–60% through three complementary
techniques, ordered by implementation dependency:

| Priority | Technique | Effort | File(s) |
|---|---|---|---|
| P1 | lobeMask bitfield | 1 day | `get_surface_record_function.glsl.js` |
| P2 | Lite BSDF for indirect bounces | 1.5 days | `bsdf_functions.glsl.js` |
| P3 | Material LOD by depth | 1 day | `get_surface_record_function.glsl.js` |

**Dependencies**: none — fully fork-internal, builds on Phase 4 fork base.
P2 uses the `lobeMask` introduced in P1; P3 is independent of P1 and P2.

**DoD summary**: ≥40% ms/sample reduction on a glass-and-came scene, all
three bitfield/flag features present on shader inspection.

---

## P1 — lobeMask bitfield

### File: `src/shader/functions/get_surface_record_function.glsl.js`

**Current behavior**: `getSurfaceRecord` populates a `SurfaceRec` struct with
all Disney BSDF lobe parameters unconditionally, even when sheen, clearcoat,
or iridescence is zero. Downstream `bsdfEval` branches on each lobe and
evaluates math that returns zero anyway.

**Change description**:

1. Add a `uint lobeMask` field to `SurfaceRec` (or `surface_record_struct.glsl.js`
   if the struct is defined there separately — locate the actual file):
   ```glsl
   uint lobeMask; // bitfield: bit 0 = diffuse, bit 1 = specular/GGX,
                  //           bit 2 = sheen, bit 3 = clearcoat,
                  //           bit 4 = iridescence, bit 5 = transmission
   ```

2. In `getSurfaceRecord`, compute `lobeMask` from the material parameters:
   ```glsl
   rec.lobeMask = 0u;
   if (rec.roughness > 0.0 || rec.metallic < 1.0) rec.lobeMask |= 1u;  // diffuse
   rec.lobeMask |= 2u;                                                    // specular always
   if (rec.sheen > 0.001)      rec.lobeMask |= 4u;
   if (rec.clearcoat > 0.001)  rec.lobeMask |= 8u;
   if (rec.iridescence > 0.001) rec.lobeMask |= 16u;
   if (rec.transmission > 0.001) rec.lobeMask |= 32u;
   ```

3. In `bsdf_functions.glsl.js`, gate each lobe evaluation on its bit:
   ```glsl
   if ((rec.lobeMask & 4u) != 0u) { /* sheen eval */ }
   if ((rec.lobeMask & 8u) != 0u) { /* clearcoat eval */ }
   if ((rec.lobeMask & 16u) != 0u) { /* iridescence eval */ }
   ```
   Diffuse and specular/GGX are always evaluated (bits 0+1 always set for
   glass-and-came scenes).

**Struct file location**: verify at implementation time. The struct may be in
`src/shader/structs/surface_record_struct.glsl.js` or inlined in the function
file — check with `grep -rn "struct SurfaceRec" src/`.

**DoD verification** (GPU required):
- Shader source inspection: `lobeMask` field present in SurfaceRec; `bsdfEval`
  contains `if ((rec.lobeMask & 4u) != 0u)` guards.
- On a stained-glass scene (sheen=0, clearcoat=0, iridescence=0): verify
  equivalent output to pre-sprint (pixel mean error < 0.1%).
- On a test scene with sheen material: verify sheen still renders correctly
  (lobeMask bit 2 set, sheen branch executes).

---

## P2 — Lite BSDF for indirect bounces

### File: `src/shader/functions/bsdf_functions.glsl.js`

**Current behavior**: every bounce — whether primary (eye ray) or indirect
(bounce depth ≥ 2) — evaluates the full Disney BSDF including sheen, clearcoat,
iridescence, and multiscatter GGX. Indirect bounces are physically the least
important contribution; perceptual sensitivity drops sharply after depth 1.

**Change description**:

1. Add a `bool liteMode` parameter to `bsdfEval` (or, preferably, a `bool
   liteMode` field on `SurfaceRec` so it flows naturally through the call chain
   without signature changes).

2. In the main path tracing loop (`path_tracer.glsl.js` or equivalent), set
   `liteMode = state.depth > 1`. At depth ≤ 1 (primary hit and first bounce),
   full BSDF. At depth > 1, lite BSDF.

3. In `bsdfEval`, when `liteMode == true`:
   - Skip sheen lobe entirely (lobeMask bit 2 gated already by P1, just skip
     the sheen weight).
   - Skip clearcoat lobe entirely.
   - Skip iridescence lobe.
   - Replace multiscatter GGX compensation (`Ess` / `EssAvg` terms) with
     single-scatter GGX. This is the main cost driver — the multiscatter
     compensation requires two additional GLSL function calls.
   - Keep: Lambertian diffuse, single-scatter GGX specular, transmission/
     refraction lobe (transmission is visually important at all depths for
     glass).

4. To guard against perceptual degradation on opal/glueChip/ringMottled panels
   (which rely on subtle indirect shading):
   - Add a `bool forceFullBSDF` flag to the material, defaulting to false.
   - If `material.forceFullBSDF`, override `liteMode = false` at the
     `getSurfaceRecord` call site.
   - Expose `forceFullBSDF` as a `PhysicalPathTracingMaterial` parameter so
     `@vitrum/pt-webgl` can set it per-material via the existing material
     upload path.

**Risk**: The perceptual threshold for "degraded opal" vs "acceptable simplification"
is scene-dependent. Capture an A/B reference render of an opal panel scene
before and after P2. If visible degradation is present, apply `forceFullBSDF`
to those material types in the host material builder.

**DoD verification** (GPU required):
- Profile ms/sample on a glass-and-came scene at 192 samples, with all three
  Sprint 4 changes applied: target ≥40% reduction vs pre-sprint baseline.
- Shader inspection: `bsdfEval` has a `liteMode` branch; indirect paths
  (`state.depth > 1`) skip sheen/clearcoat/iridescence/multiscatter.
- A/B render of an opal panel at 64 samples: if visible degradation, enable
  `forceFullBSDF` for that material class.

---

## P3 — Material LOD by depth

### File: `src/shader/functions/get_surface_record_function.glsl.js`

**Current behavior**: texture lookups (baseColorMap, roughnessMap, normalMap,
transmissionMap) are performed unconditionally at every bounce depth. At depth ≥ 3,
the texture sample contributes negligibly to the final pixel (the throughput
is already reduced by N Fresnel/roughness multiplications). The GPU texture
fetch cost is fixed per sample regardless of throughput.

**Change description**:

1. Add a `int materialLodDepth` uniform to `PhysicalPathTracingMaterial`
   (default value: 2). This controls the depth threshold beyond which
   texture fetches are replaced by the material's average/constant values.

2. In `getSurfaceRecord`, compare the current path depth (`state.depth` or
   equivalent) against `materialLodDepth`:
   ```glsl
   bool useTextures = (state.depth <= materialLodDepth);
   ```

3. Gate each texture fetch on `useTextures`:
   ```glsl
   vec4 baseColor = useTextures
       ? texture2D(baseColorMap, uv)
       : vec4(material.baseColor, 1.0);  // authored flat color fallback
   float roughness = useTextures
       ? texture2D(roughnessMap, uv).g
       : material.roughness;
   // ... etc for normalMap, transmissionMap, emissiveMap
   ```

4. For `normalMap` specifically: when `!useTextures`, skip the TBN tangent-space
   transform entirely (which also avoids the tangent attribute fetch). Use the
   geometric face normal directly.

**Vitrum-side plumbing** (no code change needed — extensions is already open):
The `materialLodDepth=2` uniform should be set from `@vitrum/pt-webgl`'s
`EngineOptions.extensions` field with key `'three-gpu-pathtracer.materialLodDepth'`.
The `extensions` field on `EngineOptions` in `@vitrum/core/src/engine.ts` is
already an open `Record<string, unknown>`; no core type change is required.
The pt-webgl backend reads `options.extensions?.['three-gpu-pathtracer.materialLodDepth']`
and passes it to `PhysicalPathTracingMaterial`'s uniform setter.

**DoD verification** (GPU required):
- Profile ms/sample on a 6-bounce glass scene: texture fetches at depth ≥ 3
  are skipped. Verify via browser GPU profiling (Chrome DevTools WebGL timeline):
  texture read bandwidth at depth ≥ 3 should be near zero.
- A/B render at 192 samples, same scene: pixel mean error < 0.5% vs.
  pre-sprint baseline (texture lookup at every depth). Depth ≥ 3 contributes
  < 5% of total pixel energy in typical scenes; the LOD swap should be
  nearly invisible.
- Set `materialLodDepth = 0` (disable LOD, textures at all depths): output
  must match pre-sprint baseline exactly (regression guard).

---

## Sprint 4 DoD checklist

- [ ] P1: `lobeMask` bitfield in SurfaceRec; sheen/clearcoat/iridescence
       branches gated.
- [ ] P2: `liteMode` flag active at `state.depth > 1`; reduces `bsdfEval`
       to Lambertian + GGX + transmission.
- [ ] P3: `materialLodDepth=2` uniform; texture fetches skipped when
       `state.depth > 2`.
- [ ] Profile: ms/sample ≥40% reduction on a glass-and-came scene vs.
       pre-sprint baseline. Captured in `plan/sprint-4-benchmark.md`.
- [ ] No visual regression on opaque scenes (< 0.1% mean pixel error).
- [ ] `forceFullBSDF` flag documented for opal/glueChip/ringMottled if
       degradation is observed.
