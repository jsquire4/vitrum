# Sprint 10c — BDPT Fork Patch Specification

**Status**: vitrum-side scaffold COMPLETE. Fork patch BLOCKED — prerequisite
Sprints 4, 5, 6 fork patches do not exist (no spec files in plan/, no commits in
fork). Sprint 5 MRT G-buffer (`WebGLMultipleRenderTargets`, gColor/gNormalDepth/gAlbedo)
is required by the ping-pong light-subpath texture architecture and is absent from the
fork. Implementation attempt on 2026-05-12 confirmed blocked at this gate.
Sprint 2 (commit 5388ef0) and Sprint 3 (commit e656a73) are applied.

**Replaces**: `plan/sprint-10c-deferred.md` (archived below as an appendix).

**Created**: 2026-05-09
**Decision 3 / Decision 8 anchor**: vanilla BDPT only — ReSTIR BDPT is DXR/RTX/Falcor-bound, porting estimate 3–5 months. See phase-6-roadmap.md §6 Decision 8 for full citation.

---

## Prerequisite state

Before applying this patch:

1. Sprints 2–7 fork patches must be applied in order (see their `plan/sprint-N-pt-fork-patch.md`).
2. Sprint 5 MRT G-buffer scaffold must be active (`WebGLMultipleRenderTargets` allocated
   with at least 3 channels — gColor, gNormalDepth, gAlbedo).
3. `npm run build` clean in `~/projects/three-gpu-pathtracer/`.
4. Sprint 7 caustic-gap re-evaluation must have been performed (see Appendix A).

---

## Vitrum-side deliverables (DONE)

The following are now shipped in `@vitrum/shared-samplers`:

| Export | File | Purpose |
|---|---|---|
| `BDPTVertex` | `src/bdptVertex.ts` | Vertex type with exact float-offset doc |
| `BDPT_KIND_LIGHT/EYE/CONNECTION/INVALID` | `src/bdptVertex.ts` | Kind constants (0–3) |
| `BDPT_VERTEX_FLOATS` (12) | `src/bdptVertex.ts` | Floats per packed vertex |
| `BDPT_VERTEX_BYTES` (48) | `src/bdptVertex.ts` | Bytes per packed vertex |
| `BDPT_MAX_LIGHT_BOUNCES` (3) | `src/bdptVertex.ts` | Max light subpath bounces |
| `BDPT_MAX_EYE_BOUNCES` (12) | `src/bdptVertex.ts` | Max eye subpath bounces |
| `packBDPTVertex` | `src/bdptVertex.ts` | CPU→GPU packing |
| `unpackBDPTVertex` | `src/bdptVertex.ts` | GPU→CPU unpacking (testing) |
| `bdptConnectionMIS` | `src/bdptMIS.ts` | Power-heuristic MIS weight |
| `buildBDPTStrategyPDFs` | `src/bdptMIS.ts` | Per-strategy PDF table builder |

---

## Vertex layout (CPU ↔ GLSL contract)

The 12-float layout is fixed. Both the CPU pack functions and every GLSL shader
must use these offsets — any deviation breaks the CPU/GPU handshake.

```
Float index  GLSL texel              Field
──────────── ─────────────────────── ──────────────────────────────────────────
base +  0    texelFetch(tex, c, 0).x position.x  (world space)
base +  1    texelFetch(tex, c, 0).y position.y
base +  2    texelFetch(tex, c, 0).z position.z
base +  3    texelFetch(tex, c, 0).w kind  (0=light, 1=eye, 2=conn, 3=invalid)
base +  4    texelFetch(tex, c, 1).x normal.x    (unit-length shading normal)
base +  5    texelFetch(tex, c, 1).y normal.y
base +  6    texelFetch(tex, c, 1).z normal.z
base +  7    texelFetch(tex, c, 1).w pdfFwd      (forward path PDF, solid angle)
base +  8    texelFetch(tex, c, 2).x throughput.r (accumulated path weight)
base +  9    texelFetch(tex, c, 2).y throughput.g
base + 10    texelFetch(tex, c, 2).z throughput.b
base + 11    texelFetch(tex, c, 2).w pdfRev      (reverse PDF for MIS)
```

`c` = column index = vertex index within the light subpath (0 to BDPT_MAX_LIGHT_BOUNCES-1).
Texture size: width = BDPT_MAX_LIGHT_BOUNCES = 3, height = 3 (one row per RGBA32F texel group).

---

## Fork files to create / modify

### NEW: `src/shader/shaders/pathtracing/light_subpath_kernel.glsl.js`

**Purpose**: traces a light subpath from each emitter up to `BDPT_MAX_LIGHT_BOUNCES`
bounces and writes each vertex into a ping-pong texture.

**Exports**: `LIGHT_SUBPATH_KERNEL_GLSL` (template literal string, consumed by
`PhysicalPathTracingMaterial.js`).

**Key GLSL logic**:

```glsl
// Called once per frame before the main eye-ray dispatch.
// Outputs to a WebGL2 texture bound as a framebuffer attachment.
//
// Texture layout (RGBA32F, width=BDPT_MAX_LIGHT_BOUNCES, height=3):
//   row 0: position.xyz, kind
//   row 1: normal.xyz, pdfFwd
//   row 2: throughput.rgb, pdfRev
//
// Each draw call writes one column (one vertex) at texel.x = vertexIndex.

uniform sampler2D uLightTexture;   // emitter texture (Sprint 2/3)
uniform int uNumLights;
uniform int uBDPTMaxLightBounces;  // = BDPT_MAX_LIGHT_BOUNCES = 3

layout(location = 0) out vec4 gVertex0; // position.xyz, kind
layout(location = 1) out vec4 gVertex1; // normal.xyz, pdfFwd
layout(location = 2) out vec4 gVertex2; // throughput.rgb, pdfRev

void main() {
  int vtxIndex = int(gl_FragCoord.x);
  // 1. Sample emitter (light_tree.glsl.js CDF lookup)
  // 2. Trace ray from emitter surface, store vertex at vtxIndex=0
  // 3. Bounce (BSDF sample), store vertex at vtxIndex=1, vtxIndex=2
  // 4. Write pdfFwd for this vertex; pdfRev filled by next pass or connection
  //    (approximated as bsdfPdf of reverse direction for initial implementation)
}
```

**WebGL2 vertex storage decision**: **texture ping-pong** (NOT MRT-only).

Rationale: MRT would write all N light vertices simultaneously in one pass, but
requires `WebGLMultipleRenderTargets` with N × 3 attachments (N=3 → 9 targets),
which exceeds the `MAX_DRAW_BUFFERS` limit of 8 on many WebGL2 implementations.
Texture ping-pong uses 3 successive draw calls (one per vertex/bounce) into the
same 3-attachment MRT target with the column index advancing via a `uVertexCol`
uniform. This stays within the 8-target limit and keeps the framebuffer setup
from Sprint 5 reusable.

Implementation steps:
1. Allocate a `WebGLTexture` with `internalFormat = RGBA32F`, width=3, height=3.
   This is the "light path buffer" — one column per bounce, one row per texel group.
2. Create a framebuffer with 3 color attachments (gVertex0/1/2 → 3 RGBA32F targets).
   This reuses the Sprint 5 `WebGLMultipleRenderTargets` pattern.
3. For each bounce `k` in [0, BDPT_MAX_LIGHT_BOUNCES-1]:
   - Set `uVertexCol = k`.
   - Set the viewport scissor to column `k`, all rows.
   - Draw a fullscreen quad → `light_subpath_kernel.glsl.js` runs for that column.
4. After all bounces, unbind framebuffer. The light path buffer texture is ready
   for the connection pass.

---

### NEW: `src/shader/shaders/pathtracing/eye_subpath_kernel.glsl.js`

**Purpose**: traces from camera into scene, attempting a connection to every stored
light vertex at each eye-subpath bounce.

**Exports**: `EYE_SUBPATH_KERNEL_GLSL`.

**Key change from the existing `path_tracer.glsl.js`**: at each indirect bounce,
before continuing the eye ray, call `evaluateBDPTConnection()` (see below) for
each stored light vertex and accumulate the contribution.

```glsl
uniform sampler2D uLightPathBuffer; // texture from light_subpath_kernel
uniform int uBDPTMaxLightBounces;   // = 3

vec3 evaluateBDPTConnection(
    vec3 eyePos, vec3 eyeNormal, vec3 eyeThroughput, float eyePdfFwd,
    int lightVtxIdx
) {
    // 1. Fetch light vertex from uLightPathBuffer using texelFetch
    // 2. Visibility test (shadow ray from eyePos toward lightVertex.position)
    // 3. Evaluate BSDF at eye vertex toward light vertex direction
    // 4. Evaluate BSDF at light vertex toward eye vertex direction
    // 5. Compute geometry factor G = |cos θ_e| × |cos θ_l| / dist²
    // 6. Compute MIS weight (see connection.glsl.js)
    // 7. Return contribution = lightThroughput × BSDF_e × G × BSDF_l × MIS × eyeThroughput
}
```

---

### NEW: `src/shader/shaders/pathtracing/connection.glsl.js`

**Purpose**: visibility test + MIS weight for a single eye↔light vertex connection.

**Exports**: `CONNECTION_GLSL`.

**MIS weight replicates `bdptConnectionMIS` semantics from `@vitrum/shared-samplers`**.
The CPU-side `buildBDPTStrategyPDFs` / `bdptConnectionMIS` are the reference
implementations; the GLSL here is a direct port. Both must agree bit-for-bit
on MIS weight for the same PDF inputs.

GLSL MIS weight (inline power heuristic, β=2):

```glsl
float bdptMISWeight(float pdfSelected, float pdfSum2) {
    // pdfSum2 = sum of p_i^2 across all strategies (pre-computed in caller)
    // pdfSelected = p_k for the selected strategy
    float p2 = pdfSelected * pdfSelected;
    return (pdfSum2 > 0.0) ? p2 / pdfSum2 : 0.0;
}
```

Shadow ray:
```glsl
bool isVisible(vec3 origin, vec3 target) {
    // Use existing traceScene() with a shadow-ray mode flag.
    // Sprint 5's traceScene analytic-came path is included automatically.
    vec3 dir = normalize(target - origin);
    float dist = length(target - origin);
    // ... occlusion test, return false if any hit before (dist - epsilon)
}
```

---

### MODIFY: `src/shader/shaders/pathtracing/path_tracer.glsl.js`

**Changes** (minimal; preserve all existing Sprint 2–8 patches):

1. Add `uniform sampler2D uLightPathBuffer` declaration.
2. At each indirect-bounce iteration (depth > 0), call `evaluateBDPTConnection()`
   for each of the 3 stored light vertices and accumulate into `radiance`.
3. Do NOT call the connection routine on the primary hit (depth == 0) — this
   avoids double-counting with direct NEE from `direct_lighting.glsl.js`.

---

### MODIFY: `PhysicalPathTracingMaterial.js`

Add new uniforms:

```javascript
// Sprint 10c — BDPT
{ name: 'uLightPathBuffer', type: 'sampler2D' },
{ name: 'uBDPTMaxLightBounces', type: 'int', default: 3 },
{ name: 'uBDPTEnabled', type: 'bool', default: false },  // off by default until verified
```

Add the light-path-buffer texture allocation (similar to the MRT G-buffer
allocation from Sprint 5):

```javascript
// In PathtracerSceneSync or equivalent host-side setup:
const lightPathBuffer = renderer.gl.createTexture();
renderer.gl.bindTexture(gl.TEXTURE_2D, lightPathBuffer);
renderer.gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, BDPT_MAX_LIGHT_BOUNCES, 3);
// width=3 columns (one per bounce), height=3 rows (3 texel groups per vertex)
```

---

## WebGL2 vertex storage — detailed constraint analysis

| Approach | Max vertices | Drawbacks | Why chosen / rejected |
|---|---|---|---|
| MRT (all bounces at once) | 8 × MAX_DRAW_BUFFERS | MAX_DRAW_BUFFERS = 8; 9 targets needed for 3 bounces → overflows | Rejected |
| Texture ping-pong (one column per draw call) | Unlimited (limited by texture width) | 3 draw calls per frame; viewport scissor required | **Chosen** |
| SSBO / transform feedback | Theoretically cleaner | Not available in WebGL2 (only WebGL2 compute via extensions unavailable in Safari) | Rejected |

The texture ping-pong approach adds ~3 draw calls per frame to the existing
PT_FINAL accumulation loop. Each call is a fullscreen quad (~2 triangles); the
bottleneck is the ray-trace work in the fragment shader, not the draw-call overhead.

---

## Definition of Done (fork-side)

- [ ] `light_subpath_kernel.glsl.js`: traces 3 bounces from emitters; writes all
      3 vertices to `uLightPathBuffer` with correct layout (float offsets match
      `BDPTVertex` spec above).
- [ ] `eye_subpath_kernel.glsl.js` / `path_tracer.glsl.js`: at each indirect bounce,
      calls `evaluateBDPTConnection()` for each stored light vertex.
- [ ] `connection.glsl.js`: shadow ray + MIS weight; no NaN/Inf on any test scene.
- [ ] `PhysicalPathTracingMaterial.js`: `uLightPathBuffer`, `uBDPTMaxLightBounces`,
      `uBDPTEnabled` uniforms declared and wired.
- [ ] `uBDPTEnabled = false` by default — BDPT is an opt-in mode for PT_FINAL.
- [ ] Visual A/B: floor caustic from sun-through-panel converges at ~256 samples
      with `uBDPTEnabled = true` vs. ~1024+ samples for pure NEE.
- [ ] Reference renders captured to `tools/reference-renders/sprint-10c-before.png`
      (pure NEE at 256 samples) and `tools/reference-renders/sprint-10c-after.png`
      (BDPT at 256 samples).
- [ ] No regression on existing Sprint 2–8 test scenes with `uBDPTEnabled = false`.

---

## Risk callouts

1. **MAX_DRAW_BUFFERS constraint** (highest risk): the ping-pong approach is robust,
   but each draw call's framebuffer must be validated with `checkFramebufferStatus`.
   If any target format is unsupported on a device, BDPT silently falls back to
   `uBDPTEnabled = false`. Add a capability probe at `PathtracerSceneSync` startup.

2. **Per-frame cost**: each BDPT frame requires 3 extra draw calls (light subpath
   kernel) plus N_light_vertices × connection-shadow-rays per eye-vertex. For N=3
   light vertices and a 1920×1080 render, connection shadow rays add ~6M ray tests
   per sample. On a mid-range GPU this is ~50–100 ms/sample overhead. If overhead
   exceeds 2× baseline, reduce BDPT_MAX_LIGHT_BOUNCES to 1 or 2.

3. **MIS weight correctness**: the GLSL `bdptMISWeight` inline must agree with the
   TypeScript `bdptConnectionMIS(pdfs, k, 2)` reference. Write a CPU-side test
   that directly calls both implementations on the same PDF inputs and asserts
   bit-level agreement (within Float32 tolerance).

4. **pdfRev approximation**: the initial implementation approximates `pdfRev` as the
   BSDF PDF of the reverse direction evaluated at pack time. Full Veach MIS requires
   re-evaluating this after path completion. The approximation is visually acceptable
   for caustic verification but will bias the variance estimate. Track as a
   known issue in `phase-6-status.md`.

5. **Budget**: minimum 2 weeks for an experienced implementer familiar with the fork.
   Adding WebGL2 ping-pong boilerplate, shadow-ray integration, and MIS tuning each
   add a day. Budget 3 weeks with buffer for GPU debugging.

---

## Appendix A — Archived deferred status

The original deferred status doc (`plan/sprint-10c-deferred.md`) contained the
trigger criterion and un-defer instructions. Those instructions remain valid:

> Sprint 10c trigger: run the Sprint 7 DoD hero render at 256 samples PT_FINAL.
> Measure per-pixel noise standard deviation in the floor caustic region.
> If the floor caustic noise SD is visually prominent, proceed with this patch.
> If noise SD is acceptable (converged at 256 samples without BDPT), close 10c
> permanently — Sprint 7 equi-angular sampling was sufficient.

The user has un-deferred Sprint 10c explicitly for library-side scaffolding.
The fork patch itself remains conditional on the visual re-evaluation step above.
