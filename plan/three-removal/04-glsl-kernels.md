# WS4 — GLSL kernel port + the frozen uniform/`#define` schema (`src/glsl/`)

> The ~4,663 LOC of GLSL kernels are verified THREE-free and port **as-is** (0 imports / 0 `from 'three'` / 0 `#include`). The work here is: (a) freeze + reproduce the uniform + `#define` schema the kernels read; (b) port `BVHShaderGLSL` from three-mesh-bvh; (c) reproduce the compose order; (d) remap the binding conventions (THREE's `ShaderMaterial` auto-binding → our explicit `GlProgram` sampler-unit/UBO binding). The kernels themselves are copied byte-for-byte.

## 0. File inventory (copy the fork's `src/shader/**` + `src/materials/pathtracing/glsl/**` verbatim)
```
src/glsl/
  bvh/         # the BVHShaderGLSL port (3 traversal chunks + materialIndex) — §2
  structs/     # camera/lights/equirect/material/surface_record (448 LOC) — copied verbatim
  sampling/    # light/shape/equirect sampling (435) — copied
  common/      # texture_sample/fresnel/util/math/shape_intersection (450) — copied
  rand/        # sobol/pcg/stratified (363) — copied
  bsdf/        # ggx/sheen/iridescence/fog/volume_march/spectral_accumulator/thin_film_tmm/bsdf_functions (1472) — copied
  render/      # render_structs/camera_util/trace_scene/attenuate_hit/direct_light/get_surface_record/bdpt_* (1432) — copied
  composeTraceGlsl.ts   # the compose root (analog of composePtWebgpuTraceWgsl) — §3
  # (fullscreenVert.ts removed 2026-06-10 — dead; runtime uses gl/fullscreenQuad.ts FULLSCREEN_VERT, fork VS preserved as a comment there)
```

## 1. THE FROZEN SCHEMA — uniforms + defines the kernels read (the contract WS2/WS3/WS5 satisfy)

### 1a. `#define` flags (16; verified `PhysicalPathTracingMaterial.js:55-90`)
```
FEATURE_MIS=1  FEATURE_RUSSIAN_ROULETTE=1  FEATURE_DOF=1  FEATURE_BACKGROUND_MAP=0  FEATURE_FOG=1
FEATURE_BDPT=0  FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION=0  RANDOM_TYPE=2 (0=PCG,1=Sobol,2=Stratified)
CAMERA_TYPE=0 (0=Persp,1=Ortho,2=Equirect)  DEBUG_MODE=0  FEATURE_ADDITIVE_ACCUM=0
ATTR_NORMAL=0  ATTR_TANGENT=1  ATTR_UV=2  ATTR_COLOR=3
```
Runtime-toggled (must be change-gated to avoid per-frame recompile): `FEATURE_DOF` (bokeh!=0), `FEATURE_BACKGROUND_MAP`, `FEATURE_FOG`, `FEATURE_MIS` (from `multipleImportanceSampling`), `FEATURE_ADDITIVE_ACCUM` (from `configureAdditiveAccumulation`), `CAMERA_TYPE`.

### 1b. Uniforms (≈55 across 11 groups; verified `PhysicalPathTracingMaterial.js:92-210`)
The `GlProgram` (WS2) must expose a `set*`/`bind*` for each. **Do not re-add the removed `u_sss*` uniforms** — per-material SSS reads `surf.*` from the material texture.
| Group | uniforms |
|---|---|
| Path trace | `resolution`(vec2), `opacity`(f), `bounces`(i), `transmissiveBounces`(i), `filterGlossyFactor`(f), `uRadianceClamp`(f), `seed`(i) |
| Camera | `physicalCamera`(struct→packed), `cameraWorldMatrix`(mat4), `invProjectionMatrix`(mat4) |
| Scene (samplers) | `bvh`(struct of 4 samplers — §2), `attributesArray`(sampler2DArray), `materialIndexAttribute`(usampler2D), `materials`(sampler2D), `textures`(sampler2DArray) |
| Light | `lights`(sampler2D struct), `iesProfiles`(sampler2DArray), `environmentIntensity`(f), `environmentRotation`(mat4), `envMapInfo`(struct: map+marginal+conditional samplers + `totalSum`/`width`/`height`) |
| Background | `backgroundBlur`(f), `backgroundMap`(sampler2D, gated), `backgroundAlpha`(f), `backgroundIntensity`(f), `backgroundRotation`(mat4) |
| Randomness | `sobolTexture`(sampler2D), `stratifiedTexture`(sampler2D), `stratifiedOffsetTexture`(sampler2D, blue-noise) |
| Volume | `u_volumeDensity`(f), `u_scatterAlbedo`(vec3), `u_anisotropyG`(f) |
| Spectral | `u_jakobCoeffs`(vec3), `uCmfX/Y/Z`(float[81]), `uXCmfCdf/uYCmfCdf/uZCmfCdf`(float[82]), `uXCmfIntegral/uYCmfIntegral/uZCmfIntegral`(f=106.857), `uSpectralRendering`(i), `iorCauchyA/B/C`(f) |
| Caustic | `uCausticStrategy`(i: 0/1/2), `uMneeMaxIterations`(f=8), `uMneeMaxChainLength`(f=3) |
| Material LOD | `materialLodDepth`(i=2) |
| BDPT (gated) | `uBdptEnabled`(bool), `uBdptMaxLightBounces`(i=3), `uBdptLightPathTex`(sampler2D), `uBdptLightSubpathPass`(i), `uBdptVertexCol`(i) |

Most per-frame scalars (`resolution`, `bounces`, `seed`, matrices, counts) move into the **FrameParams std140 UBO** (WS5) for batched upload; the samplers bind per-program via `GlProgram.bindTexture` (sampler-unit assignment at link). The fork sets these as individual `uniform*` calls — the UBO is our greenfield improvement (see [`08`](./08-freebies-and-future.md)); both are valid, UBO is faster.

## 2. The `BVHShaderGLSL` port (drop three-mesh-bvh; `src/glsl/bvh/`)

Copy verbatim (MIT) from `node_modules/three-mesh-bvh/src/webgl/glsl/`:
- `common_functions.glsl.js` (94) → `BVH_COMMON_FUNCTIONS` — includes `uTexelFetch1D(sampler2D, uint)` row-major fetch.
- `bvh_struct_definitions.glsl.js` (25) → `BVH_STRUCT` — `struct BVH { usampler2D index; sampler2D position; sampler2D bvhBounds; usampler2D bvhContents; };`
- `bvh_ray_functions.glsl.js` (224) → `BVH_RAY_FUNCTIONS` — `bvhIntersectFirstHit`, `bvhIntersectsLastHit`.
- `bvh_distance_functions.glsl.js` (207) → `BVH_DISTANCE_FUNCTIONS` — only if the fork uses closest-point queries (it does for some features).

These read ONLY their `BVH` struct samplers — no THREE. The `materialIndexAttribute` (`usampler2D`) the fork uploads via `UIntVertexAttributeTexture` becomes our `materialIndex` texture from the BVH adapter (WS3 §2). The GLSL `intersectTriangles` reads `uTexelFetch1D(bvh.index, i).xyz` then `bvh.position` for the 3 verts — the WS3 adapter packs both compatibly.

## 3. `composeTraceGlsl.ts` — the compose root (analog of `composePtWebgpuTraceWgsl`)

Reproduce the exact fork order (`PhysicalPathTracingMaterial.js:227-441`). This is load-bearing (struct-before-use):
```ts
export function composeTraceGlsl(f: TraceFeatures): string {
  return `
    #define RAY_OFFSET 1e-4
    #define INFINITY 1e20
    precision highp isampler2D; precision highp usampler2D; precision highp sampler2DArray;
    vec4 envMapTexelToLinear( vec4 a ) { return a; }
    // MRT outputs (location 0 = pc_fragColor declared by GlProgram preamble)
    layout(location = 1) out vec4 gNormalDepth;
    layout(location = 2) out vec4 gAlbedo;
    ${BVH_COMMON_FUNCTIONS}${BVH_STRUCT}${BVH_RAY_FUNCTIONS}
    ${STRUCTS.camera}${STRUCTS.lights}${STRUCTS.equirect}${STRUCTS.material}${STRUCTS.surface_record}
    ${randBlock(f.randomType)}                 // RANDOM_TYPE branch: stratified | sobol | pcg + rand/rand2/.. macros
    ${COMMON.texture_sample}${COMMON.fresnel}${COMMON.util}${COMMON.math}${COMMON.shape_intersection}
    ${UNIFORM_DECLS}                           // env/lighting/background/camera/geometry/path-tracer/image uniform decls
    ${SAMPLING.shape}${SAMPLING.equirect}${SAMPLING.light}
    ${spectralCausticBdptUniformDecls(f)}      // u_jakobCoeffs/iorCauchy*/uCausticStrategy/uMnee* + (FEATURE_BDPT) uBdpt*
    ${PTBVH.inside_fog_volume}
    ${BSDF.ggx}${BSDF.sheen}${BSDF.iridescence}${BSDF.fog}${BSDF.volume_march}${BSDF.spectral_accumulator}${BSDF.thin_film_tmm}${BSDF.bsdf_functions}
    ${RENDER.render_structs}${RENDER.camera_util}${RENDER.trace_scene}${RENDER.attenuate_hit}${RENDER.direct_light}${RENDER.get_surface_record}
    ${f.bdpt ? RENDER.bdpt_light_subpath + RENDER.bdpt_connection : ''}
    ${RENDER.main}                             // the main() loop (render orchestration GLSL)
  `;
}
```
Note: the fork relies on THREE injecting `#include <common>` (a small set of THREE GLSL helpers) and `pc_fragColor` at location 0. We must (a) declare `pc_fragColor` ourselves (done in `GlProgram` preamble), and (b) **inline the few `<common>` helpers** the kernels actually use (gamma/luminance/packing) — grep the kernels for the `<common>`-provided symbols (`saturate`, `pow2`, etc.) and provide a `glsl/common/three_common_shim.glsl.ts`. This shim is the one genuinely-new GLSL we author (≈30-50 LOC).

## 4. Binding-convention remap (THREE auto-bind → explicit)

| THREE (fork) | Our equivalent |
|---|---|
| `material.<uniform> = v` (alias) | `prog.set*('<uniform>', v)` |
| `material.bvh.updateFrom(meshBVH)` | bind the 4 BVH-adapter textures to the `BVH` struct's sampler members |
| `material.materials = MaterialsTexture` | `prog.bindTexture('materials', sceneTex.materials)` |
| `material.attributesArray` (DataArrayTexture) | `prog.bindTexture('attributesArray', sceneTex.attributesArray, TEXTURE_2D_ARRAY)` |
| auto sampler-unit assignment | `GlProgram` assigns units sequentially at link (WS2 `assignSamplerUnits`) |
| `setDefine` → recompile event | `GlProgram.setDefine` → `#dirty` → relink on next `use()` |
| `#version 300 es` (implicit) | emitted by `GlProgram` preamble |

## 5. WS4 done-when
- `composeTraceGlsl({...defaults})` compiles + links on the capture host with no GLSL errors (the program builder + the schema are consistent).
- A diffuse-only feature subset (`bsdf_functions` Lambertian branch + `trace_scene` + `get_surface_record`) renders the S0 Cornell matching the fork silhouette + diffuse GI.
- The BVH GLSL port + the WS3 adapter pass the brute-force oracle together (GPU traversal correct).
- The kernels are byte-identical to the fork's (a `diff` gate against `packages/three-gpu-pathtracer/src/shader/**` — they're copied, not edited).
