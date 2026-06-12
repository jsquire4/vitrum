# WS3 — Scene ingestion: shared-bvh reuse + BVH texture adapter + ported packers (`src/scene/`)

> Goal: turn a `@vitrum/core` `Scene` into a bundle of WebGL2 textures the GLSL kernels read via `texelFetch1D`. Reuse `@vitrum/shared-bvh` for geometry+BVH+TLAS; write a texture-packing adapter; port the fork's material/light/env packers off `DataTexture` onto raw GL textures. **No THREE.**

## 0. File inventory
```
src/scene/
  uploadSceneTextures.ts   # buildSceneTextures(scene) → UploadedSceneTextures (the bundle + destroy())
  bvhTextureAdapter.ts     # shared-bvh arrays → the 4 BVH data textures (the inverse of MeshBVHUniformStruct.bvhToTextures)
  materialsTexture.ts      # port of MaterialsTexture.js (85px RGBA32F/material) — drive from core MaterialSpec
  lightsTexture.ts         # port of LightsInfoUniformStruct.js (6px RGBA32F/light) — drive from core SceneEmitter
  equirectHdrInfo.ts       # port of EquirectHdrInfoUniform.js (map + marginal/conditional CDF RedFormat-HalfFloat)
  attributesTextureArray.ts# normal/tangent/uv/color → 4-layer RGBA32F sampler2DArray (from shared-bvh normals/uvs + derived tangents)
  partition.ts             # partitionSceneBySupport(scene, PT_WEBGL2_SUPPORT) wrapper
```

## 1. What `@vitrum/shared-bvh` already gives us (REUSE — `ScenePackResult`, verified `scenePack.ts:40-59`)

```ts
interface ScenePackResult {
  positions: Float32Array;   // stride 4, .xyz + .w=0
  normals:   Float32Array;   // stride 4, .xyz + .w=0 (fallback (0,1,0))
  uvs:       Float32Array;   // stride 4, .xy=uv0 .zw=uv1
  indices:   Uint32Array;    // stride 4, .xyz = GLOBAL vertex indices, .w=0
  triMaterialIds: Uint32Array; // 1 u32/tri = resolveMaterialId(primitiveId)
  bvhNodes:  Float32Array;   // 8 words/node (see §2)
  triangleCount: number;
  tlasNodes/tlasInstanceIndices/tlasBlasRoots/tlasInstance{WorldToLocal,LocalToWorld}/tlasNodeCount; // multi-mesh
  primitiveTlasBindings; warnings;
}
```
Build via `packSceneFromCore(scene, { tlas: true, resolveMaterialId })` (`scenePack.ts:812`). This covers geometry + BVH + TLAS + per-tri material id — replacing three-mesh-bvh's `MeshBVH` build AND the fork's positions/normals/uv collection. Material *data*, light data, env CDF, and the tangent+color attribute layers are NOT in shared-bvh — port those (§4-§7).

## 2. BVH texture adapter (`bvhTextureAdapter.ts`) — the inverse of `MeshBVHUniformStruct.bvhToTextures`

shared-bvh's 32-byte node is **byte-identical** to three-mesh-bvh's (verified `buildArrayBvh.ts:13-27` + `bvhIntersect.wgsl.ts:180-181`):
```
f32[0..2] boundsMin   f32[3..5] boundsMax
u32[6] = interior: RELATIVE right-child offset | leaf: absolute tri offset
u32[7] = interior: split axis (0/1/2)          | leaf: 0xFFFF0000 | triCount
```
The fork's GLSL traversal (`BVHShaderGLSL.bvh_ray_functions`) reads 4 data textures (`bvh_struct_definitions`: `usampler2D index; sampler2D position; sampler2D bvhBounds; usampler2D bvhContents`). Pack shared-bvh's arrays into them (square `dim×dim`, row-major `uv.x=i%dim, uv.y=i/dim`, NEAREST):

| Texture | GL format | texels/node | Content from shared-bvh |
|---|---|---|---|
| `bvhBounds` | RGBA32F | 2/node | texel `2i` = `boundsMin.xyz` (.a unused); texel `2i+1` = `boundsMax.xyz` |
| `bvhContents` | RG32UI | 1/node | leaf → `.x = u32[7]` (already `0xFFFF0000\|count`), `.y = u32[6]` (offset); interior → `.x = u32[7]` (splitAxis), `.y = u32[6]` (RELATIVE right offset). GLSL does `currNode + boundsInfo.y` — works because both store relative. |
| `position` | RGBA32F | 1/vertex | shared-bvh `positions` (stride-4); **set `.w = 1.0`** (three-mesh-bvh's `texelFetch1D(...).rgb` expects w=1, vs shared-bvh's w=0 — a `.w` fixup pass). |
| `index` | RGBA32UI | 1/tri | shared-bvh `indices` (stride-4, GLOBAL); `.xyz` = the 3 vertex indices (.w ignored). The fork's `intersectTriangles` reads `.xyz`. |

```ts
export function packBvhTextures(gl, pack: ScenePackResult): { bounds, contents, position, index, materialIndex } {
  const nodeCount = pack.bvhNodes.length / 8;
  const u32 = new Uint32Array(pack.bvhNodes.buffer);
  // bvhBounds: RGBA32F, 2 texels/node
  const boundsDim = squareDim(nodeCount * 2);
  const bounds = new Float32Array(boundsDim * boundsDim * 4);
  for (let i = 0; i < nodeCount; i++) { const n = i*8; bounds.set([pack.bvhNodes[n],pack.bvhNodes[n+1],pack.bvhNodes[n+2],0], (2*i)*4); bounds.set([pack.bvhNodes[n+3],pack.bvhNodes[n+4],pack.bvhNodes[n+5],0],(2*i+1)*4); }
  // bvhContents: RG32UI, 1 texel/node — note: RG32UI texture stores 2 channels; pack into RGBA32UI with .zw=0 if RG unavailable
  const contDim = squareDim(nodeCount);
  const contents = new Uint32Array(contDim * contDim * 2);
  for (let i = 0; i < nodeCount; i++) { contents[i*2] = u32[i*8+7]; contents[i*2+1] = u32[i*8+6]; }
  // position: shared-bvh positions with .w → 1.0
  const position = pack.positions.slice(); for (let v = 3; v < position.length; v += 4) position[v] = 1.0;
  // index: shared-bvh indices (stride-4 GLOBAL), upload as RGBA32UI
  return { bounds: tex2D(gl, RGBA32F, boundsDim, bounds), contents: texU(gl, RG32UI, contDim, contents),
           position: tex2D(gl, RGBA32F, squareDim(position.length/4), position),
           index:    texU(gl, RGBA32UI, squareDim(pack.indices.length/4), pack.indices),
           materialIndex: texU(gl, R32UI, squareDim(pack.triMaterialIds.length), pack.triMaterialIds) };
}
```
**Validation hook:** reuse the existing CPU brute-force BVH oracle (`wsl-gpu/scripts/restir-tlas-bvh-bruteforce-ab.ts` class) — a correct traversal returns identical closest-hits over any valid tree; gate this adapter at 100% match before trusting any render. This is the F-TLAS1/F-RC1 stride-bug guard applied to WebGL2.

## 3. The `BVHShaderGLSL` port (string modules — drop three-mesh-bvh)

three-mesh-bvh's 4 GLSL chunk files (`common_functions` 94, `bvh_struct_definitions` 25, `bvh_ray_functions` 224, `bvh_distance_functions` 207 LOC; MIT) are render-framework-free. Copy them verbatim into `src/glsl/bvh/` as exported strings (`export const BVH_COMMON_FUNCTIONS = /* glsl */\`...\``). They read only their own `BVH` struct samplers via `texelFetch1D` — no THREE. The struct (`bvh_struct_definitions:16-24`) stays `struct BVH { usampler2D index; sampler2D position; sampler2D bvhBounds; usampler2D bvhContents; };`. Detail + the materialIndex attribute in [`04`](./04-glsl-kernels.md) §2.

## 4. `materialsTexture.ts` — 85-pixel RGBA32F per material (port of `MaterialsTexture.js`)

`MATERIAL_PIXELS = 85` (verified), stride = 340 floats; format RGBA32F, ClampToEdge, NEAREST; square `dim = ceil(sqrt(materials.length * 85))`. **Drive from `@vitrum/core` `MaterialSpec`** (not THREE materials) — the field MAP is the verified texel table:

| px | r / g / b / a | px | r / g / b / a |
|---|---|---|---|
| s0 | color.rgb / map | s10 | specularColor.rgb / specularColorMap |
| s1 | metalness / metalnessMap / roughness / roughnessMap | s11 | specularIntensity / specularIntensityMap / isThinFilm / — |
| s2 | ior / transmission / transmissionMap / emissiveIntensity | s12 | attenuationColor.rgb / attenuationDistance |
| s3 | emissive.rgb / emissiveMap | s13 | alphaMap / opacity / alphaTest / side* |
| s4 | normalMap / normalScale.xy / clearcoat | s14 | matte / castShadow / vertexColors\|(flat<<1) / flags* |
| s5 | clearcoatMap / ccRoughness / ccRoughnessMap / ccNormalMap | s15 | sssSigmaT / sssAnisotropyG / dispersionStrength / thinFilmEnabled |
| s6 | ccNormalScale.xy / pad / sheen | s16 | sssAlbedo.rgb / thinFilmLayerCount |
| s7 | sheenColor.rgb / sheenColorMap | s17 | thinFilmIncidentIor / angleDependent / 0 / packedFeatureFlags* |
| s8 | sheenRoughness / sheenRoughnessMap / iridescenceMap / iridThicknessMap | s18 | frontLayerTransmission.rgb / frontLayerRoughness |
| s9 | iridescence / iridescenceIOR / iridThicknessRange.xy | s19 | backLayerTransmission.rgb / backLayerRoughness |

- `s13.a side*`: 1/-1/0; **0 when `!isThinFilm && transmission > 0`**.
- `s14.a flags*`: `Number(transparent) | (vitrumScatteringCoefficient>0 ? TRANSLUCENT_BIT(1<<4) : 0)`; `s14.b` = `vertexColors | (flatShading<<1)`; GLSL also reads `fogVolume = int(s14.b) & 4`.
- `s17.a packedFeatureFlags`: `(hasSpectral?1) | (hasFrontLayer?2) | (hasBackLayer?4)`.
- **s15.c dispersion** = `dispersionStrengthFromAbbe(ior, abbe)` (F=486.1, C=656.3 nm), 0 when `abbe<=0||ior<=1`.
- **samples 20..27 (32 floats)** = spectral attenuation grid, 380→780 nm, `t=i/31`.
- **samples 28..54 (108 floats)** = thin-film stack, 35 layers × `[ior, thicknessNm, extinction]` + 3-pad.
- **samples 55..84 (30 texels)** = 15 texture-transform mat3s (2 texels each: map, metalness, roughness, transmission, emissive, normal, clearcoat, clearcoatNormal, clearcoatRoughness, sheenColor, sheenRoughness, iridescence, iridescenceThickness, specularColor, specularIntensity).
- Texture ids stored as **plain floats** (Pixel-6 `floatBitsToInt` is broken); GLSL reads `int(round(...))`, -1 = none.

The core `MaterialSpec` carries most of these; the fork's `userData.vitrum*` extension fields (scattering coeff, dispersion Abbe, thin-film stack) map from `MaterialSpec` equivalents or `Material.extensions` (design principle #3). **The GLSL `material_struct` decoder is KEPT unchanged** (`structs/material_struct.glsl.js:122-241`) — so this packer must reproduce the byte layout exactly.

## 5. `lightsTexture.ts` — 6-pixel RGBA32F per light (port of `LightsInfoUniformStruct.js`)

`LIGHT_PIXELS = 6` (24 floats/light); types `RECT_AREA=0, CIRC_AREA=1, SPOT=2, DIR=3, POINT=4`. `luminance = 0.2126r+0.7152g+0.0722b`. Texel layout (verified):
- s0: world pos.xyz / type; s1: color.rgb / intensity; s2: u-vector / power; s3: v-vector / area; s4: radius·decay·distance / coneCos; s5: penumbraCos / iesProfile / — / —.
- RectArea/Circular: `u=(w,0,0)·q`, `power = luminance·intensity·(w·h)·(circular?π/4:1)`, `v=(0,h,0)·q`, `area = |u×v|·(circular?π/4:1)`.
- Spot: q from `lookAt`, `power=luminance·intensity`, `area=π·r²`, s4=radius/decay/distance/cos(angle), s5=cos(angle·(1-penumbra))/iesIndex.
- Point: s2.rgb=pos, `power=luminance·intensity`, s4.g=decay, s4.b=distance.
- Directional: s2.rgb=normalize(pos-target), `power=luminance·intensity`.

Map from core `SceneEmitter` (kinds directional/point/spot/rect-area/disc-area/mesh-area). Note: shared-bvh's emitter packing (`bvhCore.ts`) is a *different* layout for the walkaround/pt-webgpu storage-buffer path — this is the fork's 6px texture layout; keep them distinct (the GLSL `lights_struct.glsl.js:39-86` decoder is kept).

## 6. `equirectHdrInfo.ts` — env map + importance-sampling CDF (port of `EquirectHdrInfoUniform.js`)

Three textures: `map` (equirect, HalfFloat RGBA, Repeat-S/ClampEdge-T after preprocess), `marginalWeights` (HalfFloat **RedFormat**, 1×height: maps random→row-center-v), `conditionalWeights` (HalfFloat **RedFormat**, width×height: maps random→column-center-u per row). CDF build: per-pixel `weight = 0.2126r+0.7152g+0.0722b`; row-normalize conditional, total-normalize marginal; `binarySearchFindClosestIndexOf` + half-texel `(i+1)/N + 0.5` recentering; `totalSum` = unnormalized luminance integral. Driven from the core `SceneEnvironment` (kind `hdri`) HDR pixel data. GLSL `equirect_struct` + `equirect_sampling` kept.

## 7. `attributesTextureArray.ts` — the 4-layer attribute array (normal/tangent/uv/color)

4-layer `sampler2DArray`, RGBA32F, internalFormat `RGBA32F`, NEAREST: **layer 0=normal, 1=tangent, 2=uv, 3=color** (verified `AttributesTextureArray.js:5-33`). Per layer a near-square holding `vertexCount` texels; `itemSize 3 → 4` promotion (vec3 in RGBA, .a=0); integer attrs normalized by `2^bpe-1`. GLSL reads `texelFetch1D(attributesArray, layer, index)` (`texture_sample_functions.glsl.js:5-21`).
- **normal** ← shared-bvh `normals` (stride-4, already RGBA-ready).
- **uv** ← shared-bvh `uvs` (stride-4, .xy=uv0; the fork uses 2-component uv → put uv0 in .xy).
- **tangent** — shared-bvh does NOT emit tangents. Derive them CPU-side from positions+uvs (standard per-triangle tangent accumulation) at pack time, OR mark `ATTR_TANGENT` unused and have the GLSL fall back to a screen-space/geometric tangent (normal-mapping degrades gracefully). Decision: derive tangents (needed for normal/clearcoat-normal maps).
- **color** — closed in the native pt-webgl2 path (2026-06-12). Core mesh `colors`
  are merged into the 5-layer attribute array using `mergeWorldSpaceFromCore`
  vertex ranges; absent colors stay default `(1,1,1,1)`, and material slots used
  by colored primitives set `material.vertexColors=1` so GLSL multiplies
  `COLOR_0 × baseColor`.

## 8. `UploadedSceneTextures` bundle + `uploadSceneTextures.ts`
```ts
export interface UploadedSceneTextures {
  bvhBounds, bvhContents, bvhPosition, bvhIndex, materialIndex: WebGLTexture;  // §2
  materials: WebGLTexture;            // §4 (sampler2D)
  attributesArray: WebGLTexture;      // §7 (sampler2DArray)
  lights: WebGLTexture; lightCount: number;   // §5
  envMap, envMarginal, envConditional: WebGLTexture | null; envTotalSum: number;  // §6
  textures2DArray: WebGLTexture | null;   // material texture atlas (RenderTarget2DArray.texture → a real array texture)
  iesProfiles: WebGLTexture | null;
  destroy(): void;                    // deleteTexture for all
}
export function buildSceneTextures(gl, scene, caps): { textures: UploadedSceneTextures; pack: ScenePackResult; warnings: string[] } {
  const { supported, warnings } = partitionSceneBySupport(scene, caps);
  const pack = packSceneFromCore(supported, { tlas: true, resolveMaterialId });
  // packBvhTextures(gl, pack) + materialsTexture + lightsTexture + equirectHdrInfo + attributesTextureArray
  return { textures, pack, warnings };
}
```
`#repackScene` (WS1 §5) destroys the old bundle, builds the new one, stores `#geoPack = pack` (for incremental patches), and calls `reset()`. The material-texture atlas (the fork's `RenderTarget2DArray` `textures` uniform) becomes a real WebGL2 `TEXTURE_2D_ARRAY` uploaded from the core materials' image data.

## 9. WS3 done-when
- `buildSceneTextures` returns a bundle for a Cornell core scene; the BVH adapter passes the CPU brute-force oracle at 100%.
- `materialsTexture` round-trips a known `MaterialSpec` to the exact 85px byte layout (unit-test against a golden the way `materialPackingCoreEquivalence` does).
- All textures bind without GL errors and the S0 diffuse kernel reads correct geometry (silhouette matches the fork capture).
