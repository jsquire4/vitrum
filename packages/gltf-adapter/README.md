# @vitrum/gltf-adapter

glTF 2.0 → `@vitrum/core` Scene adapter.

Hand-rolled GLB container parsing and accessor unpacking. Browser + Node
compatible; image decoding is pluggable, with small built-in Node PNG/JPEG/WebP
fallbacks for `loadGltfAndDecodeTextures()`. Draco / meshopt compressed geometry
is supported via host-supplied decoder hooks (the package bundles no compressed
geometry decoder — see [Compressed geometry](#compressed-geometry)).

---

## Installation

```ts
// workspace-local (monorepo)
import { gltfToScene } from '@vitrum/gltf-adapter';
```

---

## Quick start

```ts
import { gltfToScene } from '@vitrum/gltf-adapter';
import { createEngine } from '@vitrum/engine';

// Load a GLB file (any means you prefer — fetch, fs.readFile, etc.)
const buffer = await fetch('/scene.glb').then(r => r.arrayBuffer());

// Convert to a core Scene.
const { scene, warnings } = await gltfToScene(buffer);
if (warnings.length) console.warn('[gltf-adapter]', warnings);

// Pass to any vitrum backend.
const engine = await createEngine(device, canvas);
engine.setScene(scene);
```

---

## API

### `gltfToScene(input, opts?) → Promise<{ scene, warnings }>`

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `ArrayBuffer \| GltfJson` | Raw GLB bytes, raw JSON bytes (UTF-8), or a parsed `GltfJson` object |
| `opts.buffers` | `Map<number, ArrayBuffer> \| Record<number, ArrayBuffer>` | Pre-loaded external buffers for `.gltf` files (keyed by buffer index). **The adapter does NOT fetch URIs.** |
| `opts.decodeImage` | `(bytes: Uint8Array, mimeType: string) => Promise<unknown>` | Optional image decode callback (see Texture handles below) |
| `opts.sceneIndex` | `number` | Which glTF scene to import (default: `gltf.scene ?? 0`) |
| `opts.dracoDecode` | `DracoDecodeFn` | Host-supplied `KHR_draco_mesh_compression` decoder hook (see Compressed geometry) |
| `opts.meshoptDecode` | `MeshoptDecodeFn` | Host-supplied `EXT_meshopt_compression` decoder hook (see Compressed geometry) |

Returns `{ scene, animations, animationTargets, warnings }`:

| Field | Type | Description |
|-------|------|-------------|
| `scene` | `Scene` | The converted `@vitrum/core` scene (rest pose) |
| `animations` | `AnimationClip[]` | glTF animations as core clips (empty when none). Evaluate with `sampleAnimationClip` from `@vitrum/core` |
| `animationTargets` | `Record<string, string[]>` | Maps a channel node id (`gltf-node-<index>`) → the `ScenePrimitive.id`s created from that node's mesh |
| `warnings` | `string[]` | Non-fatal conversion issues |

### `loadGltfAsset(input, opts?)`

High-level URL/GLB/JSON loader. It resolves external `.bin` buffers and image
bytes, converts to a core `Scene`, analyzes feature use, ranks shipping
backends, and returns a `textureDecodeReport` before the first frame renders.

```ts
import { loadGltfAsset } from '@vitrum/gltf-adapter';

const asset = await loadGltfAsset('/assets/scene.gltf', {
  fetch,
  decodeImage: async (bytes, mimeType) => decodePixels(bytes, mimeType),
  cache: myAssetByteCache,
});

console.log(asset.recommendedBackend.backend);
console.table(asset.textureDecodeReport.entries);
```

`LoadGltfAssetOptions.cache` is keyed by the fully resolved URL plus resource
kind (`asset`, `buffer`, or `image`). Fetch/resource failures throw typed
`GltfFetchFailed` / `GltfResourceNotFound` errors with `{ url, kind }` fields.

### `loadGltfForEngine(input, opts?)`

One-call adapter-owned engine preparation. Hosts inject an existing engine or a
factory; the adapter selects/checks the backend, creates a
`GltfSceneController`, attaches the scene when requested, and forwards the same
`textureDecodeReport` returned by `loadGltfAsset`.

When texture decode is requested, `loadGltfForEngine()` decodes inactive
`KHR_materials_variants` materials as well as the currently active scene. Variant
switches therefore patch backend-ready decoded handles instead of reintroducing
raw image handles later; those material-table entries appear in
`textureDecodeReport` as synthetic `gltf-material-<index>` rows.

When the controller applies animation or material-variant patches through an
incremental `updatePrimitive()` target, it calls the target's optional
`reset()` hook after the patch batch. Engine targets therefore invalidate PT
accumulation / temporal GI history on animated glTF mutations without requiring
the host to remember an extra reset call; full-scene `setScene()` fallbacks keep
their normal scene-replacement invalidation semantics.

`compatibilityMode: 'reject-unsupported' | 'reject-degraded'` rejects before
engine construction when the selected backend cannot satisfy the imported
asset's feature report.

---

## Support matrix

### Geometry

| Feature | Status |
|---------|--------|
| GLB binary container | Supported |
| .gltf JSON + pre-fetched buffers | Supported |
| POSITION / NORMAL / TEXCOORD_0 / TEXCOORD_1 | Supported. If a primitive material references exactly one higher glTF UV set (`TEXCOORD_N`, `N > 1`) and does not also need material-visible `TEXCOORD_1`, the adapter losslessly loads that accessor into core `uv1` and rewrites the primitive-local `TextureRef.texCoord` values to `1`. Missing/conflicting higher UV sets emit structured diagnostics and drop the affected texture fields instead of sampling the wrong UV channel. |
| TANGENT / COLOR_0 | Supported. Authored TANGENT is preserved; tangent-space mapped primitives without TANGENT synthesize xyzw tangents from POSITION/NORMAL/TEXCOORD_0. COLOR_0 is imported and compatibility-reported: pt-webgl2 and full-tier pt-webgpu consume it natively; walkaround-hybrid consumes it approximately in visible baseColor/alpha; pt-webgpu lite reports a structured unsupported issue. Secondary vertex color sets (`COLOR_1+`) are not imported and emit structured ignored-data diagnostics. |
| Indices (UINT16 / UINT32 / UINT8) | Supported |
| Flat normal generation (NORMAL absent) | Supported |
| Sparse accessors | Supported (all component types) |
| Multiple primitives per mesh | Supported → one `MeshPrimitive` per glTF primitive |
| Node hierarchy / nested TRS + matrix | Supported → flattened world transforms |
| Primitive mode TRIANGLES (4) | Supported |
| Primitive modes TRIANGLE_STRIP (5) / TRIANGLE_FAN (6) | Supported → triangulated to an indexed triangle list (glTF §3.7.2.1 winding; degenerates dropped; indexed + non-indexed) |
| Point/line modes (POINTS, LINES, LINE_LOOP, LINE_STRIP) | Supported as `fallback-generated-mesh`: POINTS become tiny cubes and line modes become thin rectangular prisms. `reject-unsupported` accepts them; `reject-degraded` rejects the topology approximation. Override the generated half-width with `pointLineFallbackRadius` when asset scale requires it. |
| KHR_draco_mesh_compression | Supported via `opts.dracoDecode` hook. Without a hook: uncompressed fallback accessors when present (warn), else warn + skip; throws if in `extensionsRequired` with no fallback |
| EXT_meshopt_compression | Supported via `opts.meshoptDecode` hook (bufferView-level — geometry, animation and image consumers all see decompressed data). Without a hook: spec fallback buffer when present (warn), else warn + skip; throws if in `extensionsRequired` with no fallback |
| EXT_mesh_gpu_instancing | Supported for mesh nodes → core `InstancedMeshPrimitive` with `nodeWorld * instanceTRS` baked into each instance matrix. Required use is accepted. `GltfSceneController` patches `instances[]` when the instanced node or an ancestor animates. Malformed accessors warn and import the base mesh once. Skinned/morphed instancing is supported as a renderable `fallback-generated-mesh` route: the importer expands it to one `SkinnedMeshPrimitive` per authored instance, preserves instance-local controller bindings, `reject-unsupported` accepts it, and `reject-degraded` rejects it as a non-native approximation. Native instanced skinning remains a future performance/core-contract feature. |
| Morph targets (POSITION + NORMAL + TANGENT + TEXCOORD_0 plus the UV semantic mapped to core `uv1`, sparse OK) | Supported → `SkinnedMeshPrimitive.morphTargets` / `.morphTargetNormals` / `.morphTargetTangents` / `.morphTargetUvs` / `.morphTargetUv1s` / `.morphWeights` (node/mesh weights; unskinned morphed meshes get a synthesized identity skeleton). The shared CPU skin solver applies TANGENT deltas to solved tangent-space shading when rest tangents exist and blends UV deltas into posed texture coordinates for backend upload. If a primitive material losslessly remaps a single high glTF UV set (`TEXCOORD_N`, `N > 1`) into core `uv1`, matching `TEXCOORD_N` morph deltas feed `.morphTargetUv1s`. Compatibility remains approximate for tangent deltas because GPU-native tangent skinning falls back to CPU. Missing-base UV morphs, conflicting high UV sets, and morph UV lanes not assigned to core `uv1` emit source-pathed `ignored-morph-target-texcoord` diagnostics plus `morphTargetTexcoords=unsupported` compatibility issues. |
| Skins / JOINTS_0 (u8 + u16) / WEIGHTS_0 | Supported when a mesh node binds `skin` and the primitive provides both `JOINTS_0` and `WEIGHTS_0` → `SkinnedMeshPrimitive` at rest pose (incl. `bindMatrix`/`bindMatrixInverse`). Joint/weight attributes without a bound node skin, or incomplete joint/weight pairs, are structured unsupported compatibility issues and best-effort import falls back to a static mesh with diagnostics. |
| Animations (LINEAR / STEP / CUBICSPLINE; T/R/S/weights channels) | Supported → `result.animations` as core `AnimationClip[]` (see Animations below). Geometry imports at rest pose; the host drives playback |
| Cameras | Warn + ignored |

### Materials

| glTF field | Core `MaterialSpec` field |
|-----------|--------------------------|
| `pbrMetallicRoughness.baseColorFactor` | `baseColor` (RGB) |
| `pbrMetallicRoughness.metallicFactor` | `metallic` |
| `pbrMetallicRoughness.roughnessFactor` | `roughness` |
| `pbrMetallicRoughness.baseColorTexture` | `baseColorMap` |
| `pbrMetallicRoughness.metallicRoughnessTexture` | `roughnessMap` (ORM: G=rough, B=metal) |
| `normalTexture` + `.scale` | `normalMap` + `normalScale` |
| `occlusionTexture` + `.strength` | `aoMap` + `aoMapIntensity` |
| `emissiveFactor` | `emissive` |
| `emissiveTexture` | `emissiveMap` |
| `KHR_materials_emissive_strength.emissiveStrength` | `emissiveIntensity` |
| `alphaMode` (OPAQUE/MASK/BLEND) | `alphaMode` ('opaque'/'mask'/'blend') |
| `alphaCutoff` | `alphaCutoff` |
| `doubleSided` | `extensions.doubleSided` |
| `KHR_materials_transmission.transmissionFactor` | `transmission` |
| `KHR_materials_transmission.transmissionTexture` | `transmissionMap` |
| `KHR_materials_ior.ior` | `ior` |
| `KHR_materials_volume.thicknessFactor` | `thickness` |
| `KHR_materials_volume.thicknessTexture` | `thicknessMap` (pt-webgl2, pt-webgpu, and walkaround-hybrid approximate; backend exactness varies) |
| `KHR_materials_volume.attenuationDistance` | `attenuationDistance` |
| `KHR_materials_volume.attenuationColor` | `attenuationColor` |
| `KHR_materials_specular.specularFactor` | `specularIntensity` |
| `KHR_materials_specular.specularColorFactor` | `specularColor` |
| `KHR_materials_specular.specularTexture` | `specularIntensityMap` |
| `KHR_materials_specular.specularColorTexture` | `specularColorMap` |
| `KHR_materials_pbrSpecularGlossiness.diffuseFactor` | `baseColor` + `opacity` (legacy conversion) |
| `KHR_materials_pbrSpecularGlossiness.specularFactor` | `specularColor` |
| `KHR_materials_pbrSpecularGlossiness.glossinessFactor` | `roughness = 1 - glossinessFactor` |
| `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture` | `specularColorMap`; `loadGltfAndDecodeTextures()` / `decodeSceneTextures()` can bake alpha glossiness into a linear `roughnessMap` for `cpu-linear` and `webgpu` texture targets |
| `KHR_materials_sheen.sheenColorFactor` | `sheenColor` |
| `KHR_materials_sheen.sheenRoughnessFactor` | `sheenRoughness` |
| `KHR_materials_sheen.sheenColorTexture` | `sheenColorMap` |
| `KHR_materials_sheen.sheenRoughnessTexture` | `sheenRoughnessMap` |
| `KHR_materials_clearcoat.clearcoatFactor` | `clearcoat` |
| `KHR_materials_clearcoat.clearcoatRoughnessFactor` | `clearcoatRoughness` |
| `KHR_materials_clearcoat.*Texture` | `clearcoatMap`, `clearcoatRoughnessMap`, `clearcoatNormalMap` |
| `KHR_materials_iridescence.iridescenceFactor` | `iridescence` |
| `KHR_materials_iridescence.iridescenceIor` | `iridescenceIor` |
| `KHR_materials_iridescence.iridescenceThicknessMinimum/Maximum` | `iridescenceThicknessRange` |
| `KHR_materials_anisotropy.anisotropyStrength` | `anisotropy` |
| `KHR_materials_anisotropy.anisotropyRotation` | `anisotropyRotation` |
| `KHR_materials_anisotropy.anisotropyTexture` | `anisotropyMap` |

### Emitters

| glTF feature | Core `SceneEmitter` type |
|---|---|
| `KHR_lights_punctual` point light | `PointEmitter` (`intensity` = candela, `decay = 2`) |
| `KHR_lights_punctual` spot light | `SpotEmitter` (`angle` = `outerConeAngle`, `penumbra` derived from inner/outer ratio) |
| `KHR_lights_punctual` directional | `DirectionalEmitter` (`intensity` = lux) |

**Intensity units:** glTF punctual uses candela (cd) for point/spot and lux (lx) for directional.
These photometric values are passed directly as `EmitterBase.intensity`. Vitrum backends treat
`intensity` as a dimensionless linear multiplier (`color × intensity`). For SI-calibrated scenes
divide by a reference level before passing to the engine (e.g., 100 000 lx for a sunny exterior sky).

### Animations

glTF animations are converted to core `AnimationClip`s on `result.animations`.
Each channel's `target.node` is the stable id `gltf-node-<index>` (the glTF
node index; also exported as `animationNodeId(index)`), **not** a
`ScenePrimitive.id` — use `result.animationTargets` to resolve it to the
primitives created from that node's mesh. The engine does not advance clips;
the host evaluates `sampleAnimationClip(clip, time)` (`@vitrum/core`) each
frame and pushes the results:

- **translation / rotation / scale** — recompose the node transform and call
  `engine.updatePrimitive(primId, { transform })` for each mapped primitive.
  The adapter flattens the node hierarchy at import, so channels animating an
  *ancestor* of a mesh node have no mapped primitives; hosts needing full
  scene-graph animation must retain the `GltfJson` hierarchy and recompute
  world transforms themselves.
- **weights** — write the sampled vector into the skinned primitive's
  `morphWeights`, re-run `solveSkin`, and push `positions`/`normals`.
- **Skeletal (joint-node) channels** — the channel names the joint's glTF node;
  after sampling the skeleton pose, rebuild `SkinnedMeshPrimitive.bones` and
  re-run `solveSkin`. The adapter does not retarget skeletal clips.

### Out of scope (documented)

- **Cameras**: ignored. Emitting a warning.
- **Bundled Draco / MeshOpt decoders**: the package stays dependency-free; compressed
  geometry requires the host to inject `opts.dracoDecode` / `opts.meshoptDecode`
  (see Compressed geometry). Without a hook the spec fallbacks apply, else warn + skip
  (or throw when the extension is in `extensionsRequired`).
- **Morph TANGENT deltas**: preserved on `SkinnedMeshPrimitive.morphTargetTangents`;
  compatibility reports current backend tangent-space shading as approximate until
  solvers/renderers consume those deltas directly.
- **`KHR_materials_volume.thicknessTexture`**: imported as `thicknessMap`;
  pt-webgl2 and pt-webgpu consume it approximately as Beer-Lambert distance
  clamps, while walkaround-hybrid samples readable maps approximately by
  exponentiating its scalar Beer tint with `thicknessTexture.g`.
- **Low-level URI-based buffers / images**: `gltfToScene` does not fetch. Use
  `loadGltfAsset` for URL/base-URI loading, or pre-load and supply
  `opts.buffers` / `opts.imageBytes`.

---

## Compressed geometry

The adapter resolves `KHR_draco_mesh_compression` and `EXT_meshopt_compression`
through **host-supplied decoder hooks** — the package itself bundles no decoder,
keeping it dependency-free. `gltfToScene` is async, so hooks may return their
result synchronously or as a Promise (both are awaited).

### Hook contract

```ts
// KHR_draco_mesh_compression — per-primitive. Receives the compressed blob and
// the extension's semantic → Draco-attribute-unique-id map. Returned arrays
// must match the primitive's DECLARED accessors (which per spec describe the
// decoded data): length = accessor.count × components, and either the
// accessor's exact componentType (the adapter then applies `normalized`
// itself) or an already-dequantized Float32Array.
type DracoDecodeFn = (
  compressed: Uint8Array,
  attributeIds: Readonly<Record<string, number>>, // e.g. { POSITION: 0, NORMAL: 1 }
) => DracoDecodeResult | Promise<DracoDecodeResult>;

interface DracoDecodeResult {
  attributes: Record<string, Int8Array | Uint8Array | Int16Array | Uint16Array | Uint32Array | Float32Array>;
  indices?: Uint8Array | Uint16Array | Uint32Array; // length = index accessor count
}

// EXT_meshopt_compression — per-bufferView (so accessors, animations and
// images all transparently see decompressed data). Mirrors meshoptimizer's
// MeshoptDecoder.decodeGltfBuffer; must return exactly count × byteStride bytes.
type MeshoptDecodeFn = (
  compressed: Uint8Array,
  count: number,
  byteStride: number,
  mode: 'ATTRIBUTES' | 'TRIANGLES' | 'INDICES',
  filter: 'NONE' | 'OCTAHEDRAL' | 'QUATERNION' | 'EXPONENTIAL',
) => Uint8Array | Promise<Uint8Array>;
```

### Failure modes (honest by design)

| Situation | Behavior |
|---|---|
| Extension present, hook supplied | Decoded; geometry identical to an uncompressed export |
| No hook, spec fallback exists (Draco fallback accessors / meshopt non-stub fallback buffer) | Fallback used + warning |
| No hook, no fallback, extension in `extensionsUsed` only | Warning + affected primitives skipped |
| No hook, no fallback, extension in `extensionsRequired` | **Throws** a clear Error |
| Hook throws / returns wrong-sized data | Warning + primitive skipped (throws when the extension is required) |

### Wiring `draco3d`

```ts
import DracoDecoderModule from 'draco3d/draco_decoder_nodejs.js'; // or the web build

const draco = await DracoDecoderModule();

const dracoDecode: DracoDecodeFn = (compressed, attributeIds) => {
  const decoder = new draco.Decoder();
  const buf = new draco.DecoderBuffer();
  buf.Init(compressed, compressed.byteLength);
  const mesh = new draco.Mesh();
  try {
    if (!decoder.DecodeBufferToMesh(buf, mesh).ok()) throw new Error('Draco decode failed');

    const attributes: Record<string, Float32Array> = {};
    for (const [semantic, uniqueId] of Object.entries(attributeIds)) {
      const attr = decoder.GetAttributeByUniqueId(mesh, uniqueId);
      const n = mesh.num_points() * attr.num_components();
      const out = new draco.DracoFloat32Array();
      decoder.GetAttributeFloatForAllPoints(mesh, attr, out);
      const arr = new Float32Array(n); // dequantized floats are always accepted
      for (let i = 0; i < n; i++) arr[i] = out.GetValue(i);
      draco.destroy(out);
      attributes[semantic] = arr;
    }

    const indices = new Uint32Array(mesh.num_faces() * 3);
    const face = new draco.DracoInt32Array();
    for (let f = 0; f < mesh.num_faces(); f++) {
      decoder.GetFaceFromMesh(mesh, f, face);
      indices[f * 3] = face.GetValue(0);
      indices[f * 3 + 1] = face.GetValue(1);
      indices[f * 3 + 2] = face.GetValue(2);
    }
    draco.destroy(face);
    return { attributes, indices };
  } finally {
    draco.destroy(mesh);
    draco.destroy(buf);
    draco.destroy(decoder);
  }
};

const { scene } = await gltfToScene(glb, { dracoDecode });
```

(`@gltf-transform`-style setups expose the same `draco3d` decoder module —
pass it through the identical adapter.)

### Wiring `meshoptimizer`

```ts
import { MeshoptDecoder } from 'meshoptimizer';

await MeshoptDecoder.ready;

const meshoptDecode: MeshoptDecodeFn = (compressed, count, byteStride, mode, filter) => {
  const target = new Uint8Array(count * byteStride);
  MeshoptDecoder.decodeGltfBuffer(target, count, byteStride, compressed, mode, filter);
  return target;
};

const { scene } = await gltfToScene(glb, { meshoptDecode });
```

---

## Texture handles

The adapter decodes image bytes to opaque handles. Which handle shape you get
depends on the environment:

| Environment | `opts.decodeImage` | Handle shape | Works with |
|-------------|-------------------|-------------|-----------|
| Browser | absent | `ImageBitmap` for `loadGltfAsset()`; platform canvas-readback pixels for `loadGltfAndDecodeTextures()` / engine `decodeTextures` | pt-webgpu external-image upload, or CPU-atlas payloads when the decoded path can use browser image + canvas APIs |
| Node / Worker | absent | `{ kind: 'raw-image', mimeType, data: Uint8Array }` | Needs custom backend upload |
| Any | provided | Return value of callback | Whatever the callback returns |

Supply `opts.decodeImage` to control the decode target precisely. The adapter
calls it once per unique image index, in parallel.

### sRGB vs linear

The default TextureRef bridge passes bytes/opaque handles as-is. **The backend is responsible for colorspace-correct upload:**

- `baseColorMap`, `emissiveMap` → **sRGB** (backends must use `sRGB` texture format or gamma-decode in shader).
- `normalMap`, `roughnessMap` (ORM), `aoMap`, `lightMap`, `bumpMap`, `anisotropyMap` → **linear** (must NOT sRGB-decode).

`decodeSceneTextures(target: 'cpu-linear')` is the opt-in exception: raw image
handles are converted to linear `Float32Array` RGBA payloads using this same
color/data policy before backend upload. In browser hosts the default path uses
`createImageBitmap` plus canvas/OffscreenCanvas readback when available; Node,
workers without canvas readback, compressed texture sources, and custom formats
still supply `decodePixels`.

`decodeSceneTextures(target: 'webgpu')` also resolves raw image handles through
the same platform/custom pixel path, but preserves the backend upload color
space: sRGB material maps stay sRGB-valued so WebGPU sRGB texture formats can
perform the hardware decode, while linear/data maps stay linear. This is the
predictable pre-upload path for glTF loads that would otherwise leave
`{ kind: 'raw-image' }` handles opaque to pt-webgpu.

### Sampler metadata

`TextureRef` preserves glTF sampler address and filter intent:

- `wrapS` / `wrapT` carry repeat, clamp-to-edge, or mirrored-repeat.
- `magFilter` and `minFilter` carry authored nearest/linear filtering.
- `mipFilter` carries authored mip policy (`none`, `nearest`, or `linear`).

`textureDecodeReport.entries[]` includes the same sampler fields plus
`usesMipmaps` when the asset authored a mipmapped minification mode. It also
separates the material role's color space (`colorSpace`) from the decoded handle
hint (`handleColorSpace`) when known, so hosts can tell a CPU-linear bake from a
WebGPU-ready sRGB-preserved payload. Browser default `ImageBitmap` handles from
ordinary `loadGltfAsset()` are reported through `imageBitmapCount` /
`imageBitmapRefs`; decoded loads report CPU-readable `pixel-data` handles when
browser readback or a custom decoder succeeds, and preserve a structured
diagnostic when readback is unavailable. Current backends already consume per-map
UV, transform, and wrap metadata where their material map rows are supported;
per-texture filter/mipmap enforcement remains backend policy and is reported
through `analyzeGltfAsset()` / `evaluateGltfBackendCompatibility()` as
`*.samplerPolicy` compatibility issues when a selected backend can only
approximate the authored filter/mipmap policy.

### ORM texture

glTF stores roughness and metallic in a single combined texture
(`metallicRoughnessTexture`). The adapter maps the SAME `TextureRef` to both
`roughnessMap` and `metallicMap` (WEBGL2-04 closure: pt-webgl2 reads metalness
from `metallicMap`); backends sample G for roughness and B for metallic, and
atlas packers dedupe by handle so storage is not duplicated.

---

## Credits

Based on the [glTF 2.0 specification](https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html) by the Khronos Group (CC-BY 4.0).
See `CREDITS.md` in the repository root for the full attribution list.
