# @vitrum/gltf-adapter

glTF 2.0 → `@vitrum/core` Scene adapter.

Hand-rolled GLB container parsing and accessor unpacking. Browser + Node
compatible; image decoding is pluggable, with deterministic built-in PNG/JPEG
decoders in both hosts and a Node WebP fallback for
`loadGltfAndDecodeTextures()`. Draco and meshopt compressed geometry
decode through lazy package built-ins by default; optional host hooks can replace
either codec (see [Compressed geometry](#compressed-geometry)).

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
const buffer = await fetch('/scene.glb').then((r) => r.arrayBuffer());

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

| Parameter                       | Type                                                        | Description                                                                                                                          |
| ------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `input`                         | `ArrayBuffer \| GltfJson`                                   | Raw GLB bytes, raw JSON bytes (UTF-8), or a parsed `GltfJson` object                                                                 |
| `opts.buffers`                  | `Map<number, ArrayBuffer> \| Record<number, ArrayBuffer>`   | Pre-loaded external buffers for `.gltf` files (keyed by buffer index). **The adapter does NOT fetch URIs.**                          |
| `opts.resourceLimits`           | `GltfImportResourceLimits`                                  | Import-wide geometry, encoded-resource, decoded-texture, and resource-operation ceilings. See [Resource governance](#resource-governance). |
| `opts.decodeImage`              | `(bytes: Uint8Array, mimeType: string) => Promise<unknown>` | Optional image decode callback (see Texture handles below)                                                                           |
| `opts.sceneIndex`               | `number`                                                    | Which glTF scene to import (default: `gltf.scene ?? 0`)                                                                              |
| `opts.dracoDecode`              | `DracoDecodeFn`                                             | Optional host override for the built-in `KHR_draco_mesh_compression` decoder                                                         |
| `opts.meshoptDecode`            | `MeshoptDecodeFn`                                           | Optional host override for the built-in `EXT/KHR_meshopt_compression` decoder                                                        |
| `opts.compressionDecoderPolicy` | `'builtin' \| 'host-only'`                                  | Codec policy. Defaults to `'builtin'`; `'host-only'` disables built-ins while preserving explicit hooks and validated spec fallbacks |

Returns `{ scene, animations, animationTargets, warnings }`:

| Field              | Type                       | Description                                                                                              |
| ------------------ | -------------------------- | -------------------------------------------------------------------------------------------------------- |
| `scene`            | `Scene`                    | The converted `@vitrum/core` scene (rest pose)                                                           |
| `animations`       | `AnimationClip[]`          | glTF animations as core clips (empty when none). Evaluate with `sampleAnimationClip` from `@vitrum/core` |
| `animationTargets` | `Record<string, string[]>` | Maps a channel node id (`gltf-node-<index>`) → the `ScenePrimitive.id`s created from that node's mesh    |
| `warnings`         | `string[]`                 | Non-fatal conversion issues                                                                              |

### `loadGltfAsset(input, opts?)`

High-level URL/GLB/JSON loader. It resolves external `.bin` buffers and image
bytes, converts to a core `Scene`, analyzes feature use, ranks shipping
backends, and returns a `textureDecodeReport` before the first frame renders.

```ts
import {
  loadGltfAsset,
  releaseGltfResources,
} from '@vitrum/gltf-adapter';

const asset = await loadGltfAsset('/assets/scene.gltf', {
  fetch,
  decodeImage: async (bytes, mimeType) => decodePixels(bytes, mimeType),
  cache: myAssetByteCache,
});

try {
  console.log(asset.recommendedBackend.backend);
  console.table(asset.textureDecodeReport.entries);
} finally {
  releaseGltfResources(asset);
}
```

`LoadGltfAssetOptions.cache` is keyed by the fully resolved URL plus resource
kind (`asset`, `buffer`, or `image`). Fetch/resource failures throw typed
`GltfFetchFailed` / `GltfResourceNotFound` errors with `{ url, kind }` fields.

### Resource governance

Every public import path creates one monotonic resource ledger. The high-level
loader shares it across fetches, compression, accessors, generated geometry,
image acquisition, and texture normalization instead of resetting the budget
between stages.

| `resourceLimits` field | Default | Meaning |
| --- | ---: | --- |
| `maxDecodedGeometryBytes` | 512 MiB | Aggregate decoded and adapter-generated typed geometry |
| `maxEncodedResourceBytes` | 256 MiB | One asset, buffer, image, or raw-image payload |
| `maxTotalEncodedBytes` | 512 MiB | Aggregate distinct encoded resources observed by one import |
| `maxDecodedTexturePixels` | 16,777,216 | One decoded or derived texture surface |
| `maxTotalDecodedTexturePixels` | 16,777,216 | Aggregate accepted/adapter-generated texture surfaces |
| `maxConcurrentResourceOperations` | 4 | Simultaneous fetch or image-decode operations |

An explicit `0` disables a byte or pixel ceiling. Concurrency must remain a
positive safe integer. `loadGltfAndDecodeTextures()` and
`decodeSceneTextures()` also expose the flat
`maxDecodedTexturePixels`, `maxTotalDecodedTexturePixels`, and
`maxImageDecodeConcurrency` options; when supplied, those values override the
matching structured fields.

Later `configureTextureDecode` policy is validated against every resource
already charged and commits transactionally. Undefined nested fields do not
relax inherited limits, while defined flat aliases retain their documented
precedence.

Encoded streams are checked incrementally, capped at 65,536 reads, cancelled on
failure, and never inserted into the cache after rejection. Unique image work is
deduplicated and concurrency-limited. Geometry and encoded-resource overflows
throw `GltfResourceLimitError` with `limitKind`, `limit`, `actual`, and `path`.
Texture-output overflows leave the affected texture unchanged and emit the
corresponding structured decode diagnostic before an adapter output allocation.
Embedded `bufferView` images are non-owning views and reuse the parent buffer’s
encoded-resource identity rather than charging the same bytes again.

Decoded image handles are transaction-owned. A failed import closes every
decoder-created closable identity once. A successful import keeps reachable
handles alive until `releaseGltfResources(result)` is called; that release is
idempotent and also accepts engine-bridge results through their nested `asset`.
Successful texture normalization closes an acquired handle immediately when the
normalized result no longer references it.

For engine-bridge results, first dispose or detach the engine/controller that
consumes the imported scene, then call `releaseGltfResources(result)`.
`VitrumCanvas` performs that ordering automatically, including cancellation
while `attachVitrum()` is still pending.

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

| Feature                                                                                                      | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GLB binary container                                                                                         | Supported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| .gltf JSON + pre-fetched buffers                                                                             | Supported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| glTF asset version                                                                                           | Supports glTF `2.x` assets whose optional `asset.minVersion` is no newer than both the declared asset version and this adapter's glTF 2.0 implementation. Malformed versions, other major versions, and a required minimum newer than 2.0 reject before scene publication.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| POSITION / NORMAL / TEXCOORD_N                                                                               | Supported. Every valid sparse `TEXCOORD_N` stream is preserved under the same authored index in core `uvSets`; `uvs` and `uv1` remain aliases for sets 0 and 1. Material `TextureRef.texCoord` values keep their authored indices through all renderer families. Missing referenced streams reject instead of sampling a different lane. With `KHR_mesh_quantization`, POSITION accepts normalized or unnormalized signed/unsigned 8- and 16-bit storage, while NORMAL follows the extension's signed-normalized encoding rules.                                                                                                                                                                                                                                                                                                                                                                                         |
| TANGENT / COLOR_N                                                                                            | Supported. Authored TANGENT is preserved; tangent-space mapped primitives without TANGENT synthesize xyzw tangents from the material-selected UV set. Every valid `COLOR_N` stream is preserved in core `colorSets`, with `colors` as the `COLOR_0` alias. `COLOR_0` participates in backend shading according to the selected renderer profile; higher color sets remain available to hosts and extensions but have no implicit base-color meaning in core glTF material semantics. pt-webgpu lite accepts the primitive-constant opaque `COLOR_0` case by baking it into `material.baseColor` and rejects varying/alpha-bearing cases it cannot represent.                                                                                                                                                                                                                                                                      |
| Application-specific primitive attributes (`_NAME`)                                                          | Accepted with a source-pathed diagnostic and left out of the core Scene because no generic attribute channel is declared. Unknown non-application semantics reject instead of being silently discarded.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Indices (UINT16 / UINT32 / UINT8)                                                                            | Supported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Flat normal generation (NORMAL absent)                                                                       | Supported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Sparse accessors                                                                                             | Supported (all component types)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Multiple primitives per mesh                                                                                 | Supported → one `MeshPrimitive` per glTF primitive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Node hierarchy / nested TRS + matrix                                                                         | Supported → flattened world transforms                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Primitive mode TRIANGLES (4)                                                                                 | Supported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Primitive modes TRIANGLE_STRIP (5) / TRIANGLE_FAN (6)                                                        | Supported → triangulated to an indexed triangle list (glTF §3.7.2.1 winding; degenerates dropped; indexed + non-indexed)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Point/line modes (POINTS, LINES, LINE_LOOP, LINE_STRIP)                                                      | Supported as `fallback-generated-mesh`: POINTS become tiny cubes and line modes become thin rectangular prisms. `reject-unsupported` accepts them; `reject-degraded` rejects the topology approximation. Override the generated half-width with `pointLineFallbackRadius` when asset scale requires it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| KHR_draco_mesh_compression                                                                                   | Supported by the lazy built-in Draco decoder in browser and Node. `opts.dracoDecode` overrides it. Draco face lists are published as indexed TRIANGLES; a missing base `primitive.indices` accessor is synthesized rather than dropping connectivity                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| EXT/KHR_meshopt_compression                                                                                  | Supported by the lazy built-in meshoptimizer decoder at bufferView level, so geometry, animation and image consumers all see decompressed data. `opts.meshoptDecode` overrides it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| EXT_mesh_gpu_instancing                                                                                      | Supported for mesh nodes → core `InstancedMeshPrimitive` with `nodeWorld * instanceTRS` baked into each instance matrix. Required use is accepted. `GltfSceneController` patches `instances[]` when the instanced node or an ancestor animates. Malformed accessors warn and import the base mesh once. Skinned/morphed instancing is supported as a renderable `fallback-generated-mesh` route: the importer expands it to one `SkinnedMeshPrimitive` per authored instance, preserves instance-local controller bindings, `reject-unsupported` accepts it, and `reject-degraded` rejects it as a non-native approximation. Native instanced skinning remains a future performance/core-contract feature.                                                                                                                                                                                                                                                                                                    |
| Morph targets (POSITION + NORMAL + TANGENT + TEXCOORD_N, sparse OK)                                         | Supported → scalable core morph streams plus compatibility aliases for UV sets 0 and 1. The shared solver blends every represented UV-set delta into posed texture coordinates and applies tangent deltas when rest tangents exist. A `TEXCOORD_N` delta without a matching base stream rejects. Optional `COLOR_N` and application-specific morph deltas are valid glTF data that the current core morph contract cannot represent; they are preserved at the base-attribute level, diagnosed with exact source paths, and omitted from morph evaluation. Unknown non-application morph semantics reject.                                                                                                                                                                                                                                                                                                           |
| Skins / JOINTS_0 (u8 + u16) / WEIGHTS_0                                                                      | Supported when a mesh node binds `skin` and the primitive provides both `JOINTS_0` and `WEIGHTS_0` → `SkinnedMeshPrimitive` at rest pose (incl. `bindMatrix`/`bindMatrixInverse`). Joint/weight attributes without a bound node skin, or incomplete joint/weight pairs, are structured unsupported compatibility issues and best-effort import falls back to a static mesh with diagnostics.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Animations (LINEAR / STEP / CUBICSPLINE; T/R/S/weights channels)                                             | Supported → `result.animations` as core `AnimationClip[]` (see Animations below). Geometry imports at rest pose; the host drives playback                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Cameras                                                                                                      | Projection metadata is validated against glTF 2.0 and returned on `result.cameras`; it is not injected into the core Scene camera contract, so a structured `ignored-camera` compatibility diagnostic remains                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

### Materials

| glTF field                                                      | Core `MaterialSpec` field                                                                                                                                                        |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pbrMetallicRoughness.baseColorFactor`                          | `baseColor` (RGB)                                                                                                                                                                |
| `pbrMetallicRoughness.metallicFactor`                           | `metallic`                                                                                                                                                                       |
| `pbrMetallicRoughness.roughnessFactor`                          | `roughness`                                                                                                                                                                      |
| `pbrMetallicRoughness.baseColorTexture`                         | `baseColorMap`                                                                                                                                                                   |
| `pbrMetallicRoughness.metallicRoughnessTexture`                 | `roughnessMap` (ORM: G=rough, B=metal)                                                                                                                                           |
| `normalTexture` + `.scale`                                      | `normalMap` + `normalScale`                                                                                                                                                      |
| `occlusionTexture` + `.strength`                                | `aoMap` + `aoMapIntensity`                                                                                                                                                       |
| `emissiveFactor`                                                | `emissive`                                                                                                                                                                       |
| `emissiveTexture`                                               | `emissiveMap`                                                                                                                                                                    |
| `KHR_materials_emissive_strength.emissiveStrength`              | `emissiveIntensity`                                                                                                                                                              |
| `alphaMode` (OPAQUE/MASK/BLEND)                                 | `alphaMode` ('opaque'/'mask'/'blend')                                                                                                                                            |
| `alphaCutoff`                                                   | `alphaCutoff`                                                                                                                                                                    |
| `doubleSided`                                                   | `extensions.doubleSided`                                                                                                                                                         |
| `KHR_materials_transmission.transmissionFactor`                 | `transmission`                                                                                                                                                                   |
| `KHR_materials_transmission.transmissionTexture`                | `transmissionMap`                                                                                                                                                                |
| `KHR_materials_ior.ior`                                         | `ior`                                                                                                                                                                            |
| `KHR_materials_volume.thicknessFactor`                          | `thickness`                                                                                                                                                                      |
| `KHR_materials_volume.thicknessTexture`                         | `thicknessMap` (pt-webgl2, pt-webgpu, and walkaround-hybrid approximate; backend exactness varies)                                                                               |
| `KHR_materials_volume.attenuationDistance`                      | `attenuationDistance`                                                                                                                                                            |
| `KHR_materials_volume.attenuationColor`                         | `attenuationColor`                                                                                                                                                               |
| `KHR_materials_specular.specularFactor`                         | `specularIntensity`                                                                                                                                                              |
| `KHR_materials_specular.specularColorFactor`                    | `specularColor`                                                                                                                                                                  |
| `KHR_materials_specular.specularTexture`                        | `specularIntensityMap`                                                                                                                                                           |
| `KHR_materials_specular.specularColorTexture`                   | `specularColorMap`                                                                                                                                                               |
| `KHR_materials_pbrSpecularGlossiness.diffuseFactor`             | `baseColor` + `opacity` (legacy conversion)                                                                                                                                      |
| `KHR_materials_pbrSpecularGlossiness.specularFactor`            | `specularColor`                                                                                                                                                                  |
| `KHR_materials_pbrSpecularGlossiness.glossinessFactor`          | `roughness = 1 - glossinessFactor`                                                                                                                                               |
| `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture` | `specularColorMap`; `loadGltfAndDecodeTextures()` / `decodeSceneTextures()` can bake alpha glossiness into a linear `roughnessMap` for `cpu-linear` and `webgpu` texture targets |
| `KHR_materials_sheen.sheenColorFactor`                          | `sheenColor`                                                                                                                                                                     |
| `KHR_materials_sheen.sheenRoughnessFactor`                      | `sheenRoughness`                                                                                                                                                                 |
| `KHR_materials_sheen.sheenColorTexture`                         | `sheenColorMap`                                                                                                                                                                  |
| `KHR_materials_sheen.sheenRoughnessTexture`                     | `sheenRoughnessMap`                                                                                                                                                              |
| `KHR_materials_clearcoat.clearcoatFactor`                       | `clearcoat`                                                                                                                                                                      |
| `KHR_materials_clearcoat.clearcoatRoughnessFactor`              | `clearcoatRoughness`                                                                                                                                                             |
| `KHR_materials_clearcoat.*Texture`                              | `clearcoatMap`, `clearcoatRoughnessMap`, `clearcoatNormalMap`                                                                                                                    |
| `KHR_materials_iridescence.iridescenceFactor`                   | `iridescence`                                                                                                                                                                    |
| `KHR_materials_iridescence.iridescenceIor`                      | `iridescenceIor`                                                                                                                                                                 |
| `KHR_materials_iridescence.iridescenceThicknessMinimum/Maximum` | `iridescenceThicknessRange`                                                                                                                                                      |
| `KHR_materials_anisotropy.anisotropyStrength`                   | `anisotropy`                                                                                                                                                                     |
| `KHR_materials_anisotropy.anisotropyRotation`                   | `anisotropyRotation`                                                                                                                                                             |
| `KHR_materials_anisotropy.anisotropyTexture`                    | `anisotropyMap`                                                                                                                                                                  |

### Emitters

| glTF feature                      | Core `SceneEmitter` type                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| `KHR_lights_punctual` point light | `PointEmitter` (`intensity` = candela, `decay = 2`)                                   |
| `KHR_lights_punctual` spot light  | `SpotEmitter` (`angle` = `outerConeAngle`, `penumbra` derived from inner/outer ratio) |
| `KHR_lights_punctual` directional | `DirectionalEmitter` (`intensity` = lux)                                              |

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
  _ancestor_ of a mesh node have no mapped primitives; hosts needing full
  scene-graph animation must retain the `GltfJson` hierarchy and recompute
  world transforms themselves.
- **weights** — write the sampled vector into the skinned primitive's
  `morphWeights`, re-run `solveSkin`, and push `positions`/`normals`.
- **Skeletal (joint-node) channels** — the channel names the joint's glTF node;
  after sampling the skeleton pose, rebuild `SkinnedMeshPrimitive.bones` and
  re-run `solveSkin`. The adapter does not retarget skeletal clips.

### Out of scope (documented)

- **Cameras**: ignored. Emitting a warning.
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

`KHR_draco_mesh_compression` and `EXT/KHR_meshopt_compression` work without
host setup. The default `'builtin'` policy lazy-loads each codec only when an
asset uses it; uncompressed assets do not initialize either decoder.
Public imports use the import-wide resource ledger as the authority and do not
reimpose the standalone 512 MiB compression guard when the corresponding public
byte or geometry ceiling is explicitly disabled.

- Draco uses a package-owned Google Draco 1.5.7 decoder WASM and a
  browser-clean ESM wrapper. The same wrapper and exact WASM bytes run in
  browser, worker, and Node hosts. The vendored decoder chunk contains no
  `fs`, `path`, encoder, or CommonJS compatibility branch.
- meshopt lazy-loads `MeshoptDecoder` from meshoptimizer 1.1.1 (MIT) and waits
  for its `ready` promise before decoding.
- `opts.dracoDecode` and `opts.meshoptDecode` are optional overrides. An explicit
  hook always wins over the built-in.
- `compressionDecoderPolicy: 'host-only'` disables both built-ins. It is useful
  for hosts that centrally provision codecs or intentionally forbid WASM.

Failed codec initialization is not cached permanently: a later conversion
retries the Draco module/WASM load or the meshoptimizer runtime initialization.
Each conversion is copy-on-write; a late codec or validation
failure cannot publish half-rewritten JSON or synthetic buffers. Optional
extensions may use only fully validated spec fallbacks. Required extensions,
or compressed data without an exact fallback, fail closed with an error.

### Optional override contract

```ts
type DracoDecodeFn = (
  compressed: Uint8Array,
  attributeIds: Readonly<Record<string, number>>,
  context?: DracoDecodeContext,
) => DracoDecodeResult | Promise<DracoDecodeResult>;

interface DracoDecodeContext {
  attributes: Readonly<
    Record<
      string,
      {
        componentType: 5120 | 5121 | 5122 | 5123 | 5126;
        normalized: boolean;
        count: number;
        type: 'SCALAR' | 'VEC2' | 'VEC3' | 'VEC4';
      }
    >
  >;
}

interface DracoDecodeResult {
  attributes: Record<
    string,
    Int8Array | Uint8Array | Int16Array | Uint16Array | Float32Array
  >;
  // Triangle-list faces. When primitive.indices is absent, the adapter creates
  // a synthetic SCALAR index accessor instead of discarding connectivity.
  indices: Uint8Array | Uint16Array | Uint32Array;
}

type MeshoptDecodeFn = (
  compressed: Uint8Array,
  count: number,
  byteStride: number,
  mode: 'ATTRIBUTES' | 'TRIANGLES' | 'INDICES',
  filter: 'NONE' | 'OCTAHEDRAL' | 'QUATERNION' | 'EXPONENTIAL' | 'COLOR',
) => Uint8Array | Promise<Uint8Array>;
```

Decoded Draco attributes must exactly match their declared accessor shape and
component type, or be already-dequantized `Float32Array`s. `JOINTS_n` is the
strict exception: joint indices must preserve the accessor's declared unsigned
integer type (`Uint8Array` or `Uint16Array`) so they cannot be rounded or
normalized. Decoded indices are
validated against the accessor and vertex count. Draco exposes connectivity as
face lists, so decoded `TRIANGLE_STRIP` inputs are normalized to indexed
`TRIANGLES` before downstream topology conversion. meshopt output must be
exactly `count × byteStride` bytes.

The two meshopt extension names are validated against their distinct contracts.
`KHR_meshopt_compression` accepts the ATTRIBUTES codec v1
header and `COLOR` filter; `EXT_meshopt_compression` accepts only its legacy
codec/header set and rejects `COLOR`. KHR permits its codec `byteStride` to
differ from the parent bufferView stride, while an authored EXT parent stride
must match the codec stride. A buffer-level `fallback: true` value is an
ownership marker for actual uncompressed fallback storage, not proof that bytes
are absent: every view referencing that buffer must carry the matching extension,
and no compressed-source declaration may point at it. Fallback is selected only
after the declared byte range is present and valid.

### Failure behavior

| Situation                                                  | Behavior                                                    |
| ---------------------------------------------------------- | ----------------------------------------------------------- |
| Built-in policy, no override                               | Lazy built-in decode                                        |
| Explicit hook                                              | Hook overrides the built-in                                 |
| Decoder/hook fails, optional extension with exact fallback | Fully validated fallback + structured degraded diagnostic   |
| Decoder/hook fails, required extension                     | Throws; required extension is never silently bypassed       |
| Decoder/hook fails, no exact fallback                      | Throws; malformed or incomplete geometry is never published |

### Vendored Draco provenance

The decoder is Google Draco 1.5.7 under Apache-2.0. The complete license is at
`src/assets/LICENSE.draco.txt`. `src/vendor/draco_decoder_browser.js` is a
mechanical derivation of upstream `draco_decoder_nodejs.js`: its Node `fs`/`path`
branch was removed and the UMD footer was replaced by an ESM default export.
Decoder logic is otherwise unchanged.

| File                                  | SHA-256                                                            |
| ------------------------------------- | ------------------------------------------------------------------ |
| Upstream `draco_decoder_nodejs.js`    | `e8049906ef3f8f75d3456c22a3f31bfdfe5b5b5bd09ccdec613b9e9a49d554d8` |
| `src/vendor/draco_decoder_browser.js` | `7ec0115432825e898de8796e696b2b6a424405307295b4ab4eae14ade8b2d375` |
| `src/assets/draco_decoder.wasm`       | `2516a4e43526d71787bf2f678f951329f7f858f8f15f42d4bc9e370b31a0da3a` |

---

## Texture handles

The adapter decodes image bytes to opaque handles. Which handle shape you get
depends on the environment:

| Environment   | `opts.decodeImage` | Handle shape                                                                                                                     | Works with                                                                                                       |
| ------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Browser       | absent             | `ImageBitmap` for `loadGltfAsset()`; deterministic PNG/JPEG pixels for decoded loads, with platform image/canvas fallback for WebP and other supported formats | pt-webgpu external-image upload or CPU-atlas payloads |
| Node / Worker | absent             | `{ kind: 'raw-image', mimeType, data: Uint8Array }` for `loadGltfAsset()`; deterministic PNG/JPEG pixels for decoded loads; Node also has a built-in WebP decoder | Raw-handle backend upload or decoded CPU payloads |
| Any           | provided           | Return value of callback                                                                                                         | Whatever the callback returns                                                                                    |

Supply `opts.decodeImage` to control the decode target precisely. The adapter
calls it once per unique image index under the import's bounded concurrency
limit. Each resolved callback result transfers one fresh ownership unit to that
import/result. A host cache that shares one closable identity across imports must
retain ownership itself (for example by returning a non-closable wrapper) or
provide idempotent/reference-counted `close()` semantics. Identity sharing within
one import is deduplicated automatically.

### sRGB vs linear

The default TextureRef bridge passes bytes/opaque handles as-is. **The backend is responsible for colorspace-correct upload:**

- `baseColorMap`, `emissiveMap` → **sRGB** (backends must use `sRGB` texture format or gamma-decode in shader).
- `normalMap`, `roughnessMap` (ORM), `aoMap`, `lightMap`, `bumpMap`, `anisotropyMap` → **linear** (must NOT sRGB-decode).

`decodeSceneTextures(target: 'cpu-linear')` is the opt-in exception: raw image
handles are converted to linear `Float32Array` RGBA payloads using this same
color/data policy before backend upload. PNG and JPEG use the same
pure-JavaScript decoder and options in browser and Node, preventing browser
color management or canvas implementation differences from changing decoded
pixels. Browser WebP uses the platform image/canvas path while Node WebP uses
`webp-wasm`; a host that requires byte-identical WebP output across hosts
supplies one shared `decodePixels` implementation, which always takes
precedence. Workers without canvas readback, compressed texture sources, and
custom formats likewise use `decodePixels`.

The built-in PNG/JPEG and Node WebP paths preflight encoded dimensions against
the exact pixel policy before decoder import or output allocation. The JPEG wrapper
passes explicit non-stricter `jpeg-js` resolution and memory options, so an
explicit zero or a high finite pixel ceiling does not silently restore that
codec’s smaller defaults.

`decodeSceneTextures(target: 'webgpu')` also resolves raw image handles through
the same platform/custom pixel path, but preserves the backend upload color
space: sRGB material maps stay sRGB-valued so WebGPU sRGB texture formats can
perform the hardware decode, while linear/data maps stay linear. This is the
predictable pre-upload path for glTF loads that would otherwise leave
`{ kind: 'raw-image' }` handles opaque to pt-webgpu.

When `maxTextureSize` or `npotRepeatWrapPolicy: 'resize-to-pot'` changes
dimensions, RGB is reconstructed in linear light: exact texel-area integration
is used while downsampling and pixel-centred bilinear reconstruction while
upsampling. sRGB payloads are encoded again only after filtering. Alpha/data
channels remain independent linear values.

### Sampler metadata

`TextureRef` preserves glTF sampler address and filter intent:

- `wrapS` / `wrapT` carry repeat, clamp-to-edge, or mirrored-repeat.
- `magFilter` and `minFilter` carry authored nearest/linear filtering.
- `mipFilter` carries authored mip policy (`none`, `nearest`, or `linear`).

`textureDecodeReport.entries[]` includes the same sampler fields plus
`usesMipmaps` when the asset authored a mipmapped minification mode. It also
separates the material role's color space (`colorSpace`) from the decoded handle
hint (`handleColorSpace`) when known, so hosts can tell a CPU-linear bake from a
WebGPU-ready sRGB-preserved payload. It also reports decoded handle payload shape
(`handleChannels`, `handleDataType`) when known, so one-call loaders can inspect
whether a backend-ready map is the expected RGBA float payload before upload.
Browser default `ImageBitmap` handles from
ordinary `loadGltfAsset()` are reported through `imageBitmapCount` /
`imageBitmapRefs`; decoded loads report CPU-readable `pixel-data` handles when
a built-in, platform, or custom decoder succeeds, and preserve a structured
diagnostic when the selected fallback cannot read pixels. Current backends already consume per-map
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
