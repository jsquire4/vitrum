# @vitrum/gltf-adapter

glTF 2.0 → `@vitrum/core` Scene adapter.

**Zero runtime dependencies.** Hand-rolled GLB container parsing and accessor
unpacking. Browser + Node compatible; image decoding is pluggable, and Draco /
meshopt compressed geometry is supported via host-supplied decoder hooks (the
package bundles no decoder — see [Compressed geometry](#compressed-geometry)).

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
| POSITION / NORMAL / TEXCOORD_0 / TEXCOORD_1 | Supported |
| TANGENT / COLOR_0 | Supported. Authored TANGENT is preserved; tangent-space mapped primitives without TANGENT synthesize xyzw tangents from POSITION/NORMAL/TEXCOORD_0. COLOR_0 is imported and compatibility-reported: pt-webgl2 consumes it natively; pt-webgpu and walkaround-hybrid currently report structured unsupported issues. |
| Indices (UINT16 / UINT32 / UINT8) | Supported |
| Flat normal generation (NORMAL absent) | Supported |
| Sparse accessors | Supported (all component types) |
| Multiple primitives per mesh | Supported → one `MeshPrimitive` per glTF primitive |
| Node hierarchy / nested TRS + matrix | Supported → flattened world transforms |
| Primitive mode TRIANGLES (4) | Supported |
| Primitive modes TRIANGLE_STRIP (5) / TRIANGLE_FAN (6) | Supported → triangulated to an indexed triangle list (glTF §3.7.2.1 winding; degenerates dropped; indexed + non-indexed) |
| Point/line modes (POINTS, LINES, LINE_LOOP, LINE_STRIP) | Warn + skip (core has no point/line primitive) |
| KHR_draco_mesh_compression | Supported via `opts.dracoDecode` hook. Without a hook: uncompressed fallback accessors when present (warn), else warn + skip; throws if in `extensionsRequired` with no fallback |
| EXT_meshopt_compression | Supported via `opts.meshoptDecode` hook (bufferView-level — geometry, animation and image consumers all see decompressed data). Without a hook: spec fallback buffer when present (warn), else warn + skip; throws if in `extensionsRequired` with no fallback |
| Morph targets (POSITION + NORMAL deltas, sparse OK) | Supported → `SkinnedMeshPrimitive.morphTargets` / `.morphTargetNormals` / `.morphWeights` (node/mesh weights; unskinned morphed meshes get a synthesized identity skeleton). TANGENT deltas warn + skip |
| Skins / JOINTS_0 (u8 + u16) / WEIGHTS_0 | Supported → `SkinnedMeshPrimitive` at rest pose (incl. `bindMatrix`/`bindMatrixInverse`) |
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
| `KHR_materials_volume.thicknessTexture` | `thicknessMap` (reserved; backend support varies) |
| `KHR_materials_volume.attenuationDistance` | `attenuationDistance` |
| `KHR_materials_volume.attenuationColor` | `attenuationColor` |
| `KHR_materials_specular.specularFactor` | `specularIntensity` |
| `KHR_materials_specular.specularColorFactor` | `specularColor` |
| `KHR_materials_specular.specularTexture` | `specularIntensityMap` |
| `KHR_materials_specular.specularColorTexture` | `specularColorMap` |
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
- **Morph TANGENT deltas**: warn + skipped (core `SkinnedMeshPrimitive` has no morph-tangent field).
- **`KHR_materials_volume.thicknessTexture`**: imported as reserved
  `thicknessMap`; backend support varies and compatibility reports surface
  unsupported targets.
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
| Browser | absent | `ImageBitmap` | pt-webgpu, pt-webgl2 |
| Node / Worker | absent | `{ kind: 'raw-image', mimeType, data: Uint8Array }` | Needs custom backend upload |
| Any | provided | Return value of callback | Whatever the callback returns |

Supply `opts.decodeImage` to control the decode target precisely. The adapter
calls it once per unique image index, in parallel.

### sRGB vs linear

The adapter passes bytes as-is. **The backend is responsible for colorspace-correct upload:**

- `baseColorMap`, `emissiveMap` → **sRGB** (backends must use `sRGB` texture format or gamma-decode in shader).
- `normalMap`, `roughnessMap` (ORM), `aoMap`, `lightMap`, `bumpMap`, `anisotropyMap` → **linear** (must NOT sRGB-decode).

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
