# @vitrum/gltf-adapter

glTF 2.0 → `@vitrum/core` Scene adapter.

**Zero runtime dependencies.** Hand-rolled GLB container parsing and accessor
unpacking (no Draco, no meshopt). Browser + Node compatible; image decoding is
pluggable.

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

Returns `{ scene, animations, animationTargets, warnings }`:

| Field | Type | Description |
|-------|------|-------------|
| `scene` | `Scene` | The converted `@vitrum/core` scene (rest pose) |
| `animations` | `AnimationClip[]` | glTF animations as core clips (empty when none). Evaluate with `sampleAnimationClip` from `@vitrum/core` |
| `animationTargets` | `Record<string, string[]>` | Maps a channel node id (`gltf-node-<index>`) → the `ScenePrimitive.id`s created from that node's mesh |
| `warnings` | `string[]` | Non-fatal conversion issues |

---

## Support matrix

### Geometry

| Feature | Status |
|---------|--------|
| GLB binary container | Supported |
| .gltf JSON + pre-fetched buffers | Supported |
| POSITION / NORMAL / TEXCOORD_0 / TEXCOORD_1 | Supported |
| TANGENT / COLOR_0 | Supported (imported; backends may or may not consume) |
| Indices (UINT16 / UINT32 / UINT8) | Supported |
| Flat normal generation (NORMAL absent) | Supported |
| Sparse accessors | Supported (all component types) |
| Multiple primitives per mesh | Supported → one `MeshPrimitive` per glTF primitive |
| Node hierarchy / nested TRS + matrix | Supported → flattened world transforms |
| Primitive mode TRIANGLES (4) | Supported |
| Primitive modes TRIANGLE_STRIP (5) / TRIANGLE_FAN (6) | Supported → triangulated to an indexed triangle list (glTF §3.7.2.1 winding; degenerates dropped; indexed + non-indexed) |
| Point/line modes (POINTS, LINES, LINE_LOOP, LINE_STRIP) | Warn + skip (core has no point/line primitive) |
| KHR_draco_mesh_compression | Warn + skip |
| EXT_meshopt_compression | Warn + skip |
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
| `KHR_materials_volume.thicknessTexture` | — (warn + ignored; core has no thickness map field) |
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
- **Draco / MeshOpt compression**: warns + primitive skipped. Decode externally first.
- **Morph TANGENT deltas**: warn + skipped (core `SkinnedMeshPrimitive` has no morph-tangent field).
- **`KHR_materials_volume.thicknessTexture`**: warn + ignored (core `MaterialSpec` has no thickness map field).
- **URI-based buffers / images**: the adapter does not fetch. Pre-load and supply via `opts.buffers` or `opts.decodeImage`.

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
