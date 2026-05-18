# Sketch: Babylon.js scene binding (future)

**Status:** design pressure-test only — no `@vitrum/babylon-bindings` package until PT + walkaround baselines are stable.

## Purpose

Validate that `@vitrum/core` **`Scene`** field names are sufficient when the host is **not** three.js. If a field is three-specific (e.g. assumes `Texture.image` exists), it belongs in **`Material.extensions`** or a **binding-local** wrapper, not in core.

## Mapping table (conceptual)

| `@vitrum/core`                                | Babylon analogue (sketch)                                                                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `MeshPrimitive.positions/normals/uvs/indices` | `VertexBuffer` / `Geometry.getVerticesData` — copy into `Float32Array` / typed index buffer                     |
| `MeshPrimitive.transform`                     | `mesh.getWorldMatrix()` → column-major `Float32Array(16)`                                                       |
| `Material` PBR fields                         | `PBRMaterial` / `StandardMaterial` albedo, roughness, metalness, etc.                                           |
| `TextureRef`                                  | Babylon `Texture` handle as **opaque** `unknown`, or serializable `{ url, colorSpace }` if backend loads by URL |
| `RectAreaEmitter`                             | Area light or emissive plane mesh → map `uAxis`/`vAxis` from width/height + world rotation                      |
| `DirectionalEmitter`                          | `DirectionalLight.direction` (invert if Babylon uses “direction to scene”)                                      |
| `SceneEnvironment` `hdri`                     | `CubeTexture` / equirect `Texture` as opaque ref or file ref                                                    |

## Contract rules

1. **`@vitrum/core` never imports Babylon or THREE.**
2. Bindings may **throw** on unsupported host features; they must **document** caps in package README (mirror `three-bindings`).
3. **`Engine` implementations** that only understand three today (`pt-webgl`, `walkaround-hybrid`) stay behind **`sceneFromThreeJS`** + optional future `sceneFromBabylon`; no requirement that every backend accept every binding on day one.

## Next step (when scheduled)

Add an empty `packages/babylon-bindings` with `sceneFromBabylon(scene): Scene` **stub** that throws `Not implemented`, to reserve the name and force TS consumers to opt in explicitly.
