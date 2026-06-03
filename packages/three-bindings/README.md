# @vitrum/three-bindings

Adapter between `THREE.Scene` and `@vitrum/core`'s `Scene` type. Plus a glTF loader and a re-export of the CPU skinning solver (implemented in `@vitrum/core`).

## Public surface

- `sceneFromThreeJS(threeScene)` — traverses a `THREE.Scene`, returns a `@vitrum/core` `Scene` with `primitives`, `emitters`, and `environment`. Accepts `THREE.Mesh` (Standard / Physical / Basic material), `THREE.SkinnedMesh` (rest pose + skeleton + optional morph targets), `THREE.InstancedMesh` (→ `InstancedMeshPrimitive`), and supported `THREE.Light` types; throws on `ShaderMaterial` / `RawShaderMaterial` / unsupported material types.
- `vitrumSceneToThree(scene)` — reverse direction for hosts displaying a vitrum scene through a THREE renderer. Skinned meshes are pre-solved at conversion; subsequent per-frame pose changes flow through `engine.updatePrimitive(id, { positions, normals })`.
- `disposeVitrumThreeSceneRoot(root)` — releases the `THREE.Material` / `THREE.BufferGeometry` instances `vitrumSceneToThree` allocated.
- `applyEnvironment(threeScene, environment)` — wires `THREE.Scene.environment` from `@vitrum/core` `SceneEnvironment` (HDRI / procedural-sky / none).
- `loadGltfScene(source, opts?)` — convenience glTF → `@vitrum/core` Scene loader (URL / Blob / File / ArrayBuffer).
- `solveSkin(prim, outPositions?, outNormals?)` / `combineSkinMatrices` / `mat3InverseTranspose` — CPU linear-blend skinning solver. Implemented in `@vitrum/core`; re-exported here for convenience. Takes a `SkinnedMeshPrimitive` (with current pose + optional morph weights + optional `bindMatrix`) and produces deformed positions / normals ready for `engine.updatePrimitive`.
- `convertMaterial(threeStdMat)` / `convertBasicMaterial(threeBasicMat)` — material-only conversion (used by hot-reload paths and downstream tests).
- `extractThreePbrScalars(material)` — read PBR scalar fields from a THREE material into a typed record. Used by `walkaround-hybrid` + `pt-webgpu` material packing.
- `VITRUM_USER_DATA_KEYS` — string-key table for the `userData` round-trip protocol.

## Extending material conversion

Host-app-specific userData round-tripping (e.g., stained-glass dichroic LUTs) flows through three.js's standard `material.userData` slot plus the `Material.extensions` discriminated extension point in `@vitrum/core`. `convertMaterial` reads the well-known keys defined in `VITRUM_USER_DATA_KEYS` and stamps them into the returned `MaterialSpec.extensions`. See `external_requests/IMPLEMENTATION-STATUS.md` for the active extension list.

## Status

Pre-1.0. The conversion contract is stable; new userData fields plug in through `VITRUM_USER_DATA_KEYS` + `MaterialSpec.extensions` without modifying the adapter core.
