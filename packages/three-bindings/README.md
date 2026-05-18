# @vitrum/three-bindings

Adapter between `THREE.Scene` and `@vitrum/core`'s `Scene` type. Plus a thin glTF loader.

## Public surface

- `sceneFromThreeJS(root, opts?)` — walks a `THREE.Object3D` graph, returns a `@vitrum/core` `Scene` with `primitives`, `emitters`, and `environment`.
- `vitrumSceneToThree(scene, opts?)` — the reverse direction for hosts that want to display a vitrum scene through a THREE renderer.
- `vitrumMaterialToThree(material, opts?)` — material-only conversion (used by hot-reload paths).
- `disposeVitrumThreeSceneRoot(root)` — cleans up the THREE.Material / THREE.BufferGeometry instances `vitrumSceneToThree` allocated.
- `applyEnvironment(scene, env)` — wires `THREE.Scene.environment` from `@vitrum/core` `SceneEnvironment` (HDRI / procedural-sky / none).
- `loadGLTF(url, opts?)` — convenience glTF → `@vitrum/core` Scene loader.
- `extractThreePbrScalars(material)` — read PBR scalar fields from a THREE material into a typed record. Used by walkaround-hybrid + pt-webgpu material packing.

## Extension converters

Host-app-specific userData round-tripping (e.g., stained-glass dichroic LUTs) does NOT live here — it's opt-in via `MaterialExtensionConverter` plugged through the `convertMaterial` / `vitrumMaterialToThree` options. See `@vitrum/stained-glass-extensions` for a reference implementation.

## Status

Pre-1.0. The conversion contract is stable; new userData fields plug in through the extension-converter mechanism without modifying the adapter core.
