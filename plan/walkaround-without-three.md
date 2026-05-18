# Walkaround hybrid: what breaks if THREE disappears?

**Audience:** maintainers generalizing `@vitrum/walkaround-hybrid` beyond three.js hosts.

This is the Milestone **M4** answer to: _“What would break if THREE disappeared from walkaround?”_

## Summary

| Layer                     | THREE today?                                           | Replacement concept                                                                                                                                                                           |
| ------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Public engine options** | `HybridEngineOptions.threeScene: THREE.Scene`          | A **host scene graph** exposing mesh geometry + world transforms + materials the BVH merge understands; DDGI still expects a **full `Scene`** until `SceneBvh` generalizes.                   |
| **ReSTIR BVH**            | `buildSceneBVH([scene], …)` via `@vitrum/shared-bvh`   | Already accepts `THREE.Scene \| Object3D[]`; a non-THREE host could supply **`Object3D`-shaped** data only if **`StaticGeometryGenerator`** / mesh-bvh path is reimplemented for raw buffers. |
| **DDGI**                  | `DDGIFrameInputs.scene: THREE.Scene`                   | **ProbeUpdatePass** + **`SceneBvh.update(scene)`** traverse three meshes; needs **`WalkaroundDDGIScene`** abstraction (see `hostScene/types.ts`).                                             |
| **TSL / Node materials**  | `GIReceiver`, `applyDDGIShading`, RC material wrappers | **Hard-coupled** to `three/webgpu` + `three/tsl`; a Babylon host would not use these entrypoints — separate material bridge.                                                                  |
| **WGSL pipelines**        | None (strings + bind layouts)                          | **Unaffected** — backend is WebGPU + WGSL.                                                                                                                                                    |

## Module-by-module

| Module                              | THREE coupling                                                                                                         |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `HybridEngine.ts`                   | Holds **`THREE.Scene`**; passes to DDGI + `restir/buildSceneBVH`; triangle-count readiness uses **`scene.traverse`**.  |
| `restir/bvhCompute.ts`              | **`THREE.Vector3`, `THREE.Color`**, mesh materials, UV/attenuation helpers — all mesh-bvh + three material model.      |
| `shared-bvh` usage                  | **`buildSceneBVH`** from three scene graph (internally **three-mesh-bvh**).                                            |
| `ddgi/*`                            | Scene traversal, **`SceneBvh`**, probe pass expects three renderer adapter shape.                                      |
| `pipeline/WalkaroundGPUPipeline.ts` | Mostly GPU — coupling via **buffer layouts** built from BVH emitters (emitter format today derived from three scenes). |
| `rc/*`                              | Cascade math + **TSL** hooks — three/webgpu stack.                                                                     |
| `shaders/*.wgsl.ts`                 | None.                                                                                                                  |

## RC re-composition (hybrid shade pass)

**Historical:** RC cascade compute was **removed from the hybrid stack** (shade pass no longer samples `Lo_rc`); RC modules remain for the **standalone RC engine path**.

**To re-enable in `HybridEngine`:**

1. Restore **uniform / texture** slots that feed RC radiance into **`SHADE_WGSL`** (or composite pass).
2. Schedule **cascade dispatch** in `renderFrame` when `HybridEngine`’s layer toggles include RC (mirror pre–2026-05-08 ordering).
3. Validate **DDGI + ReSTIR + RC** memory budget (bind group limits) after merge.

Until then, no runtime TODO in hot paths — behavior is **intentionally** DDGI + ReSTIR only in hybrid.

## Related

- `packages/walkaround-hybrid/src/hostScene/types.ts` — exported seam types for future non-THREE BVH roots.
- [binding-babylon-sketch.md](./binding-babylon-sketch.md) — second-binding field checklist.
