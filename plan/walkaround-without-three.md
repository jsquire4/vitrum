# Walkaround hybrid: what breaks if THREE disappears?

**Audience:** maintainers generalizing `@vitrum/walkaround-hybrid` beyond three.js hosts.

This is the Milestone **M4** answer to: *“What would break if THREE disappeared from walkaround?”*

## Summary

| Layer | THREE today? | Replacement concept |
|-------|----------------|----------------------|
| **Public engine options** | `HybridEngineOptions.threeScene: THREE.Scene` | A **host scene graph** exposing mesh geometry + world transforms + materials the BVH merge understands; DDGI still expects a **full `Scene`** until `SceneBvh` generalizes. |
| **ReSTIR BVH** | `buildSceneBVH([scene], …)` via `@vitrum/shared-bvh` | Already accepts `THREE.Scene \| Object3D[]`; a non-THREE host could supply **`Object3D`-shaped** data only if **`StaticGeometryGenerator`** / mesh-bvh path is reimplemented for raw buffers. |
| **DDGI** | `DDGIFrameInputs.scene: THREE.Scene` | **ProbeUpdatePass** + **`SceneBvh.update(scene)`** traverse three meshes; needs **`WalkaroundDDGIScene`** abstraction (see `hostScene/types.ts`). |
| **TSL / Node materials** | `GIReceiver`, `applyDDGIShading`, RC material wrappers | **Hard-coupled** to `three/webgpu` + `three/tsl`; a Babylon host would not use these entrypoints — separate material bridge. |
| **WGSL pipelines** | None (strings + bind layouts) | **Unaffected** — backend is WebGPU + WGSL. |

## Module-by-module

| Module | THREE coupling |
|--------|----------------|
| `HybridEngine.ts` | Holds **`THREE.Scene`**; passes to DDGI + `restir/buildSceneBVH`; triangle-count readiness uses **`scene.traverse`**. |
| `restir/bvhCompute.ts` | **`THREE.Vector3`, `THREE.Color`**, mesh materials, UV/attenuation helpers — all mesh-bvh + three material model. |
| `shared-bvh` usage | **`buildSceneBVH`** from three scene graph (internally **three-mesh-bvh**). |
| `ddgi/*` | Scene traversal, **`SceneBvh`**, probe pass expects three renderer adapter shape. |
| `pipeline/WalkaroundGPUPipeline.ts` | Mostly GPU — coupling via **buffer layouts** built from BVH emitters (emitter format today derived from three scenes). |
| `rc/*` | Cascade math + **TSL** hooks — three/webgpu stack. |
| `shaders/*.wgsl.ts` | None. |

## RC re-composition (hybrid shade pass)

**Historical:** RC cascade compute was **removed from the hybrid stack** (shade pass no longer samples `Lo_rc`); RC modules remained for the **standalone RC engine path**.

**Status (2026-05-18):** W8 sprint in flight. See [w8-rc-mis-composition.md](./w8-rc-mis-composition.md) for the full phased plan.

- **Phase 1A** ✅ — `cascadePyramid.ts` + `cascadeBuffers.ts` THREE-free (plain `CascadeAABB` `{min,max}` + `[x,y,z]` tuples replace `THREE.Box3` / `THREE.Vector3`).
- **Phase 1B** ✅ — `cascadeDispatch.ts` gains a parallel `dispatchFrameRaw(opts: RCDispatchOptsRaw)` entry that takes `GPUDevice` + raw `GPUBuffer`s + plain tuples (no THREE imports in the new path).
- **Phase 2**  ✅ — `HybridEngineOptions.rcEnabled` toggle + per-engine `RCSubsystem` sidecar that builds its own BVH + cascade `GPUBuffer`s and dispatches each frame. **Cascade-0 output is NOT yet sampled in `shade.wgsl` (Phase 3).**
- **Phase 3**  ⏳ — `shade.wgsl` reads cascade-0, MIS composition with DDGI / ReSTIR-GI.
- **Phase 4**  ⏳ — Reference renders + acceptance test (Cornell with `rcEnabled: true` vs DDGI-only).

Until Phase 3 lands, RC dispatch runs when `rcEnabled: true` but does not affect the visible image — useful only for memory-budget + dispatch-path validation.

## Related

- `packages/walkaround-hybrid/src/hostScene/types.ts` — exported seam types for future non-THREE BVH roots.
- [binding-babylon-sketch.md](./binding-babylon-sketch.md) — second-binding field checklist.
