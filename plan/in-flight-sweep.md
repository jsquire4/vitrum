# In-flight complexity sweep
**Started:** 2026-05-26 | **Branch:** `main` @ `0a8cea6` (reconcile after C2 TLAS landings)

**Scope:** 596 TS/TSX files under `packages/`, `examples/`, `tools/` (tools: 0 `.ts`; benchmark-runner is `.mjs`). Nine domain agents + two integration agents + knip dead-code pass. Code is sole truth.

## Partition

| Group | Paths | Files |
|-------|--------|------:|
| G1 | core, engine | 38 |
| G2 | three-bindings, scene-lighting, stained-glass-extensions | 29 |
| G3 | shared-bvh, shared-samplers, shared-denoisers | 86 |
| G4 | walkaround-hybrid | 159 |
| G5 | walkaround-rc, dev | 30 |
| G6 | pt-webgl | 33 |
| G7 | pt-webgpu | 66 |
| G8 | three-gpu-pathtracer | 132 |
| G9 | examples (+ tools note) | 23 |

**Boundary overlaps:** `frame.ts`/`Engine` (G1↔G4), `scenePack.ts` (G3↔G4↔G7), `common.wgsl.ts` (G4↔G7), `forkUniformBridge` (G6↔G8↔G3 samplers).

---

## Dead code (knip 6.14.2)

**Config noise:** knip fails loading `three-gpu-pathtracer/vite.config.js` (missing `example/` scan dir).

**Unused files (43)** — includes vitest tests knip misclassifies as unused entrypoints, fork `example/*` demos, `packages/pt-webgpu/src/bdpt/fillBdptLightPathCpu.ts`, `packages/walkaround-hybrid/src/rc/rcParamsLayout.ts` + `.generated.ts` (false positive if only dynamic import).

**Unused devDependencies (20)** — root + package-level eslint/react/playwright peers.

**Unlisted dependencies:** `@vitrum-pathtracer/src/uniforms/MaterialsTexture.js` in pt-webgl test; `@webgpu/types` in tsconfig.

**Unused exports (30+)** — includes `percentile95` (prBenchHarness), `_resetCacheUnsafe` (gpuDetection), many `PT_WEBGPU_*` WGSL re-exports, `collectDDGIPointLightsFromRoot`, PPG constants, restir helpers, `MOTION_VECTORS_WGSL`, etc.

**Unused exported types (14)** — FrameResources sub-interfaces, neural layer types, OIDNModelTensorNames, SurfaceTextureId/Name.

**Duplicate exports:** `PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP` vs `PT_WEBGPU_REQUIRED_STORAGE_BUFFERS` in webgpuLimits.ts.

---

## Raw agent findings

See user-facing report in chat (full enumeration). Domains: G1–G9 + INTEGRATION core↔hybrid + INTEGRATION shared↔backends.

**Verified in HEAD (orchestrator):**
- `Mat4` brand + `FrameOutput` discriminated union + `BackendTexture` brands **present** in `packages/core/src` (AGENTS "NOT IN HEAD" is stale).
- `reuseSharedWebGpuDevice` default **opt-in** (`=== true`) in shared-denoisers.
- `PCG_WGSL` canonical in `shared-samplers` only; walkaround/pt-webgpu `common.wgsl.ts` import PCG via module graph (no local `fn pcgInit` duplicate in those files).
- `forkAccess.ts` **exists** in pt-webgl.
- `window.__WGPU__` write **still present** in HybridEngine debug branch (~1154).
- `window.__DDGI__` in probeUpdatePass + DDGI.ts.
