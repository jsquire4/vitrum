# Layered hybrid v2 — DDGI + RC + ReSTIR composing additively

## Goal

Compose three real-time GI techniques into a single ReSTIR-driven shade pass:

```
final_color = direct_lighting × sun_visibility(RC)
            + diffuse_indirect(DDGI atlas)
            + emissive
            + specular_BRDF
```

Where:

- **ReSTIR DI** owns the shade kernel — primary-ray-cast, importance-sampled
  reservoir resampling for direct lighting from emissive lights.
- **DDGI** runs as a compute prerequisite. Updates the irradiance/visibility
  probe atlas per-frame (1/4 round-robin). Atlas binds into the shade pass
  as ray-miss radiance — secondary bounces sample the probe instead of
  tracing further. HDRP-RT pattern.
- **RC** runs as a compute prerequisite. Updates a coarse sun-visibility
  cascade pyramid per-frame. Shade pass samples the cascade for sun
  shadow attenuation instead of tracing a per-pixel shadow ray.

v1 ships with ReSTIR alone routing under `walkaroundEngine: 'hybrid'`. v2
keeps that visual quality while adding DDGI + RC layers as compute
prerequisites that make the shade pass faster (target: 60 FPS where v1
sits at ~30 FPS).

## Architecture choice — Path B (compute prerequisite layering)

Path A (TSL injection per material via `material.outputNode`) was the
original "glorious hybrid" plan. It crashed because:

- WebGPU renderer's RenderObject cache desynced when the composer reassigned
  `outputNode` mid-session. Per-frame `Nodes.delete(undefined).usedTimes`
  storm froze Chrome.
- The TSL graph composition pattern (`add(output, vec4(layerSum, 0))`) is
  fragile across Three.js NodeMaterial versions.

Path B (compute prerequisite, ReSTIR owns the frame) avoids both pitfalls:

- ReSTIR's `WalkaroundGPUPipeline` already manually composites to the swap
  chain. No TSL involvement.
- DDGI / RC compute results live in shared storage textures + buffers.
  Shade pass reads them via standard bind groups.
- Materials are NEVER upgraded mid-session. The renderer cache stays stable.

The orphaned files (`HybridStage.tsx`, `HybridContext.ts`,
`useHybridFrameLoop.ts`, `applyHybridShading.ts`, `layers/*.ts`) implement
Path A and stay in tree as historical reference. Path B uses a different
architecture — a unified `HybridLayeredStage` that owns all three compute
pipelines + the shade pass.

## Phases

### Phase 1 — DDGI as ray-miss radiance

Status: ✅ shipped (commits 913eefd through 89903e7, 2026-05-07).

Variation from original plan: DDGI integrated as ambient on opaque
receivers (`Lo_ddgi = ddgi × albedo × INV_PI` added to combined
radiance) rather than as ray-miss replacement on the sky pixel branch.
The opaque-receiver integration is more visually meaningful — the
panel-emitter contributions reach walls/floor via the probe atlas
instead of being limited to single-bounce ReSTIR-GI.

Sub-phases:

- 1.1 (913eefd): HybridLayeredStage skeleton — useDDGI + ReSTIR coexisting
- 1.1.1 (ee5d309): DDGI gate + OrbitControls
- 1.2A (1bfc599): WalkaroundGPUPipeline DDGI bind group infrastructure
- 1.2B-shader (d86cbef): shade.wgsl declarations + smoke-test gate
- 1.2B-real+wire (75265b0): ddgiSampleFromBindings + per-frame setDDGIInputs
- 1.2B-toggle (89903e7): **HYBRID_LAYERS**.ddgi=false properly reverts UBO

Changes:

1. Add `HybridLayeredStage.tsx` (new). Mounts when
   `walkaroundEngine === 'hybrid'`. Replaces the v1 ReSTIR routing.
   Owns:
   - DDGI compute pipeline (`probeUpdatePass.ts` + `probeGrid.ts`)
   - ReSTIR pipeline (`WalkaroundGPUPipeline.ts`)
   - Shared BVH + scene
2. Per-frame loop (priority=1, ReSTIR-style frame ownership):
   - Run DDGI 1/4 round-robin probe update
   - Run ReSTIR full pipeline (RIS + temporal + spatial + shade + atrous + composite)
3. Shade pass extension (`shade.wgsl`):
   - Add DDGI atlas + uniform bind group entries
   - Replace hardcoded `vec4f(0.5, 0.7, 1.0, 1.0)` sky with DDGI sample at
     ray-miss position
   - For secondary indirect bounces (`Lo_at_xs`), sample DDGI atlas at the
     bounce-hit position when bounce ray misses the BVH (sky escape)
4. Hardware-validate on Lovelace via Chrome MCP.

Acceptance gate: `__HYBRID_LAYERS__.ddgi = false` produces v1 visual;
`= true` adds visible diffuse-indirect bounces (e.g. wall opposite the
panel picks up colored bounce light from the panel).

### Phase 2 — RC as sun visibility

#### Phase 2A — RC compute running alongside DDGI ✅ shipped (9008ea8)

Sub-phase: HybridLayeredStage now mounts RC's useSceneBVH +
useCascadeBuffers + per-frame dispatchCascadePasses at priority=0.
Visual unchanged because shade.wgsl doesn't yet sample the cascade.
Toggle: `window.__HYBRID_LAYERS__.rc = false` disables.

#### Phase 2B — shade.wgsl reads cascade ❌ BLOCKED on WebGPU limits

Wiring the cascade buffers into shade.wgsl hits the default
`maxStorageBuffersPerShaderStage = 8`. Current shade pipeline already
binds 8 storage buffers (5 frame: reservoirs + 3 scene: BVH/idx/pos +
emitters/CDF). Adding 5 cascade buffers (one per cascade level C0–C4)
would total 13 — exceeds the limit.

Two viable approaches for the blocker:

1. **Pack-into-one-buffer**: concatenate all 5 cascades into a single
   storage buffer + a per-level-offset UBO. Single binding. Cleanest
   from a budget perspective. Requires writing a packer in
   HybridLayeredStage and a sampler in shade.wgsl that reads from the
   right offset given (cascadeLevel, cellIdx). Recommended.

2. **Request higher device limit**: in StudioScene's gl factory,
   create the WebGPURenderer with `requiredLimits:
{ maxStorageBuffersPerShaderStage: 16 }`. Modern hardware supports
   it; software fallback (SwiftShader) doesn't. Re-tests would need
   to gate on `isHardwareGpu` AND the requested feature.

Once unblocked:

1. Add cascade bind group (group 4) to WalkaroundGPUPipeline.
2. Add `pipeline.setRCInputs(packedBuffer, levelOffsetsUbo)`.
3. Modify shade.wgsl: add `cascadeSampleSunVisibility(worldPos)` →
   replaces or multiplies the existing per-pixel
   `bvhTraceTintedVisibility` call in `Lo_sunCaustic`.
4. HybridLayeredStage: extract GPUBuffer from
   `cascadeBuffers.gpuCascades[i]` (StorageBufferAttribute) and pack
   into the single buffer once per BVH rebuild. Push level offsets
   each frame.
5. Hardware-validate.

Acceptance gate: `__HYBRID_LAYERS__.rc = false` falls back to
per-pixel `bvhTraceTintedVisibility`. `= true` uses the cascade for a
faster (but blockier) sun-visibility sample. FPS should be
measurably higher with rc=true on a complex scene.

### Phase 3 — Layer toggles + perf measurement

1. `__HYBRID_LAYERS__ = { ddgi: bool|'isolate', rc: bool|'isolate', restir: bool|'isolate' }`
2. `'isolate'` mode: zero out all other layers; show ONLY this layer's
   contribution (debug visual).
3. Add per-layer ms timing to `__WGPU__.walkaround.frameTimings`.
4. Validate target: 60 FPS at 1080p with all three layers active on
   NVIDIA Lovelace.

### Phase 4 (optional) — Specular split-sum + denoising polish

Per the original glorious-hybrid plan; Phase 5-6 of that plan.

## Anti-cheese rules

- DON'T re-introduce the TSL `outputNode` reassignment pattern. Materials
  stay un-upgraded; ReSTIR owns the frame.
- DON'T modify the working ReSTIR shade math. Each layer adds new code
  paths; existing math stays.
- DON'T break the v1 routing. `'hybrid'` keeps producing the path-traced
  caustic visual at every step — even if a phase is mid-build, the user-
  facing button stays correct.
- After every phase: hardware-validate on Lovelace. Visual regression =
  rollback that phase.

## File map

### Changed

- `src/rendering/scene/StudioScene.tsx` — route 'hybrid' to
  HybridLayeredStage instead of RestirStage.
- `src/rendering/scene/walkaround/engines/restir/shaders/shade.wgsl.ts` —
  add DDGI / RC bind group reads + sample helpers.
- `src/rendering/scene/walkaround/engines/restir/WalkaroundGPUPipeline.ts` —
  parameterize bind groups so HybridLayeredStage can swap in DDGI / RC
  inputs.

### New

- `src/rendering/scene/walkaround/HybridLayeredStage.tsx` — Path-B stage
  that owns DDGI compute + RC compute + ReSTIR pipeline.

### Untouched (orphaned Path-A scaffolding kept for reference)

- `HybridStage.tsx`, `HybridContext.ts`, `useHybridFrameLoop.ts`,
  `applyHybridShading.ts`, `layers/*.ts` — Path-A approach, kept in tree.
