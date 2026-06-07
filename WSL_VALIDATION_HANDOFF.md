# WSL validation handoff

Current local context: this tree moved while the Codex pass was running.
`HEAD` is `b1e7470` (`docs(validation): BOTH RC gates PASS - vitem harness
winding+axes fixed, no threshold recalibration needed`), and that commit already
contains the bulk of the requested contract/hygiene pass. The remaining
working-tree code fix from this agent is the strict-indexing correction in
`packages/shared-denoisers/__tests__/indexExports.test.ts`.

Validate from a WSL-native Node/npm install. If `which npm` resolves under
`/mnt/c/`, fix PATH or install Node inside WSL first; Windows npm cannot run the
workspace reliably from the WSL filesystem.

## What changed

- `@vitrum/three-bindings`
  - `sceneFromThreeJS` now applies the same renderability guards to plain
    meshes, `InstancedMesh`, and `SkinnedMesh`: invisible objects/materials,
    transparent near-zero `MeshBasicMaterial`, and `ShaderMaterial` /
    `RawShaderMaterial` rejection.
  - Regression tests cover transparent instanced/skinned skips and shader
    material throws before converter-specific work.
  - `vitrumSceneToThree` now clones a shared `THREE.Texture` handle whenever a
    `TextureRef` carries per-material UV transform or `texCoord` state, avoiding
    last-writer-wins mutation across materials.
  - Regression tests cover transformed texture round-trip and two material specs
    sharing one source texture with different transforms.

- `@vitrum/engine`
  - `attachVitrum` frame composition was extracted to
    `composeAttachVitrumFrameInput`, and `swapChainPlumbing.test.ts` now pins the
    real production composition site instead of reconstructing the spread inline.
  - `telemetryProxy.test.ts` and `debugSurface.test.ts` now exercise
    `wrapWithIdempotentDispose` instead of only fake engines.
  - `lifecycle.test.ts` header was corrected to match the import/type smoke it
    actually performs.

- `@vitrum/shared-denoisers` / `@vitrum/shared-samplers`
  - Canonical `demodulateAlbedo` / `remodulateAlbedo` are exported from the
    package barrel with a regression test.
  - The dead `SPATIAL_FILTER_WGSL` module and tests were removed; package
    metadata and live luminance comments no longer advertise it.
  - Follow-up fix: the new barrel test uses non-null indexed reads so
    `noUncheckedIndexedAccess` stays green.

- `three-gpu-pathtracer`
  - `src/index.d.ts` now matches the JS barrel more closely, including
    `PhysicalPathTracingMaterial`, `PathTracingRenderer`,
    `AmbientOcclusionMaterial`, `WebGLPathTracer.isCompiling`,
    `stableNoise`, and `renderBdptLightSubpathPass`.
  - `volumeMarch` now takes only `(tSurface, u)`, matching what the shader
    actually uses; the lone call site was updated.

## Local checks already run

These were run from the Windows Codex shell using bundled Node plus repo
dependencies where possible:

```bash
# Passed
node node_modules/typescript/bin/tsc --noEmit -p packages/three-bindings/tsconfig.json
node node_modules/typescript/bin/tsc --noEmit -p packages/shared-denoisers/tsconfig.json
node node_modules/typescript/bin/tsc --noEmit -p packages/three-gpu-pathtracer/tsconfig.json

# Passed
node packages/three-gpu-pathtracer/scripts/shader-smoke-check.js
```

Vitest could not be executed from this Windows shell because Vite resolves UNC /
mapped-drive test paths to `/wsl.localhost/...` and fails before collecting
tests. Native WSL should not hit that path-translation bug.

## Targeted validation commands

Run from `/home/jsquire4/projects/vitrum`:

```bash
npm run typecheck
npm test
```

Focused packages/suites for this pass:

```bash
npm run typecheck --workspace @vitrum/three-bindings
npm run test --workspace @vitrum/three-bindings -- \
  src/__tests__/sceneFromThreeJS.test.ts \
  src/__tests__/materialTextureRef.test.ts

npm run typecheck --workspace @vitrum/engine
npm run test --workspace @vitrum/engine -- \
  __tests__/telemetryProxy.test.ts \
  __tests__/debugSurface.test.ts \
  __tests__/lifecycle.test.ts \
  __tests__/swapChainPlumbing.test.ts

npm run typecheck --workspace @vitrum/shared-denoisers
npm run test --workspace @vitrum/shared-denoisers -- \
  __tests__/indexExports.test.ts

npm run shader-smoke --workspace three-gpu-pathtracer
```

If you run the broader repo gate, also include the already-green fork smoke:

```bash
npm run fork-shader-smoke
```

## Still needs GPU-backed judgment

Do not mark these closed from mechanical tests alone:

- `ResolvePass` still runs every frame as an inert copy because
  `checkerboardOn` is hardcoded `0`. GitNexus marked this path HIGH risk, and a
  safe skip requires changing downstream `resolvedTexture` ownership
  (`CompositePass`, `presentLastFrame`, progressive seed texture, timestamp
  layout). Treat this as a GPU pipeline design/validation item, not a CPU-only
  hygiene edit.
- DDGI probe white-bounce/albedo model: needs radiometric A/B before changing.
- `pt-webgpu` photon-map radius/strategy weighting: needs caustic gap-closure
  renders, not just unit tests.
- SVGF-real conservative object-id fallback: larger G-buffer/object-id work.
- NRC stale-record clearing: validate record buffer behavior before patching.
- Stained-glass normal-map shadow perturbation: needs a product decision on
  whether to gate, remove, or replace with a physically justified option.
- OIDN readback race/dimension guard: narrow lifecycle edge; verify under
  resize/dispose stress before patching.

## API ergonomics proposals only

Do not implement these until the contract shape is agreed in `@vitrum/core`.

- Backend-typed factories:
  - `createEngine(opts): Promise<Engine>` stays the erased facade.
  - Add overloads or named factories returning backend-specific intersections,
    for example `createPTEngine_WebGL2(...): Promise<Engine & PTEngineWebGL2>`.
  - Prefer named backend factories over a generic conditional return type unless
    the public option discriminator is already stable enough.

- Scene read-back:
  - Add optional `engine.getScene?(): Scene` to `Engine`, with copy/freeze
    semantics documented. Backends that maintain internal THREE scenes should
    return the canonical core `Scene`, not a host object reference.
  - This would let hero-viewer and similar hosts stop shadowing scene state.

- Raw WebGPU device negotiation:
  - Add a helper parallel to `attachVitrum`, for example
    `negotiateWebGPUDevice(options): Promise<{ adapter, device, format, profile }>`
    or a narrower `createWebGPUBackendDevice`.
  - The helper should preserve the host-lifecycle rule: it may acquire a device
    for convenience, but engines must continue to accept host-owned devices.

## Notes

- Pre-existing mode-only changes remain in several `scripts/` and `tools/`
  files. They were present before this pass and should not be mixed into a
  semantic commit unless intentionally normalized.
- GitNexus impact was run before edits. The index warned it was stale / sibling
  mapped, but still reported the expected high-risk surfaces:
  `sceneFromThreeJS` and `vitrumSceneToThree` CRITICAL,
  `wrapWithIdempotentDispose` and `WebGLPathTracer` HIGH, `ResolvePass` HIGH.
