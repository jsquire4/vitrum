# tools/behavioral-gate

End-to-end behavioral gate.  Boots the real `pt-webgpu` and `walkaround-hybrid` engines
on lavapipe (CPU Vulkan), renders a minimal Cornell-box scene for each config in the
matrix, reads back the output texture, and asserts the result meets the per-config
expectation.

This is the gate class that caught the F1–F3 total-runtime-breakage bugs (full-tier
bind-group crash, SPPM placeholder min-binding-size, DDGI sampler strip) that the
3,000-assertion test suite and both shader compile gates were completely blind to.

## How to run locally

**Prerequisites:** Deno ≥2.8, Mesa lavapipe (or any Vulkan ICD).

```bash
# Ubuntu — install prerequisites once:
sudo apt-get install -y mesa-vulkan-drivers

# Run from the repo root:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run behavioral-gate

# The npm script is a Node wrapper around the Deno gate. Normal successful runs
# stream the underlying gate output unchanged; a known Deno native-WebGPU host
# panic is classified as HOST-BLOCKED in `behavioral-gate-host-status.json`.

# Self-test mode (injects a synthetic BLACK result and verifies detection):
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run behavioral-gate -- --self-test

# Focus a label subset while developing one lane:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run behavioral-gate -- --filter gltf

# Require selected non-lite pt-webgpu lanes to resolve to the full tier:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run behavioral-gate -- --filter gltf-material-sweep --require-full-tier

# Run the same gate through the companion WSL dzn runtime for full-tier proof:
npm run behavioral-gate:dzn -- --filter gltf --require-full-tier

# The dzn wrapper writes a filter-specific status artifact on PASS and
# fail-closes long host hangs as HOST-BLOCKED, for example
# behavioral-gate-dzn-gltf-status.json for --filter gltf. Override the default
# 180s budget when deliberately collecting a slower full-tier lane:
VITRUM_BEHAVIORAL_GATE_DZN_TIMEOUT_MS=600000 \
  npm run behavioral-gate:dzn -- --filter gltf --require-full-tier

# Verify committed dzn status artifacts:
npm run behavioral-gate:dzn-status-check

# H32 standalone oracle: TLAS shadow rays skip glass but still hit opaque geometry.
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run behavioral-gate:tlas-glass-shadow
```

The committed dzn status set includes sixteen passing focused artifacts
(`gltf-material-sweep`, `mutation`, `default`, `lite-tier`, `spectral`,
`skinned`, `analytic`, `bdpt`, `restirPtReuse`, `caustic`, `photon`, `light`,
`directional`, `hdri`, `procedural-sky`, `wh/`) plus the broader `--filter gltf`
full-suite lane. Together they cover every real behavioral-gate label; the only
uncovered label is the synthetic `__self-test/always-black` failure-injection row.
The broad glTF status confirms full-tier execution with zero GPU errors for all
selected glTF lanes; real-asset comparisons use the explicit `dzn-full` golden
variant under `tools/reference-renders/gltf-real-behavioral-dzn-full/`. The
default, lite-tier, spectral, skinned, and analytic lanes prove baseline
pt/walkaround, explicit pt-webgpu lite fallback, spectral combos,
skinned/glTF-skinned animation, and full-tier analytic sphere boot/render health.
The BDPT lane is a full-tier boot/render proof for `pt/bdpt` and
`pt/spectral+bdpt`; material furnace and multi-vertex promotion remain separate
radiometric proof work. The ReSTIR-PT lane is a full-tier boot/render proof for
off-default `pt/restirPtReuse`; equal-spp variance and specialty radiometric
promotion remain separate proof work. The caustic/photon lanes prove full-tier
boot/render health for `pt/caustic-manifold`, `pt/caustic-photon`, and
`pt/spectral+photon`; caustic radiometric convergence remains a separate A/B
proof. The light, directional, HDRI, and procedural-sky lanes pin dzn execution
for analytic emitters and environment modes across the relevant pt-webgpu
full/lite and walkaround rows; reference-quality radiometric sweeps remain
separate. The `wh/` lane pins all ten walkaround behavioral rows on dzn,
including RC, PPG, checkerboard, skinned, rect-area, directional, HDRI, and
glass-GI modes.

## What it covers

### pt-webgpu configs (34)

| Label | Engine opts | Notes |
|-------|-------------|-------|
| `pt/default` | — | baseline |
| `pt/spectral` | `spectral:true` | hero-λ spectral transport |
| `pt/bdpt` | `bdpt:true` | bidirectional PT |
| `pt/caustic-manifold` | `causticStrategy:'manifold-nee'` | MNEE manifold caustics |
| `pt/caustic-photon` | `causticStrategy:'photon-map'` | SPPM photon map |
| `pt/spectral+photon` | `spectral:true, causticStrategy:'photon-map'` | spectral SPPM combo |
| `pt/lite-tier` | `traceTier:'lite'` | lite binding budget |
| `pt/restirPtReuse` | `restirPtReuse:true` | ReSTIR-PT (wired, off-default) |
| `pt/skinned-mesh` | — | scene with skinned-mesh primitive |
| `pt/analytic-sphere` | — | scene with analytic sphere primitive |
| `pt/point-light` | — | point emitter only |
| `pt/disc-light` | — | analytic disc emitter |
| `pt/spot-light` | — | spot emitter only |
| `pt/directional-2` | — | 2 directional emitters |
| `pt/hdri-env` | — | synthetic flat-white HDRI environment |
| `pt/procedural-sky` | — | Preetham procedural sky |
| `pt/spectral+bdpt` | `spectral:true, bdpt:true` | combo |
| `pt/lite+hdri` | `traceTier:'lite'` | lite + HDRI |
| `pt/lite+point-light` | `traceTier:'lite'` | lite + point light |
| `pt/gltf-unlit` | — | glTF `KHR_materials_unlit` import + render boot |
| `pt/gltf-textured-pbr` | — | glTF baseColorTexture decode hook + render boot |
| `pt/gltf-transmission` | — | glTF `KHR_materials_transmission` import + render boot |
| `pt/gltf-skinned-animation` | — | glTF skin + animation channel import + render boot |
| `pt/gltf-draco-mock` | — | glTF `KHR_draco_mesh_compression` mock decoder + render boot |
| `pt/gltf-point-line-fallback` | — | glTF POINTS/LINES/LINE_LOOP/LINE_STRIP fallback-generated mesh diagnostics + golden PNG |
| `pt/gltf-triangle-strip-fan` | — | glTF TRIANGLE_STRIP/TRIANGLE_FAN triangulation through `loadGltfForEngine()` + golden PNG |
| `pt/gltf-material-sweep` | — | synthetic glTF material-map decode/report sweep + golden PNG |
| `pt/gltf-real-box-textured` | — | real Khronos BoxTextured GLB import/decode/render + golden PNG |
| `pt/gltf-real-draco` | — | real Draco-compressed Khronos asset via host decoder + golden PNG |
| `pt/gltf-real-meshopt` | — | real meshopt-compressed Khronos asset via host decoder + golden PNG |
| `pt/mutation-material` | — | real adapter render → `updatePrimitive()` material patch → render, requires readback delta |
| `pt/mutation-environment` | — | real adapter render → same-sized `updateEnvironment()` HDRI patch → render, requires readback delta |
| `pt/mutation-emitter` | — | real adapter render → `updateEmitter()` point-light patch → render, requires readback delta |
| `pt/mutation-transform` | — | real adapter render → `updatePrimitive()` transform/TLAS patch → render, requires readback delta |

The glTF rows are end-to-end import/engine smoke fixtures: they assert that the
adapter preserves the named feature, boots the selected engine, uploads the scene,
and produces finite non-black output. On lavapipe's WSL adapter, `pt-webgpu`
auto-selects the lite tier because the adapter exposes only the lite WebGPU storage
limits; full material-lobe fidelity remains covered by the renderer fidelity matrix
and package-level material tests.

The pt-webgpu rows print `tier=full|lite`. Use `--require-full-tier` for
full-tier capture work; it fails selected non-lite pt-webgpu rows as `WRONG-TIER`
before any golden update/compare if the backend resolves to lite.

### walkaround-hybrid configs (10)

| Label | Engine opts | Notes |
|-------|-------------|-------|
| `wh/default` | — | baseline |
| `wh/rcEnabled` | `rcEnabled:true` | Radiance Cascades |
| `wh/ppgEnabled` | `ppgEnabled:true` | PPG path guiding |
| `wh/gtao-off` | `gtaoEnabled:false` | GTAO disabled |
| `wh/checkerboard` | `checkerboardEnabled:true` | checkerboard resolve |
| `wh/skinned-mesh` | — | scene with skinned-mesh primitive |
| `wh/hdri-env` | — | synthetic flat-white HDRI environment |
| `wh/rect-area-emitter` | — | rect-area ceiling light |
| `wh/directional-sun` | `primaryLightDir, primaryLightIntensity` | direct sun NEE |
| `wh/glass-gi` | — | refracted GI through glass |

## Assertions per config

1. **Zero GPU errors** — no validation or out-of-memory errors from the device error scopes.
2. **Finite pixels** — no NaN values in the readback.
3. **Non-black output** — mean luminance ≥ 0.005 (after 8 frames at 64×64).
4. **Per-lane invariants** — e.g. glTF golden PNG tolerance checks and
   mutation-lane readback deltas.

All required checks for a lane must pass for a result of `OK`.

## Expectation table

Each config in `gate.mjs` has an entry in `EXPECTATION_TABLE`:

| Value | Meaning |
|-------|---------|
| `expected: 'ok'` | Gate **fails** if result is anything other than OK |
| `expected: 'known-residual'` | Gate **passes** regardless of result; includes `reason` and `planItem` fields |

`known-residual` entries are **temporary scaffolding**, not permanent exceptions.  They
encode open bugs being fixed in parallel so the gate is green on CI now and flips to
enforcing as fixes land.

**Graduating a residual:** when the fix for a `known-residual` config lands, the agent
making that fix updates the entry from `'known-residual'` to `'ok'` and removes the
`reason`/`planItem` fields.  A config that was expected to be a residual but now renders
correctly will start showing `PASS | KNOWN-RESIDUAL` (which is fine — it just means the
fix landed without the table being updated; update the table in the same commit as the fix).

### Current known-residuals

None. Every config in `EXPECTATION_TABLE` is currently expected to return `ok`.

Local WSL/Deno note: Deno 2.8.1's WebGPU GLES path can panic in wgpu-hal on
this adapter while running the existing walkaround-hybrid configs, including
`wh/default`. The npm wrapper writes `behavioral-gate-host-status.json` and
exits non-zero when that known host panic occurs. The focused `--filter gltf`
lane is currently a pt-webgpu lane and passes on WSL lavapipe; walkaround
render-gate promotion should be re-enabled after that Deno/WebGPU harness issue
is cleared.

## Naga gap patches

lavapipe's Vulkan/naga layer rejects a small set of WGSL constructs that Tint/Dawn
accepts.  The gate applies the same patches as the pre-push T1 GPU smoke:

- **pt-webgpu** — strips the 3-arg `textureLoad(bdptLightPath, …, mip)` mip argument;
  adds `isNan`/`isInf` polyfills; downgrades `bdptLightPath` from `texture_storage_2d`
  to `texture_2d<f32>` when `bdpt=false`.
- **walkaround-hybrid** — uses `tools/shader-gate/nagaFix.mjs` (the production fix
  shared with the shader compile gate and the T1 smoke); primarily rewrites
  `ptr<storage>` function parameters.

## CI

The `behavioral-gate` job runs in `.github/workflows/ci.yml` on every push to `main`
and every pull request.  It is **not** `continue-on-error` — a regression in any
`'ok'` config hard-fails the build.

## Relationship to other gates

| Gate | What it checks |
|------|---------------|
| `shader-gate` (WGSL) | Every composed WGSL string compiles without errors (static) |
| `shader-gate` (GLSL) | Every pt-webgl2 feature combination compiles with glslangValidator (static) |
| **`behavioral-gate`** | End-to-end engine boots, renders, produces finite non-black output (dynamic) |
| T1 GPU smoke (`wsl-gpu`) | Full convergence + radiometric oracles on real GPU (wsl-gpu, pre-push) |

The behavioral gate fills the gap between static shader compilation and full convergence
tests: it exercises the engine factory, BVH build, scene upload, UBO packing, and
readback — the class of bug that manifests as a crash or black render rather than a
shader compile error.

## Standalone oracles

Some Road-to-100 closures need a narrower behavioral proof than the full engine
matrix. These scripts live beside the main gate and are opt-in:

| Command | Covers |
|---------|--------|
| `npm run behavioral-gate:tlas-glass-shadow` | H32 TLAS glass-shadow traversal: a real WebGPU dispatch imports the production shared-bvh BVH/TLAS WGSL and proves `traceTlasAny(..., skipGlass=true)` ignores the glass triangle while still finding opaque geometry behind it. |
