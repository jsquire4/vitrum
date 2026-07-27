# tools/shader-gate

In-repo shader compile gate.  Two scripts:

- **`gate.mjs`** — WGSL gate (WebGPU + naga/Tint).  Enumerates every composed WGSL shader
  across `@vitrum/pt-webgpu`, `@vitrum/walkaround-hybrid`, and `@vitrum/shared-denoisers`,
  feeds each one through `createShaderModule` + `getCompilationInfo`, and fails on any
  `'error'`-severity message from the driver's shader compiler.  It then creates
  adapter-backed compute pipelines for named entry points and walks
  `@vitrum/walkaround-hybrid`'s production `compilePipelines()` variants so binding
  layouts, stage visibility, and optional pass layouts are validated too.
- **`glslGate.mjs`** — GLSL ES 3.00 gate (`glslangValidator`).  Composes the full
  production GLSL program (preamble + compat defines + body) for every production-reachable
  feature-flag combination of `@vitrum/pt-webgl2` and validates each one with
  `glslangValidator`.

This directory is also an npm workspace package.  Root `npm test` runs the CPU-only
GLSL gate plus its injected-error self-test through `@vitrum/shader-gate`; the WGSL
gate remains an explicit `npm run shader-gate` / CI step because it needs a WebGPU
adapter.

## How to run locally

**Prerequisites:** Deno ≥2.8, Mesa lavapipe (or any Vulkan ICD), `glslang-tools`.

```bash
# Ubuntu — install prerequisites once:
sudo apt-get install -y mesa-vulkan-drivers glslang-tools

# Run the WGSL gate (from the repo root):
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run shader-gate

# WGSL self-test: injects one intentionally broken shader and asserts the gate detects it:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run shader-gate -- --self-test

# Debug only: compile shader modules but skip the pipeline-creation phase:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run shader-gate -- --no-pipeline-gate

# Run the GLSL gate (no Vulkan required — glslangValidator is a CPU-only binary):
npm run shader-gate:glsl

# GLSL self-test: injects a broken chunk and asserts detection:
npm run shader-gate:glsl -- --self-test
```

The WGSL and GLSL gates print live shader-module, pipeline-variant, and feature-
combination totals at the end of each run; keep the command output as the count
source of truth rather than hard-coding those numbers in planning docs.

The underlying npm scripts are:

```
# WGSL gate
deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env tools/shader-gate/gate.mjs

# GLSL gate
deno run --sloppy-imports --allow-read --allow-env --allow-run --allow-write=/tmp tools/shader-gate/glslGate.mjs
```

`--sloppy-imports` lets Deno import `.ts` files from the vitrum workspace
packages without a build step.

## What is covered

### WGSL (gate.mjs)

| Subsystem | Shaders compiled |
|-----------|-----------------|
| `pt-webgpu` — full-tier path trace | `composePtWebgpuTraceWgsl(false)` (SSS), `composePtWebgpuTraceWgsl(true)` (BDPT) |
| `pt-webgpu` — composite megakernel | `composePtWebgpuCompositeTraceWgsl(false/true)` |
| `pt-webgpu` — standalone consts | `PT_WEBGPU_TRACE_WGSL`, `PT_WEBGPU_TRACE_LITE_WGSL`, `PT_WEBGPU_SEED_BLIT_WGSL` |
| `pt-webgpu` — ReSTIR-PT | producer, temporal, spatial, resolve |
| `walkaround-hybrid` — full pass graph | ~28 passes via `composeWgsl(rootModule, WGSL_MODULES)` |
| `walkaround-hybrid` — PPG update | `buildPpgUpdateWgsl(341)` with layout + luminance preamble |
| `walkaround-hybrid` — NRC | `buildRisGiNrcModule(cfg)` (skipped on adapters with maxBindGroups < 5) |
| `shared-denoisers` — standalone | temporal accumulation, à-trous variance, SVGF reprojection + variance + 7x7 fallback, BMFR, HDR luminance bilateral, Welford variance |

The `composeWgsl` walkaround-hybrid shaders and the walkaround-rc exports are
compiled exactly as emitted. The gate does not rewrite WGSL for naga. A fatal
30-shader portability inventory prevents either source drift or a hidden
validator-specific derivative.

After shader-module compilation, `gate.mjs` creates compute pipelines for every
named compute entry in the pt-webgpu, shared-denoiser, and walkaround-rc inventory.
It also calls walkaround-hybrid's production `compilePipelines()` with the same
explicit bind-group layouts used by the engine for these variants:

| Production pipeline-layout variant | What is validated |
|------------------------------------|-------------------|
| `walkaround-hybrid/production-default` | always-on RIS/temporal/spatial/shade/GTAO/GI/indirect/composite layout set |
| `walkaround-hybrid/production-gris` | GRIS/ReSTIR-PT reuse temporal+spatial GI two-group layouts |
| `walkaround-hybrid/production-ppg` | PPG update pipeline and guided-sampling tree layout |
| `walkaround-hybrid/production-regir` | ReGIR grid-build read/write layout |
| `walkaround-hybrid/production-nrc` | NRC fifth bind group when the adapter reports `maxBindGroups >= 5` |

### GLSL (glslGate.mjs)

| Subsystem | Combinations compiled |
|-----------|----------------------|
| `pt-webgl2` — full fragment program | Production-reachable feature combinations (see table below) |

**Validator:** `glslangValidator` (KhronosGroup/glslang) without `--target-env`.
This mode validates GLSL in-place (parse + type-check) without generating SPIR-V,
which correctly accepts `#version 300 es` and enforces GLSL ES 3.00 semantics.
(With `--target-env opengl` or `vulkan` it refuses `#version 300 es` and requires 310+.)

**What is validated:** the exact GLSL program production compiles at runtime —
`buildFragmentSource(defines, composeTraceGlsl(features))` and
`buildVertexSource(defines, FULLSCREEN_VERT)` — with the production preamble
(`#version 300 es`, precision qualifiers, `layout(location=0) out vec4 pc_fragColor`,
GLSL3 compat `#define` bridges) prepended exactly as `GlProgram.#relink()` does.

**Feature-combination matrix:**

| Combination | bdpt | dof | cameraType | stainedGlass | Notes |
|-------------|------|-----|------------|--------------|-------|
| `baseline` | false | false | 0 (persp) | false | Default production path |
| `bdpt-on` | **true** | false | 0 | false | BDPT chunks included in composer |
| `dof-on` | false | **true** | 0 | false | `FEATURE_DOF=1` #if paths |
| `cameraType-ortho` | false | false | **1** | false | `CAMERA_TYPE=1` #if paths |
| `cameraType-equirect` | false | false | **2** | false | `CAMERA_TYPE=2` #if paths |
| `stained-glass` | false | false | 0 | **true** | `FEATURE_STAINED_GLASS_*=1` path |
| `sobol-on` | false | false | 0 | false | `RANDOM_TYPE=1` Sobol RNG path |

The only compose-time branch is `bdpt` (includes/excludes `bdpt_light_subpath` +
`bdpt_connection` chunks).  All other flags only affect `#define` values that the
GLSL preprocessor resolves.  The combinations above exercise every distinct
compile-time code path in the composed program.

**Limits of this gate** (known gaps, not regressions):
- Validates GLSL ES semantics but not WebGL2-specific driver extensions or uniform block
  interface rules that ANGLE/Mesa would enforce at link time.  The wsl-gpu T1 GPU smoke
  provides that gate (real WebGL2 context on lavapipe).
- `FEATURE_FOG`, `FEATURE_BACKGROUND_MAP`, and `RANDOM_TYPE=2` are pinned-off in
  production (`featureTypes.ts`) and are therefore not exercised here.

## What is NOT covered

**Runtime-state-dependent compositions.** A small number of shaders accept
runtime booleans or numeric parameters that gate entire code paths (e.g.
`composePtWebgpuTraceWgsl(bdptEnabled)` — both `false` and `true` variants are
compiled here).  Compositions that would require a live `Scene` or device
feature query at import time are not reachable from this static gate.

## CI

Both gates run as steps in the `shader-gate` job in `.github/workflows/ci.yml`
on every push to `main` and every pull request.  The job uses `ubuntu-latest` +
Mesa lavapipe (software Vulkan, for the WGSL gate) + `glslang-tools` (for the
GLSL gate) so no real GPU is required.

The `shader-gate` job is independent of the `mechanical-checks` job (no `needs:`
dependency) so both run in parallel and a typecheck/test failure does not suppress
a shader compiler error.

## Relationship to the behavioral gate

The shader gate is **static**: it compiles WGSL/GLSL source strings and creates WebGPU
pipeline objects, but never boots an engine or renders a frame.  It cannot catch engine
factory crashes, UBO upload gaps, or total-black renders caused by unbound resources.

The **`tools/behavioral-gate/`** fills that gap: it boots the real `createPTEngine_WebGPU`
and `createWalkaroundEngine_Hybrid` factory functions, renders a Cornell-box scene for
every config in the matrix, and asserts zero GPU errors and finite non-black output.  A
separate CI job (`behavioral-gate` in `ci.yml`) runs it on every push.  See
[`tools/behavioral-gate/README.md`](../behavioral-gate/README.md) for the full config
matrix and expectation-table contract.
