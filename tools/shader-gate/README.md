# tools/shader-gate

In-repo WGSL compile gate.  Enumerates every composed WGSL shader across
`@vitrum/pt-webgpu`, `@vitrum/walkaround-hybrid`, and `@vitrum/shared-denoisers`,
feeds each one through `createShaderModule` + `getCompilationInfo`, and fails
on any `'error'`-severity message from the driver's shader compiler (naga/Tint).

## How to run locally

**Prerequisites:** Deno ≥2.8, Mesa lavapipe (or any Vulkan ICD).

```bash
# Ubuntu — install lavapipe once:
sudo apt-get install -y mesa-vulkan-drivers

# Run the gate (from the repo root):
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run shader-gate

# Self-test: injects one intentionally broken shader and asserts the gate detects it:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run shader-gate -- --self-test
```

On the dev rig the gate currently compiles **47 shaders** (47 OK, 0 FAILED).

The underlying npm script is:

```
deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env tools/shader-gate/gate.mjs
```

`--sloppy-imports` lets Deno import `.ts` files from the vitrum workspace
packages without a build step.  `--unstable-webgpu` exposes the WebGPU API.

## What is covered

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

The `composeWgsl` walkaround-hybrid shaders are transformed by `nagaFix.mjs`
before compilation.  The transforms handle the naga/wgpu gap around
`ptr<storage>` function parameters, which are valid in Tint/Dawn but rejected
by naga.  See comments in `nagaFix.mjs` for the full list of transforms.

## What is NOT covered

**GLSL (pt-webgl2).** `@vitrum/pt-webgl2` composes GLSL strings, not WGSL.  A
separate glslang-based gate is tracked as a follow-up in
`plan/road-to-100.md §7.3`.

**Runtime-state-dependent compositions.** A small number of shaders accept
runtime booleans or numeric parameters that gate entire code paths (e.g.
`composePtWebgpuTraceWgsl(bdptEnabled)` — both `false` and `true` variants are
compiled here).  Compositions that would require a live `Scene` or device
feature query at import time are not reachable from this static gate.

**WebGL2 shaders.** GLSL is validated by the driver at draw-call time; there is
no offline GLSL compile path in Deno without bundling glslang as a native
extension.

## CI

The gate runs as the `shader-gate` job in `.github/workflows/ci.yml` on every
push to `main` and every pull request.  It uses `ubuntu-latest` + Mesa
lavapipe (software Vulkan) so no real GPU is required.

The job is independent of the `mechanical-checks` job (no `needs:` dependency)
so both run in parallel and a typecheck/test failure does not suppress a shader
compiler error.
