# @vitrum/shared-denoisers

Denoiser building blocks consumed by the walkaround-hybrid, pt-webgpu, and other core-scene render backends.

## What's here

- **À-trous + variance** — `ATROUS_WGSL`, `ATROUS_VARIANCE_WGSL`. The default denoiser baseline for walkaround-hybrid.
- **SVGF (real)** — `SVGF_REPROJECTION_WGSL`, `SVGF_VARIANCE_FROM_MOMENTS_WGSL`, `SVGF_7X7_SPATIAL_FALLBACK_WGSL` (Schied 2017). Selected via `EngineOptions.denoiser: 'svgf-real'`.
- **BMFR** — `BMFR_WGSL` + `runBmfrWebGPU()` + `bmfrRegression.ts` (Koskela et al. 2019, "Blockwise Multi-Order Feature Regression for Real-Time Path-Tracing Reconstruction," ACM TOG 38(5)). Overlapping blockwise least-squares fits of noisy 1-spp color to a `[1, p.xyz, n.xyz, p².xyz]` feature matrix use cooperative direct Householder TSQR on the regularized rectangular system, followed by a deterministic overlap-resolve pass and temporal EMA. `bmfrRegression.ts` is the CPU-unit-testable solver oracle; `wgsl/bmfr.wgsl.ts` is the GPU kernel. Walkaround exposes it only as an explicit full-tier selection and records its two passes under the `bmfr-fit` / `bmfr-resolve` timestamp-query labels. Performance is adapter-, resolution-, and scene-dependent; the implementation does not make an unconditional real-time claim.
- **Welford temporal variance** — `WELFORD_VARIANCE_WGSL`. Foundational primitive shared by atrous-variance and SVGF.
- **Temporal accumulator** — `TEMPORAL_ACCUM_WGSL`. EMA blend with optional TCBB clip.
- **HDR luminance bilateral** — `HDR_LUMINANCE_BILATERAL_WGSL`. Edge-stop bilateral preview filter.
- **OIDN bridge** — `denoiseFinal()` calls Intel Open Image Denoise via `onnxruntime-web` (optional peer dep). `OIDNDispatcherCore` is the shared cohort state machine for converged-backend OIDN dispatchers; walkaround-hybrid uses the same bridge primitives for its `'oidn-final'` denoiser mode.

## Texture helpers

`webGpuTextureUpload.ts` exposes typed upload specializations (`uploadRgbAsRgba16f`, `uploadRgbAsRgba32f`, `uploadR32f`, `uploadR32Uint`, `uploadR16Uint`, `uploadRg32f`, `fillRgba32f`, `fillRg32f`, `fillR16Uint`, `readRgba16fToRgb`) backed by a file-local `uploadTexture2D<T>` primitive — single source of WebGPU-aligned (256-byte) row-pitched upload arithmetic.

## Status

Pre-1.0. The OIDN bridge is the most volatile surface; signature may change as the model-loading story matures.
