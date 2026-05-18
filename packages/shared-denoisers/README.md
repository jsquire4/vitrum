# @vitrum/shared-denoisers

Denoiser building blocks consumed by the walkaround-hybrid pipeline (and, via the OIDN bridge, by pt-webgl).

## What's here

- **À-trous + variance** — `ATROUS_WGSL`, `ATROUS_VARIANCE_WGSL`. The default denoiser baseline for walkaround-hybrid.
- **SVGF (real)** — `SVGF_REPROJECTION_WGSL`, `SVGF_VARIANCE_FROM_MOMENTS_WGSL`, `SVGF_7X7_SPATIAL_FALLBACK_WGSL` (Schied 2017). Selected via `denoiser: { kind: 'svgf-real' }`.
- **Welford temporal variance** — `WELFORD_VARIANCE_WGSL`. Foundational primitive shared by atrous-variance and SVGF.
- **Temporal accumulator** — `TEMPORAL_ACCUM_WGSL`. EMA blend with optional TCBB clip.
- **HDR luminance bilateral** — `HDR_LUMINANCE_BILATERAL_WGSL`. Edge-stop bilateral preview filter.
- **OIDN bridge** — `denoiseFinal()` calls Intel Open Image Denoise via `onnxruntime-web` (optional peer dep). Used by HybridEngine's `'oidn-final'` denoiser mode and pt-webgl's converged-frame path.

## Texture helpers

`webGpuTextureUpload.ts` exposes `uploadTexture2D<T>` and specializations (`uploadRgbAsRgba16f`, `fillRg32f`, etc.) consumed by all three GPU drivers — single source of std140-aware row-pitched upload arithmetic.

## Status

Pre-1.0. The OIDN bridge is the most volatile surface; signature may change as the model-loading story matures.
