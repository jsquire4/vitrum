# Dataset Specification — Vitrum Neural Denoiser

## Overview

The neural denoiser requires paired noisy/clean renders of the same scene.
Each pair consists of:
- A **noisy** 1 spp path-traced render (WebGPU walkaround output)
- Auxiliary G-buffers: **albedo** and **world-space normals**
- A **clean** 4096 spp reference render (same camera, scene, lighting)

Auxiliary buffers come from the walkaround pipeline's G-buffer outputs
(the same buffers used by the SVGF/atrous-variance denoiser).

## Directory Layout

```
data/
  cornell_box/
    noisy/
      frame_0001.png           # 1 spp noisy color (RGB 8-bit PNG, tonemapped to LDR)
      frame_0001_albedo.png    # albedo G-buffer (RGB 8-bit PNG, [0,1] diffuse reflectance)
      frame_0001_normal.png    # world-space normals (RGB 8-bit PNG, encoded [0,1])
    clean/
      frame_0001.png           # 4096 spp reference (RGB 8-bit PNG, same tonemapping)
  multi_material/
    noisy/
      ...
    clean/
      ...
```

## Image Format

All images are **8-bit RGB PNG**. HDR values are tonemapped before saving using
the Reinhard operator `L / (1 + L)` applied per channel, where `L` is the raw
path-traced linear energy.

Normals are encoded as `(n * 0.5 + 0.5)` to fit world-space normals in [0, 1].
Decode with `n * 2 - 1` to recover world-space [-1, 1] normals.

## Capturing Training Data

1. Render the same scene from diverse camera positions and lighting conditions.
2. For each camera/lighting configuration:
   a. Render at 1 spp (walkaround-hybrid HDR color tonemapped) → save as `noisy/frame_NNNN.png`
   b. Save the albedo G-buffer → `noisy/frame_NNNN_albedo.png`
   c. Save the world normals G-buffer → `noisy/frame_NNNN_normal.png`
   d. Render at 4096 spp (pt-webgl2 or pt-webgpu reference) → save as `clean/frame_NNNN.png`
3. Aim for 1000–5000 unique camera positions per scene.

Reference scenes for collecting training data:
- `examples/pt-webgl2-direct/` or `examples/pt-webgpu-direct/` (Vite projects —
  `npm run dev --workspace @vitrum-examples/pt-webgl2-direct` from the repo root).
  Both noisy (walkaround-hybrid 1 spp) and clean (pt-webgl2/pt-webgpu 4096 spp)
  captures come from these scenes. A batched-capture runner is a future addition
  (see "Capture-runner gap" below).

### Capture-runner gap

This repo does NOT currently ship a batched G-buffer capture script. To
build a real dataset a host has to:

1. Drive the walkaround engine through 1000+ camera positions and save
   `noisy/`, `noisy/*_albedo`, `noisy/*_normal` PNGs each frame
   (the walkaround-hybrid pipeline writes `hdrColorTexture`,
   `hdrAlbedoTexture`, and `gNormalDepthTexture` — read back via
   `device.queue.submit` + `copyTextureToBuffer` + `mapAsync(READ)`).
2. Re-render the same camera positions through `pt-webgl2` (or `pt-webgpu`) at
   4096 spp for the clean reference, save as `clean/`.
3. Apply the same tonemapping operator to both noisy and clean before
   saving (Reinhard `L / (1 + L)` is the default; whatever you pick must
   match what `train.py` expects since it loads PNGs as already-tonemapped
   LDR).

The vitrum benchmark-runner (`tools/benchmark-runner/`) already wraps a
Playwright headless capture loop and could be extended with a dataset-
collection mode. Filed as a future enhancement; not blocking training
for any host willing to roll their own capture.

## Minimum Dataset Size

- 500 pairs minimum (training may overfit below this)
- 5000 pairs recommended for production quality

## Notes on Alignment

The noisy and clean images MUST be rendered from the exact same camera position,
scene state, and lighting. Any misalignment (different camera, moved geometry,
different light) will produce an incorrectly supervised example and degrade quality.
