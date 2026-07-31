# Dataset Specification — Vitrum Neural Denoiser

## Overview

The neural denoiser requires paired noisy/clean renders of the same scene.
Each pair consists of:

- A **noisy** single stochastic renderer estimate, stored as linear HDR RGB.
- Auxiliary **albedo** and **world-space normal** buffers from that exact frame.
- A **clean** reference accumulated from at least 4096 accepted renderer
  estimates at the same camera, scene, and lighting state.

The shipped renderer capture reads the exact three textures consumed by the
runtime neural denoiser: pre-denoise shade radiance, albedo, and encoded
world-normal/depth. It does not reconstruct them in a separate adapter.

## Directory Layout

```
data/
  cornell_box/
    noisy/
      frame_0001.bin           # one-sample noisy linear-HDR RGB (VHDR)
      frame_0001_albedo.png    # albedo G-buffer (RGB 8-bit PNG, [0,1] diffuse reflectance)
      frame_0001_normal.png    # world-space normals (RGB 8-bit PNG, encoded [0,1])
    clean/
      frame_0001.bin           # accumulated clean linear-HDR RGB (VHDR)
  multi_material/
    noisy/
      ...
    clean/
      ...
```

## Image Format

Noisy and clean color targets use the repository's little-endian **VHDR v1**
format: four `u32` header words (`'VHDR'`, version, width, height), followed by
tightly packed row-major float32 RGB. Values remain in linear HDR and are not
tonemapped or clamped.

Albedo and normals are **8-bit RGB PNG** because both are bounded auxiliaries.

Normals are encoded as `(n * 0.5 + 0.5)` to fit world-space normals in [0, 1].
Decode with `n * 2 - 1` to recover world-space [-1, 1] normals.

## Dataset Manifest

Production quality evidence must include a dataset manifest JSON referenced by
`quality-ab-production.json` as `artifacts.datasetManifestPath`. The learned
systems proof checker validates this manifest when a production checkpoint is
registered.

```json
{
  "schema": "vitrum.neural-denoiser.dataset.v1",
  "id": "walkaround-renderer-seed-1984",
  "sceneCount": 1,
  "sampleCount": 500,
  "noisySpp": 1,
  "cleanReferenceSpp": 4096,
  "includesAlbedo": true,
  "includesNormals": true,
  "captureSource": "vitrum-walkaround-hybrid-neural-input-readback",
  "tonemap": "linear-hdr",
  "estimatorSampleUnit": "walkaround-renderer-frame",
  "warmupFrames": 8,
  "scenes": [
    {
      "id": "cornell_box",
      "sampleCount": 500,
      "noisyPath": "data_renderer/cornell_box/noisy/",
      "cleanPath": "data_renderer/cornell_box/clean/",
      "albedoPath": "data_renderer/cornell_box/noisy/*_albedo.png",
      "normalPath": "data_renderer/cornell_box/noisy/*_normal.png"
    }
  ]
}
```

The top-level fields must match the `dataset` block in the production quality
manifest exactly. `scenes[].sampleCount` values must sum to `sampleCount`, and
each scene must name noisy, clean, albedo, and normal artifact locations.

## Capturing Training Data

On a machine where Deno exposes a WebGPU adapter, run from the repository root:

```bash
npm run capture:neural-renderer-dataset -- \
  --out data_renderer \
  --pairs 500 \
  --size 128 \
  --clean-frames 4096 \
  --warmup-frames 8 \
  --seed 1984
```

`capture-renderer-dataset.mjs` instantiates
`createWalkaroundEngine_Hybrid` directly. For every deterministic Cornell-box
camera/light variant it:

1. Renders the configured warmup frames without accepting training samples.
2. Saves the first accepted pre-denoise radiance estimate as noisy VHDR.
3. Saves the albedo and decoded world normal captured in the same GPU
   submission as that noisy estimate.
4. Accumulates all accepted radiance estimates in float64 on the CPU, divides
   by the exact accepted count, and saves the result as clean VHDR.
5. Builds every artifact and both manifests in a same-filesystem sibling staging
   directory, then publishes the complete dataset with one directory rename.

The `--out` path must not already exist. This strict rule is intentional:
replacing a populated directory is not a single atomic operation on both POSIX
and Windows. Refusing an in-place rerun prevents an old manifest from describing
a mixture of old and newly overwritten pair artifacts after a failure. Choose a
new output path (or explicitly remove the old generation) for each capture.

The walkaround renderer is a temporally reused real-time estimator, so successive
frames can be correlated. The manifest records
`estimatorSampleUnit: "walkaround-renderer-frame"` rather than pretending the
accepted frames are independent path-tracer samples. The schema's historical
`noisySpp` and `cleanReferenceSpp` fields carry the corresponding one-frame and
accepted-frame counts.

`capture-dataset.mjs` remains useful as a CPU-only format and training-plumbing
smoke. It is not the production renderer dataset generator.

## Minimum Dataset Size

- 500 pairs minimum (training may overfit below this)
- 5000 pairs recommended for production quality

## Notes on Alignment

The noisy and clean images MUST be rendered from the exact same camera position,
scene state, and lighting. Any misalignment (different camera, moved geometry,
different light) will produce an incorrectly supervised example and degrade quality.
