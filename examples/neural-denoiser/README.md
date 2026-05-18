# @vitrum-examples/neural-denoiser

W10 demo: walkaround-hybrid with selectable denoiser mode.

## What it shows

- `@vitrum/walkaround-hybrid`'s `HybridEngine` driving a noisy Cornell box
- Three denoiser modes selectable via URL param `?denoiser=…`:
  - `atrous-variance` (default) — Welford temporal accumulator + à-trous
  - `svgf-real` — real Schied 2017 SVGF (T2.H1)
  - `neural` — U-Net neural denoiser (T2.H2 / W10)

## Running

```bash
npm install
npm run dev --workspace @vitrum-examples/neural-denoiser
```

Then open `http://localhost:5176/?denoiser=neural` (or any of the three modes).

## Neural weights

The `?denoiser=neural` mode requires `neuralWeights` to be passed to the
`HybridEngine` constructor. This demo synthesises deterministic-random
He-initialised weights via `buildRandomWeightsForSpec` so the pipeline can be
exercised end-to-end without a trained checkpoint.

The denoised output will NOT be visually clean with random weights — that is
expected. To get real denoising:

1. Train a U-Net matching `WALKAROUND_DENOISER_UNET_SPEC` (see
   `tools/neural-denoiser-training/README.md`).
2. Export to the `.vitrum-model` binary format
   (`tools/neural-denoiser-training/export_weights.py`).
3. Replace the `buildRandomWeightsForSpec(...)` call in `src/main.ts` with
   `loadWeightsFromArrayBuffer(await (await fetch('/vi-neural-weights.bin')).arrayBuffer())`.

## Citations

- Chaitanya et al. 2017, "Interactive Reconstruction of Monte Carlo Image
  Sequences using a Recurrent Denoising Autoencoder" — U-Net auxiliary feature
  recipe.
- Ronneberger et al. 2015, "U-Net: Convolutional Networks for Biomedical Image
  Segmentation" — U-Net architecture.
