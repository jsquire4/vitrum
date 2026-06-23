# Vitrum Neural Denoiser — Training Tools

This directory contains the workflow for training and exporting the vitrum U-Net denoiser.

## Quick verified end-to-end smoke (no GPU, no PyTorch required)

This exercises the whole **capture → train(export) → load** pipeline on a tiny
synthetic dataset. It does NOT produce useful denoiser weights — it proves the
plumbing and the on-disk weight format are correct (road-to-100 A10).

```bash
# 1. Capture a tiny deterministic dataset (4 noisy/clean pairs at 128²).
#    Uses a self-contained CPU path tracer — no GPU, no browser.
node capture-dataset.mjs --out data_smoke --pairs 4 --size 128 \
  --noisy-spp 1 --clean-spp 256 --seed 1984

# 2. Validate dataset loading + tensor shapes + the export format, and write a
#    canonical 535,107-param .vitrum-model binary. NUMPY ONLY — no PyTorch.
python3 train.py --dry-run --data data_smoke --out-bin weights_dryrun.bin

# 3. Prove the exported binary loads through the runtime loader + allocates an
#    InferenceGraph (vitest, stub device, from the repo root):
npx vitest run packages/walkaround-hybrid/__tests__/neuralWeightsRoundTrip.test.ts
```

The dry-run prints `Param count: 535,107 == canonical 535,107  ✓` and the
round-trip test loads the checkpoint through the real `loadWeightsFromArrayBuffer`
and feeds it into a real `InferenceGraph.initialize`.

> The weight format is the **`.vitrum-model` binary** (magic `0xDEAF1984`,
> version 1) read by `loadWeightsFromArrayBuffer`. Earlier notes that referred to
> a `vi-neural-weights.json` are stale — the runtime loader is binary, not JSON.

## Prerequisites (for *real* training)

```bash
pip install torch torchvision Pillow numpy
```

PyTorch >= 2.0 is required to actually train. `--dry-run` needs only `numpy` +
`Pillow` and is the path used on CPU-only / no-torch boxes.

## Workflow

### 1. Collect a dataset

Render noisy (1 spp) + clean (4096 spp) pairs. See `dataset_spec.md` for the
required directory layout and image format.

- **Smoke / format-exercise dataset (CPU, this box):** `node capture-dataset.mjs`
  (above). Self-contained CPU path tracer; deterministic; tiny.
- **Real dataset (GPU session, see "GPU capture" below):** drive the pt-webgpu /
  walkaround engines headlessly through ~/projects/wsl-gpu's deno render worker.

### 2. Train the model

```bash
# Tiny end-to-end smoke (requires torch): ≤2 epochs, ≤64 patch.
python3 train.py --smoke --data data_smoke --out-bin weights.bin

# Full training run:
python3 train.py \
  --data path/to/your/dataset/ \
  --epochs 50 \
  --batch 4 \
  --lr 1e-4 \
  --out-pth model.pth \
  --out-bin weights.bin
```

`train.py` saves both a PyTorch checkpoint (`model.pth`) and the vitrum binary
weights (`weights.bin`) at the end of training. Both `--smoke` and a full run
assert the model's param count equals the canonical **535,107** before training,
so an architecture drift from `unetArchitecture.ts` fails fast.

### 3. Export weights (optional standalone step)

If you already have a trained `.pth` checkpoint and want to re-export:

```bash
python export_weights.py --pth model.pth --out weights.bin
```

### 4. Load at runtime

```typescript
import { loadWeightsFromArrayBuffer } from '@vitrum/walkaround-hybrid';

const response = await fetch('/path/to/weights.bin');
const weights = loadWeightsFromArrayBuffer(await response.arrayBuffer());

const engine = await createWalkaroundEngine_Hybrid({
  device,
  denoiser: 'neural',
  neuralWeights: weights,
  // ...other options
});
```

## Architecture

The U-Net follows Ronneberger et al. 2015 with the Chaitanya et al. 2017
9-channel input layout (noisy RGB + albedo + world normals).

| Stage         | Shape               |
|---------------|---------------------|
| Input         | H × W × 9           |
| Encoder L1    | H/2 × W/2 × 24      |
| Encoder L2    | H/4 × W/4 × 48      |
| Encoder L3    | H/8 × W/8 × 96      |
| Bottleneck    | H/8 × W/8 × 192     |
| Decoder L3    | H/4 × W/4 × 96      |
| Decoder L2    | H/2 × W/2 × 48      |
| Decoder L1    | H × W × 24          |
| Output        | H × W × 3 (RGB)     |

**Parameters:** ~535,107 (~2.04 MB at f32, ~1.02 MB at f16)

Skip connections are element-wise addition (not concatenation). The skip
source for decoder level N is the pre-downsampling encoder feature map
at the same spatial resolution as the decoder output.

## Weight Posture

The published walkaround package ships no production neural weights. The repo
tracks limited research checkpoints under `checkpoints/` so the binary format
and runtime loader can be exercised, but they are small CPU-trained artifacts
and are not suitable as a production default. The walkaround engine uses
`'atrous-variance'` denoising by default; `'neural'` is opt-in and requires
validated weights to be provided at engine creation.

`checkpoints/manifest.json` is the authoritative checkpoint classification
ledger. It records every committed `.vitrum-model` file's role, byte size,
SHA-256, parameter count, and production-default eligibility. The root
`npm run learned-systems-proof-check` command fails if a checkpoint is
unregistered, if its bytes drift from the manifest, or if any production/default
checkpoint appears without a passing `quality-ab-production.json` A/B manifest.
That production quality manifest must also cite reproducibility artifacts:
the dataset manifest, aggregate result summary, candidate outputs, and reference
outputs used for the A/B. Metric-only manifests are rejected because they cannot
support a production checkpoint claim.

## GPU capture (real dataset — for the GPU session)

`capture-dataset.mjs` ships a **CPU** path-traced smoke set so the format and
training plumbing can be validated anywhere. For a *real* training dataset you
want the actual production renderers (pt-webgpu noisy/clean + the walkaround
G-buffers). On a box with a working WebGPU adapter, drive the deno render worker:

- `~/projects/wsl-gpu/capture-worker/render-pt-webgpu.ts` — renders a
  `ScenarioDescriptor` through `@vitrum/pt-webgpu` to a PNG (reaches lavapipe).
  Run it at low spp for `noisy/` and high spp (≥4096) for `clean/`.
- `~/projects/wsl-gpu/capture-worker/render-hybrid.ts` — the walkaround pipeline,
  for the `albedo` / world-`normal` aux G-buffers (read back `hdrAlbedoTexture` /
  `gNormalDepthTexture`).

Apply the same Reinhard `L/(1+L)` tonemap to noisy + clean before saving (this is
what `capture-dataset.mjs` does and what `train.py` assumes — see `dataset_spec.md`).

**This step was NOT runnable on the box where A10 was implemented** (no clean
headless WebGPU adapter from here), so it is documented rather than executed. The
CPU `capture-dataset.mjs` covers the FORMAT path; the GPU worker covers radiometric
fidelity and is the remaining real-hardware tail.

## What's shipped vs what you build

| Piece                                    | Status                        |
|------------------------------------------|-------------------------------|
| U-Net architecture (PyTorch + WGSL)      | Shipped (`train.py`, `@vitrum/walkaround-hybrid/src/neural/`) |
| Weight format + binary serialiser        | Shipped (`export_weights.py` + `loadWeightsFromArrayBuffer`)  |
| InferenceGraph (GPU runtime)             | Shipped (`@vitrum/walkaround-hybrid` W10)                     |
| Training script (+ `--smoke`/`--dry-run`)| Shipped (`train.py`)                                          |
| Dataset format spec                      | Shipped (`dataset_spec.md`)                                   |
| **CPU smoke capture runner**             | Shipped (`capture-dataset.mjs` — format-exercise, not a real dataset) |
| **Round-trip test (export → load)**      | Shipped (`walkaround-hybrid/__tests__/neuralWeightsRoundTrip.test.ts`) |
| **Pre-trained weights**                  | **Not shipped — host trains their own**                       |
| **Checkpoint manifest**                  | Shipped (`checkpoints/manifest.json` — research/production classification + hashes) |
| **GPU batched capture runner**           | **Documented, not wired here** — needs a WebGPU adapter (see "GPU capture") |
| **Reference Cornell test dataset**       | **Not shipped — too large + scene-licensing concerns**        |

### Honest gaps (what still needs real hardware / torch)

1. **Real radiometric dataset** — the shipped capture is a CPU synthetic Cornell
   at tiny resolution. A production dataset needs the GPU worker above (WebGPU
   adapter required) and 500–5000 pairs per `dataset_spec.md`.
2. **Actual training** — `--smoke` and full training need PyTorch installed.
   `--dry-run` (numpy only) validates everything *except* the gradient loop.
3. **Trained weights** — still none ship; the produced `.vitrum-model` from
   `--dry-run` contains He-init random weights (correct shapes, useless output).

The reference dataset is intentionally absent because even a small Cornell-spp
dataset is ~50–500 MB of PNG and is best collected fresh by each consumer against
the rendering pipeline they actually intend to ship.

## References

- Ronneberger, Fischer, Brox 2015. U-Net. MICCAI. https://arxiv.org/abs/1505.04597
- Chaitanya et al. 2017. Interactive Reconstruction of Monte Carlo Image
  Sequences using a Recurrent Denoising Autoencoder. SIGGRAPH.
  https://doi.org/10.1145/3072959.3073601
