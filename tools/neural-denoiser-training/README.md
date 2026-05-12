# Vitrum Neural Denoiser — Training Tools

This directory contains the workflow for training and exporting the vitrum U-Net denoiser.

## Prerequisites

```bash
pip install torch torchvision Pillow numpy
```

PyTorch >= 2.0 is required.

## Workflow

### 1. Collect a dataset

Render noisy (1 spp) + clean (4096 spp) pairs from the vitrum example scenes.
See `dataset_spec.md` for the required directory layout and image format.

### 2. Train the model

```bash
python train.py \
  --data path/to/your/dataset/ \
  --epochs 50 \
  --batch 4 \
  --lr 1e-4 \
  --out-pth model.pth \
  --out-bin weights.bin
```

`train.py` saves both a PyTorch checkpoint (`model.pth`) and the vitrum binary
weights (`weights.bin`) at the end of training.

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

**Parameters:** ~426,075 (~1.63 MB at f32)

Skip connections are element-wise addition (not concatenation). The skip
source for decoder level N is the pre-downsampling encoder feature map
at the same spatial resolution as the decoder output.

## No Pre-trained Weights

Vitrum ships no trained weights. You must collect a dataset and train.
The walkaround engine uses `'atrous-variance'` denoising by default;
`'neural'` is opt-in and requires weights to be provided at engine creation.

## References

- Ronneberger, Fischer, Brox 2015. U-Net. MICCAI. https://arxiv.org/abs/1505.04597
- Chaitanya et al. 2017. Interactive Reconstruction of Monte Carlo Image
  Sequences using a Recurrent Denoising Autoencoder. SIGGRAPH.
  https://doi.org/10.1145/3072959.3073601
