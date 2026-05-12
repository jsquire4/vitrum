# Neural Denoiser Training Pipeline

This directory documents how to train the walkaround neural denoiser model
for use with `@vitrum/walkaround-hybrid`'s `InferenceGraph`.

The training pipeline is researcher/host concern. The vitrum library ships
inference infrastructure only — no Python code, no model weights.

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Training runtime |
| PyTorch | ≥ 2.1 | Neural network training |
| CUDA (optional) | ≥ 12.0 | GPU-accelerated training |
| NumPy | ≥ 1.24 | Data pipeline |
| Pillow | ≥ 10.0 | Image I/O |
| ONNX | ≥ 1.14 | Model export (optional) |

Install via conda or venv:

```bash
conda create -n vitrum-denoiser python=3.11 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate vitrum-denoiser
pip install numpy pillow onnx
```

---

## Architecture

The model architecture is defined in TypeScript — the canonical source of truth is:

```
packages/walkaround-hybrid/src/neural/unetArchitecture.ts
```

**Summary:**
- UNet with 3 encoder levels + bottleneck + 3 decoder levels + output projection.
- Channel widths: 9 → 24 → 48 → 96 → 192 (bottleneck) → 96 → 48 → 24 → 3.
- Input: 9 channels (noisy RGB + albedo RGB + world-space normals RGB).
- Output: 3 channels (denoised RGB).
- Parameter count: **426,075** (~1.63 MB f32).

The PyTorch model must match this architecture exactly, or the exported weights
will not load into `InferenceGraph`.

See `train.py.md` for the full architecture spec in PyTorch pseudocode.

---

## Workflow overview

1. **Generate dataset** — render noisy/clean pairs from the walkaround engine and
   offline path tracer. See `dataset_spec.md`.

2. **Train** — run the training loop on the dataset. See `train.py.md`.

3. **Export weights** — convert PyTorch weights to the binary format consumed by
   `ModelWeights`. See `export_weights.md`.

4. **Load in browser** — the host application fetches the exported `.vitrum-weights`
   file and constructs a `ModelWeights` object, then passes it to `InferenceGraph`.

---

## File index

| File | Purpose |
|---|---|
| `README.md` | This file — setup and workflow |
| `dataset_spec.md` | How to generate noisy/clean training pairs |
| `train.py.md` | PyTorch architecture spec + training loop pseudocode |
| `export_weights.md` | How to export trained weights to the vitrum binary format |

---

## Inference-time integration

Once weights are exported, loading them into the library:

```typescript
import { InferenceGraph, ModelWeights } from '@vitrum/walkaround-hybrid';
import { WALKAROUND_DENOISER_UNET_SPEC } from '@vitrum/walkaround-hybrid';

// Load your exported weights file (host-side fetch / ArrayBuffer).
const weights: ModelWeights = await loadWeightsFromFile('/models/denoiser.vitrum-weights');

const graph = new InferenceGraph(WALKAROUND_DENOISER_UNET_SPEC, weights);
await graph.initialize(device);

// Per-frame in renderFrame:
graph.run(device, encoder, inputBuffers, outputBuffers);
```

See `plan/sprint-13-walkaround-integration.md` for the full wiring plan into
`HybridEngine.renderFrame`.

---

## Bail-out criterion

Per `plan/archive/phase-6-roadmap.md` §Sprint 13:

> "Bail-out criterion: if month-1 inference benchmarks don't hit <50 ms, abort and wait for WebNN."

Run inference-time benchmarks on your target GPU after the first month of integration.
If frame time from `InferenceGraph.run()` exceeds 50 ms at your target resolution,
abort neural denoising and rely on SVGF (Sprint 10a) alone until WebNN ships across
browsers without a flag.
