# Neural Denoiser Dataset Specification

Training data for the walkaround neural denoiser consists of **noisy/clean image
pairs** captured from the vitrum rendering pipeline. Each pair provides a
supervision signal: the network learns to map noisy walkaround output to a
clean reference image.

---

## Dataset size target

Per `plan/archive/phase-6-roadmap.md` §Sprint 13 Definition of Done:

> "Training pipeline generates ~10K noisy/clean pairs."

**Minimum viable dataset**: 8,000 pairs (enough to train without severe overfitting
at this model size).  
**Recommended**: 10,000–20,000 pairs from diverse scenes and lighting conditions.

---

## Pair structure

Each training sample consists of:

| Tensor | Channels | Format | Source |
|---|---|---|---|
| `noisy`   | 3 (RGB) | float32 | Walkaround output at 4–8 SPP |
| `albedo`  | 3 (RGB) | float32 | G-buffer albedo (Sprint 5 MRT, Sprint 10a SVGF) |
| `normals` | 3 (RGB) | float32 | G-buffer world-space normals (Sprint 5 MRT) |
| `clean`   | 3 (RGB) | float32 | PT_FINAL output at ≥192 SPP (ground truth) |

All tensors are at the **same resolution** (e.g. 512×512 for training patches,
or full 1080p if VRAM permits). For memory efficiency, use **random 128×128 crops**
during training.

**Total storage per pair at 512×512**: 4 tensors × 3 channels × 512² × 4 bytes = ~12.6 MB.  
**Total dataset at 10K pairs**: ~126 GB. Use 128×128 crops to reduce to ~1.3 GB.

---

## Rendering procedure

### Noisy frames (walkaround engine)

1. Load the stained-glass scene in walkaround mode.
2. Disable SVGF and PPG denoising (raw noisy output only).
3. Render with the walkaround ReSTIR DI pipeline at **4–8 samples per pixel**.
4. Export from the `HybridEngine` frame output:
   - `rawColor` (RGBA16F → convert to float32 RGB)
   - G-buffer albedo (from Sprint 5 MRT, binding 2)
   - G-buffer normals (from Sprint 5 MRT, binding 1, decode to world-space vec3)

### Clean frames (PT_FINAL)

1. Render the **same camera viewpoint** in PT_FINAL mode at **≥192 SPP**.
2. Export the converged RGB output as float32 (no post-process denoising applied).

### Camera diversity

To prevent overfitting to a single viewpoint:
- Orbit the camera around the scene at ~200 random orientations.
- Vary sun direction, sky tint, and fixture intensities across subsets.
- Include both interior (caustics-dominant) and exterior (HDRI-dominant) lighting conditions.

---

## File format

Store each pair as a single `.npz` (NumPy) file:

```python
np.savez_compressed(
    f'pairs/pair_{i:05d}.npz',
    noisy   = noisy_rgb_float32,   # (H, W, 3)
    albedo  = albedo_float32,       # (H, W, 3)
    normals = normals_float32,      # (H, W, 3)  world-space, range [-1, 1]
    clean   = clean_rgb_float32,    # (H, W, 3)
)
```

The dataset loader in `train.py.md` reads these `.npz` files.

---

## Preprocessing

Apply these transforms before feeding to the network:

1. **Random 128×128 crops** — crop the same region from all four tensors simultaneously.
2. **Horizontal flip** — 50% probability, applied identically to all tensors.
3. **Tone-mapping (Reinhard)** — apply `c / (1 + c)` to both `noisy` and `clean` before
   computing the loss. Train in tone-mapped space to prevent high-brightness pixels
   from dominating the MSE loss. Invert at test time if needed.
4. **Normal encoding** — normals are in world-space `[-1, 1]`; encode as `(n + 1) / 2`
   to map to `[0, 1]` for network input.

---

## Quality filtering

Reject pairs where:
- `clean` mean luminance < 0.01 (fully dark frame — camera inside geometry).
- `clean` max luminance > 100 (blown-out; likely a firefly in the PT reference).
- `noisy` NaN or Inf values present.

A 5% rejection rate is normal for a real scene.

---

## Hardware notes

Capturing 10K pairs at 192 SPP PT_FINAL:
- At 10 seconds/frame on a desktop GPU: ~28 CPU-hours.
- Parallelise with headless Chrome instances if available.
- Store raw exports immediately — re-rendering is expensive.
