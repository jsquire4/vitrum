> ARCHIVED 2026-05-28 — SHIPPED. Neural U-Net in walkaround-hybrid/src/neural/InferenceGraph.ts; denoiser: 'neural' + neuralWeights. Training tools at tools/neural-denoiser-training/.

# Neural Denoiser — Future Sprint Placeholder

**Status:** Future / unscheduled
**Created:** 2026-05-11
**Context:** This functionality was deleted during sweep-2026-05-11 because the existing
implementation was not runnable. This doc preserves the design intent for when a real
implementation sprint is scheduled.

---

## What was deleted and why

The following were removed (D4, decisions doc):

- `packages/walkaround-hybrid/src/neural/` (entire directory: `InferenceGraph.ts`,
  `unetArchitecture.ts`, and WGSL kernels: `conv2d.wgsl.ts`, `relu.wgsl.ts`,
  `bilinearUpsample.wgsl.ts`, `skipConnection.wgsl.ts`, `transposedConv2d.wgsl.ts`)
- `packages/walkaround-hybrid/__tests__/sprint13-neural.test.ts`
- `tools/neural-denoiser-training/` (entire directory)

Re-exports of `InferenceGraph`, `unetArchitecture`, `buildUNetSpec` were removed from
`packages/walkaround-hybrid/src/index.ts`.

The architecture files were educationally valuable but misleading as code: 8 structural
bugs prevented the scaffold from running, there was no `'neural'` mode in `HybridEngine`,
and `train.py.md` was a Markdown file pretending to be a Python script. Half-implementations
were explicitly called out as unwelcome by the user.

---

## Paper references

**Primary architecture precedent:**
Chaitanya, C.R.A., Kaplanyan, A., Schied, C., Salvi, M., Lefohn, A., Nowrouzezahrai, D.,
Aila, T. "Interactive Reconstruction of Monte Carlo Image Sequences using a Recurrent
Denoising Autoencoder." *SIGGRAPH*, 2017.
https://doi.org/10.1145/3072959.3073601

**UNet architecture:**
Ronneberger, O., Fischer, P., Brox, T. "U-Net: Convolutional Networks for Biomedical
Image Segmentation." *MICCAI*, 2015.
https://arxiv.org/abs/1505.04597

**OIDN (alternative to compare against — see below):**
Chaitanya et al. 2017 (above); Intel Open Image Denoise v2 — uses a U-Net variant with
temporal accumulation. Apache-licensed.
https://www.openimagedenoise.org/

---

## U-Net architecture (from the deleted `unetArchitecture.ts`)

The architecture was verified to be within the 1–3 MB weight budget and feasible for
WebGPU inference. It is worth preserving exactly.

**Inputs:** 9 channels — noisy RGB (3) + denoised albedo (3) + world-space normals (3).
Albedo and normals come from the SVGF/atrous-variance G-buffer pipeline.

**Encoder** (3 levels, stride-2 conv, 2× downsampling per level):
```
Level 1: conv2d(9→24,  3×3, stride 2) + ReLU  →  H/2  × W/2  × 24
Level 2: conv2d(24→48, 3×3, stride 2) + ReLU  →  H/4  × W/4  × 48
Level 3: conv2d(48→96, 3×3, stride 2) + ReLU  →  H/8  × W/8  × 96
```

**Bottleneck** (stride 1, no spatial change):
```
conv2d(96→192, 3×3, stride 1) + ReLU           →  H/8  × W/8  × 192
```

**Decoder** (3 levels, transposed conv 2×2, stride 2, skip-add from encoder):
```
Level 3: transposedConv2d(192→96, 2×2, stride 2)
         + skip-add(enc3: H/8×W/8×96)  →  H/4 × W/4 × 96
         + conv2d(96→96, 3×3) + ReLU
Level 2: transposedConv2d(96→48, 2×2, stride 2)
         + skip-add(enc2: H/4×W/4×48)  →  H/2 × W/2 × 48
         + conv2d(48→48, 3×3) + ReLU
Level 1: transposedConv2d(48→24, 2×2, stride 2)
         + skip-add(enc1: H/2×W/2×24)  →  H   × W   × 24
         + conv2d(24→24, 3×3) + ReLU
```

**Output projection:**
```
conv2d(24→3, 1×1, stride 1)  →  denoised RGB
```

**Parameter count (verified in deleted source):**
```
enc1:        9 × 24 × 9 + 24   =   1,968
enc2:       24 × 48 × 9 + 48   =  10,416
enc3:       48 × 96 × 9 + 96   =  41,568
bottleneck: 96 × 192 × 9 + 192 = 166,080
dec3_tconv: 192 × 96 × 4 + 96  =  73,824   (2×2 kernel)
dec3_conv:   96 × 96 × 9 + 96  =  83,040
dec2_tconv:  96 × 48 × 4 + 48  =  18,480
dec2_conv:   48 × 48 × 9 + 48  =  20,784
dec1_tconv:  48 × 24 × 4 + 24  =   4,632
dec1_conv:   24 × 24 × 9 + 24  =   5,208
proj:        24 ×  3 × 1 +  3  =      75
─────────────────────────────────────────
TOTAL:                            426,075 parameters
BYTES (f32):                    1,704,300 ≈ 1.63 MB
```

This fits within the 1–3 MB DoD target.

**Intermediate GPU memory at 1080p (f32):**
```
enc1 output:  540 ×  960 × 24  × 4 ≈  50 MB
enc2 output:  270 ×  480 × 48  × 4 ≈  25 MB
enc3 output:  135 ×  240 × 96  × 4 ≈  12 MB
bottleneck:   135 ×  240 × 192 × 4 ≈  25 MB
dec3 output:  270 ×  480 × 96  × 4 ≈  50 MB
dec2 output:  540 ×  960 × 48  × 4 ≈ 100 MB
dec1 output: 1080 × 1920 × 24  × 4 ≈ 199 MB
─────────────────────────────────────────────
Total intermediate: ~461 MB at 1080p (f32)
```

Production mitigation: fp16 halves this to ~230 MB, or channel pruning reduces width.
For WebGPU targets with 8 GB VRAM, fp16 intermediates are required.

---

## The 8 scaffold bugs to avoid on re-implementation

These were the structural correctness blockers in the deleted code. A re-implementation
must avoid all 8 on day one, not fix them post-hoc.

**Bug 1 — Skip-connection spatial mismatch.**
Decoder Level 3 (`dec3_up`) is at `H/4 × W/4 × 96`. The skip connection adds `enc3`
which is at `H/8 × W/8 × 96` — 4× mismatch in both spatial dimensions. The transposed
conv that produces `dec3_up` from the H/8 bottleneck must output H/4, not H/8. Fix: the
transposed conv `inputH=135, inputW=240` with stride 2 produces `270 × 480`, which is
H/4 — correct. The spec arithmetic was right but the comment describing it was wrong
(calling `enc3` output "H/8" and `dec3_up` "H/4" correctly, but then pairing them in
the skip layer confused the reviewers). Verify shapes at every skip-add site before
dispatch.

**Bug 2 — `enc_input` tensor never assembled.**
The layer spec references `'enc_input'` as the 9-channel packed input to `enc1_conv`,
but no layer or host code packs the three input tensors (`noisyColor`, `albedo`,
`normals`) into a single HxWx9 buffer. The host must explicitly assemble this buffer
(via a packing compute shader or CPU-side copy) before calling `InferenceGraph.run()`.

**Bug 3 — Binding index mismatch between `InferenceGraph.run()` and WGSL kernels.**
`InferenceGraph.run()` places the output buffer at `binding = layer.inputs.length`,
the uniform buffer at `+1`, weights at `+2`, biases at `+3`. The WGSL kernels in
`conv2d.wgsl.ts` declared: input=0, weights=1, bias=2, output=3, uniform=4 — a
different layout. Re-implementation must define a single canonical binding convention and
enforce it in both the TypeScript dispatch and the WGSL declarations. Recommended:
match the WGSL declarations (input=0, weights=1, bias=2, output=3, uniform=4) and
update the TypeScript side to match.

**Bug 4 — Uniform buffer never written.**
`InferenceGraph.initialize()` allocates a 32-byte uniform buffer per layer but never
calls `device.queue.writeBuffer()` with actual shape params (inputH, inputW, inputC,
outputC, stride, etc.) before dispatch. The uniform buffer is zeroed; the WGSL kernels
read zero-dim shapes and produce no output. Fix: write the uniform buffer in
`initialize()` from `layer.params` immediately after allocation.

**Bug 5 — `train.py.md` cannot be executed.**
The training pipeline was documented as a `.md` file in `tools/neural-denoiser-training/`.
Re-implementation requires a real `train.py` (PyTorch recommended, see exporter spec
below). The `.md` extension blocked all training tooling.

**Bug 6 — Bind-group cache not stable across buffer swaps.**
`InferenceGraph.run()` caches bind groups after first creation, but bind groups hold
references to specific `GPUBuffer` objects. If the host swaps output buffers (e.g. on
resize), the cached bind group binds the destroyed original. Re-implementation must
invalidate the cache on any buffer swap, or use explicit cache keys based on buffer
identity.

**Bug 7 — `dispose()` does not clear the bind-group cache slot-by-slot.**
`dispose()` set `this._cachedBindGroups = []` after destroying underlying buffers.
GPUBindGroups hold references to destroyed buffers; the JS GC may not release these
immediately. Re-implementation should set each slot to `undefined` explicitly and
replace with a new `Array(spec.layers.length).fill(undefined)`.

**Bug 8 — No `'neural'` denoiser mode in `HybridEngine`.**
The engine accepted only `'atrous-variance'` (formerly `'svgf'`). The neural path was
entirely unwired. Re-implementation must add a `'neural'` mode to `HybridEngineOptions`
and route it through the pipeline compiler before declaring the feature complete.

---

## `train.py` exporter spec

The training script must export weights in the format expected by `InferenceGraph`:

**Input:**
- Training pairs: (noisy path-traced render, denoised ground-truth render) from the
  vitrum reference scenes in `examples/`. Auxiliary buffers: albedo G-buffer and
  world-space normals G-buffer.
- Data format: RGBA PNG pairs, loaded as float32.

**Architecture:** exact topology above. PyTorch `nn.Conv2d` / `nn.ConvTranspose2d` with
no batch norm (inference is per-frame; batch norm would require re-normalization at
inference time).

**Loss:** L1 + SSIM composite, or a perceptual loss (VGG feature matching) if quality
is insufficient. Match what Chaitanya 2017 uses for the path-tracing denoising case.

**Export format (`.vitrum-model` binary):**
```
Header: [u32 magic=0xDEAF1984, u32 version=1, u32 layerCount]
Per layer: [u32 nameLen, char[nameLen] name,
            u32 weightCount, f32[weightCount] weights,
            u32 biasCount,   f32[biasCount]   biases]
```
Weight layout: `[outputC × inputC × kH × kW]` (standard PyTorch layout, row-major).
The loader in `InferenceGraph` reads this format to populate `ModelWeights`.

**Training target:** inference time < 50 ms at 1080p on a mid-range discrete GPU.
Quantize to fp16 post-training if the f32 model exceeds the latency budget.

---

## OIDN-bridge alternative (worth comparing before scheduling this sprint)

Intel Open Image Denoise (OIDN v2, Apache 2.0) provides a pre-trained WebAssembly + WASM
SIMD path via `@intel/oidn-wasm`. It is CPU-based (no GPU), but its quality at 1 spp is
state-of-the-art and it requires no training data.

**Trade-off vs vitrum neural:**
| Factor | OIDN-bridge | Vitrum neural |
|---|---|---|
| Training required | No | Yes (scene-specific) |
| Inference speed | ~200–500 ms CPU (1080p) | Target <50 ms GPU |
| Integration | FFI wrapper, WASM | Native WebGPU compute |
| Weight portability | Intel-hosted, not modifiable | User-trained, customizable |
| License | Apache 2.0 | MIT (vitrum) |

Recommendation: if the walkaround latency budget tolerates 200–500 ms (e.g., a
"screenshot quality" denoising mode rather than real-time per-frame), the OIDN WASM
bridge is worth prototyping first. If real-time per-frame denoising is required, the
vitrum GPU neural path is the correct long-term answer. Compare against SVGF (renamed
`atrous-variance`) to quantify the quality gap before scheduling either.

---

## Pre-requisites before scheduling

1. Albedo G-buffer texture wired into the walkaround pipeline (needed for the 9-channel
   neural input).
2. The `atrous-variance` pipeline (from the rename) providing the "denoised albedo"
   auxiliary input — the neural denoiser denoises the noisy color with the clean albedo
   as a side channel.
3. A training dataset — at minimum the Cornell box and multi-material scene reference
   renders at 1 spp (noisy) vs 4096 spp (clean). These must be rendered before
   `train.py` can run.
4. Fix Bug 2 (host-side 9-channel packing) early; it blocks every other integration step.
