# Sprint 13 — Walkaround Neural Denoiser: Integration Spec

**Status**: Vitrum-side scaffold complete. GPU-verified integration deferred.  
**Replaces**: `plan/sprint-13-deferred.md`  
**Created**: 2026-05-09  
**Source**: `plan/phase-6-roadmap.md` §5, Sprint 13

---

## What this document covers

Sprint 13 shipped the inference scaffold (WGSL kernels, InferenceGraph, UNet spec,
training pipeline docs). This document specifies exactly how to wire `InferenceGraph`
into `HybridEngine.renderFrame` once GPU verification is available.

---

## Integration point in renderFrame

The neural denoiser runs **post-pipeline, before composite**. The execution order
in `HybridEngine.renderFrame` after integration:

```
1. BVH / ReSTIR DI (existing)
2. SVGF variance pass          (Sprint 10a — svgfVarianceMain)
3. SVGF à-trous pass × 5       (Sprint 10a — svgfAtrousMain)
4. ── NEURAL DENOISER ──────────── NEW (Sprint 13, gated)
   4a. Pack G-buffers → 9-channel enc_input buffer
   4b. InferenceGraph.run(device, encoder, inputs, outputs)
5. Composite pass              (existing composite.wgsl)
```

The denoiser is gated by `HybridEngineOptions.neuralDenoiserEnabled` (default: false).
When disabled, the composite pass reads the SVGF output directly (Sprint 10a behavior).

---

## Input buffer packing

`InferenceGraph` expects one 9-channel input tensor named `enc_input` (H × W × 9, f32):

| Channels  | Source                                                  | Buffer                                |
| --------- | ------------------------------------------------------- | ------------------------------------- |
| 0–2 (RGB) | SVGF denoised color (after 5 à-trous passes)            | `atrous_outputColor` ping-pong result |
| 3–5 (RGB) | G-buffer albedo (Sprint 5 MRT, binding 2)               | `gAlbedo` texture read                |
| 6–8 (RGB) | G-buffer normals (Sprint 5 MRT, binding 1, decode .xyz) | `gNormalDepth` texture read           |

**Pack pass**: A small compute shader (not in Sprint 13 scope — author in a follow-up
single-day patch) reads the three source textures and writes a flat f32 buffer in
channels-last format. Workgroup 8×8 × 1, dispatch (ceil(W/8), ceil(H/8), 1).

Normal encoding: world-space normals from G-buffer are already in [-1, 1]; encode
as `(n + 1) / 2` to map to [0, 1] before passing to the network (matches the
`dataset_spec.md` preprocessing).

---

## Bind group additions to pipelineCompiler.ts

Two new bind groups for the neural denoiser dispatch chain:

**Bind group 3** — shared inputs (read across all layers that need them):

- binding 0: `enc_input` storage buffer (H × W × 9 f32) — packed G-buffer
- binding 1: `denoisedColor` storage buffer (H × W × 3 f32) — final output

**Layer-level bind groups** (bind group 0 per layer, recreated per dispatch):

- Per the `InferenceGraph.run()` implementation — auto-created from `pipeline.getBindGroupLayout(0)`.

The `WalkaroundGPUPipeline` must allocate:

- `enc_input` buffer: `width × height × 9 × 4` bytes
- All intermediate tensor buffers: managed internally by `InferenceGraph.initialize()`
- `denoisedColor` output buffer: `width × height × 3 × 4` bytes

---

## Gating: `setNeuralDenoiserEnabled(on: boolean)`

Add to `HybridEngine`:

```typescript
private _neuralDenoiserEnabled = false;

/** Toggle neural denoiser at runtime. Default: off.
 *  The inference graph is initialized lazily on first enable call.
 *  Disable to fall back to SVGF output for the composite pass. */
setNeuralDenoiserEnabled(on: boolean): void {
  this._neuralDenoiserEnabled = on;
}

get neuralDenoiserEnabled(): boolean {
  return this._neuralDenoiserEnabled;
}
```

Add to `HybridEngineOptions`:

```typescript
/** Sprint 13 — Enable neural denoiser. Default: false. */
readonly neuralDenoiserEnabled?: boolean;
/** Sprint 13 — Pre-loaded model weights for the neural denoiser. */
readonly neuralDenoiserWeights?: ModelWeights;
```

The `InferenceGraph` is constructed + initialized lazily in `renderFrame` on the
first frame where `neuralDenoiserEnabled === true` and `neuralDenoiserWeights` is set.

---

## Inference time DoD

Per `plan/phase-6-roadmap.md` §Sprint 13:

| Metric                            | Target    | Abort threshold         |
| --------------------------------- | --------- | ----------------------- |
| `InferenceGraph.run()` frame time | < 50 ms   | ≥ 50 ms (month-1 check) |
| Typical discrete GPU (desktop)    | 10–30 ms  | —                       |
| Integrated GPU / mobile           | 50–150 ms | Disable, await WebNN    |

**Month-1 benchmarks**: after the first month of integration, time `InferenceGraph.run()`
using WebGPU timestamp queries. If p95 > 50 ms on the target GPU, abort neural denoising
and rely on SVGF alone. See `plan/phase-6-roadmap.md` §7 item 5 for the bail-out question.

---

## Memory budget

**Weights (CPU + GPU copy)**: ~1.63 MB f32 (426,075 params × 4 bytes).

**Intermediate tensors at 1080p (float32)**:

| Tensor        | Shape            | Size        |
| ------------- | ---------------- | ----------- |
| enc_input     | 1080 × 1920 × 9  | 71.3 MB     |
| enc1          | 540 × 960 × 24   | 49.8 MB     |
| enc2          | 270 × 480 × 48   | 24.9 MB     |
| enc3          | 135 × 240 × 96   | 12.4 MB     |
| btn           | 135 × 240 × 192  | 24.9 MB     |
| dec3          | 270 × 480 × 96   | 49.8 MB     |
| dec2          | 540 × 960 × 48   | 99.5 MB     |
| dec1          | 1080 × 1920 × 24 | 199 MB      |
| denoisedColor | 1080 × 1920 × 3  | 25 MB       |
| **Total**     |                  | **~557 MB** |

**Mitigation options** (apply in order until budget fits):

1. **fp16 inference**: halve all intermediate tensor sizes (~278 MB). Requires
   WGSL kernel changes to use `f16` (behind a WebGPU extension; verify support).
2. **Channel pruning**: reduce bottleneck from 192 to 128 channels. Saves ~40% of
   the intermediate memory with ~3% quality loss. Requires retraining.
3. **Tile-based inference**: process the image in 256×256 tiles with 32px overlap.
   Reduces peak VRAM to ~80 MB at the cost of multiple dispatch calls per frame.

---

## Dispatch sizing (resolution-adaptive)

`WALKAROUND_DENOISER_UNET_SPEC` hardcodes dispatch counts for 1080p. For other
resolutions, scale the `dispatchX` / `dispatchY` in each layer's `params` field
before constructing the `InferenceGraph`:

```typescript
function adaptSpecForResolution(
  spec: InferenceGraphSpec,
  width: number,
  height: number,
): InferenceGraphSpec {
  // Recompute all dispatchX/Y from the resolution and layer input dimensions.
  // Left as host-side concern per the library/host separation principle.
  // See plan/sprint-13-walkaround-integration.md for the sizing formula.
}
```

Sizing formula per layer kind:

- `conv2d` / `transposed_conv2d`: `(ceil(outputW / 8), ceil(outputH / 8), outputC)`
- `relu` / `skip`: `(ceil(H × W × C / 256), 1, 1)`

---

## Uniform buffer updates (per-frame)

The `Conv2DParams` and `TransposedConv2DParams` uniforms are static for a given
resolution — they do not change frame-to-frame. Write them once in `initialize()`
and leave them constant. Only re-upload on resolution change (HybridEngine.reset()).

---

## Integration checklist

When GPU verification is available, work through this checklist in order:

- [ ] Add `neuralDenoiserEnabled` / `neuralDenoiserWeights` to `HybridEngineOptions`
- [ ] Add `setNeuralDenoiserEnabled` / `neuralDenoiserEnabled` getter to `HybridEngine`
- [ ] Author the G-buffer pack compute shader (one-day patch)
- [ ] Allocate `enc_input` and `denoisedColor` buffers in `resourceManager.ts`
- [ ] Call `InferenceGraph.initialize(device)` lazily on first enable
- [ ] Insert the denoiser dispatch between SVGF and composite in renderFrame
- [ ] Modify composite pass to read `denoisedColor` when denoiser is enabled
- [ ] Run month-1 inference benchmarks with WebGPU timestamp queries
- [ ] A/B compare denoised vs. raw SVGF at 4 and 8 SPP
- [ ] Update `plan/phase-6-status.md` with benchmark results

---

## Why GPU-verified integration is deferred

Sprint 13's library-side scaffold is complete. The wiring cannot be GPU-verified in
the current autonomous-mode session (no live WebGPU device). The three trigger criteria
for Sprint 13 (SVGF gap visible, WebNN still behind flag, PPG didn't close noise gap)
are user-evaluated prerequisites, not library-code prerequisites — the library can and
should be ready before evaluation completes. This integration spec is the readiness gate.
