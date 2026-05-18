# Sprint 10a — Walkaround side: replacing à-trous with SVGF

**Mode scope**: walkaround (WebGPU path, `@vitrum/walkaround-hybrid`).
**Status**: deferred — integration blocked pending GPU verification environment.

---

## Context

The walkaround engine currently uses `ATROUS_WGSL` from `@vitrum/shared-denoisers` as
its denoising pass, with 3 à-trous iterations (step widths 1, 2, 4). Sprint 10a
replaces this with `SVGF_WGSL`, adding variance guidance from the Sprint 9 Welford
buffer and temporal stability through per-pixel variance estimation.

The vitrum-side WGSL (`svgf.wgsl.ts`) and TypeScript bindings (`svgfBindings.ts`) are
complete. This document covers the walkaround-side wiring.

---

## Walkaround SVGF wiring points

### 1. `packages/walkaround-hybrid/src/pipeline/pipelineCompiler.ts`

Add two new compute pipelines to the compiled pipeline set:

```typescript
// Import from shared-denoisers
import { SVGF_WGSL } from '@vitrum/shared-denoisers';

// In compilePipelines():
const svgfVariancePipeline = device.createComputePipeline({
  layout: 'auto',
  compute: {
    module: device.createShaderModule({ code: SVGF_WGSL }),
    entryPoint: 'svgfVarianceMain',
  },
});

const svgfAtrousPipeline = device.createComputePipeline({
  layout: 'auto',
  compute: {
    module: device.createShaderModule({ code: SVGF_WGSL }),
    entryPoint: 'svgfAtrousMain',
  },
});
```

Both entry points are in the same WGSL module string — one `createShaderModule` call,
two `createComputePipeline` calls with different `entryPoint` values.

### 2. `packages/walkaround-hybrid/src/pipeline/bindGroupBuilders.ts`

Add two new bind group builder functions:

**`buildSVGFVarianceBindGroup(device, textures, welfordBuffer, uniformBuffer)`**

Bind group matching `SVGFVarianceBindGroupLayout`:

```
binding 0 — inputColor:    current frame noisy color texture (rgba16float)
binding 1 — prevRadiance:  previous frame accumulated color (rgba16float)
binding 2 — gbufferNormal: G-buffer normal (rgba16float, .xyz)
binding 3 — gbufferDepth:  G-buffer depth (rgba16float, .r) or r32float
binding 4 — motionVectors: screen-space motion vectors (rg32float)
binding 5 — varianceIn:    Welford variance buffer from Sprint 9 (rg32float)
binding 6 — varianceOut:   estimated variance output (rg32float, storage write)
binding 7 — ubo:           SVGFVarianceUBO (16 bytes, frameCount)
```

The Welford buffer at binding 5 is the same `RG32Float` texture that Sprint 9's
`accumulate.wgsl` writes to each frame. Its `.r` = Welford mean, `.g` = M2.

**`buildSVGFAtrousBindGroup(device, inputColor, outputColor, normal, depth, varianceMap, uniformBuffer)`**

Bind group matching `SVGFAtrousBindGroupLayout`:

```
binding 0 — inputColor:  ping-pong input (rgba16float)
binding 1 — outputColor: ping-pong output (rgba16float, storage write)
binding 2 — gbufferNormal
binding 3 — gbufferDepth
binding 4 — varianceMap: output of svgfVarianceMain (rg32float, .r = variance)
binding 5 — ubo:         SVGFAtrousUBO (16 bytes, iteration + sigmas)
```

### 3. `packages/walkaround-hybrid/src/engines/hybrid/HybridEngine.ts`

Replace the current à-trous dispatch loop with the SVGF two-pass dispatch:

```typescript
// Sprint 9 à-trous: 3 iterations, step widths 1/2/4 — REPLACE with:

// Pass 1: variance estimation
packSVGFVarianceUniforms({ frameCount: this._frameCount }, this._svgfVarianceUniformBuf);
commandEncoder.setPipeline(this._svgfVariancePipeline);
commandEncoder.setBindGroup(0, this._svgfVarianceBindGroup);
commandEncoder.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));

// Pass 2: à-trous wavelet, 5 iterations
for (let iter = 0; iter < 5; iter++) {
  packSVGFUniforms({ iteration: iter, ...SVGF_DEFAULT_UNIFORMS }, this._svgfAtrousUniformBuf);
  const atrousBindGroup = this._buildAtrousPingPong(iter);
  commandEncoder.setPipeline(this._svgfAtrousPipeline);
  commandEncoder.setBindGroup(0, atrousBindGroup);
  commandEncoder.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
}
```

Ping-pong: maintain two `rgba16float` textures (`_atrousPing`, `_atrousPong`).

- Even iterations: read from ping, write to pong.
- Odd iterations: read from pong, write to ping.
- Input to iteration 0: the noisy accumulated color from the accumulator pass.
- Output of iteration 4: the final denoised frame (written to the display target).

### 4. `_frameCount` tracking in HybridEngine

Add a `_frameCount: number` field to `HybridEngine`. Increment each frame.
Reset to 0 when the camera moves (detect via `FrameInput.cameraMatrix` delta > ε).

Pass `frameCount` to `packSVGFVarianceUniforms` each frame.

---

## Welford buffer compatibility

The Sprint 9 Welford variance buffer is a `RG32Float` texture where:

- `.r` = Welford mean (running average of luminance)
- `.g` = Welford M2 (sum of squared deltas)

This is directly compatible with SVGF's `varianceIn` binding (binding 5 in the variance
pass). No format conversion needed. The SVGF shader reads `.r` and `.g` and wraps them
in a `WelfordVariance` struct locally.

**Key constraint (Decision 13)**: the `WelfordVariance` struct layout in `svgf.wgsl.ts`
must remain byte-for-byte identical to the canonical definition in
`walkaround-hybrid/src/shaders/common.wgsl.ts @version 1`. Both declare:

```wgsl
struct WelfordVariance { mean: f32, m2: f32 };
```

If the canonical layout changes, bump the @version comment and update SVGF's local copy.

---

## À-trous removal

Once SVGF is wired and verified, remove the existing à-trous pipeline from
`pipelineCompiler.ts` and its bind group builder from `bindGroupBuilders.ts`.
Keep `ATROUS_WGSL` in `@vitrum/shared-denoisers` (do not delete the export) — it
may be referenced by tests or future denoisers. Only remove the walkaround wiring.

---

## Uniform defaults

Use `SVGF_DEFAULT_UNIFORMS` for initial tuning:

```
sigmaColor  = 10.0   — variance-guided; relaxed to handle caustic variance
sigmaNormal = 128.0  — preserves came/lead strip edges aggressively
sigmaDepth  = 1.0    — tune relative to room scale (try 0.5 for small scenes)
```

Budget 3 days for re-tuning σ values as noted in the Sprint 10a risk estimate.
Tuning criterion: at 8 SPP, diffuse floor surfaces should be indistinguishable
from a 64-SPP reference. Caustic edges on glass panels are acceptable at 8 SPP.

---

## Definition of done (walkaround side)

- [ ] SVGF two-pass pipeline (variance + à-trous×5) compiles without validation errors
- [ ] Variance bind group correctly references Sprint 9 Welford buffer
- [ ] Ping-pong textures allocated and swapped correctly across 5 iterations
- [ ] `_frameCount` resets on camera move, increments each frame
- [ ] Old à-trous pipeline removed from walkaround wiring
- [ ] Visual A/B at 8 SPP: diffuse surfaces cleaner than à-trous baseline
- [ ] No WebGPU validation layer errors in DevTools GPU section
- [ ] Walkaround frame rate ≥ 20 fps on reference GPU (measure with timestamp queries)

---

## Integration risk: GPU verification blocked

All TypeScript and WGSL code is complete and type-clean. GPU execution cannot be
verified in this environment. Integration testing requires a browser with WebGPU
enabled and the walkaround engine running. Recommended verification sequence:

1. Open DevTools → GPU panel; check for pipeline compilation errors.
2. Capture `__WG__` debug bridge output for variance estimates (should be non-zero
   after frame 2 on surfaces with any lighting variation).
3. Run the visual A/B: screenshot walkaround at 8 SPP before and after the swap.
4. Check WebGPU validation layer for bind group layout mismatches.
