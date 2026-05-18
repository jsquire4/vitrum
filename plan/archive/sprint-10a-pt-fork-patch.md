# Sprint 10a — PT preview side: swapping Sprint 6's hexagonal filter for SVGF

**Mode scope**: PT preview (fork-side + host glue).
**Status**: deferred — integration blocked pending GPU verification environment.

---

## Context

Sprint 6 shipped a 37-tap hexagonal-kernel bilateral filter (`PTSpatialDenoiser.tsx`)
that auto-disables above 24 accumulated samples. Sprint 10a replaces this with SVGF's
spatiotemporal filter to improve temporal stability and variance-guided edge stopping.

The vitrum-side WGSL shader (`@vitrum/shared-denoisers/src/wgsl/svgf.wgsl.ts`) and
TypeScript bindings (`svgfBindings.ts`) are complete. This document covers the
**fork-side and host-side** work needed to feed SVGF the correct G-buffers.

---

## Prerequisite: Sprint 5 MRT G-buffer scaffold

SVGF requires three G-buffer channels per pixel at primary hit:

- **Color** (noisy accumulated radiance) — already available from the PT accumulation target.
- **Normal** (world-space) — available from Sprint 5's `gNormalDepth` MRT channel (`.xyz`).
- **Depth** (linear camera-space) — available from Sprint 5's `gNormalDepth` MRT channel (`.w`).

Sprint 5 allocated `WebGLMultipleRenderTargets` with three channels and populated them
from the primary-hit surface record in `PhysicalPathTracingMaterial.js`. Verify that
`gNormalDepth` is correctly filled before wiring SVGF.

Additionally, SVGF requires **motion vectors** (screen-space UV delta per pixel).
The PT preview currently has no motion vector pass — this is a new requirement.

---

## Fork-side changes required

### 1. Motion vector output

In `PhysicalPathTracingMaterial.js`, add a fourth MRT output `gMotion` (RG32F):

```glsl
// In the fragment shader, at primary hit:
vec4 prevClip = prevViewProjMatrix * vec4(worldPos, 1.0);
vec2 prevNDC  = prevClip.xy / prevClip.w;
vec2 currNDC  = gl_FragCoord.xy / vec2(targetWidth, targetHeight) * 2.0 - 1.0;
gMotion = vec4((currNDC - prevNDC) * 0.5, 0.0, 1.0); // UV delta in [-1, 1]
```

Uniforms needed: `prevViewProjMatrix` (mat4), `targetWidth`, `targetHeight`.

### 2. MRT layout update

Update `WebGLMultipleRenderTargets` allocation from 3 to 4 channels:

```
target 0: gColor       (rgba16float) — noisy accumulated radiance
target 1: gNormalDepth (rgba16float) — .xyz = world normal, .w = linear depth
target 2: gAlbedo      (rgba16float) — base-color unlit
target 3: gMotion      (rg32float)   — screen-space motion delta
```

File: `PathTracingLayer.tsx` (or wherever `WebGLMultipleRenderTargets` is allocated).

### 3. Previous-frame radiance buffer

SVGF's variance estimation pass reads `prevRadiance` for temporal feedback.
Add a second ping-pong accumulation target and blit the current frame's result
into it at frame end (before resetting for the next sample batch).

---

## Host-side changes required

### PTSpatialDenoiser.tsx replacement

Remove `PTSpatialDenoiser.tsx` (the Sprint 6 hexagonal filter).

Create a new `PTSVGFDenoiser.tsx` as a `postprocessing` Effect subclass:

```typescript
// Pseudocode — not a complete implementation.
class PTSVGFDenoiser extends Effect {
  private svgfVariancePipeline: WebGLProgram | null = null;
  private svgfAtrousPipeline: WebGLProgram | null = null;
  private pingBuffer: WebGLTexture | null = null;
  private pongBuffer: WebGLTexture | null = null;

  // Called per-frame when pathtracer.samples < AUTO_DISABLE_THRESHOLD.
  override render(renderer, inputBuffer, outputBuffer) {
    // 1. Run svgfVarianceMain to estimate per-pixel variance.
    // 2. Run svgfAtrousMain 5× with ping-pong, iteration 0→4.
    // 3. Write final filtered color to outputBuffer.
  }
}
```

Auto-disable threshold: keep at `pathtracer.samples > 24` (same as Sprint 6).

### EffectComposer wiring

Replace `PTSpatialDenoiser` in the composer chain with `PTSVGFDenoiser`:

```typescript
// In PathTracingLayer.tsx:
// Before: composer.addPass(new EffectPass(camera, spatialDenoiser));
// After:  composer.addPass(new EffectPass(camera, svgfDenoiser));
```

SVGF must remain first in the chain (before Bloom), same as the hexagonal filter.

---

## Uniform defaults for PT preview

Use `SVGF_DEFAULT_UNIFORMS` from `@vitrum/shared-denoisers/svgfBindings`:

```
sigmaColor  = 10.0   (variance-guided; wider tolerance at low sample counts)
sigmaNormal = 128.0  (preserve sharp panel edges)
sigmaDepth  = 1.0    (scene-scale appropriate for interior room)
```

Tune `sigmaDepth` downward (toward 0.5) if blurring occurs across came/glass boundaries.

---

## Definition of done (PT preview side)

- [ ] Motion vector MRT target allocated and populated in fork
- [ ] Previous-frame radiance ping-pong buffer in place
- [ ] `PTSVGFDenoiser.tsx` replaces `PTSpatialDenoiser.tsx` in EffectComposer
- [ ] SVGF runs 5 à-trous iterations with variance guidance
- [ ] Auto-disables above 24 accumulated samples
- [ ] Visual A/B at 8 samples: cleaner than Sprint 6 hexagonal on glass surfaces
- [ ] No regression on Sprint 5 MRT binding point assignments

---

## Integration risk: GPU verification blocked

The WGSL shader and TypeScript descriptors are complete and type-clean.
GPU execution cannot be verified in this environment (no WebGPU/WebGL available).
Integration testing must happen in a browser environment with:

1. A running host application with PT preview enabled.
2. DevTools GPU timestamp queries or `EXT_disjoint_timer_query_webgl2` for pass timing.
3. Reference screenshots before/after the filter swap.

**Do not merge the fork-side changes until GPU integration is verified in a browser.**
