# @vitrum/pt-webgl2

THREE-free native WebGL2 path-tracing backend for the `@vitrum/core` engine contract.

## Backend ID

`'pt-webgl2'`

## Entry point

```ts
import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl2';
```

## Device requirement

A host-owned `WebGL2RenderingContext`. The engine allocates GL resources against it
but **never** loses or destroys the context — the host owns the lifecycle.

The extension `EXT_color_buffer_float` (RGBA32F render targets) is **required**;
`createPTEngine_WebGL2` throws if it is absent.

## Trace tiers

| Tier | Condition | Difference |
|------|-----------|------------|
| `full` | Default after mandatory WebGL2 kernel limits pass | Auxiliary G-buffer (normal/depth, albedo at MRT attachments 1–2) enabled |
| `lite` | Explicit `traceTier:'lite'` lower-memory profile | G-buffer outputs disabled (`normalDepth`, `albedo` are `null` on `FrameRendered`). The path-tracing kernel — all bounces, full BSDF, spectral, textures, all emitter types — runs **unchanged**. |

Force a tier with `traceTier: 'full' | 'lite'` in options.
Both tiers require four draw buffers and four color attachments for the
four-attachment NEE candidate handoff, plus 15 fragment texture units for the
maximum selectable shader graph (WebGL2's required minimum is 16).
Material/BVH texture dimensions are validated against the live context when
each scene resource is allocated; `MAX_TEXTURE_SIZE` does not silently change
the output tier.

## Capabilities summary

| Feature | Status |
|---------|--------|
| Mesh / instanced-mesh / skinned-mesh primitives | Supported |
| Analytic shapes (sphere, box, capsule, cylinder, h-channel came) | Supported as generated mesh fallback (`fallback-generated-mesh`) |
| Emitters: directional, point, spot, rect-area, disc-area, mesh-area | Supported |
| Environment: none, hdri | Supported (HDRI requires raw `{width, height, data}` RGB float payload) |
| Spectral hero-wavelength (`spectral: true`) | Supported — CIE CMF importance sampling/reconstruction, per-material Jakob-Hanika reflectance, packed spectral attenuation, thin-film, and Cauchy dispersion are uploaded and consumed by the active trace path. |
| Bidirectional path tracing (`bdpt: true`) | Supported bounded general BDPT — 1–8 stored light vertices (default 4), finite c=0 and c≥1 surface/medium connections, nested-medium Beer/HG transport, and Veach power-heuristic MIS. Analytic, mesh-area, and HDRI light subpaths are supported. |
| `backgroundAlpha` | Supported (0 = transparent background; coverage is preserved by the portable alpha-aware running mean on every device) |
| Analytic lights NEE (`lights.count`) | Supported (H1 fix) |
| Texture atlas (material maps) | Supported — raw `{width,height,data}` or DataTexture-shaped |
| Caustic strategy `'bdpt'` | Supported alias for the same bounded general estimator as `bdpt: true`; tune with `bdptOptions.maxLightBounces`. |
| MNEE / SPPM (`'manifold-nee'`, `'photon-map'`) | Unsupported and rejected by the strict option validator; use the pt-webgpu full tier for these estimators. |
| Denoiser `auto` / `oidn-final` | `auto` always resolves to `oidn-final`. Omitted `oidn.modelUrl` uses the pinned Intel RT HDR alb+nrm ONNX; hosts may override. Requires optional `onnxruntime-web` at the first denoise cycle. Full tier supplies HDR + albedo + normal aux, lite tier supplies HDR color. Retrieve with `getLatestDenoised()` and observe state via `FrameStats.denoiserState`. |
| Inverse (`createInverseSession`) | Finite-difference only. `method:'path-replay'` throws. Certified path-replay is a pt-webgpu full-tier niche (emissive RGB, one bounce). |

## Deliberate backend boundaries

- **Realtime denoisers**: `atrous`, `atrous-variance`, `svgf-real`, `bmfr`, and `neural` are outside this converged backend and are rejected by the strict construction validator. Use `auto` or explicit `oidn-final` for final-pass denoising.
- **Caustic-estimator breadth**: this backend provides BDPT. Newton MNEE and SPPM are not option values and fail closed; those estimators are implemented by the pt-webgpu full tier.
- **SSS model**: translucent surfaces use one back-face single-scatter event with a scalar free-flight majorant, per-channel scattering albedo, and Henyey–Greenstein phase. This is intentionally narrower than pt-webgpu's native volume path.
- **Mesh-area stream**: mesh-area emitters use the dedicated triangle-light NEE/MIS stream rather than the six-texel analytic-light stream. They are sampled area/power-weighted and remain visible through the emissive fold.
- **Environment rotation**: `rotationY` is uploaded as `environmentRotation`; GLSL applies the same negative lookup rotation convention as pt-webgpu and walkaround-hybrid.

## Minimal usage snippet

```ts
import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl2';
import type { Scene } from '@vitrum/core';

const canvas = document.querySelector('canvas')!;
const gl = canvas.getContext('webgl2')!;

const engine = await createPTEngine_WebGL2({ device: gl });

const scene: Scene = {
  primitives: [/* your MeshPrimitive objects */],
  emitters: [],
  environment: { kind: 'none' },
};

engine.setScene(scene);

function tick(t: number) {
  const output = engine.renderFrame({
    viewMatrix: /* Float32Array column-major */ yourViewMatrix,
    projMatrix: /* Float32Array column-major */ yourProjMatrix,
    cameraPosition: [0, 0, 5],
    viewport: { width: canvas.width, height: canvas.height, devicePixelRatio: window.devicePixelRatio },
    frameIndex: frameCount++,
    frameSeed: (t * 1000) & 0xffffffff,
  });

  if (output.kind === 'rendered') {
    // output.primaryRadiance is a WebGLTexture — blit it to screen.
  }

  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
```

## Provenance

Ported from `gkjohnson/three-gpu-pathtracer` (MIT). See `CREDITS.md` at the repo root.
