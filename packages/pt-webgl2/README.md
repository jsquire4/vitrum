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
| `full` | MAX_DRAW_BUFFERS≥3, MAX_TEXTURE_IMAGE_UNITS≥12, MAX_TEXTURE_SIZE≥8192 | Auxiliary G-buffer (normal/depth, albedo at MRT attachments 1–2) enabled |
| `lite` | Any limit below full-tier threshold | G-buffer outputs disabled (`normalDepth`, `albedo` are `null` on `FrameRendered`). The path-tracing kernel — all bounces, full BSDF, spectral, textures, all emitter types — runs **unchanged**. |

Force a tier with `traceTier: 'full' | 'lite'` in options.

## Capabilities summary (post H1/H2/H3 fixes, 2026-06-09)

| Feature | Status |
|---------|--------|
| Mesh / instanced-mesh / skinned-mesh primitives | Supported |
| Analytic shapes (sphere, box, etc.) | Not supported (mesh fallback via `@vitrum/engine`) |
| Emitters: directional, point, spot, rect-area, disc-area, mesh-area | Supported |
| Environment: none, hdri | Supported (HDRI requires raw `{width, height, data}` RGB float payload) |
| Spectral hero-wavelength (`spectral: true`) | Lit, achromatic-flat until Jakob coefficient upload (H2 partial) |
| Bidirectional path tracing (`bdpt: true`) | Supported (A5, 2026-06-10): host opt-in (`bdpt: true`); light-subpath passes are driven on any driver. No ANGLE-specific gating exists — `EXT_disjoint_timer_query` is NOT used as a gate. |
| `backgroundAlpha` | Supported (0 = transparent background; <1 forces alpha-composite regime) |
| Analytic lights NEE (`lights.count`) | Supported (H1 fix) |
| Texture atlas (material maps) | Supported — raw `{width,height,data}` or DataTexture-shaped |
| Caustic strategy `'manifold-nee'` | Heuristic refraction-walk (NOT Newton-solve MNEE — see options.ts) |
| Caustic strategy `'photon-map'` | Deterministic cone-traced estimate (known ~21% energy approximation) |
| Denoiser `oidn-final` | Supported as an async final-pass CPU result. Requires host `oidn: { modelUrl }` and optional `onnxruntime-web`; full tier supplies HDR + albedo + normal aux, lite tier supplies HDR color. Retrieve with `getLatestDenoised()` and observe state via `FrameStats.denoiserState`. |

## Known gaps

- **BDPT** (`bdpt: true`): BDPT is implemented and host-driven (A5). No ANGLE-specific gating exists — there is no `EXT_disjoint_timer_query` gate and no driver detection path. Pass `bdpt: true` to enable; any conformant WebGL2 driver runs full BDPT.
- **Realtime denoisers**: `atrous`, `atrous-variance`, `svgf-real`, `bmfr`, and `neural` remain unsupported on this converged WebGL2 backend and warn/degrade to no-denoise. Use `oidn-final` for final-pass denoising.
- **`rotationY` implemented (H6)**: `makeRotationYMat4(-rotationY)` is uploaded as `environmentRotation`; the GLSL equirect lookup applies `mat3(environmentRotation) * worldDir` so the environment dome rotates CCW. Default `rotationY = 0` is byte-identical to pre-H6. pt-webgpu implements the same convention via `params.environmentTint.w` (packed rotY) consumed by `rotateYNeg`/`rotateYPos` helpers in `connect.wgsl.ts`. walkaround-hybrid also consumes `rotationY` (HybridEngine.ts:2061,2072 pass it through to DDGI and DDGI-probe-update; `environmentSample.wgsl.ts` applies `envRotateYNeg`).
- **Mesh-area NEE**: mesh-area emitters are sampled via explicit triangle-light NEE (B4, 2026-06-10) — area-weighted random triangle selection with shadow ray. Also visible via emissive fold on direct camera hits.
- **Spectral: achromatic-flat**: spectral mode traces the hero-wavelength path and reconstructs RGB via CIE CMF tables (H2 fix), but Jakob–Hanika material coefficients are not yet uploaded so spectral reflectance is a uniform tint over RGB. Tracked in road-to-100.

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
