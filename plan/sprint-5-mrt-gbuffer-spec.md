# Sprint 5 — MRT G-buffer Layout Specification

**Status**: locked (Decision 12).  Downstream sprints (6, 10a, 10b) are
expected to read from these locations verbatim without re-declaring the layout.

---

## Buffer allocation

Three `WebGLMultipleRenderTargets` texture slots, allocated at engine creation
(or on first `setScene`), re-used across all frames.

| Location | Name | Internal format | Notes |
|---|---|---|---|
| 0 | `gColor` | RGBA16F | Accumulated radiance — the path-traced color image. Same as the existing `primaryRadiance` output. After Sprint 5 this is a slot in the MRT rather than a standalone render target. |
| 1 | `gNormalDepth` | RGBA16F | World-space normal (xyz) + linear depth (w). |
| 2 | `gAlbedo` | RGBA8 | Demodulated base color (no lighting). May use RGBA16F if the host requires wide-gamut / HDR texture colors (e.g., emissive panels). Default RGBA8 is sufficient for non-emissive base colors. |

---

## Channel details

### `gNormalDepth` (location 1, RGBA16F)

- **`.rgb` — world-space normal (encoded to [0, 1])**  
  Cartesian unit vector in world space encoded as `(n * 0.5 + 0.5)` by the
  primary-hit surface record (see `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`,
  the `textureStore(gNormalDepthOut, ..., vec4f(normal * 0.5 + 0.5, depthSigned))` write).
  All consumers MUST decode with `xyz * 2.0 - 1.0` before use; see
  `atrous.wgsl.ts` and `spatialFilter.wgsl.ts` for the canonical decode pattern.
  Sky/miss pixels are encoded as `vec3f(0.5, 1.0, 0.5)` which decodes to `(0, 1, 0)` (world-up).
  NOT octahedral-encoded in Sprint 5 (octahedral encoding is a Sprint 9/10a
  consideration when bandwidth becomes critical — see note below).

- **`.w` — linear depth (camera-space, always positive)**  
  `linearDepth = -dot(hitPoint_cameraSpace, forwardVec_cameraSpace)`.
  Equivalently, `linearDepth = -viewMatrix[2][0..2] · hitPoint_world - viewMatrix[2][3]`
  using the row-major convention Three.js uses for view matrices.
  Sky/miss pixels write `linearDepth = 0.0` (or a large sentinel like `1e6`).
  The sign convention matches the existing à-trous denoiser
  (`packages/shared-denoisers/src/wgsl/atrous.wgsl.ts`, line 55: "depth lives
  in .w … sky pixels write depth=0").

**Encoding note for Sprint 9/10a**: octahedral encoding (Cigolle et al. 2014,
"Survey of Efficient Representations for Independent Unit Vectors") would pack
the normal into two float16 channels, freeing `.zw` for motion vectors.
That change would break the `.xyz = world normal` convention here.  Sprint 9/10a
MUST check this spec before changing the encoding.  As a forward-compatibility
bridge: `gNormalDepth.w` must always be linear depth regardless of Sprint 9
normal encoding choice.

### `gAlbedo` (location 2, RGBA8 default / RGBA16F optional)

- **`.rgb` — base color × ambient occlusion, unlit**  
  Sampled from `surfaceRecord.albedo` at primary-hit resolution.  "Demodulated"
  in the OIDN sense: all direct and indirect lighting contribution is excluded.
  For emissive panels, the emissive term is excluded from this buffer (it is
  part of `gColor` only).

- **`.a` — currently 1.0 (reserved)**  
  May be used in a future sprint for roughness or metallic encoding.

**Format note**: RGBA8 represents values in [0, 1] with 8-bit precision per
channel.  For displays with wide color gamut or for panels whose base color
textures are authored in P3/Rec2020, the host should override to RGBA16F.
The `@vitrum/pt-webgl` engine does not force a specific format — it uses
whatever the host allocates in the MRT.  The default (documented here) is RGBA8.

---

## How downstream sprints consume these buffers

### Sprint 6 — spatial denoiser

Reads `gNormalDepth.rgb` (edge-stop normal) and `gNormalDepth.w` (edge-stop
depth).  Does NOT use `gAlbedo`.

### Sprint 10a — SVGF

Reads all three channels.  Uses `gNormalDepth` for temporal reprojection edge
stops; uses `gAlbedo` to demodulate lighting before temporal accumulation, then
re-multiplies at display.

### Sprint 10b — OIDN final pass

Passes `gAlbedo.rgb` and `gNormalDepth.rgb` as auxiliary inputs to the OIDN
ONNX model.  The model subtracts the albedo from the noisy color, denoises the
lighting, then re-multiplies.

---

## `FrameOutput` bindings

The `@vitrum/core` `FrameOutput` interface (`packages/core/src/frame.ts`)
already declares:

```typescript
readonly normalDepth?: BackendTexture;   // gNormalDepth slot
readonly albedo?: BackendTexture;        // gAlbedo slot
```

The `@vitrum/pt-webgl` engine's `renderFrame` return value fills these with
the MRT texture handles after Sprint 5 MRT allocation is complete.

Before Sprint 5, both fields are `undefined`.  Hosts that need them should
check for `undefined` defensively:

```typescript
const out = engine.renderFrame(input);
if (out.normalDepth != null) {
  // safe to pass to SVGF / OIDN
}
```
