# Sprint 10b — Host-side checklist: OIDN final-pass denoising

**Mode scope**: PT final only. Not real-time.
**Status**: vitrum-side bridge complete (`oidnBridge.ts`). Host-side items below.

---

## What's done (vitrum-side)

- `@vitrum/shared-denoisers/src/oidnBridge.ts` — lazy ONNX Runtime Web wrapper.
  Exports: `denoiseFinal`, `preloadOIDNModel`, `clearOIDNCache`.
- Execution providers configured as `['webnn', 'webgpu', 'wasm']` per Decision 11.
- `onnxruntime-web` declared as optional peerDependency in `shared-denoisers/package.json`.
- Session caching by model URL; cache clearable via `clearOIDNCache()`.

---

## Host-side items

### 1. Install onnxruntime-web

In the host application's `package.json`:
```json
"dependencies": {
  "onnxruntime-web": "^1.18.0"
}
```

Then `npm install`. This is not installed in the vitrum workspace (optional peer dep
— the library compiles without it; the host opts in).

### 2. Bundle the OIDN ONNX model file

Obtain the OIDN ONNX model from the official OpenImageDenoise project:
- Project: https://github.com/OpenImageDenoise/oidn
- Look for the `oidn-X.Y.Z.x86_64.linux.tar.gz` release, which includes ONNX exports.
- Typical model variants:
  - `oidn_rt_hdr.onnx` — color only (5 MB)
  - `oidn_rt_hdr_alb.onnx` — color + albedo (6 MB)
  - `oidn_rt_hdr_alb_nrm.onnx` — color + albedo + normal (7 MB, best quality)

The model with albedo + normal inputs produces the highest quality output; these
G-buffers are available from the Sprint 5 MRT scaffold (`gAlbedo`, `gNormalDepth`).

Place the model file in the host's public/static assets directory:
```
public/models/oidn_rt_hdr_alb_nrm.onnx
```

Pass the path to `denoiseFinal`:
```typescript
await denoiseFinal(inputs, { modelUrl: '/models/oidn_rt_hdr_alb_nrm.onnx' });
```

### 3. Add the "Denoise" button to the PT final UI

In the PT final controls panel (e.g., `PhotorealismControls.tsx` or a dedicated
`PTFinalControls.tsx`):

```tsx
const [isDenoising, setIsDenoising] = useState(false);

async function handleDenoise() {
  setIsDenoising(true);
  try {
    const color  = captureFloat32FromAccumBuffer();   // HxWx3 RGB float
    const normal = captureFloat32FromGBufferNormal();  // HxWx3 float (optional)
    const albedo = captureFloat32FromGBufferAlbedo();  // HxWx3 float (optional)

    const denoised = await denoiseFinal(
      { color, normal, albedo, width: renderWidth, height: renderHeight },
      { modelUrl: '/models/oidn_rt_hdr_alb_nrm.onnx' },
    );

    // Display / save the denoised output.
    displayDenoisedResult(denoised);
  } finally {
    setIsDenoising(false);
  }
}

return (
  <button onClick={handleDenoise} disabled={isDenoising}>
    {isDenoising ? 'Denoising…' : 'Denoise (OIDN)'}
  </button>
);
```

The button should be visible only in PT_FINAL mode and only after the accumulation
has reached a minimum sample count (e.g., ≥ 64 samples).

### 4. Capture float buffers from WebGL render targets

The host must read back float pixel data from the WebGL MRT targets:

```typescript
// Color: read from the PT accumulation buffer (FrameOutput.primaryRadiance).
function captureFloat32FromAccumBuffer(): Float32Array {
  const buf = new Float32Array(width * height * 4);
  renderer.readRenderTargetPixels(accumTarget, 0, 0, width, height, buf);
  // Convert RGBA → RGB by dropping alpha.
  return stripAlpha(buf, width, height);
}

// Normal: read from gNormalDepth target, extract .xyz channels.
function captureFloat32FromGBufferNormal(): Float32Array { /* similar pattern */ }

// Albedo: read from gAlbedo target.
function captureFloat32FromGBufferAlbedo(): Float32Array { /* similar pattern */ }
```

Note: `renderer.readRenderTargetPixels` is synchronous and blocks the main thread.
For non-blocking behavior, prefer `WebGLSync` + `readPixels` with `PACK_SKIP_ROWS`.
For PT final (single shot, not real-time), synchronous readback is acceptable.

### 5. Pre-warm the model (optional)

To avoid latency on first click, pre-warm during the PT accumulation phase:

```typescript
// When PT_FINAL mode is entered or accumulation starts:
preloadOIDNModel({ modelUrl: '/models/oidn_rt_hdr_alb_nrm.onnx' })
  .catch(err => console.warn('[OIDN] Preload failed:', err));
```

This downloads and initializes the ONNX session in the background while samples
accumulate. On first `denoiseFinal` call, the session is already cached.

### 6. Save the denoised output alongside the raw render

The existing PT final save-to-file flow should offer two options:
- "Save raw" — the undenoised accumulation buffer (existing behavior).
- "Save denoised" — appears after denoising runs; saves the OIDN output.

```typescript
function saveDenoisedAsPNG(denoised: Float32Array, width: number, height: number) {
  // Convert Float32 linear → 8-bit sRGB on a canvas.
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    imageData.data[i * 4 + 0] = linearToSRGB(denoised[i * 3 + 0] ?? 0) * 255;
    imageData.data[i * 4 + 1] = linearToSRGB(denoised[i * 3 + 1] ?? 0) * 255;
    imageData.data[i * 4 + 2] = linearToSRGB(denoised[i * 3 + 2] ?? 0) * 255;
    imageData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  canvas.toBlob(blob => triggerDownload(blob, 'render-denoised.png'), 'image/png');
}
```

### 7. Browser support fallback

| Browser | WebNN | WebGPU | WASM |
|---------|-------|--------|------|
| Chrome 120+ (WebNN flag) | ✓ | ✓ | ✓ |
| Edge 120+ (WebNN flag) | ✓ | ✓ | ✓ |
| Firefox 120+ | — | ✓ | ✓ |
| Safari 17+ | — | ✓ | ✓ |
| Any browser | — | — | ✓ |

ORT-Web picks the first available provider in the configured list. No host-side
feature detection is needed — ORT handles the fallback automatically.

If WebNN is detected as available, OIDN inference on a 2K frame should complete
in <2 seconds. On WebGPU fallback: 2–5 seconds. On WASM: 10–30 seconds.

Consider showing an estimated time tooltip when the denoising is triggered:
```typescript
const providerHint = await detectFastestProvider(); // 'webnn'|'webgpu'|'wasm'
const estimatedSeconds = { webnn: 2, webgpu: 5, wasm: 30 }[providerHint] ?? 30;
showToast(`Denoising… (~${estimatedSeconds}s)`);
```

### 8. Memory management

The OIDN model session occupies ~100–500 MB of GPU/CPU memory while loaded.
Call `clearOIDNCache()` when:
- The host navigates away from the PT final view.
- The user switches to a different render mode (walkaround, raster).
- The device is under memory pressure (listen to `document.addEventListener('memorywarning', ...)`).

---

## Definition of done (host side)

- [ ] `onnxruntime-web` installed in host's `package.json`
- [ ] OIDN ONNX model file (color+albedo+normal variant) bundled in public/models/
- [ ] "Denoise (OIDN)" button in PT final UI, gated on ≥ 64 accumulated samples
- [ ] Float32 readback from accumulation + G-buffer targets correct
- [ ] `preloadOIDNModel` called on PT final mode entry
- [ ] Denoised PNG save option alongside raw render save
- [ ] Loading/progress indicator during inference
- [ ] `clearOIDNCache()` called on mode exit or memory pressure
- [ ] Inference time <2s on a reference GPU with WebNN, <5s with WebGPU

---

## Sprint 10b acceptance benchmark

Capture before/after comparison on a 2K hero render of the stained-glass scene:
- **Before**: raw PT_FINAL at 192 samples (standard convergence setting).
- **After**: same render denoised with OIDN.
- Expected: visible Monte Carlo noise eliminated; sharp glass boundaries preserved;
  caustic detail retained (OIDN's normal + albedo aux inputs guide edge preservation).
- Save both renders to `tools/reference-renders/sprint-10b-<before|after>.png`.
