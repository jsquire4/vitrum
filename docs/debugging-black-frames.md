# Black-frame debugging runbook

A "black frame" is any render where the canvas stays black for more than one or two
frames when a lit scene is loaded.  The failure modes below cover every root cause
found in this codebase's history.  Work through the checks in order — earlier checks
are cheaper and eliminate the most common causes first.

---

## Check 0 — GPU errors (the fastest signal)

Before any other check, look for GPU validation errors.  A single error scope
violation can silence the render pipeline entirely.

**How to check:**
- Open the browser DevTools **Console** tab.  Chrome's WebGPU layer emits validation
  errors as `GPUDevice.pushErrorScope` / `popErrorScope` results.  vitrum surfaces
  these via `engine.capabilities` (see `onError` below).
- If you're using `createEngine` or `attachVitrum`, pass `onError`:

  ```ts
  createEngine({
    canvas, scene,
    onError(err, event) {
      console.error('[vitrum] engine error:', event.phase, err);
    },
  });
  ```

  `phase: 'attach:renderFrame'` + `recoverable: false` means the RAF loop has
  halted after 5 consecutive throws.

- If you're driving the backend directly (`createPTEngine_WebGPU`,
  `createWalkaroundEngine_Hybrid`), call `device.popErrorScope()` manually, or
  enable `device.addEventListener('uncapturederror', …)` at the device level.

**Why this is first:** the three actual total-blackout bugs fixed in this repo (F1
full-tier bind-group crash, F1b SPPM min-binding-size, F3 DDGI sampler strip) all
produced GPU validation errors.  The shader compile gate and Vitest suite were
completely blind to them.  The behavioral gate (`npm run behavioral-gate`) runs
25 configs and hard-fails on a non-zero GPU error count.

---

## Check 1 — No scene set

**Symptom:** Canvas stays black from frame 1; no console errors.

**Cause:** `engine.renderFrame()` was called before `engine.setScene(scene)`.

- `createEngine` calls `setScene` automatically from the options object —
  you must pass `scene` to `createEngineOptions`.
- `createPTEngine_WebGPU` and `createPTEngine_WebGL2` do **not** set a scene at
  construction.  Call `await engine.setScene(scene)` before the first `renderFrame`.
- `attachVitrum` calls `createEngine` which sets the scene; no extra call needed.
- `<VitrumCanvas>` also handles this automatically.

**Check:** Add a temporary `console.log(engine.getScene?.())` before the render
loop starts.  If it returns `undefined`, the scene was not set.

**Fix:**
```ts
const engine = await createPTEngine_WebGPU({ device });
await engine.setScene(scene);          // ← required before renderFrame
requestAnimationFrame(tick);
```

---

## Check 2 — `swapChainView` missing on the walkaround-hybrid backend

**Symptom:** `engine.capabilities.presentationMode === 'swapchain-required'` but
every `renderFrame` returns `{ kind: 'skipped' }`.  Canvas stays black.

**Cause:** `HybridEngine.renderFrame` silently returns `kind:'skipped'` when
`input.swapChainView` is absent (HybridEngine.ts, ~line 2130).  The walkaround-hybrid
backend writes directly into the swap-chain texture; without a fresh view it has
nowhere to write.

**Check:** Gate on `capabilities.presentationMode`:
```ts
if (engine.capabilities.presentationMode === 'swapchain-required') {
  // MUST provide swapChainView every frame.
}
```

**Fix:** Acquire a fresh `GPUTextureView` inside each RAF tick:
```ts
const ctx = canvas.getContext('webgpu')!;
function tick(): void {
  const view = ctx.getCurrentTexture().createView();   // fresh every frame
  engine.renderFrame({
    // …
    swapChainView: asBackendTexture<'webgpu', GPUTextureView>(view),
    swapChainFormat: asBackendTextureFormat<'webgpu', GPUTextureFormat>(format),
  });
  requestAnimationFrame(tick);
}
```

Do **not** cache the view across frames — the WebGPU spec treats a texture view as
single-use per frame and caching it produces a validation error.

`attachVitrum` handles this automatically via `detectWebGPUSwapChain` +
`acquireSwapChainView` in `lifecycle/vanilla.ts`.

---

## Check 3 — `setSize` not called on walkaround-hybrid after resize

**Symptom:** Renders correctly at the initial window size, goes black (or
stretches / crops incorrectly) after the browser window is resized.

**Cause:** Unlike the pt-webgl2 / pt-webgpu backends (which honour
`FrameInput.viewport` per-frame), `HybridEngine` sizes its render targets
(DDGI atlas, ReSTIR reservoirs, accumulation buffer) at construction time.
Pushing a new `viewport` via `FrameInput` is silently ignored.  When the
canvas backing-store grows larger than the construction size, the output
covers only the top-left corner.

**Check:** `engine.capabilities.presentationMode === 'swapchain-required'`
means the engine needs explicit resize notification.

**Fix:** Call `engine.setSize(w, h)` whenever the canvas dimensions change.
`attachVitrum` does this via `ResizeObserver`:
```ts
new ResizeObserver(([entry]) => {
  const { width, height } = entry.contentRect;
  // Update backing store
  canvas.width  = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  engine.setSize?.(canvas.width, canvas.height);
}).observe(canvas);
```

---

## Check 4 — lite-tier silent feature absence

**Symptom:** Switching `createEngine` to `prefer:'realtime'` with an adapter that
falls into the lite tier (mobile iGPU, software renderer) produces black or
noticeably dimmer output; a warning appears in the console.

**Cause:** The lite-tier binding budget omits several passes (RC, PPG, NRC,
neural denoiser).  Before H15 / H16, the tier gate was silent.  Since those
fixes the engine logs a warning when a feature is requested that the lite tier
cannot support, e.g.:

```
[vitrum/walkaround-hybrid] tier:'lite' forbids rcEnabled; ignoring
```

The render is not black in the pure-lite case, but may appear dim if the host
was relying on RC for indirect lighting.

**Check:** Look for `tier:'lite'` warnings in the console.  Use
`probeAdapterProfile` to read the resolved tier before constructing the engine:
```ts
import { probeAdapterProfile } from '@vitrum/engine';
const profile = await probeAdapterProfile();
if (profile.realtimeTier === 'lite') {
  // Offer a quality caveat or fall back to pt-webgl2.
}
```

---

## Check 5 — NaN-poisoned environment

**Symptom:** pt-webgpu or pt-webgl2 renders black or produces `NaN` in the
pixel readback; the behavioral gate reports "NaN pixels".

**Cause:** An all-zero or infinite normal being passed to `normalize(vec3(0))`
in the neural pack pass (sky pixels with no surface hit) produced NaN in the
output buffer.  This guard was added in the neuralPack NaN-guard fix (plan
item 12 / H46-A area).

Also check: `FrameInput.cameraPosition` NaN or Inf, an `environment` whose
`intensity` is `NaN` / `Infinity`, or a material with a `baseColor` component
outside `[0, 1]` (values outside range are clamped by the engine but NaN is
not).

**Check:** Add a temporary assertion before the render call:
```ts
console.assert(isFinite(cameraPosition[0]), 'camera NaN');
console.assert(isFinite(scene.environment?.intensity ?? 0), 'env NaN');
```

The behavioral gate checks `isNaN(pixel)` on the readback for every config —
run `npm run behavioral-gate` to confirm a NaN-clean output.

---

## Check 6 — WebGL2 context lost

**Symptom:** pt-webgl2 renders correctly for a while then goes black; the
console emits `WebGL context lost` (or the browser DevTools shows a red X on
the canvas).

**Cause:** The browser evicts the WebGL context when GPU memory pressure is
high, the tab is backgrounded, or the GPU driver resets.  The pt-webgl2 backend
does not automatically recover from a context loss.

**Check:**
```ts
canvas.addEventListener('webglcontextlost', (e) => {
  e.preventDefault();   // required to allow restore
  console.warn('[vitrum] WebGL2 context lost');
});
canvas.addEventListener('webglcontextrestored', () => {
  console.warn('[vitrum] WebGL2 context restored — recreate the engine');
  // Dispose the old engine and construct a new one.
});
```

**Fix:** Dispose the engine and call `createPTEngine_WebGL2` again once the
context is restored.  The `attachVitrum` helper does not currently auto-recover
from context loss; the host must handle it.

---

## Check 7 — Device limit exceeded (SPPM / photon-map)

**Symptom:** `causticStrategy:'photon-map'` renders black with a GPU validation
error in the console (`Buffer binding size … exceeds maxStorageBufferBindingSize`
or similar).

**Cause:** The SPPM photon-map variant allocates per-photon buffers sized from
`maxSamplesPerPixel` and the scene's triangle count.  On low-end adapters the
required buffer size exceeds `device.limits.maxStorageBufferBindingSize` (often
128 MB on mobile / integrated).

**Check:** Look for GPU errors from `device.popErrorScope()` or the
`behavioral-gate` report (`pt/caustic-photon` is currently a `known-residual`
in the gate expectation table).

**Fix:** Either reduce `maxSamplesPerPixel` / scene complexity, or fall back to
`causticStrategy:'manifold-nee'` (MNEE is on by default and has no extra
buffer budget):
```ts
const engine = await createPTEngine_WebGPU({
  device,
  causticStrategy: adapter.isMobile ? 'manifold-nee' : 'photon-map',
});
```

Use `probeAdapterProfile()` to read `profile.maxStorageBufferBindingSize` and
decide at runtime.

---

## Quick-reference symptom table

| Symptom | Most likely cause | Check |
|---------|-------------------|-------|
| Black from frame 1, no errors | No scene set | Check 1 |
| `kind:'skipped'` every frame on WebGPU | `swapChainView` missing | Check 2 |
| Correct at load, black after resize | `setSize` not called | Check 3 |
| Dim / missing indirect on mobile | lite-tier feature absence | Check 4 |
| NaN in pixel readback | NaN-poisoned env or normals | Check 5 |
| Black after tab background | WebGL2 context lost | Check 6 |
| Black + GPU errors on SPPM | Device limit exceeded | Check 7 |
| Black + GPU errors on any config | Validation error in pipeline | Check 0 |

---

## Running the behavioral gate

The behavioral gate (`tools/behavioral-gate/`) boots the real engines on
lavapipe and asserts non-black + zero GPU errors for 25 configs.  It is the
fastest way to confirm that a local change has not introduced a new black-frame
regression across the config matrix:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run behavioral-gate
```

See [`tools/behavioral-gate/README.md`](../tools/behavioral-gate/README.md) for
the full config list, assertion criteria, and the expectation table.
