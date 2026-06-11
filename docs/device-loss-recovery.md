# Device-loss recovery

A GPU device loss (WebGPU `GPUDevice.lost`) or WebGL context loss (`webglcontextlost`)
makes all engine-owned GPU resources invalid. This document describes how vitrum
detects the event, what state survives, and the correct host sequence to restore
rendering.

---

## Detection

Every shipping backend wires its own platform event and surfaces it as an
[`EngineError`][engine-error] delivered through `engine.onError`:

| Backend | Platform event | `EngineError.kind` |
|---------|---------------|-------------------|
| `walkaround-hybrid` | `device.lost.then(…)` | `'device-lost'` |
| `pt-webgpu` | `device.lost.then(…)` | `'device-lost'` |
| `pt-webgl2` | `webglcontextlost` canvas event | `'context-lost'` |

Both `'device-lost'` and `'context-lost'` set `fatal: true`, which also
transitions `engine.state` to `'error'`. Subsequent `renderFrame` calls return
`{ kind: 'skipped' }` — no GPU work is submitted.

`attachVitrum` forwards fatal engine errors to the host's `onEngineError` callback:

```ts
attachVitrum({
  canvas, scene, camera,
  onEngineError(err) {
    if (err.fatal) {
      // engine.state is now 'error'; the RAF loop continues but renders nothing.
      // Host should dispose and recreate.
      console.error('[vitrum] fatal engine error:', err.kind, err.message);
    }
  },
});
```

[engine-error]: ../packages/core/src/engine/telemetry.ts

---

## What state survives vs. what does not

| State | Survives? | Notes |
|-------|-----------|-------|
| `Scene` (geometry, materials, emitters) | YES — it is host-side CPU data | Call `setScene(scene)` after recreating |
| Camera matrices / viewport | YES — re-pushed every frame | No action needed |
| `CreateEngineOptions` / `AttachVitrumOptions` | YES — host-owned | Retained by `attachVitrum`; used for auto-recreate |
| DDGI probe atlases / GI state | NO — GPU textures are destroyed | Export BEFORE device loss via `exportGIState()`, import AFTER recreating |
| PT accumulation buffer (SPP history) | NO — GPU textures are destroyed | Accumulation restarts from sample 0 |
| ReSTIR temporal reservoirs | NO | Rebuilt automatically over a few frames |
| Frame index / `prevView` / `prevProj` | NO | Reset to 0 / undefined on recreate |

---

## Recreate sequence

### `createEngine` path (most hosts)

```ts
import { createEngine, attachVitrum } from '@vitrum/engine';

let handle = await attachVitrum({ canvas, scene, camera, onEngineError });

async function onEngineError(err) {
  if (!err.fatal) return;                      // non-fatal: informational only

  // 1. Export GI state BEFORE dispose (walkaround backend only; no-op on PT)
  const giState = await handle.engine.exportGIState?.() ?? null;

  // 2. Dispose the broken engine and stop the RAF loop.
  handle.dispose();

  // 3. Recreate — createEngine mints a fresh device automatically.
  handle = await attachVitrum({ canvas, scene, camera, onEngineError });

  // 4. Restore warm GI (if available and the engine supports it).
  if (giState) handle.engine.importGIState?.(giState);
}
```

**Note on accumulation:** the PT accumulation buffer does not survive device
loss. After recreating, `engine.state` starts at `'ready'` with
`samplesAccumulated = 0`. There is no seed mechanism that can safely survive
across device boundaries because the source texture lives on the destroyed device.

### Raw backend factory path

For hosts that construct `createWalkaroundEngine_Hybrid` / `createPTEngine_WebGPU`
directly rather than via `createEngine`:

```ts
// 1. Wire onError before the first frame.
const unsub = engine.onError?.((err) => {
  if (!err.fatal) return;
  unsub?.();
  handleLoss(err);
});

async function handleLoss(_err) {
  // 2. Dispose the broken engine (frees all GPU resources against the lost device).
  engine.dispose();

  // 3. Re-negotiate a fresh device — the old device is invalid.
  const { device: newDevice } = await negotiateWebGPUDevice({ target: 'walkaround-hybrid' });

  // 4. Reconstruct the engine against the new device.
  engine = await createWalkaroundEngine_Hybrid({ device: newDevice, /* … */ });
  engine.setScene(scene);
}
```

---

## Auto-recreate via `attachVitrum`

`attachVitrum` supports automatic device-loss recovery via the
`autoRecreateOnDeviceLoss` option (default `false`).

When enabled, on any fatal `EngineError` with `kind: 'device-lost'` or
`kind: 'context-lost'`:

1. `onEngineError` is called first — the host sees the event regardless.
2. The RAF loop is stopped.
3. If the engine exposes `exportGIState` (walkaround-hybrid), the GI state is
   exported.
4. The engine is disposed.
5. `createEngine` is called again with the same options.
6. If GI state was exported and the new engine exposes `importGIState`, it is
   imported (warm DDGI probes survive the recreate).
7. The RAF loop resumes.

Retries are capped: at most **2 recreates within a 30-second window**. After
the cap is exceeded, the final `onEngineError` is delivered and the loop stays
stopped.

```ts
const handle = await attachVitrum({
  canvas, scene, camera,
  autoRecreateOnDeviceLoss: true,
  onEngineError(err) {
    // Called for EVERY error — before any auto-recreate, and for the final
    // error if the retry cap is exceeded.
    console.warn('[vitrum] engine error:', err.kind, err.message, 'fatal:', err.fatal);
  },
});
```

`VitrumCanvas` accepts the same prop:

```tsx
<VitrumCanvas
  scene={scene}
  camera={camera}
  autoRecreateOnDeviceLoss
  onEngineError={(err) => console.warn(err)}
/>
```

---

## Browser triggers

| Trigger | Backend | Notes |
|---------|---------|-------|
| GPU process crash / driver hang | All WebGPU | Chrome: GPU process restart → `device.lost` on all live devices |
| Tab backgrounded (some drivers) | walkaround-hybrid | Rare; depends on driver and OS power management policy |
| `device.destroy()` called by host | All WebGPU | Intentional teardown; also fires `device.lost` |
| Canvas removed from DOM + GC | pt-webgl2 | Browser may issue `webglcontextlost` if the GL context is GC'd while an engine holds it |
| Mobile foreground/background cycle | All | Some Android drivers lose the GPU context on app backgrounding |

---

## `negotiateWebGPUDevice` and device loss

`negotiateWebGPUDevice` is a **one-shot convenience** — it acquires an adapter,
requests a device, and returns both. It registers **no `device.lost` handler**
and holds **no reference** to the returned device. The host owns the device
lifecycle and must handle `device.lost` (or use `autoRecreateOnDeviceLoss` via
`attachVitrum`).

---

## Connection to the black-frames runbook

If you suspect a device loss is causing black frames, add `onEngineError` (or
check `engine.state`) as **Check 0** in the
[black-frames runbook](./debugging-black-frames.md) before debugging shader or
upload issues.

---

## Host-owns-lifecycle principle

vitrum's core design is **the host owns the device lifecycle**. The engine
accepts a device handle but does NOT create, track, or destroy it (except inside
`createEngine`, which owns the device it mints). A host that shares one
`GPUDevice` across subsystems (e.g. the progressive walkaround→PT facade) must
destroy the device AFTER disposing all engines built on it. Device loss does not
change this rule — the broken device is already functionally destroyed; the host
disposes the engine and then creates a fresh device.
