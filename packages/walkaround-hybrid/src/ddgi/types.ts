/**
 * DDGILight — the light-source shape consumed by ProbeUpdatePass.
 *
 * This type is intentionally NOT @vitrum/core's SceneEmitter. The DDGI
 * probe-update pass consumes a host-domain projection of lights that is
 * specifically tagged with the runtime kinds the walkaround pipeline
 * recognises ('sun' | 'fixture' | 'teaLight'). SceneEmitter uses an
 * abstract emitter taxonomy ('directional' | 'point' | 'spot' | …) that
 * the core contract defines independently of any specific renderer's
 * conventions. Keeping DDGILight separate lets the host bridge map its
 * domain objects without polluting the core contract with walkaround-
 * specific kind strings.
 *
 * Fields verified against probeUpdatePass.ts _uploadLights (the only
 * consumer in this package):
 *   - kind      — switched on: 'sun' | 'fixture' | 'teaLight'
 *   - intensity — multiplied by _sunIntensityMul for sun lights
 *   - position  — accessed via unsafe cast on fixture/teaLight lights
 *   - on        — filter: only lights where on===true are uploaded
 */
export interface DDGILight {
  /** Runtime kind tag. Only 'sun', 'fixture', and 'teaLight' are handled;
   *  any other kind is silently skipped in _uploadLights. */
  readonly kind: 'sun' | 'fixture' | 'teaLight' | string;

  /** Photometric intensity value (arbitrary units; host normalises to
   *  whatever scale the renderer expects). */
  readonly intensity: number;

  /** Whether this light is active. ProbeUpdatePass filters to only
   *  lights where on === true before uploading to the GPU UBO. */
  readonly on: boolean;

  /** World-space position. Only used for 'fixture' and 'teaLight' kinds.
   *  Accessed via an unsafe cast in the original source because LightSource
   *  did not expose position on the base type — DDGILight makes it explicit
   *  and optional (sun lights have no meaningful position). */
  readonly position?: { readonly x: number; readonly y: number; readonly z: number };
}

/**
 * Minimal device handle accepted by DDGI.updateFrame. The probe-update
 * pass needs a `GPUDevice` to bind compute pipelines; previously DDGI's
 * input shape required a Three.js renderer adapter (`{ backend: { device,
 * isWebGPUBackend } }`), forcing HybridEngine to synthesise an empty fake.
 * This shape removes that coupling — DDGI no longer cares whether the
 * device came from a Three.js renderer or directly from `navigator.gpu`.
 *
 * `renderer` is still accepted (and forwarded to ProbeUpdatePass.init for
 * the legacy three.js-backend lazy-init path), but it is optional. When
 * absent, ProbeUpdatePass falls back to `navigator.gpu.requestAdapter`.
 */
export interface DDGIDeviceHandle {
  /** Raw WebGPU device (or null/undefined for the navigator.gpu fallback). */
  readonly device?: GPUDevice;
  /** Optional Three.js WebGPURenderer-shaped object (legacy path). */
  readonly renderer?: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } };
}
