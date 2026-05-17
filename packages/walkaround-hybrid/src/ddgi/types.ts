/**
 * DDGI internal types — re-exports DDGILight from @vitrum/three-bindings
 * (where it lives alongside its THREE.RectAreaLight extractor) and defines
 * the local-only DDGIDeviceHandle.
 *
 * DDGILight was relocated to three-bindings in W7-H4 because its sole
 * producer (`collectDDGILightsFromRectAreaLights`) is THREE-only light
 * extraction and belongs in the three-bindings adapter layer. The type
 * shape is still walkaround-domain-tagged (kinds 'sun' | 'fixture' |
 * 'teaLight') and is re-exported here so existing imports inside the
 * walkaround-hybrid package don't need to retarget.
 */

export type { DDGILight } from '@vitrum/three-bindings';

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
