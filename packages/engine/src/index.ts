// @vitrum/engine — drop-in entry point.
//
// Re-exports the public Engine surface from @vitrum/core so hosts can take
// a single dependency for the common case.

export {
  createEngine,
  pickBackend,
  deriveScaleDefaults,
  mergeWalkaroundTlasExtension,
  type CreateEngineAdvancedByBackend,
  type CreateEngineBackendId,
  type CreateEngineErrorEvent,
  type CreateEngineErrorPhase,
  type CreateEngineOptions,
  type EngineWithBackendId,
  type EnginePreference,
  type RuntimeEngineBackendId,
  type RuntimeEngineWithBackendId,
  type ScaleDefaults,
} from './createEngine.js';
export { auditSceneNeedsTlas, type SceneTlasAudit } from '@vitrum/core';
export { computeSceneAABB, type SceneAABB } from './sceneAABB.js';

// Phase-0 productization — graceful-degradation adapter probe. The function
// lives here (it imports the real backend limit thresholds); the data shape
// (`AdapterProfile`) is re-exported from @vitrum/core for one-import ergonomics.
export { probeAdapterProfile } from './adapterProfile.js';
export type {
  AdapterProfile,
  RealtimeTier,
  HeroBackendRec,
  PtWebgpuTierRec,
} from '@vitrum/core';

// Lifecycle helpers — vanilla attachVitrum() is also available as a
// dedicated entrypoint via `@vitrum/engine/lifecycle`. React's
// <VitrumCanvas> ships under `@vitrum/engine/react`.
export {
  attachVitrum,
  type AttachVitrumOptions,
  type AttachVitrumHandle,
  // CameraLike: the structural camera interface for the RAF loop. A real
  // THREE.PerspectiveCamera / THREE.OrthographicCamera satisfies it structurally.
  // Exported so hosts can type their own camera adapters without redefining inline.
  type CameraLike,
  // D1.5 — named QualityOption type for the per-frame quality value-or-getter union.
  type QualityOption,
} from './lifecycle/vanilla.js';

// Raw WebGPU device negotiation — the lifecycle-layer peer of attachVitrum for
// hosts that drive the backend factories directly. Acquires a HOST-OWNED
// adapter + device + preferred-format + AdapterProfile; the host owns the
// device's lifecycle (must device.destroy() it). Reuses the same limit
// thresholds the backend factories apply — no hidden ownership.
export {
  negotiateWebGPUDevice,
  type NegotiateWebGPUDeviceOptions,
  type NegotiatedWebGPUDevice,
  type NegotiateTarget,
} from './negotiateWebGPUDevice.js';

// Convenience re-exports — hosts can `import { createEngine, type Engine } from '@vitrum/engine'`.
export type {
  Engine,
  EngineState,
  EngineCapabilities,
  Scene,
  ScenePrimitive,
  SceneEmitter,
  MaterialSpec,
  FrameInput,
  FrameOutput,
  Viewport,
} from '@vitrum/core';

// Progressive-refinement handoff: drive a real-time engine while the camera
// moves, hand off to a converged path tracer when it settles. Pure host-side
// coordination over two engines sharing one scene (P8 cross-engine frontier).
export {
  ProgressiveHandoffCoordinator,
  type ProgressiveHandoffOptions,
  type HandoffPhase,
  type HandoffFrameResult,
} from './progressiveHandoff.js';

// Progressive engine FACADE (P8 Track A increment 3): mints ONE shared GPUDevice
// and stands the realtime (walkaround-hybrid) + converged (pt-webgpu) engines up
// on it, then wires a ProgressiveHandoffCoordinator with seed-on-handoff. The
// shared device is what makes the realtime→PT seed handoff legal (cross-engine
// texture compatibility). `computeProgressiveLimitUnion` is exported for adapter
// preflight.
export {
  createProgressiveEngine,
  computeProgressiveLimitUnion,
  checkProgressiveLimitUnion,
  progressiveHandleAsEngine,
  type CreateProgressiveEngineOptions,
  type ProgressiveEngineHandle,
} from './createProgressiveEngine.js';

// Backend-specific lighting vocabulary re-export. `Engine.updateLighting` is
// contractually opaque in @vitrum/core (Readonly<Record<string, unknown>>),
// so hosts that drive HybridEngine's per-frame time-of-day scrub can import
// this concrete type for compile-time key checking without @vitrum/core baking
// a backend-specific shape into the universal contract.
export type { LightingOptions } from '@vitrum/walkaround-hybrid';

// GI-state persistence (cached light field) — the walkaround-hybrid backend's
// DDGI probe-atlas export/import, forwarded by the createEngine() facade so
// hosts can persist/restore converged GI without dropping to the concrete
// HybridEngine. `createEngine()` returns `Engine & Partial<GIStatePersistable>`
// (the methods are present only on the walkaround backend). `serializeGIState` /
// `deserializeGIState` convert a snapshot to/from a transferable ArrayBuffer.
export type { GIStatePersistable } from './idempotentDispose.js';
export {
  serializeGIState,
  deserializeGIState,
  type GIStateSnapshot,
} from '@vitrum/walkaround-hybrid';
