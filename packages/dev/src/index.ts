// @vitrum/dev — public façade.
//
// Dev-only debug overlay components. Add @vitrum/dev as a devDependency.
// Never import this package in production code.
//
// React surface:
//   import { FrameTimeHUD, DDGIAtlasViewer, ... } from '@vitrum/dev';
//
// Vanilla surface:
//   import { attachDebugOverlays } from '@vitrum/dev';
//
// Types:
//   import type { DebuggableEngine, FrameStats, ProgressStats } from '@vitrum/dev';

// ── Shared types ─────────────────────────────────────────────────────────────
export type {
  FrameStats,
  ProgressStats,
  EngineDebugSurface,
  DebuggableEngine,
} from './types.js';

// ── React components ─────────────────────────────────────────────────────────
export { FrameTimeHUD, RingBuffer } from './react/FrameTimeHUD.js';
export type { FrameTimeHUDProps } from './react/FrameTimeHUD.js';

export { DDGIAtlasViewer } from './react/DDGIAtlasViewer.js';
export type { DDGIAtlasViewerProps } from './react/DDGIAtlasViewer.js';

export { BVHVisualizer } from './react/BVHVisualizer.js';
export type { BVHVisualizerProps } from './react/BVHVisualizer.js';

export { GISignalSplit } from './react/GISignalSplit.js';
export type { GISignalSplitProps } from './react/GISignalSplit.js';

export { DenoiserABToggle } from './react/DenoiserABToggle.js';
export type { DenoiserABToggleProps } from './react/DenoiserABToggle.js';

export { MaterialInspector } from './react/MaterialInspector.js';
export type { MaterialInspectorProps } from './react/MaterialInspector.js';

// ── Vanilla surface ──────────────────────────────────────────────────────────
export { attachDebugOverlays } from './vanilla.js';
export type { AttachDebugOverlaysOptions, DebugOverlaysHandle } from './vanilla.js';
