// Single source of truth for the set of walkaround engines. Used by
// StudioScene (component dispatch) and RasterStage (suppression gate).
//
// Sweep finding Bug 9: RasterStage previously gated only on
// `walkaroundEngine === 'rc'` so users on hybrid/ddgi/restir who hit a
// transient mount of RasterStage under WebGPURenderer could see
// GizmoHelper crash on `gl.capabilities.getMaxAnisotropy()` (the WebGL
// capabilities object doesn't exist on WebGPURenderer). One source of
// truth here so adding a new engine doesn't require remembering to
// extend that gate.

import type { WalkaroundEngine } from '@/store/viewportSlice';

export const WALKAROUND_ENGINE_KEYS = ['hybrid', 'ddgi', 'rc', 'restir'] as const;

// Compile-time guard: if someone adds a new engine to the WalkaroundEngine
// union without extending this array, the assignment fails to type-check.
const _engineKeysExhaustive: readonly WalkaroundEngine[] = WALKAROUND_ENGINE_KEYS;
void _engineKeysExhaustive;

export const WALKAROUND_ENGINE_SET: ReadonlySet<WalkaroundEngine> = new Set<WalkaroundEngine>(WALKAROUND_ENGINE_KEYS);

/** True when the active walkaround configuration is one of the known engines. */
export function isWalkaroundActive(exploreEnabled: boolean, walkaroundEngine: WalkaroundEngine): boolean {
  return exploreEnabled && WALKAROUND_ENGINE_SET.has(walkaroundEngine);
}
