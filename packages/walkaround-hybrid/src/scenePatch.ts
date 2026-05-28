/**
 * Pure helpers to patch `@vitrum/core` `Scene` snapshots in sync with
 * THREE mesh edits during incremental engine updates.
 *
 * The canonical snapshot-patch + invariant logic now lives in `@vitrum/core`
 * (theme T2 dedup) — these are thin aliases that preserve the historical
 * walkaround-hybrid names so existing call sites keep working. The core
 * variant is the strict superset (it additionally validates analytic params
 * and rejects analytic `params` on mesh-like primitives), closing the drift
 * where this backend previously skipped those checks.
 */

import {
  patchEmitterInScene,
  patchPrimitiveInScene,
} from '@vitrum/core';

export const applyPrimitivePatchToScene = patchPrimitiveInScene;
export const applyEmitterPatchToScene = patchEmitterInScene;
