/**
 * GI-state persistence (export/import) for HybridEngine.
 *
 * Extracted from `HybridEngine.ts` (R3 B-chain decomposition sweep, step 3).
 * Free functions taking explicit deps so they're testable without a full engine.
 */

import type { DDGI } from './ddgi/DDGI.js';
import type { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import type { GIStateSnapshot } from './giStateSnapshot.js';
import type { EngineWarning } from '@vitrum/core';

/** Minimal dep surface for the GI state helpers. */
export interface GIStateDeps {
  device: GPUDevice;
  ddgi: DDGI;
  pipeline: WalkaroundGPUPipeline | null;
  onWarning?: (warning: EngineWarning) => void;
}

/**
 * Export the converged DDGI global-illumination state (the "cached light
 * field") so the host can persist it (e.g. to IndexedDB via
 * {@link serializeGIState}) and restore it next session without re-converging.
 * Returns null if the probe atlases aren't allocated yet (call after the GI has
 * run at least one frame). Async (atlas readback uses mapAsync).
 */
export async function exportGIStateImpl(deps: GIStateDeps): Promise<GIStateSnapshot | null> {
  const atlas = await deps.ddgi.exportAtlasData(deps.device);
  if (!atlas) return null;
  const grid = deps.ddgi.gridParams;
  // Also snapshot the ReSTIR-GI temporal reservoirs when the pipeline is live,
  // so a restore continues the temporal+spatial GI reuse instead of dropping
  // the high-frequency indirect history and re-converging it from scratch.
  // (The RC subsystem carries no cross-frame state — it regenerates every
  // cascade from the BVH each frame — so there is nothing to persist for RC.)
  const restirGI = (await deps.pipeline?.exportRestirGIReservoirs(deps.device)) ?? undefined;
  // Also snapshot the PPG (Müller 2017) adaptive sTree / dTree guiding
  // distribution so a restore can resume guided sampling immediately from the
  // converged distribution instead of rebuilding the guide from cold. The PPG
  // section is OPTIONAL — a null return (PPG disabled or not yet initialised)
  // simply omits the section; importGIState treats its absence as a cold start.
  const ppgRaw = deps.pipeline?.exportPPGSTree() ?? null;
  const ppg = ppgRaw != null ? {
    maxSpatialCells: ppgRaw.maxSpatialCells,
    maxDTreeNodesPerCell: ppgRaw.maxDTreeNodesPerCell,
    sTreeBuf: ppgRaw.sTreeBuf,
    dTreeBuf: ppgRaw.dTreeBuf,
    dTreeOffsets: ppgRaw.dTreeOffsets,
    sceneBoundsMin: ppgRaw.sceneBoundsMin,
    sceneBoundsMax: ppgRaw.sceneBoundsMax,
  } : undefined;
  return {
    dims: { x: grid.dims.x, y: grid.dims.y, z: grid.dims.z },
    origin: [grid.origin.x, grid.origin.y, grid.origin.z],
    spacing: grid.spacing,
    ...atlas,
    ...(restirGI ? { restirGI } : {}),
    ...(ppg ? { ppg } : {}),
  };
}

/**
 * Restore a previously {@link exportGIStateImpl}-ed snapshot into the live GI state
 * (seeds the temporal blend, so rendering continues from it instead of
 * re-converging). Restores the DDGI probe atlases AND — when the snapshot
 * carries them — the ReSTIR-GI temporal reservoirs (v2+) and the PPG
 * sTree/dTree guiding distribution (v4+).
 *
 * Returns false (no-op) if the atlases aren't allocated or the snapshot's atlas
 * dims don't match the current grid. When a reservoir section is present, the
 * restore also fails (returns false) if the reservoir grid/size doesn't match
 * the live pipeline — so a partial (atlas-only) restore is never silently
 * reported as a full success. A v3 snapshot (no PPG section) restores the
 * atlases + reservoirs and returns the atlas+reservoir result unchanged; PPG
 * starts cold without error.
 */
export function importGIStateImpl(deps: GIStateDeps, snapshot: GIStateSnapshot): boolean {
  // Validate grid origin, spacing, and dims before touching GPU buffers.
  // Two scenes can have identical probe-atlas pixel dimensions but different
  // grid origin/spacing/dims — restoring into such a mismatched grid would
  // corrupt the GI with probes from the wrong world-space layout. The atlas
  // dim check in importAtlasData is necessary but not sufficient.
  const grid = deps.ddgi.gridParams;
  const epsilon = 1e-4;
  const dimsMismatch =
    snapshot.dims.x !== grid.dims.x ||
    snapshot.dims.y !== grid.dims.y ||
    snapshot.dims.z !== grid.dims.z;
  const originMismatch =
    Math.abs(snapshot.origin[0] - grid.origin.x) > epsilon ||
    Math.abs(snapshot.origin[1] - grid.origin.y) > epsilon ||
    Math.abs(snapshot.origin[2] - grid.origin.z) > epsilon;
  const spacingMismatch = Math.abs(snapshot.spacing - grid.spacing) > epsilon;
  if (dimsMismatch || originMismatch || spacingMismatch) {
    warnGIState(deps, {
      code: 'walkaround-hybrid.import-gi-state-grid-mismatch',
      backend: 'walkaround-hybrid',
      phase: 'lifecycle',
      method: 'importGIState',
      message:
        '[HybridEngine] importGIState: snapshot grid layout does not match the current grid ' +
        '(dims/origin/spacing mismatch) — restore rejected to avoid garbage GI.',
      details: {
        snapshot: {
          dims: snapshot.dims,
          origin: snapshot.origin,
          spacing: snapshot.spacing,
        },
        current: {
          dims: grid.dims,
          origin: [grid.origin.x, grid.origin.y, grid.origin.z],
          spacing: grid.spacing,
        },
      },
    });
    return false;
  }
  const atlasOk = deps.ddgi.importAtlasData(deps.device, snapshot);
  if (!atlasOk) return false;
  if (snapshot.restirGI == null) {
    // v3 (or earlier) / no reservoir section — atlas-only restore.
    // PPG section absent at this point means cold start; not a failure.
    if (snapshot.ppg != null) {
      // Best-effort: try to restore the PPG guide even without ReSTIR-GI.
      deps.pipeline?.importPPGSTree(snapshot.ppg);
    }
    return true;
  }
  // A reservoir section is present: require it to restore too, else report
  // failure rather than a misleadingly-partial success.
  const reservoirOk = deps.pipeline?.importRestirGIReservoirs(deps.device, snapshot.restirGI) ?? false;
  if (!reservoirOk) return false;
  // PPG section (v4+): restore is best-effort — a PPG mismatch (different
  // maxSpatialCells or scene bounds) causes a warm log + cold restart rather
  // than failing the whole importGIState call. The DDGI probes and ReSTIR-GI
  // reservoirs are already restored at this point; losing only the PPG guide
  // is not a correctness failure (guided sampling falls back to cosine until
  // the next training window converges).
  if (snapshot.ppg != null) {
    deps.pipeline?.importPPGSTree(snapshot.ppg);
  }
  return true;
}

function warnGIState(deps: GIStateDeps, warning: EngineWarning): void {
  if (deps.onWarning) {
    try {
      deps.onWarning(warning);
    } catch {
      // Host warning callbacks must not break GI-state import.
    }
    return;
  }
  console.warn(warning.message);
}
