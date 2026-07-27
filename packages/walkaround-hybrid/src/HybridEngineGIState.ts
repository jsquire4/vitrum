/**
 * GI-state persistence (export/import) for HybridEngine.
 *
 * Extracted from `HybridEngine.ts` (R3 B-chain decomposition sweep, step 3).
 * Free functions taking explicit deps so they're testable without a full engine.
 */

import type { DDGI } from './ddgi/DDGI.js';
import type { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import {
  f32SnapshotMetadataMatches,
  isFiniteRgba16Data,
  isValidRequiredGIStateSnapshot,
  type GIStateSnapshot,
} from './giStateSnapshot.js';
import { isValidProbeStateData } from './ddgi/probeState.js';
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
  if (!isValidRequiredGIStateSnapshot(snapshot)) {
    warnMalformedSnapshot(deps);
    return false;
  }
  // Validate grid origin, spacing, and dims before touching GPU buffers.
  // Two scenes can have identical probe-atlas pixel dimensions but different
  // grid origin/spacing/dims — restoring into such a mismatched grid would
  // corrupt the GI with probes from the wrong world-space layout. The atlas
  // dim check in importAtlasData is necessary but not sufficient.
  const grid = deps.ddgi.gridParams;
  const dimsMismatch =
    !isPositiveSafeInteger(snapshot.dims.x) ||
    !isPositiveSafeInteger(snapshot.dims.y) ||
    !isPositiveSafeInteger(snapshot.dims.z) ||
    snapshot.dims.x !== grid.dims.x ||
    snapshot.dims.y !== grid.dims.y ||
    snapshot.dims.z !== grid.dims.z;
  const originMismatch =
    !isFiniteVec3(snapshot.origin) ||
    !f32SnapshotMetadataMatches(snapshot.origin[0], grid.origin.x) ||
    !f32SnapshotMetadataMatches(snapshot.origin[1], grid.origin.y) ||
    !f32SnapshotMetadataMatches(snapshot.origin[2], grid.origin.z);
  const spacingMismatch =
    !Number.isFinite(snapshot.spacing) ||
    !(snapshot.spacing > 0) ||
    !f32SnapshotMetadataMatches(snapshot.spacing, grid.spacing);
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
  if (!canImportDDGIAtlas(deps.ddgi, snapshot)) {
    warnAtlasRejected(deps, snapshot);
    return false;
  }

  const reservoirTransaction = snapshot.restirGI == null
    ? null
    : prepareRequiredReservoirRestore(deps, snapshot);
  if (snapshot.restirGI != null && reservoirTransaction == null) {
    warnReservoirRejected(deps);
    return false;
  }

  try {
    const atlasOk = deps.ddgi.importAtlasData(deps.device, snapshot);
    if (!atlasOk) {
      reservoirTransaction?.abort();
      warnAtlasRejected(deps, snapshot);
      return false;
    }
    reservoirTransaction?.commit();
  } catch (error) {
    reservoirTransaction?.abort();
    throw error;
  }
  if (snapshot.restirGI == null) {
    // v3 (or earlier) / no reservoir section — atlas-only restore.
    // PPG section absent at this point means cold start; not a failure.
    if (snapshot.ppg != null) {
      // Best-effort: try to restore the PPG guide even without ReSTIR-GI.
      restorePPGStateBestEffort(deps, snapshot);
    }
    return true;
  }
  // PPG section (v4+): restore is best-effort. A PPG mismatch (different
  // maxSpatialCells or scene bounds) leaves the current guide unchanged, or
  // remains cold when no guide exists, rather than failing the whole
  // importGIState call. DDGI probes and ReSTIR-GI are already restored.
  if (snapshot.ppg != null) {
    restorePPGStateBestEffort(deps, snapshot);
  }
  return true;
}

function prepareRequiredReservoirRestore(
  deps: GIStateDeps,
  snapshot: GIStateSnapshot,
): ReturnType<WalkaroundGPUPipeline['prepareRestirGIReservoirImport']> {
  const reservoir = snapshot.restirGI;
  const pipeline = deps.pipeline;
  if (
    reservoir == null ||
    pipeline == null ||
    !pipeline.canImportRestirGIReservoirs(reservoir)
  ) {
    return null;
  }
  return pipeline.prepareRestirGIReservoirImport(deps.device, reservoir);
}

function canImportDDGIAtlas(ddgi: DDGI, snapshot: GIStateSnapshot): boolean {
  const grid = ddgi.gridParams;
  if (
    ddgi.getReadAtlasGPUTextures() == null ||
    !isPositiveSafeInteger(snapshot.irrW) ||
    !isPositiveSafeInteger(snapshot.irrH) ||
    !isPositiveSafeInteger(snapshot.visW) ||
    !isPositiveSafeInteger(snapshot.visH) ||
    snapshot.irrW !== grid.irradianceAtlasW ||
    snapshot.irrH !== grid.irradianceAtlasH ||
    snapshot.visW !== grid.visibilityAtlasW ||
    snapshot.visH !== grid.visibilityAtlasH ||
    snapshot.probeStateW !== grid.dims.x ||
    snapshot.probeStateH !== grid.dims.y * grid.dims.z ||
    !(snapshot.irrData instanceof Uint16Array) ||
    !(snapshot.visData instanceof Uint16Array) ||
    !(snapshot.probeStateData instanceof Float32Array) ||
    !rgbaLengthMatches(snapshot.irrW, snapshot.irrH, snapshot.irrData.length) ||
    !rgbaLengthMatches(snapshot.visW, snapshot.visH, snapshot.visData.length) ||
    !rgbaLengthMatches(
      snapshot.probeStateW,
      snapshot.probeStateH,
      snapshot.probeStateData.length,
    ) ||
    !isFiniteRgba16Data(snapshot.irrData) ||
    !isFiniteRgba16Data(snapshot.visData) ||
    !isValidProbeStateData(snapshot.probeStateData, grid.spacing)
  ) {
    return false;
  }
  return true;
}

function rgbaLengthMatches(width: number, height: number, length: number): boolean {
  if (!isPositiveSafeInteger(width) || !isPositiveSafeInteger(height)) {
    return false;
  }
  const texels = width * height;
  return (
    Number.isSafeInteger(texels) &&
    texels <= Number.MAX_SAFE_INTEGER / 4 &&
    length === texels * 4
  );
}

function isPositiveSafeInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function isFiniteVec3(value: unknown): value is readonly [number, number, number] {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    Number.isFinite(value[0]) &&
    Number.isFinite(value[1]) &&
    Number.isFinite(value[2])
  );
}

function warnAtlasRejected(
  deps: GIStateDeps,
  snapshot: GIStateSnapshot,
): void {
  warnGIState(deps, {
    code: 'walkaround-hybrid.import-gi-state-atlas-rejected',
    backend: 'walkaround-hybrid',
    phase: 'lifecycle',
    method: 'importGIState',
    message:
      '[HybridEngine] importGIState: DDGI atlas restore failed — GI state restore rejected and GI will cold-start.',
    details: {
      fallback: 'cold-start-gi',
      snapshot: {
        irradianceAtlas: [snapshot.irrW, snapshot.irrH],
        visibilityAtlas: [snapshot.visW, snapshot.visH],
      },
    },
  });
}

function warnMalformedSnapshot(deps: GIStateDeps): void {
  warnGIState(deps, {
    code: 'walkaround-hybrid.import-gi-state-malformed-snapshot',
    backend: 'walkaround-hybrid',
    phase: 'lifecycle',
    method: 'importGIState',
    message:
      '[HybridEngine] importGIState: malformed snapshot payload rejected before live GI state mutation.',
    details: {
      fallback: 'retain-current-gi',
    },
  });
}

function warnReservoirRejected(deps: GIStateDeps): void {
  warnGIState(deps, {
    code: 'walkaround-hybrid.import-gi-state-restir-reservoir-rejected',
    backend: 'walkaround-hybrid',
    phase: 'lifecycle',
    method: 'importGIState',
    message:
      '[HybridEngine] importGIState: ReSTIR-GI reservoir restore failed — GI state restore rejected to avoid a partial restore.',
    details: {
      fallback: 'cold-start-gi',
      hasPipeline: deps.pipeline != null,
    },
  });
}

function restorePPGStateBestEffort(deps: GIStateDeps, snapshot: GIStateSnapshot): void {
  if (snapshot.ppg == null) return;
  let ppgOk = false;
  let failure: unknown;
  try {
    ppgOk = deps.pipeline?.importPPGSTree(snapshot.ppg) ?? false;
  } catch (error) {
    failure = error;
  }
  if (ppgOk) return;
  warnGIState(deps, {
    code: 'walkaround-hybrid.import-gi-state-ppg-guide-rejected',
    backend: 'walkaround-hybrid',
    phase: 'lifecycle',
    method: 'importGIState',
    message:
      '[HybridEngine] importGIState: PPG guide restore failed — DDGI/ReSTIR-GI restore continues while retaining the current PPG guide (or existing cold state).',
    details: {
      fallback: 'retain-current-ppg',
      hasPipeline: deps.pipeline != null,
      maxSpatialCells: snapshot.ppg.maxSpatialCells,
      maxDTreeNodesPerCell: snapshot.ppg.maxDTreeNodesPerCell,
      ...(failure == null
        ? {}
        : {
            failure:
              failure instanceof Error
                ? failure.message
                : typeof failure === 'string'
                  ? failure
                  : 'non-Error exception',
          }),
    },
  });
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
