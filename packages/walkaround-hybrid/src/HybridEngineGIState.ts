/**
 * Complete realtime-estimator persistence for HybridEngine.
 *
 * DDGI, ReSTIR-DI, ReSTIR-GI, PPG, and NRC form one estimator generation. A
 * restore therefore validates and prepares every live subsystem before
 * publishing any of them. Optional sections are optional only when the
 * corresponding subsystem is disabled in the live pipeline.
 */

import type { EngineWarning } from '@vitrum/core';
import type { DDGI } from './ddgi/DDGI.js';
import {
  f32SnapshotMetadataMatches,
  isFiniteRgba16Data,
  isValidRequiredGIStateSnapshot,
  type GIStateSnapshot,
} from './giStateSnapshot.js';
import { giStateCompatibilityMatches } from './giStateCompatibility.js';
import { isValidProbeStateData } from './ddgi/probeState.js';
import type { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';

interface GIImportTransaction {
  commit(): void;
  rollback(): void;
  finalize(): void;
}

/** Minimal dependency surface for complete GI-state persistence. */
export interface GIStateDeps {
  readonly device: GPUDevice;
  readonly ddgi: DDGI;
  readonly pipeline: WalkaroundGPUPipeline | null;
  /** Exact key for the currently published scene, lighting, and environment. */
  readonly compatibility: Uint32Array;
  readonly onWarning?: (warning: EngineWarning) => void;
}

/**
 * Capture one complete realtime-estimator generation.
 *
 * All GPU copies are enqueued synchronously in one JavaScript turn before the
 * first await, so queue ordering prevents a host frame from being submitted
 * between the individual subsystem snapshots.
 */
export async function exportGIStateImpl(
  deps: GIStateDeps,
): Promise<GIStateSnapshot | null> {
  const pipeline = deps.pipeline;
  if (pipeline == null) return null;

  const atlasPromise = deps.ddgi.exportAtlasData(deps.device);
  const restirGIPromise = pipeline.exportRestirGIReservoirs(deps.device);
  const restirDIPromise = pipeline.exportRestirDIReservoirs(deps.device);
  const nrcPromise = pipeline.nrcStateRequired
    ? pipeline.exportNrcLearnedState()
    : Promise.resolve(null);
  const ppgRaw = pipeline.ppgStateRequired ? pipeline.exportPPGSTree() : null;

  const [atlas, restirGI, restirDI, nrc] = await Promise.all([
    atlasPromise,
    restirGIPromise,
    restirDIPromise,
    nrcPromise,
  ]);
  if (
    atlas == null ||
    restirGI == null ||
    restirDI == null ||
    (pipeline.ppgStateRequired && ppgRaw == null) ||
    (pipeline.nrcStateRequired && nrc == null)
  ) {
    return null;
  }

  const grid = deps.ddgi.gridParams;
  const ppg = ppgRaw == null
    ? undefined
    : {
        maxSpatialCells: ppgRaw.maxSpatialCells,
        maxDTreeNodesPerCell: ppgRaw.maxDTreeNodesPerCell,
        sTreeBuf: ppgRaw.sTreeBuf,
        dTreeBuf: ppgRaw.dTreeBuf,
        dTreeOffsets: ppgRaw.dTreeOffsets,
        sceneBoundsMin: ppgRaw.sceneBoundsMin,
        sceneBoundsMax: ppgRaw.sceneBoundsMax,
      };
  return {
    dims: { x: grid.dims.x, y: grid.dims.y, z: grid.dims.z },
    origin: [grid.origin.x, grid.origin.y, grid.origin.z],
    spacing: grid.spacing,
    ...atlas,
    compatibility: deps.compatibility.slice(),
    restirGI,
    restirDI,
    ...(ppg != null ? { ppg } : {}),
    ...(nrc != null ? { nrc } : {}),
  };
}

/**
 * Atomically restore a complete realtime-estimator generation.
 *
 * The method is synchronous because candidate uploads use mapped-at-creation
 * resources or encode GPU copies without waiting for execution. Publication is
 * reversible until the shared NRC command buffer has been accepted by the
 * queue. Old resources are retired only after all participants commit.
 */
export function importGIStateImpl(
  deps: GIStateDeps,
  snapshot: GIStateSnapshot,
): boolean {
  if (!isValidRequiredGIStateSnapshot(snapshot)) {
    warnMalformedSnapshot(deps);
    return false;
  }
  if (!gridMetadataMatches(deps.ddgi, snapshot)) {
    warnGridMismatch(deps, snapshot);
    return false;
  }
  if (!giStateCompatibilityMatches(snapshot.compatibility, deps.compatibility)) {
    warnCompatibilityMismatch(deps);
    return false;
  }
  if (!canImportDDGIAtlas(deps.ddgi, snapshot)) {
    warnAtlasRejected(deps, snapshot);
    return false;
  }

  const pipeline = deps.pipeline;
  if (pipeline == null) {
    warnCompleteStateRejected(deps, 'pipeline-unavailable');
    return false;
  }
  if (
    snapshot.restirGI == null ||
    snapshot.restirDI == null ||
    !pipeline.canImportRestirGIReservoirs(snapshot.restirGI) ||
    !pipeline.canImportRestirDIReservoirs(snapshot.restirDI)
  ) {
    warnCompleteStateRejected(deps, 'reservoir-cohort-incompatible');
    return false;
  }
  const ppgPresenceMatches =
    pipeline.ppgStateRequired === (snapshot.ppg != null);
  const nrcPresenceMatches =
    pipeline.nrcStateRequired === (snapshot.nrc != null);
  if (!ppgPresenceMatches || !nrcPresenceMatches) {
    warnCompleteStateRejected(
      deps,
      !ppgPresenceMatches ? 'ppg-mode-mismatch' : 'nrc-mode-mismatch',
    );
    return false;
  }
  if (
    (snapshot.ppg != null && !pipeline.canImportPPGSTree(snapshot.ppg)) ||
    (snapshot.nrc != null && !pipeline.canImportNrcLearnedState(snapshot.nrc))
  ) {
    warnCompleteStateRejected(deps, 'adaptive-state-incompatible');
    return false;
  }

  const transactions: GIImportTransaction[] = [];
  let commandBuffer: GPUCommandBuffer | null = null;
  try {
    const atlas = deps.ddgi.prepareAtlasImport(deps.device, snapshot);
    if (atlas == null) {
      warnAtlasRejected(deps, snapshot);
      return false;
    }
    transactions.push(atlas);

    const restirGI = pipeline.prepareRestirGIReservoirImport(
      deps.device,
      snapshot.restirGI,
    );
    if (restirGI == null) {
      rollbackPrepared(transactions);
      warnCompleteStateRejected(deps, 'restir-gi-prepare-rejected');
      return false;
    }
    transactions.push(restirGI);

    const restirDI = pipeline.prepareRestirDIReservoirImport(
      deps.device,
      snapshot.restirDI,
    );
    if (restirDI == null) {
      rollbackPrepared(transactions);
      warnCompleteStateRejected(deps, 'restir-di-prepare-rejected');
      return false;
    }
    transactions.push(restirDI);

    if (snapshot.ppg != null) {
      const ppg = pipeline.preparePPGSTreeImport(snapshot.ppg);
      if (ppg == null) {
        rollbackPrepared(transactions);
        warnCompleteStateRejected(deps, 'ppg-prepare-rejected');
        return false;
      }
      transactions.push(ppg);
    }

    if (snapshot.nrc != null) {
      const encoder = deps.device.createCommandEncoder({
        label: 'hybrid-complete-gi-state-import',
      });
      const nrc = pipeline.prepareNrcLearnedStateImport(
        encoder,
        snapshot.nrc,
      );
      if (nrc == null) {
        rollbackPrepared(transactions);
        warnCompleteStateRejected(deps, 'nrc-prepare-rejected');
        return false;
      }
      transactions.push(nrc);
      // Finish before publication: encoder validation failure must leave every
      // currently-live estimator resource untouched.
      commandBuffer = encoder.finish();
    }
  } catch (error) {
    rollbackPreparedOrRethrow(transactions, error);
  }

  let committed = 0;
  try {
    for (; committed < transactions.length; committed += 1) {
      transactions[committed]!.commit();
    }
    if (commandBuffer != null) {
      deps.device.queue.submit([commandBuffer]);
    }
  } catch (error) {
    rollbackPreparedOrRethrow(transactions, error);
  }

  const finalizationErrors: unknown[] = [];
  for (const transaction of [...transactions].reverse()) {
    try {
      transaction.finalize();
    } catch (error) {
      finalizationErrors.push(error);
    }
  }
  if (finalizationErrors.length > 0) {
    warnGIState(deps, {
      code: 'walkaround-hybrid.import-gi-state-finalization-failed',
      backend: 'walkaround-hybrid',
      phase: 'lifecycle',
      method: 'importGIState',
      message:
        '[HybridEngine] importGIState: the complete state was restored, but one or more retired resources could not be released.',
      details: {
        committed: true,
        failureCount: finalizationErrors.length,
      },
    });
  }
  return true;
}

function gridMetadataMatches(ddgi: DDGI, snapshot: GIStateSnapshot): boolean {
  const grid = ddgi.gridParams;
  return (
    isPositiveSafeInteger(snapshot.dims.x) &&
    isPositiveSafeInteger(snapshot.dims.y) &&
    isPositiveSafeInteger(snapshot.dims.z) &&
    snapshot.dims.x === grid.dims.x &&
    snapshot.dims.y === grid.dims.y &&
    snapshot.dims.z === grid.dims.z &&
    isFiniteVec3(snapshot.origin) &&
    f32SnapshotMetadataMatches(snapshot.origin[0], grid.origin.x) &&
    f32SnapshotMetadataMatches(snapshot.origin[1], grid.origin.y) &&
    f32SnapshotMetadataMatches(snapshot.origin[2], grid.origin.z) &&
    Number.isFinite(snapshot.spacing) &&
    snapshot.spacing > 0 &&
    f32SnapshotMetadataMatches(snapshot.spacing, grid.spacing)
  );
}

function canImportDDGIAtlas(ddgi: DDGI, snapshot: GIStateSnapshot): boolean {
  const grid = ddgi.gridParams;
  return (
    ddgi.getReadAtlasGPUTextures() != null &&
    isPositiveSafeInteger(snapshot.irrW) &&
    isPositiveSafeInteger(snapshot.irrH) &&
    isPositiveSafeInteger(snapshot.visW) &&
    isPositiveSafeInteger(snapshot.visH) &&
    snapshot.irrW === grid.irradianceAtlasW &&
    snapshot.irrH === grid.irradianceAtlasH &&
    snapshot.visW === grid.visibilityAtlasW &&
    snapshot.visH === grid.visibilityAtlasH &&
    snapshot.probeStateW === grid.dims.x &&
    snapshot.probeStateH === grid.dims.y * grid.dims.z &&
    snapshot.irrData instanceof Uint16Array &&
    snapshot.visData instanceof Uint16Array &&
    snapshot.probeStateData instanceof Float32Array &&
    rgbaLengthMatches(snapshot.irrW, snapshot.irrH, snapshot.irrData.length) &&
    rgbaLengthMatches(snapshot.visW, snapshot.visH, snapshot.visData.length) &&
    rgbaLengthMatches(
      snapshot.probeStateW,
      snapshot.probeStateH,
      snapshot.probeStateData.length,
    ) &&
    isFiniteRgba16Data(snapshot.irrData) &&
    isFiniteRgba16Data(snapshot.visData) &&
    isValidProbeStateData(snapshot.probeStateData, grid.spacing)
  );
}

function rollbackPrepared(transactions: readonly GIImportTransaction[]): void {
  const errors: unknown[] = [];
  for (const transaction of [...transactions].reverse()) {
    try {
      transaction.rollback();
    } catch (error) {
      errors.push(error);
    }
  }
  if (errors.length > 0) {
    throw new AggregateError(errors, 'GI-state candidate rollback failed.');
  }
}

function rollbackPreparedOrRethrow(
  transactions: readonly GIImportTransaction[],
  primaryError: unknown,
): never {
  try {
    rollbackPrepared(transactions);
  } catch (rollbackError) {
    const nested: unknown[] = rollbackError instanceof AggregateError
      ? rollbackError.errors.map((error: unknown) => error)
      : [rollbackError];
    throw new AggregateError(
      [primaryError, ...nested],
      'GI-state import failed and rollback also failed.',
    );
  }
  throw primaryError;
}

function rgbaLengthMatches(
  width: number,
  height: number,
  length: number,
): boolean {
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

function isFiniteVec3(
  value: unknown,
): value is readonly [number, number, number] {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    Number.isFinite(value[0]) &&
    Number.isFinite(value[1]) &&
    Number.isFinite(value[2])
  );
}

function warnGridMismatch(
  deps: GIStateDeps,
  snapshot: GIStateSnapshot,
): void {
  const grid = deps.ddgi.gridParams;
  warnGIState(deps, {
    code: 'walkaround-hybrid.import-gi-state-grid-mismatch',
    backend: 'walkaround-hybrid',
    phase: 'lifecycle',
    method: 'importGIState',
    message:
      '[HybridEngine] importGIState: snapshot grid layout does not match the current grid; restore rejected.',
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
}

function warnCompatibilityMismatch(deps: GIStateDeps): void {
  warnGIState(deps, {
    code: 'walkaround-hybrid.import-gi-state-scene-mismatch',
    backend: 'walkaround-hybrid',
    phase: 'lifecycle',
    method: 'importGIState',
    message:
      '[HybridEngine] importGIState: snapshot scene, lighting, or environment does not match the live estimator; restore rejected.',
    details: { fallback: 'retain-current-gi' },
  });
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
      '[HybridEngine] importGIState: DDGI atlas candidate was rejected; the current estimator state was retained.',
    details: {
      fallback: 'retain-current-gi',
      snapshot: {
        irradianceAtlas: [snapshot.irrW, snapshot.irrH],
        visibilityAtlas: [snapshot.visW, snapshot.visH],
      },
    },
  });
}

function warnCompleteStateRejected(
  deps: GIStateDeps,
  reason: string,
): void {
  warnGIState(deps, {
    code: 'walkaround-hybrid.import-gi-state-complete-state-rejected',
    backend: 'walkaround-hybrid',
    phase: 'lifecycle',
    method: 'importGIState',
    message:
      '[HybridEngine] importGIState: complete estimator-state restore was rejected before publication.',
    details: {
      fallback: 'retain-current-gi',
      reason,
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
      '[HybridEngine] importGIState: malformed or unverifiable snapshot rejected before live GI state mutation.',
    details: { fallback: 'retain-current-gi' },
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
