/** Single source of truth for NRC GPU diagnostic counter layout. */
export const NRC_DIAGNOSTIC_INDEX = {
  droppedRecords: 0,
  saturatedValues: 1,
  nonFiniteValues: 2,
  invalidPdfs: 3,
  droppedUpdates: 4,
} as const;

export const NRC_DIAGNOSTIC_COUNT = 5;
export const NRC_DIAGNOSTIC_BYTES = NRC_DIAGNOSTIC_COUNT * Uint32Array.BYTES_PER_ELEMENT;

export interface NrcDiagnostics {
  readonly droppedRecords: number;
  readonly saturatedValues: number;
  readonly nonFiniteValues: number;
  readonly invalidPdfs: number;
  readonly droppedUpdates: number;
  readonly hostDroppedNonFiniteRecords: number;
  readonly hostClampedTargets: number;
  readonly readbackOverlapSkips: number;
  readonly staleReadbacks: number;
  readonly trainingFailures: number;
  readonly trainedSteps: number;
  /** Exact persistent GPU-buffer residency from the shared preflight formula. */
  readonly persistentBufferCount: number;
  readonly persistentBufferBytes: number;
  /** Persistent set plus the one permitted generation-tagged readback buffer. */
  readonly peakResidentBufferCount: number;
  readonly peakResidentBufferBytes: number;
}

export const NRC_DIAGNOSTIC_CONSTANTS_WGSL = /* wgsl */`
const NRC_DIAG_DROPPED_RECORD : u32 = ${NRC_DIAGNOSTIC_INDEX.droppedRecords}u;
const NRC_DIAG_SATURATED      : u32 = ${NRC_DIAGNOSTIC_INDEX.saturatedValues}u;
const NRC_DIAG_NONFINITE      : u32 = ${NRC_DIAGNOSTIC_INDEX.nonFiniteValues}u;
const NRC_DIAG_INVALID_PDF    : u32 = ${NRC_DIAGNOSTIC_INDEX.invalidPdfs}u;
const NRC_DIAG_DROPPED_UPDATE : u32 = ${NRC_DIAGNOSTIC_INDEX.droppedUpdates}u;
`;
