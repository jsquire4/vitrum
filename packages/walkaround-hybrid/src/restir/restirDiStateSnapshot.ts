/** Serializable state and semantic validation for the live ReSTIR-DI cohort. */

import { RESERVOIR_DI_STRIDE_U32 } from './reservoirDiLayout.js';
import {
  RESTIR_RESERVOIR_LOG_ZERO,
  RESTIR_RESERVOIR_REPRESENTATION_LOG_MASS_V1,
} from './reservoirRepresentation.js';

const U32_MAX = 0xffff_ffff;
const ENV_SAMPLE_SENTINEL = U32_MAX;
const MIN_RESERVOIR_U32 = 64; // WebGPU's 256-byte minimum allocation used here

export interface RestirDISnapshot {
  /** Semantic encoding of the historical running-weight lane. */
  readonly representationVersion: typeof RESTIR_RESERVOIR_REPRESENTATION_LOG_MASS_V1;
  readonly width: number;
  readonly height: number;
  readonly strideU32: number;
  readonly current: Uint32Array;
  readonly previous: Uint32Array;
  readonly spatial: Uint32Array;
}

export function assertRestirDISnapshot(
  value: unknown,
): asserts value is RestirDISnapshot {
  if (value == null || typeof value !== 'object') {
    throw new TypeError('ReSTIR-DI snapshot must be an object.');
  }
  const snapshot = value as Record<keyof RestirDISnapshot, unknown>;
  if (
    snapshot.representationVersion !==
    RESTIR_RESERVOIR_REPRESENTATION_LOG_MASS_V1
  ) {
    throw new RangeError(
      `ReSTIR-DI representationVersion must be ${RESTIR_RESERVOIR_REPRESENTATION_LOG_MASS_V1}.`,
    );
  }
  const width = snapshot.width;
  const height = snapshot.height;
  const strideU32 = snapshot.strideU32;
  assertPositiveU32(width, 'ReSTIR-DI width');
  assertPositiveU32(height, 'ReSTIR-DI height');
  assertPositiveU32(strideU32, 'ReSTIR-DI stride');
  if (strideU32 !== RESERVOIR_DI_STRIDE_U32) {
    throw new RangeError(
      `ReSTIR-DI stride must be ${RESERVOIR_DI_STRIDE_U32} u32 values.`,
    );
  }
  const recordCount = checkedProduct(
    width,
    height,
    'ReSTIR-DI logical record count',
  );
  const expectedLength = Math.max(
    MIN_RESERVOIR_U32,
    checkedProduct(
      recordCount,
      strideU32,
      'ReSTIR-DI buffer element count',
    ),
  );
  if (expectedLength > U32_MAX) {
    throw new RangeError('ReSTIR-DI buffer length exceeds the uint32 domain.');
  }
  for (const [name, data] of [
    ['current', requireUint32Array(snapshot.current, 'ReSTIR-DI current')],
    ['previous', requireUint32Array(snapshot.previous, 'ReSTIR-DI previous')],
    ['spatial', requireUint32Array(snapshot.spatial, 'ReSTIR-DI spatial')],
  ] as const) {
    if (!(data instanceof Uint32Array)) {
      throw new TypeError(`ReSTIR-DI ${name} must be a Uint32Array.`);
    }
    if (data.length !== expectedLength) {
      throw new RangeError(
        `ReSTIR-DI ${name} length does not match the declared grid.`,
      );
    }
    assertReservoirPayload(data, recordCount, name);
  }
}

export function isValidRestirDISnapshot(
  value: unknown,
): value is RestirDISnapshot {
  try {
    assertRestirDISnapshot(value);
    return true;
  } catch {
    return false;
  }
}

function assertReservoirPayload(
  data: Uint32Array,
  recordCount: number,
  label: string,
): void {
  const floats = new Float32Array(data.buffer, data.byteOffset, data.length);
  for (let record = 0; record < recordCount; record += 1) {
    const base = record * RESERVOIR_DI_STRIDE_U32;
    const lightId = data[base]!;
    const representedAttempts = data[base + 1]!;
    const correctedLogMass = floats[base + 2]!;
    const cappedLogContributionWeight = floats[base + 3]!;
    const xiX = floats[base + 4]!;
    const xiY = floats[base + 5]!;
    const areaAttempts = data[base + 6]!;
    const environmentAttempts = data[base + 7]!;

    if (
      !Number.isFinite(correctedLogMass) ||
      !Number.isFinite(cappedLogContributionWeight) ||
      !Number.isFinite(xiX) ||
      !Number.isFinite(xiY) ||
      correctedLogMass < RESTIR_RESERVOIR_LOG_ZERO ||
      cappedLogContributionWeight < RESTIR_RESERVOIR_LOG_ZERO
    ) {
      throw new RangeError(
        `ReSTIR-DI ${label} record ${record} contains a non-finite or negative floating lane.`,
      );
    }
    if (
      representedAttempts !==
      saturatingAddU32(areaAttempts, environmentAttempts)
    ) {
      throw new RangeError(
        `ReSTIR-DI ${label} record ${record} has inconsistent proposal-support counts.`,
      );
    }
    if (representedAttempts === 0) {
      const hasColdZeroLogs =
        correctedLogMass === 0 && cappedLogContributionWeight === 0;
      const hasCanonicalEmptyLogs =
        correctedLogMass === RESTIR_RESERVOIR_LOG_ZERO &&
        cappedLogContributionWeight === RESTIR_RESERVOIR_LOG_ZERO;
      if (
        lightId !== 0 ||
        (!hasColdZeroLogs && !hasCanonicalEmptyLogs) ||
        xiX !== 0 ||
        xiY !== 0
      ) {
        throw new RangeError(
          `ReSTIR-DI ${label} record ${record} is non-empty with zero represented attempts.`,
        );
      }
      continue;
    }
    const hasSelectedOccurrence =
      correctedLogMass > RESTIR_RESERVOIR_LOG_ZERO;
    if (
      (!hasSelectedOccurrence &&
        cappedLogContributionWeight !== RESTIR_RESERVOIR_LOG_ZERO) ||
      (hasSelectedOccurrence &&
        cappedLogContributionWeight === RESTIR_RESERVOIR_LOG_ZERO)
    ) {
      throw new RangeError(
        `ReSTIR-DI ${label} record ${record} has inconsistent logarithmic mass and contribution-weight lanes.`,
      );
    }
    if (hasSelectedOccurrence) {
      if (
        xiX < 0 ||
        xiX > 1 ||
        xiY < 0 ||
        xiY > 1
      ) {
        throw new RangeError(
          `ReSTIR-DI ${label} record ${record} has an invalid selected sample.`,
        );
      }
      const selectedSupport =
        lightId === ENV_SAMPLE_SENTINEL
          ? environmentAttempts
          : areaAttempts;
      if (selectedSupport === 0) {
        throw new RangeError(
          `ReSTIR-DI ${label} record ${record} selected a sample outside its represented support.`,
        );
      }
    }
  }
  const logicalLength = recordCount * RESERVOIR_DI_STRIDE_U32;
  for (let index = logicalLength; index < data.length; index += 1) {
    if (data[index] !== 0) {
      throw new RangeError(
        `ReSTIR-DI ${label} contains non-zero allocation padding.`,
      );
    }
  }
}

/**
 * Validate and cold-migrate the pre-log-mass DI representation. The payload is
 * never reinterpreted: only its dimensions survive and every live record is
 * reset to the canonical zero state.
 */
export function coldMigrateLegacyRestirDISnapshot(value: unknown): RestirDISnapshot {
  if (value == null || typeof value !== 'object') {
    throw new TypeError('Legacy ReSTIR-DI snapshot must be an object.');
  }
  const snapshot = value as Omit<RestirDISnapshot, 'representationVersion'>;
  assertPositiveU32(snapshot.width, 'Legacy ReSTIR-DI width');
  assertPositiveU32(snapshot.height, 'Legacy ReSTIR-DI height');
  if (snapshot.strideU32 !== RESERVOIR_DI_STRIDE_U32) {
    throw new RangeError(
      `Legacy ReSTIR-DI stride must be ${RESERVOIR_DI_STRIDE_U32} u32 values.`,
    );
  }
  const recordCount = checkedProduct(
    snapshot.width,
    snapshot.height,
    'Legacy ReSTIR-DI logical record count',
  );
  const expectedLength = Math.max(
    MIN_RESERVOIR_U32,
    checkedProduct(
      recordCount,
      snapshot.strideU32,
      'Legacy ReSTIR-DI buffer element count',
    ),
  );
  if (expectedLength > U32_MAX) {
    throw new RangeError(
      'Legacy ReSTIR-DI buffer length exceeds the uint32 domain.',
    );
  }
  for (const [name, data] of [
    ['current', snapshot.current],
    ['previous', snapshot.previous],
    ['spatial', snapshot.spatial],
  ] as const) {
    if (!(data instanceof Uint32Array) || data.length !== expectedLength) {
      throw new RangeError(
        `Legacy ReSTIR-DI ${name} does not match the declared grid.`,
      );
    }
    assertLegacyReservoirPayload(data, recordCount, name);
  }
  return {
    representationVersion: RESTIR_RESERVOIR_REPRESENTATION_LOG_MASS_V1,
    width: snapshot.width,
    height: snapshot.height,
    strideU32: RESERVOIR_DI_STRIDE_U32,
    current: new Uint32Array(expectedLength),
    previous: new Uint32Array(expectedLength),
    spatial: new Uint32Array(expectedLength),
  };
}

function assertLegacyReservoirPayload(
  data: Uint32Array,
  recordCount: number,
  label: string,
): void {
  const floats = new Float32Array(data.buffer, data.byteOffset, data.length);
  for (let record = 0; record < recordCount; record += 1) {
    const base = record * RESERVOIR_DI_STRIDE_U32;
    const lightId = data[base]!;
    const representedAttempts = data[base + 1]!;
    const weightSum = floats[base + 2]!;
    const contributionWeight = floats[base + 3]!;
    const xiX = floats[base + 4]!;
    const xiY = floats[base + 5]!;
    const areaAttempts = data[base + 6]!;
    const environmentAttempts = data[base + 7]!;
    if (
      !Number.isFinite(weightSum) ||
      !Number.isFinite(contributionWeight) ||
      !Number.isFinite(xiX) ||
      !Number.isFinite(xiY) ||
      weightSum < 0 ||
      contributionWeight < 0
    ) {
      throw new RangeError(
        `Legacy ReSTIR-DI ${label} record ${record} contains an invalid floating lane.`,
      );
    }
    if (
      representedAttempts !==
      saturatingAddU32(areaAttempts, environmentAttempts)
    ) {
      throw new RangeError(
        `Legacy ReSTIR-DI ${label} record ${record} has inconsistent proposal-support counts.`,
      );
    }
    if (representedAttempts === 0) {
      if (
        lightId !== 0 ||
        weightSum !== 0 ||
        contributionWeight !== 0 ||
        xiX !== 0 ||
        xiY !== 0
      ) {
        throw new RangeError(
          `Legacy ReSTIR-DI ${label} record ${record} is non-empty with zero represented attempts.`,
        );
      }
      continue;
    }
    if (weightSum === 0 && contributionWeight !== 0) {
      throw new RangeError(
        `Legacy ReSTIR-DI ${label} record ${record} has a contribution weight without accumulated weight.`,
      );
    }
    if (contributionWeight > 0) {
      if (
        !(weightSum > 0) ||
        xiX < 0 ||
        xiX > 1 ||
        xiY < 0 ||
        xiY > 1
      ) {
        throw new RangeError(
          `Legacy ReSTIR-DI ${label} record ${record} has an invalid selected sample.`,
        );
      }
      const selectedSupport =
        lightId === ENV_SAMPLE_SENTINEL
          ? environmentAttempts
          : areaAttempts;
      if (selectedSupport === 0) {
        throw new RangeError(
          `Legacy ReSTIR-DI ${label} record ${record} selected a sample outside its represented support.`,
        );
      }
    }
  }
  const logicalLength = recordCount * RESERVOIR_DI_STRIDE_U32;
  for (let index = logicalLength; index < data.length; index += 1) {
    if (data[index] !== 0) {
      throw new RangeError(
        `Legacy ReSTIR-DI ${label} contains non-zero allocation padding.`,
      );
    }
  }
}

function saturatingAddU32(a: number, b: number): number {
  return b > U32_MAX - a ? U32_MAX : a + b;
}

function assertPositiveU32(
  value: unknown,
  label: string,
): asserts value is number {
  if (
    typeof value !== 'number' ||
    !Number.isSafeInteger(value) ||
    value <= 0 ||
    value > U32_MAX
  ) {
    throw new RangeError(`${label} must be a positive uint32 integer.`);
  }
}

function requireUint32Array(value: unknown, label: string): Uint32Array {
  if (!(value instanceof Uint32Array)) {
    throw new TypeError(`${label} must be a Uint32Array.`);
  }
  return value;
}

function checkedProduct(a: number, b: number, label: string): number {
  if (
    !Number.isSafeInteger(a) ||
    !Number.isSafeInteger(b) ||
    a < 0 ||
    b < 0 ||
    (a !== 0 && b > Number.MAX_SAFE_INTEGER / a)
  ) {
    throw new RangeError(`${label} exceeds safe-integer arithmetic.`);
  }
  return a * b;
}
