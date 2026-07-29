/** Serializable state and semantic validation for the live ReSTIR-DI cohort. */

import { RESERVOIR_DI_STRIDE_U32 } from './reservoirDiLayout.js';

const U32_MAX = 0xffff_ffff;
const ENV_SAMPLE_SENTINEL = U32_MAX;
const MIN_RESERVOIR_U32 = 64; // WebGPU's 256-byte minimum allocation used here

export interface RestirDISnapshot {
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
  const snapshot = value as RestirDISnapshot;
  assertPositiveU32(snapshot.width, 'ReSTIR-DI width');
  assertPositiveU32(snapshot.height, 'ReSTIR-DI height');
  if (snapshot.strideU32 !== RESERVOIR_DI_STRIDE_U32) {
    throw new RangeError(
      `ReSTIR-DI stride must be ${RESERVOIR_DI_STRIDE_U32} u32 values.`,
    );
  }
  const recordCount = checkedProduct(
    snapshot.width,
    snapshot.height,
    'ReSTIR-DI logical record count',
  );
  const expectedLength = Math.max(
    MIN_RESERVOIR_U32,
    checkedProduct(
      recordCount,
      snapshot.strideU32,
      'ReSTIR-DI buffer element count',
    ),
  );
  if (expectedLength > U32_MAX) {
    throw new RangeError('ReSTIR-DI buffer length exceeds the uint32 domain.');
  }
  for (const [name, data] of [
    ['current', snapshot.current],
    ['previous', snapshot.previous],
    ['spatial', snapshot.spatial],
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
    const weightSum = floats[base + 2]!;
    const unbiasedContributionWeight = floats[base + 3]!;
    const xiX = floats[base + 4]!;
    const xiY = floats[base + 5]!;
    const areaAttempts = data[base + 6]!;
    const environmentAttempts = data[base + 7]!;

    if (
      !Number.isFinite(weightSum) ||
      !Number.isFinite(unbiasedContributionWeight) ||
      !Number.isFinite(xiX) ||
      !Number.isFinite(xiY) ||
      weightSum < 0 ||
      unbiasedContributionWeight < 0
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
      if (
        lightId !== 0 ||
        weightSum !== 0 ||
        unbiasedContributionWeight !== 0 ||
        xiX !== 0 ||
        xiY !== 0
      ) {
        throw new RangeError(
          `ReSTIR-DI ${label} record ${record} is non-empty with zero represented attempts.`,
        );
      }
      continue;
    }
    if (weightSum === 0 && unbiasedContributionWeight !== 0) {
      throw new RangeError(
        `ReSTIR-DI ${label} record ${record} has a contribution weight without accumulated weight.`,
      );
    }
    if (unbiasedContributionWeight > 0) {
      if (
        !(weightSum > 0) ||
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

function saturatingAddU32(a: number, b: number): number {
  return b > U32_MAX - a ? U32_MAX : a + b;
}

function assertPositiveU32(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0 || value > U32_MAX) {
    throw new RangeError(`${label} must be a positive uint32 integer.`);
  }
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
