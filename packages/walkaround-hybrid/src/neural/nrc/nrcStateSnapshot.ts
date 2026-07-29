/**
 * Serializable learned state for the live Neural Radiance Cache.
 *
 * This is deliberately limited to estimator state that changes convergence:
 * network masters, both Adam histories, hash-grid tables and histories, and
 * the three training counters. Transient record/gradient/diagnostic buffers
 * are frame scratch and are cold-cleared when a snapshot is restored.
 */

export interface NrcStateConfig {
  readonly levels: number;
  readonly featuresPerEntry: number;
  readonly tableSize: number;
  readonly nMin: number;
  readonly growth: number;
  readonly oneBlobBins: number;
  readonly width: number;
  readonly hidden: number;
  readonly spreadC: number;
  readonly recordCap: number;
  readonly learningRate: number;
  readonly tableLearningRate: number;
  readonly useF16: boolean;
  readonly tileB: number;
  readonly warmupSteps: number;
}

export interface NrcMlpStateSnapshot {
  readonly weights: Float32Array;
  readonly biases: Float32Array;
  readonly firstMomentWeights: Float32Array;
  readonly secondMomentWeights: Float32Array;
  readonly firstMomentBiases: Float32Array;
  readonly secondMomentBiases: Float32Array;
  readonly adamT: number;
}

export interface NrcHashGridStateSnapshot {
  readonly tables: Float32Array;
  readonly firstMoment: Float32Array;
  readonly secondMoment: Float32Array;
  readonly adamT: number;
}

export interface NrcLearnedStateSnapshot {
  readonly config: NrcStateConfig;
  readonly sceneBoundsMin: readonly [number, number, number];
  readonly sceneBoundsMax: readonly [number, number, number];
  readonly trainedSteps: number;
  readonly mlp: NrcMlpStateSnapshot;
  readonly hashGrid: NrcHashGridStateSnapshot;
}

export interface NrcStateShape {
  readonly weightScalars: number;
  readonly biasScalars: number;
  readonly tableScalars: number;
}

const U32_MAX = 0xffff_ffff;
const NRC_STATE_MAGIC = 0x4e524353; // "NRCS"
const NRC_STATE_VERSION = 1;
const NRC_STATE_HEADER_BYTES = 112; // 28 × 4-byte fields

export function nrcStateShape(config: NrcStateConfig): NrcStateShape {
  const weightScalars =
    checkedProduct(config.hidden, config.width, config.width, 'NRC MLP weight count') +
    checkedProduct(3, config.width, 'NRC MLP output-weight count');
  const biasScalars =
    checkedProduct(config.hidden, config.width, 'NRC MLP bias count') + 3;
  const tableScalars = checkedProduct(
    config.levels,
    config.tableSize,
    config.featuresPerEntry,
    'NRC hash-grid scalar count',
  );
  if (!Number.isSafeInteger(weightScalars) || !Number.isSafeInteger(biasScalars)) {
    throw new RangeError('NRC state shape exceeds the safe-integer domain.');
  }
  return { weightScalars, biasScalars, tableScalars };
}

export function assertNrcLearnedStateSnapshot(
  value: unknown,
): asserts value is NrcLearnedStateSnapshot {
  if (value == null || typeof value !== 'object') {
    throw new TypeError('NRC learned-state snapshot must be an object.');
  }
  const snapshot = value as NrcLearnedStateSnapshot;
  assertNrcStateConfig(snapshot.config);
  const shape = nrcStateShape(snapshot.config);
  assertFiniteVec3(snapshot.sceneBoundsMin, 'NRC sceneBoundsMin');
  assertFiniteVec3(snapshot.sceneBoundsMax, 'NRC sceneBoundsMax');
  for (let axis = 0; axis < 3; axis++) {
    if (!(snapshot.sceneBoundsMin[axis]! < snapshot.sceneBoundsMax[axis]!)) {
      throw new RangeError(
        `NRC scene bound min[${axis}] must be smaller than max[${axis}].`,
      );
    }
  }
  assertU32(snapshot.trainedSteps, 'NRC trainedSteps');
  if (snapshot.mlp == null || typeof snapshot.mlp !== 'object') {
    throw new TypeError('NRC MLP state must be an object.');
  }
  if (snapshot.hashGrid == null || typeof snapshot.hashGrid !== 'object') {
    throw new TypeError('NRC hash-grid state must be an object.');
  }
  assertU32(snapshot.mlp.adamT, 'NRC MLP adamT');
  assertU32(snapshot.hashGrid.adamT, 'NRC hash-grid adamT');
  assertFiniteArray(snapshot.mlp.weights, shape.weightScalars, 'NRC MLP weights');
  assertFiniteArray(snapshot.mlp.biases, shape.biasScalars, 'NRC MLP biases');
  assertFiniteArray(
    snapshot.mlp.firstMomentWeights,
    shape.weightScalars,
    'NRC MLP firstMomentWeights',
  );
  assertFiniteArray(
    snapshot.mlp.secondMomentWeights,
    shape.weightScalars,
    'NRC MLP secondMomentWeights',
    true,
  );
  assertFiniteArray(
    snapshot.mlp.firstMomentBiases,
    shape.biasScalars,
    'NRC MLP firstMomentBiases',
  );
  assertFiniteArray(
    snapshot.mlp.secondMomentBiases,
    shape.biasScalars,
    'NRC MLP secondMomentBiases',
    true,
  );
  assertFiniteArray(snapshot.hashGrid.tables, shape.tableScalars, 'NRC hash-grid tables');
  assertFiniteArray(
    snapshot.hashGrid.firstMoment,
    shape.tableScalars,
    'NRC hash-grid firstMoment',
  );
  assertFiniteArray(
    snapshot.hashGrid.secondMoment,
    shape.tableScalars,
    'NRC hash-grid secondMoment',
    true,
  );
}

export function nrcStateConfigMatches(
  snapshot: NrcStateConfig,
  current: NrcStateConfig,
): boolean {
  return (
    snapshot.levels === current.levels &&
    snapshot.featuresPerEntry === current.featuresPerEntry &&
    snapshot.tableSize === current.tableSize &&
    snapshot.nMin === current.nMin &&
    f32Matches(snapshot.growth, current.growth) &&
    snapshot.oneBlobBins === current.oneBlobBins &&
    snapshot.width === current.width &&
    snapshot.hidden === current.hidden &&
    f32Matches(snapshot.spreadC, current.spreadC) &&
    snapshot.recordCap === current.recordCap &&
    f32Matches(snapshot.learningRate, current.learningRate) &&
    f32Matches(snapshot.tableLearningRate, current.tableLearningRate) &&
    snapshot.useF16 === current.useF16 &&
    snapshot.tileB === current.tileB &&
    snapshot.warmupSteps === current.warmupSteps
  );
}

export function nrcStateBoundsMatch(
  snapshot: NrcLearnedStateSnapshot,
  min: readonly [number, number, number],
  max: readonly [number, number, number],
): boolean {
  for (let axis = 0; axis < 3; axis++) {
    if (
      !f32Matches(snapshot.sceneBoundsMin[axis]!, min[axis]!) ||
      !f32Matches(snapshot.sceneBoundsMax[axis]!, max[axis]!)
    ) {
      return false;
    }
  }
  return true;
}

export function nrcLearnedStateSerializedByteLength(
  snapshot: NrcLearnedStateSnapshot,
): number {
  assertNrcLearnedStateSnapshot(snapshot);
  const shape = nrcStateShape(snapshot.config);
  const scalarCount = checkedSum(
    checkedProduct(3, shape.weightScalars, 'NRC serialized weight scalars'),
    checkedProduct(3, shape.biasScalars, 'NRC serialized bias scalars'),
    checkedProduct(3, shape.tableScalars, 'NRC serialized table scalars'),
    'NRC serialized scalar count',
  );
  const payloadBytes = checkedProduct(
    scalarCount,
    Float32Array.BYTES_PER_ELEMENT,
    'NRC serialized payload bytes',
  );
  const totalBytes = checkedSum(
    NRC_STATE_HEADER_BYTES,
    payloadBytes,
    'NRC serialized byte length',
  );
  if (totalBytes > U32_MAX) {
    throw new RangeError('NRC serialized state exceeds the uint32 byte domain.');
  }
  return totalBytes;
}

/**
 * Encode the complete learned NRC state into an owning, storage-ready buffer.
 *
 * The fixed header records every architecture/training parameter needed to
 * derive the three tensor shapes. The payload then stores three generations
 * (value, Adam first moment, Adam second moment) for weights, biases, and hash
 * tables in that order. Transient records/gradients are intentionally absent.
 */
export function serializeNrcLearnedState(
  snapshot: NrcLearnedStateSnapshot,
): ArrayBuffer {
  const totalBytes = nrcLearnedStateSerializedByteLength(snapshot);
  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);
  let offset = 0;
  const u32 = (value: number): void => {
    view.setUint32(offset, value, true);
    offset += 4;
  };
  const f32 = (value: number): void => {
    view.setFloat32(offset, value, true);
    offset += 4;
  };
  const config = snapshot.config;
  u32(NRC_STATE_MAGIC);
  u32(NRC_STATE_VERSION);
  u32(NRC_STATE_HEADER_BYTES);
  u32(totalBytes);
  u32(config.levels);
  u32(config.featuresPerEntry);
  u32(config.tableSize);
  u32(config.nMin);
  f32(config.growth);
  u32(config.oneBlobBins);
  u32(config.width);
  u32(config.hidden);
  f32(config.spreadC);
  u32(config.recordCap);
  f32(config.learningRate);
  f32(config.tableLearningRate);
  u32(config.useF16 ? 1 : 0);
  u32(config.tileB);
  u32(config.warmupSteps);
  for (const value of snapshot.sceneBoundsMin) f32(value);
  for (const value of snapshot.sceneBoundsMax) f32(value);
  u32(snapshot.trainedSteps);
  u32(snapshot.mlp.adamT);
  u32(snapshot.hashGrid.adamT);
  if (offset !== NRC_STATE_HEADER_BYTES) {
    throw new Error('NRC state header encoder drifted from its fixed layout.');
  }

  for (const values of [
    snapshot.mlp.weights,
    snapshot.mlp.firstMomentWeights,
    snapshot.mlp.secondMomentWeights,
    snapshot.mlp.biases,
    snapshot.mlp.firstMomentBiases,
    snapshot.mlp.secondMomentBiases,
    snapshot.hashGrid.tables,
    snapshot.hashGrid.firstMoment,
    snapshot.hashGrid.secondMoment,
  ]) {
    new Float32Array(buffer, offset, values.length).set(values);
    offset += values.byteLength;
  }
  if (offset !== totalBytes) {
    throw new Error('NRC state payload encoder did not fill the declared buffer.');
  }
  return buffer;
}

/** Decode and semantically validate a buffer produced by
 * {@link serializeNrcLearnedState}. Returned arrays own their storage. */
export function deserializeNrcLearnedState(
  buffer: ArrayBuffer,
): NrcLearnedStateSnapshot {
  if (!(buffer instanceof ArrayBuffer)) {
    throw new TypeError('NRC learned-state input must be an ArrayBuffer.');
  }
  if (buffer.byteLength < NRC_STATE_HEADER_BYTES) {
    throw new RangeError('NRC learned-state buffer is smaller than its header.');
  }
  const view = new DataView(buffer);
  let offset = 0;
  const u32 = (): number => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const f32 = (): number => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const magic = u32();
  if (magic !== NRC_STATE_MAGIC) {
    throw new Error('NRC learned-state buffer has an invalid magic value.');
  }
  const version = u32();
  if (version !== NRC_STATE_VERSION) {
    throw new Error(
      `NRC learned-state version ${version} is unsupported (expected ${NRC_STATE_VERSION}).`,
    );
  }
  const headerBytes = u32();
  const totalBytes = u32();
  if (
    headerBytes !== NRC_STATE_HEADER_BYTES ||
    totalBytes !== buffer.byteLength
  ) {
    throw new RangeError(
      'NRC learned-state header or total byte length is inconsistent.',
    );
  }
  const config: NrcStateConfig = {
    levels: u32(),
    featuresPerEntry: u32(),
    tableSize: u32(),
    nMin: u32(),
    growth: f32(),
    oneBlobBins: u32(),
    width: u32(),
    hidden: u32(),
    spreadC: f32(),
    recordCap: u32(),
    learningRate: f32(),
    tableLearningRate: f32(),
    useF16: decodeBool(u32(), 'NRC config useF16'),
    tileB: u32(),
    warmupSteps: u32(),
  };
  const sceneBoundsMin: [number, number, number] = [f32(), f32(), f32()];
  const sceneBoundsMax: [number, number, number] = [f32(), f32(), f32()];
  const trainedSteps = u32();
  const mlpAdamT = u32();
  const hashGridAdamT = u32();
  if (offset !== NRC_STATE_HEADER_BYTES) {
    throw new Error('NRC state header decoder drifted from its fixed layout.');
  }

  assertNrcStateConfig(config);
  const shape = nrcStateShape(config);
  const expectedScalarCount = checkedSum(
    checkedProduct(3, shape.weightScalars, 'NRC decoded weight scalars'),
    checkedProduct(3, shape.biasScalars, 'NRC decoded bias scalars'),
    checkedProduct(3, shape.tableScalars, 'NRC decoded table scalars'),
    'NRC decoded scalar count',
  );
  const expectedBytes = checkedSum(
    NRC_STATE_HEADER_BYTES,
    checkedProduct(
      expectedScalarCount,
      Float32Array.BYTES_PER_ELEMENT,
      'NRC decoded payload bytes',
    ),
    'NRC decoded byte length',
  );
  if (expectedBytes !== buffer.byteLength) {
    throw new RangeError(
      'NRC learned-state payload length does not match the encoded architecture.',
    );
  }

  const take = (length: number): Float32Array => {
    const byteLength = checkedProduct(
      length,
      Float32Array.BYTES_PER_ELEMENT,
      'NRC decoded tensor bytes',
    );
    const end = checkedSum(offset, byteLength, 'NRC decoded tensor end');
    if (end > buffer.byteLength) {
      throw new RangeError('NRC learned-state tensor exceeds its payload.');
    }
    const result = new Float32Array(buffer.slice(offset, end));
    offset = end;
    return result;
  };
  const weights = take(shape.weightScalars);
  const firstMomentWeights = take(shape.weightScalars);
  const secondMomentWeights = take(shape.weightScalars);
  const biases = take(shape.biasScalars);
  const firstMomentBiases = take(shape.biasScalars);
  const secondMomentBiases = take(shape.biasScalars);
  const tables = take(shape.tableScalars);
  const firstMoment = take(shape.tableScalars);
  const secondMoment = take(shape.tableScalars);
  if (offset !== buffer.byteLength) {
    throw new Error('NRC learned-state decoder left trailing bytes.');
  }

  const snapshot: NrcLearnedStateSnapshot = {
    config,
    sceneBoundsMin,
    sceneBoundsMax,
    trainedSteps,
    mlp: {
      weights,
      biases,
      firstMomentWeights,
      secondMomentWeights,
      firstMomentBiases,
      secondMomentBiases,
      adamT: mlpAdamT,
    },
    hashGrid: {
      tables,
      firstMoment,
      secondMoment,
      adamT: hashGridAdamT,
    },
  };
  assertNrcLearnedStateSnapshot(snapshot);
  return snapshot;
}

function assertNrcStateConfig(config: NrcStateConfig): void {
  if (config == null || typeof config !== 'object') {
    throw new TypeError('NRC snapshot config must be an object.');
  }
  for (const key of [
    'levels',
    'featuresPerEntry',
    'tableSize',
    'nMin',
    'oneBlobBins',
    'width',
    'recordCap',
    'tileB',
  ] as const) {
    assertPositiveSafeInteger(config[key], `NRC config ${key}`);
  }
  if (!Number.isSafeInteger(config.hidden) || config.hidden < 0) {
    throw new RangeError('NRC config hidden must be a non-negative safe integer.');
  }
  for (const key of ['growth', 'learningRate', 'tableLearningRate'] as const) {
    if (
      !Number.isFinite(config[key]) ||
      !Number.isFinite(Math.fround(config[key])) ||
      !(Math.fround(config[key]) > 0)
    ) {
      throw new RangeError(
        `NRC config ${key} must be positive, finite, and representable as float32.`,
      );
    }
  }
  if (
    !Number.isFinite(config.spreadC) ||
    !Number.isFinite(Math.fround(config.spreadC)) ||
    Math.fround(config.spreadC) < 0
  ) {
    throw new RangeError(
      'NRC config spreadC must be finite, non-negative, and representable as float32.',
    );
  }
  assertU32(config.warmupSteps, 'NRC config warmupSteps');
  if (typeof config.useF16 !== 'boolean') {
    throw new TypeError('NRC config useF16 must be a boolean.');
  }
  const inputWidth =
    checkedProduct(
      config.levels,
      config.featuresPerEntry,
      'NRC encoded hash-grid width',
    ) +
    checkedProduct(2, config.oneBlobBins, 'NRC encoded one-blob width') +
    7;
  if (!Number.isSafeInteger(inputWidth) || inputWidth > config.width) {
    throw new RangeError(
      `NRC encoded input width ${inputWidth} exceeds network width ${config.width}.`,
    );
  }
}

function assertFiniteArray(
  value: unknown,
  expectedLength: number,
  label: string,
  nonNegative = false,
): asserts value is Float32Array {
  if (!(value instanceof Float32Array) || value.length !== expectedLength) {
    throw new RangeError(`${label} must be a ${expectedLength}-element Float32Array.`);
  }
  for (let index = 0; index < value.length; index++) {
    const scalar = value[index]!;
    if (!Number.isFinite(scalar) || (nonNegative && scalar < 0)) {
      throw new RangeError(
        `${label}[${index}] must be finite${nonNegative ? ' and non-negative' : ''}.`,
      );
    }
  }
}

function assertFiniteVec3(value: unknown, label: string): asserts value is readonly [
  number,
  number,
  number,
] {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new RangeError(`${label} must contain exactly three scalars.`);
  }
  for (let index = 0; index < 3; index++) {
    const scalar: unknown = value[index];
    if (
      typeof scalar !== 'number' ||
      !Number.isFinite(scalar) ||
      !Number.isFinite(Math.fround(scalar))
    ) {
      throw new RangeError(
        `${label}[${index}] must be finite and representable as float32.`,
      );
    }
  }
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer.`);
  }
}

function assertU32(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > U32_MAX) {
    throw new RangeError(`${label} must be an unsigned 32-bit integer.`);
  }
}

function checkedProduct(...args: [...number[], string]): number {
  const label = args.pop() as string;
  let product = 1;
  for (const factor of args as number[]) {
    if (!Number.isSafeInteger(factor) || factor < 0) {
      throw new RangeError(`${label} contains an invalid factor.`);
    }
    product *= factor;
    if (!Number.isSafeInteger(product)) {
      throw new RangeError(`${label} exceeds the safe-integer domain.`);
    }
  }
  return product;
}

function f32Matches(a: number, b: number): boolean {
  return Number.isFinite(a) && Number.isFinite(b) && Math.fround(a) === Math.fround(b);
}

function checkedSum(...args: [...number[], string]): number {
  const label = args.pop() as string;
  let sum = 0;
  for (const term of args as number[]) {
    if (!Number.isSafeInteger(term) || term < 0) {
      throw new RangeError(`${label} contains an invalid term.`);
    }
    sum += term;
    if (!Number.isSafeInteger(sum)) {
      throw new RangeError(`${label} exceeds the safe-integer domain.`);
    }
  }
  return sum;
}

function decodeBool(value: number, label: string): boolean {
  if (value !== 0 && value !== 1) {
    throw new RangeError(`${label} must be encoded as 0 or 1.`);
  }
  return value === 1;
}
