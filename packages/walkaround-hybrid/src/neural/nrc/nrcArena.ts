/** Versioned packed GPU arenas used by the NRC query path. */

export const NRC_ARENA_ALIGNMENT = 256;

export const NRC_INFERENCE_ARENA_MAGIC = 0x4e524349; // "NRCI"
export const NRC_INFERENCE_ARENA_VERSION = 1;
export const NRC_INFERENCE_ARENA_SCHEMA = 0x6d4f4d9b;
export const NRC_INFERENCE_HEADER_WORDS = 32;
export const NRC_INFERENCE_EPOCH_WORD = 2;
export const NRC_INFERENCE_HEADER_FIELD = Object.freeze({
  weightsOffset: 5,
  weightsLength: 6,
  weightsCapacity: 7,
  biasesOffset: 8,
  biasesLength: 9,
  biasesCapacity: 10,
  tablesOffset: 11,
  tablesLength: 12,
  tablesCapacity: 13,
  levelsOffset: 14,
  levelsLength: 15,
  levelsCapacity: 16,
  arenaLength: 17,
} as const);

export const NRC_RUNTIME_ARENA_MAGIC = 0x4e524352; // "NRCR"
export const NRC_RUNTIME_ARENA_VERSION = 1;
export const NRC_RUNTIME_ARENA_SCHEMA = 0xb51952e7;
// Diagnostics deliberately begin at byte zero. The training kernels bind the
// whole runtime arena as array<atomic<u32>> and therefore retain their existing
// zero-based diagnostic indices while the query uses the versioned header below.
export const NRC_RUNTIME_DIAGNOSTICS_BYTE_OFFSET = 0;
export const NRC_RUNTIME_HEADER_BYTE_OFFSET = NRC_ARENA_ALIGNMENT;
export const NRC_RUNTIME_HEADER_WORD_OFFSET = NRC_RUNTIME_HEADER_BYTE_OFFSET / 4;
export const NRC_RUNTIME_HEADER_WORDS = 16;
export const NRC_RUNTIME_EPOCH_WORD = 2;
export const NRC_RUNTIME_HEADER_FIELD = Object.freeze({
  claimsOffset: 5,
  claimsLength: 6,
  recordsOffset: 7,
  recordsLength: 8,
  diagnosticsOffset: 9,
  diagnosticsLength: 10,
  recordCap: 11,
  recordStride: 12,
  arenaLength: 13,
} as const);

const U32_MAX = 0xffff_ffff;
const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;

function align(value: number): number {
  const aligned = Math.ceil(value / NRC_ARENA_ALIGNMENT) * NRC_ARENA_ALIGNMENT;
  return checkedInteger('aligned arena byte count', aligned);
}

function checkedInteger(label: string, value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`NRC ${label} must be a non-negative safe integer; got ${value}`);
  }
  return value;
}

function checkedBytes(label: string, value: number): number {
  checkedInteger(label, value);
  return Math.max(16, Math.ceil(value / 4) * 4);
}

function checkedEnd(label: string, offset: number, length: number): number {
  checkedInteger(`${label} offset`, offset);
  checkedInteger(`${label} length`, length);
  return checkedInteger(`${label} end`, offset + length);
}

function assertAligned(label: string, value: number, alignment: number): void {
  checkedInteger(label, value);
  if (value % alignment !== 0) {
    throw new RangeError(`NRC ${label} must be ${alignment}-byte aligned; got ${value}`);
  }
}

function assertWordRepresentable(label: string, bytes: number): number {
  assertAligned(label, bytes, U32_BYTES);
  const words = bytes / U32_BYTES;
  if (words > U32_MAX) {
    throw new RangeError(`NRC ${label} cannot be represented by an exact u32 word offset/length`);
  }
  return words;
}

function assertU32(label: string, value: number): number {
  checkedInteger(label, value);
  if (value > U32_MAX) {
    throw new RangeError(`NRC ${label} cannot be represented by an exact u32`);
  }
  return value;
}

function assertAtMost(label: string, value: number, limit: number | undefined): void {
  if (limit === undefined) return;
  checkedInteger(`${label} limit`, limit);
  if (value > limit) {
    throw new RangeError(`NRC ${label} requires ${value} bytes, limit is ${limit}`);
  }
}

export interface NrcInferenceArenaLayout {
  readonly byteSize: number;
  readonly weightsByteOffset: number;
  readonly weightsCapacityBytes: number;
  readonly biasesByteOffset: number;
  readonly biasesCapacityBytes: number;
  readonly tablesByteOffset: number;
  readonly tablesCapacityBytes: number;
  readonly levelsByteOffset: number;
  readonly levelsCapacityBytes: number;
}

export function createNrcInferenceArenaLayout(sizes: {
  readonly weightsBytes: number;
  readonly biasesBytes: number;
  readonly tablesBytes: number;
  readonly levelsBytes: number;
}): NrcInferenceArenaLayout {
  const weightsCapacityBytes = checkedBytes('inference weights bytes', sizes.weightsBytes);
  const biasesCapacityBytes = checkedBytes('inference biases bytes', sizes.biasesBytes);
  const tablesCapacityBytes = checkedBytes('inference tables bytes', sizes.tablesBytes);
  const levelsCapacityBytes = checkedBytes('inference levels bytes', sizes.levelsBytes);
  const weightsByteOffset = NRC_ARENA_ALIGNMENT;
  const biasesByteOffset = align(checkedEnd(
    'inference weights', weightsByteOffset, weightsCapacityBytes,
  ));
  const tablesByteOffset = align(checkedEnd(
    'inference biases', biasesByteOffset, biasesCapacityBytes,
  ));
  const levelsByteOffset = align(checkedEnd(
    'inference tables', tablesByteOffset, tablesCapacityBytes,
  ));
  const byteSize = align(checkedEnd(
    'inference levels', levelsByteOffset, levelsCapacityBytes,
  ));
  const layout = {
    byteSize,
    weightsByteOffset, weightsCapacityBytes,
    biasesByteOffset, biasesCapacityBytes,
    tablesByteOffset, tablesCapacityBytes,
    levelsByteOffset, levelsCapacityBytes,
  };
  validateNrcArenaLayouts({
    inference: { layout, allocationBytes: byteSize, epoch: 1, generation: 0 },
  });
  return layout;
}

export interface NrcInferencePayloadLengths {
  readonly weightsBytes: number;
  readonly biasesBytes: number;
  readonly tablesBytes: number;
  readonly levelsBytes: number;
}

export interface NrcArenaValidationLimits {
  readonly maxBufferSize?: number;
  readonly maxStorageBufferBindingSize?: number;
}

export interface NrcArenaValidationRequest {
  readonly inference?: {
    readonly layout: NrcInferenceArenaLayout;
    readonly payload?: NrcInferencePayloadLengths;
    readonly allocationBytes?: number;
    readonly epoch?: number;
    readonly generation?: number;
  };
  readonly runtime?: {
    readonly layout: NrcRuntimeArenaLayout;
    readonly allocationBytes?: number;
    readonly epoch?: number;
    readonly generation?: number;
    readonly recordCap?: number;
    readonly recordStride?: number;
  };
  readonly limits?: NrcArenaValidationLimits;
}

export function assertNrcInferencePayloadFits(
  layout: NrcInferenceArenaLayout,
  payload: NrcInferencePayloadLengths,
): void {
  validateNrcArenaLayouts({ inference: { layout, payload } });
}

export function buildNrcInferenceArenaHeader(
  layout: NrcInferenceArenaLayout,
  payload: NrcInferencePayloadLengths,
  epoch: number,
  generation: number,
): Uint32Array {
  validateNrcArenaLayouts({
    inference: { layout, payload, epoch, generation },
  });
  return encodeInferenceHeaderUnchecked(layout, payload, epoch, generation);
}

export interface NrcRuntimeArenaLayout {
  readonly byteSize: number;
  readonly diagnosticsByteOffset: number;
  readonly diagnosticsBytes: number;
  readonly headerByteOffset: number;
  readonly claimsByteOffset: number;
  readonly claimsBytes: number;
  readonly recordsByteOffset: number;
  readonly recordsBytes: number;
}

function encodeInferenceHeaderUnchecked(
  layout: NrcInferenceArenaLayout,
  payload: NrcInferencePayloadLengths,
  epoch: number,
  generation: number,
): Uint32Array {
  const f = NRC_INFERENCE_HEADER_FIELD;
  const h = new Uint32Array(NRC_INFERENCE_HEADER_WORDS);
  h[0] = NRC_INFERENCE_ARENA_MAGIC;
  h[1] = NRC_INFERENCE_ARENA_VERSION;
  h[2] = epoch;
  h[3] = NRC_INFERENCE_ARENA_SCHEMA;
  h[4] = generation;
  h[f.weightsOffset] = layout.weightsByteOffset / U32_BYTES;
  h[f.weightsLength] = payload.weightsBytes / U32_BYTES;
  h[f.weightsCapacity] = layout.weightsCapacityBytes / U32_BYTES;
  h[f.biasesOffset] = layout.biasesByteOffset / U32_BYTES;
  h[f.biasesLength] = payload.biasesBytes / U32_BYTES;
  h[f.biasesCapacity] = layout.biasesCapacityBytes / U32_BYTES;
  h[f.tablesOffset] = layout.tablesByteOffset / U32_BYTES;
  h[f.tablesLength] = payload.tablesBytes / U32_BYTES;
  h[f.tablesCapacity] = layout.tablesCapacityBytes / U32_BYTES;
  h[f.levelsOffset] = layout.levelsByteOffset / U32_BYTES;
  h[f.levelsLength] = payload.levelsBytes / U32_BYTES;
  h[f.levelsCapacity] = layout.levelsCapacityBytes / U32_BYTES;
  h[f.arenaLength] = layout.byteSize / U32_BYTES;
  return h;
}

function encodeRuntimeHeaderUnchecked(
  layout: NrcRuntimeArenaLayout,
  epoch: number,
  generation: number,
  recordCap: number,
  recordStride: number,
): Uint32Array {
  const f = NRC_RUNTIME_HEADER_FIELD;
  const h = new Uint32Array(NRC_RUNTIME_HEADER_WORDS);
  h[0] = NRC_RUNTIME_ARENA_MAGIC;
  h[1] = NRC_RUNTIME_ARENA_VERSION;
  h[2] = epoch;
  h[3] = NRC_RUNTIME_ARENA_SCHEMA;
  h[4] = generation;
  h[f.claimsOffset] = layout.claimsByteOffset / U32_BYTES;
  h[f.claimsLength] = layout.claimsBytes / U32_BYTES;
  h[f.recordsOffset] = layout.recordsByteOffset / U32_BYTES;
  h[f.recordsLength] = layout.recordsBytes / U32_BYTES;
  h[f.diagnosticsOffset] = layout.diagnosticsByteOffset / U32_BYTES;
  h[f.diagnosticsLength] = layout.diagnosticsBytes / U32_BYTES;
  h[f.recordCap] = recordCap;
  h[f.recordStride] = recordStride;
  h[f.arenaLength] = layout.byteSize / U32_BYTES;
  return h;
}

function assertHeaderField(
  header: Uint32Array,
  index: number,
  expected: number,
  label: string,
): void {
  if (header[index] !== expected) {
    throw new Error(
      `NRC ${label} header round-trip failed: encoded ${header[index]}, expected ${expected}`,
    );
  }
}

/**
 * Validate both packed NRC arenas at the actual construction/publication
 * boundary. Shader-visible offsets are u32 word addresses decoded from these
 * headers; this validator proves that every host byte range is exactly
 * representable, aligned, monotonic, non-overlapping, and inside both the
 * allocation and adapter limits before a GPUBuffer or header becomes live.
 */
export function validateNrcArenaLayouts(request: NrcArenaValidationRequest): void {
  const limits = request.limits;
  if (request.inference) {
    const spec = request.inference;
    const l = spec.layout;
    const payload = spec.payload ?? {
      weightsBytes: l.weightsCapacityBytes,
      biasesBytes: l.biasesCapacityBytes,
      tablesBytes: l.tablesCapacityBytes,
      levelsBytes: l.levelsCapacityBytes,
    };
    const epoch = spec.epoch ?? 1;
    const generation = spec.generation ?? 0;
    assertU32('inference epoch', epoch);
    if (epoch === 0) throw new RangeError('NRC inference epoch must be non-zero');
    assertU32('inference generation', generation);
    assertAligned('inference arena byte size', l.byteSize, NRC_ARENA_ALIGNMENT);
    const headerEnd = checkedEnd(
      'inference header', 0, NRC_INFERENCE_HEADER_WORDS * U32_BYTES,
    );
    const rows: ReadonlyArray<readonly [string, number, number, number]> = [
      ['weights', l.weightsByteOffset, l.weightsCapacityBytes, payload.weightsBytes],
      ['biases', l.biasesByteOffset, l.biasesCapacityBytes, payload.biasesBytes],
      ['tables', l.tablesByteOffset, l.tablesCapacityBytes, payload.tablesBytes],
      ['levels', l.levelsByteOffset, l.levelsCapacityBytes, payload.levelsBytes],
    ];
    let priorEnd = headerEnd;
    for (const [name, offset, capacity, length] of rows) {
      assertAligned(`inference ${name} byte offset`, offset, NRC_ARENA_ALIGNMENT);
      assertWordRepresentable(`inference ${name} byte offset`, offset);
      assertWordRepresentable(`inference ${name} capacity bytes`, capacity);
      assertWordRepresentable(`inference ${name} payload bytes`, length);
      if (offset < priorEnd) {
        throw new RangeError(`NRC inference ${name} region overlaps its predecessor`);
      }
      const end = checkedEnd(`inference ${name}`, offset, capacity);
      if (length > capacity) {
        throw new RangeError(`NRC inference ${name} payload exceeds arena capacity`);
      }
      if (end > l.byteSize) {
        throw new RangeError(`NRC inference ${name} region exceeds arena allocation`);
      }
      priorEnd = end;
    }
    assertWordRepresentable('inference arena byte size', l.byteSize);
    const allocation = spec.allocationBytes ?? l.byteSize;
    assertAligned('inference allocation byte size', allocation, U32_BYTES);
    assertAtMost('inference arena allocation', l.byteSize, allocation);
    assertAtMost('inference arena buffer size', l.byteSize, limits?.maxBufferSize);
    assertAtMost(
      'inference arena storage binding size',
      l.byteSize,
      limits?.maxStorageBufferBindingSize,
    );

    const h = encodeInferenceHeaderUnchecked(l, payload, epoch, generation);
    const f = NRC_INFERENCE_HEADER_FIELD;
    const expected: ReadonlyArray<readonly [number, number, string]> = [
      [0, NRC_INFERENCE_ARENA_MAGIC, 'inference magic'],
      [1, NRC_INFERENCE_ARENA_VERSION, 'inference version'],
      [2, epoch, 'inference epoch'],
      [3, NRC_INFERENCE_ARENA_SCHEMA, 'inference schema'],
      [4, generation, 'inference generation'],
      [f.weightsOffset, l.weightsByteOffset / U32_BYTES, 'inference weights offset'],
      [f.weightsLength, payload.weightsBytes / U32_BYTES, 'inference weights length'],
      [f.weightsCapacity, l.weightsCapacityBytes / U32_BYTES, 'inference weights capacity'],
      [f.biasesOffset, l.biasesByteOffset / U32_BYTES, 'inference biases offset'],
      [f.biasesLength, payload.biasesBytes / U32_BYTES, 'inference biases length'],
      [f.biasesCapacity, l.biasesCapacityBytes / U32_BYTES, 'inference biases capacity'],
      [f.tablesOffset, l.tablesByteOffset / U32_BYTES, 'inference tables offset'],
      [f.tablesLength, payload.tablesBytes / U32_BYTES, 'inference tables length'],
      [f.tablesCapacity, l.tablesCapacityBytes / U32_BYTES, 'inference tables capacity'],
      [f.levelsOffset, l.levelsByteOffset / U32_BYTES, 'inference levels offset'],
      [f.levelsLength, payload.levelsBytes / U32_BYTES, 'inference levels length'],
      [f.levelsCapacity, l.levelsCapacityBytes / U32_BYTES, 'inference levels capacity'],
      [f.arenaLength, l.byteSize / U32_BYTES, 'inference arena length'],
    ];
    for (const [index, value, label] of expected) {
      assertHeaderField(h, index, value, label);
    }
  }

  if (request.runtime) {
    const spec = request.runtime;
    const l = spec.layout;
    const epoch = spec.epoch ?? 1;
    const generation = spec.generation ?? 0;
    const recordCap = spec.recordCap ?? 0;
    const recordStride = spec.recordStride ?? 0;
    assertU32('runtime epoch', epoch);
    if (epoch === 0) throw new RangeError('NRC runtime epoch must be non-zero');
    assertU32('runtime generation', generation);
    assertU32('runtime record capacity', recordCap);
    assertU32('runtime record stride', recordStride);
    if (l.headerByteOffset !== NRC_RUNTIME_HEADER_BYTE_OFFSET) {
      throw new RangeError(
        `NRC runtime header offset must equal shader constant ${NRC_RUNTIME_HEADER_BYTE_OFFSET}`,
      );
    }
    assertAligned('runtime arena byte size', l.byteSize, NRC_ARENA_ALIGNMENT);
    assertAligned('runtime diagnostics byte offset', l.diagnosticsByteOffset, U32_BYTES);
    assertWordRepresentable('runtime diagnostics byte offset', l.diagnosticsByteOffset);
    assertWordRepresentable('runtime diagnostics bytes', l.diagnosticsBytes);
    const diagnosticsEnd = checkedEnd(
      'runtime diagnostics', l.diagnosticsByteOffset, l.diagnosticsBytes,
    );
    if (diagnosticsEnd > l.headerByteOffset) {
      throw new RangeError('NRC runtime diagnostics overlap the publication header');
    }
    assertAligned('runtime header byte offset', l.headerByteOffset, NRC_ARENA_ALIGNMENT);
    assertWordRepresentable('runtime header byte offset', l.headerByteOffset);
    const headerEnd = checkedEnd(
      'runtime header', l.headerByteOffset, NRC_RUNTIME_HEADER_WORDS * U32_BYTES,
    );
    assertAligned('runtime claims byte offset', l.claimsByteOffset, NRC_ARENA_ALIGNMENT);
    assertWordRepresentable('runtime claims byte offset', l.claimsByteOffset);
    assertWordRepresentable('runtime claims bytes', l.claimsBytes);
    if (l.claimsByteOffset < headerEnd) {
      throw new RangeError('NRC runtime claims region overlaps the publication header');
    }
    const claimsEnd = checkedEnd('runtime claims', l.claimsByteOffset, l.claimsBytes);
    assertAligned('runtime records byte offset', l.recordsByteOffset, NRC_ARENA_ALIGNMENT);
    assertWordRepresentable('runtime records byte offset', l.recordsByteOffset);
    assertWordRepresentable('runtime records bytes', l.recordsBytes);
    if (l.recordsByteOffset < claimsEnd) {
      throw new RangeError('NRC runtime records region overlaps the claims region');
    }
    const recordsEnd = checkedEnd('runtime records', l.recordsByteOffset, l.recordsBytes);
    if (recordsEnd > l.byteSize) {
      throw new RangeError('NRC runtime records region exceeds arena allocation');
    }
    const requiredClaimBytes = checkedEnd('runtime claim requirement', 0, recordCap * U32_BYTES);
    if (requiredClaimBytes > l.claimsBytes) {
      throw new RangeError('NRC runtime record capacity exceeds the claims region');
    }
    checkedInteger('runtime record scalar requirement', recordCap * recordStride);
    const requiredRecordBytes = checkedInteger(
      'runtime record byte requirement',
      recordCap * recordStride * U32_BYTES,
    );
    if (requiredRecordBytes > l.recordsBytes) {
      throw new RangeError('NRC runtime record capacity/stride exceeds the records region');
    }
    assertWordRepresentable('runtime arena byte size', l.byteSize);
    const allocation = spec.allocationBytes ?? l.byteSize;
    assertAligned('runtime allocation byte size', allocation, U32_BYTES);
    assertAtMost('runtime arena allocation', l.byteSize, allocation);
    assertAtMost('runtime arena buffer size', l.byteSize, limits?.maxBufferSize);
    assertAtMost(
      'runtime arena storage binding size',
      l.byteSize,
      limits?.maxStorageBufferBindingSize,
    );

    const h = encodeRuntimeHeaderUnchecked(
      l, epoch, generation, recordCap, recordStride,
    );
    const f = NRC_RUNTIME_HEADER_FIELD;
    const expected: ReadonlyArray<readonly [number, number, string]> = [
      [0, NRC_RUNTIME_ARENA_MAGIC, 'runtime magic'],
      [1, NRC_RUNTIME_ARENA_VERSION, 'runtime version'],
      [2, epoch, 'runtime epoch'],
      [3, NRC_RUNTIME_ARENA_SCHEMA, 'runtime schema'],
      [4, generation, 'runtime generation'],
      [f.claimsOffset, l.claimsByteOffset / U32_BYTES, 'runtime claims offset'],
      [f.claimsLength, l.claimsBytes / U32_BYTES, 'runtime claims length'],
      [f.recordsOffset, l.recordsByteOffset / U32_BYTES, 'runtime records offset'],
      [f.recordsLength, l.recordsBytes / U32_BYTES, 'runtime records length'],
      [f.diagnosticsOffset, l.diagnosticsByteOffset / U32_BYTES, 'runtime diagnostics offset'],
      [f.diagnosticsLength, l.diagnosticsBytes / U32_BYTES, 'runtime diagnostics length'],
      [f.recordCap, recordCap, 'runtime record capacity'],
      [f.recordStride, recordStride, 'runtime record stride'],
      [f.arenaLength, l.byteSize / U32_BYTES, 'runtime arena length'],
    ];
    for (const [index, value, label] of expected) {
      assertHeaderField(h, index, value, label);
    }
  }
}

export function createNrcRuntimeArenaLayout(sizes: {
  readonly diagnosticsBytes: number;
  readonly claimsBytes: number;
  readonly recordsBytes: number;
}): NrcRuntimeArenaLayout {
  const diagnosticsBytes = checkedBytes('runtime diagnostics bytes', sizes.diagnosticsBytes);
  if (diagnosticsBytes > NRC_RUNTIME_HEADER_BYTE_OFFSET) {
    throw new RangeError('NRC runtime diagnostics overlap the publication header');
  }
  const claimsBytes = checkedBytes('runtime claims bytes', sizes.claimsBytes);
  const recordsBytes = checkedBytes('runtime records bytes', sizes.recordsBytes);
  const claimsByteOffset = align(
    checkedEnd(
      'runtime header',
      NRC_RUNTIME_HEADER_BYTE_OFFSET,
      NRC_RUNTIME_HEADER_WORDS * U32_BYTES,
    ),
  );
  const recordsByteOffset = align(checkedEnd(
    'runtime claims', claimsByteOffset, claimsBytes,
  ));
  const byteSize = align(checkedEnd(
    'runtime records', recordsByteOffset, recordsBytes,
  ));
  const layout = {
    byteSize,
    diagnosticsByteOffset: NRC_RUNTIME_DIAGNOSTICS_BYTE_OFFSET,
    diagnosticsBytes,
    headerByteOffset: NRC_RUNTIME_HEADER_BYTE_OFFSET,
    claimsByteOffset,
    claimsBytes,
    recordsByteOffset,
    recordsBytes,
  };
  validateNrcArenaLayouts({
    runtime: { layout, allocationBytes: byteSize, epoch: 1, generation: 0 },
  });
  return layout;
}

export function buildNrcRuntimeArenaHeader(
  layout: NrcRuntimeArenaLayout,
  epoch: number,
  generation: number,
  recordCap: number,
  recordStride: number,
): Uint32Array {
  validateNrcArenaLayouts({
    runtime: { layout, epoch, generation, recordCap, recordStride },
  });
  return encodeRuntimeHeaderUnchecked(
    layout, epoch, generation, recordCap, recordStride,
  );
}

export function nextNrcArenaEpoch(epoch: number): number {
  const next = (epoch + 1) >>> 0;
  return next === 0 ? 1 : next;
}
