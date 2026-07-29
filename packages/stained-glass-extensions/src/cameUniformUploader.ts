/**
 * Pack analytic H-channel came geometry into std140-aligned UBO arrays.
 *
 * Exported only from `@vitrum/stained-glass-extensions/host-came-ubo`.
 * This is a complete host-owned ABI: `packCameUBO` produces the packed arrays
 * and the host uploads them to its shader UBO. Vitrum render backends do not
 * implicitly consume or mutate this host-specific binding. It is intentionally
 * distinct from core's per-primitive `h-channel-came` analytic-shape contract,
 * which has no segment/node UBO binding.
 */

export interface CameSegment {
  readonly startWorld: readonly [number, number, number];
  readonly endWorld: readonly [number, number, number];
  readonly railWidth: number;
  readonly blockHeight: number;
  readonly webThickness: number;
}

export interface CameNode {
  readonly position: readonly [number, number, number];
  readonly radius: number;
}

export interface CameUploadOptions {
  /** Positive maximum number of segment records accepted by this pack operation. */
  readonly maxSegments?: number;
  /** Positive maximum number of node records accepted by this pack operation. */
  readonly maxNodes?: number;
  /**
   * Overflow is an error by default so geometry is never silently omitted.
   * Select `truncate` explicitly when the host intentionally accepts retaining
   * only the first maxSegments/maxNodes records.
   */
  readonly overflow?: 'error' | 'truncate';
}

export interface CamePackedUBO {
  readonly segments: Float32Array;
  readonly nodes: Float32Array;
  readonly segmentCount: number;
  readonly nodeCount: number;
}

/**
 * Float layout per packed segment (std140 vec4 rows, 4 rows × 4 floats = 16):
 *
 *  Row 0 — [0] startWorld.x  [1] startWorld.y  [2] startWorld.z  [3] railWidth
 *  Row 1 — [4] endWorld.x    [5] endWorld.y    [6] endWorld.z    [7] blockHeight
 *  Row 2 — [8] webThickness  [9–11] ABI padding
 *  Row 3 — [12–15] ABI padding
 *
 * Padding slots are zero-initialized. The host shader must declare the struct
 * with matching padding (e.g. `struct CameSegment { ...; _pad0: vec3f; _pad1: vec4f; }`).
 * They are not extension fields; adding data there requires an explicitly
 * versioned ABI rather than silently changing this layout.
 */
const SEGMENT_FLOATS = 16;
const NODE_FLOATS = 4;
const DEFAULT_MAX_SEGMENTS = 500;
const DEFAULT_MAX_NODES = 200;
const MAX_PACKED_TOTAL_BYTES = 256 * 1024 * 1024;
const OPTION_KEYS = new Set(['maxSegments', 'maxNodes', 'overflow']);

function allocationSafeCap(
  value: number | undefined,
  fallback: number,
  floatsPerRecord: number,
  label: string,
): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved <= 0) {
    throw new RangeError(`packCameUBO: ${label} must be a positive safe integer.`);
  }
  const bytes = resolved * floatsPerRecord * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(bytes) || bytes > MAX_PACKED_TOTAL_BYTES) {
    throw new RangeError(
      `packCameUBO: ${label} would permit a buffer larger than ` +
      `${MAX_PACKED_TOTAL_BYTES} bytes.`,
    );
  }
  return resolved;
}

function validateCombinedAllocationBudget(maxSegments: number, maxNodes: number): void {
  const segmentBytes = maxSegments * SEGMENT_FLOATS * Float32Array.BYTES_PER_ELEMENT;
  const nodeBytes = maxNodes * NODE_FLOATS * Float32Array.BYTES_PER_ELEMENT;
  const totalBytes = segmentBytes + nodeBytes;
  if (!Number.isSafeInteger(totalBytes) || totalBytes > MAX_PACKED_TOTAL_BYTES) {
    throw new RangeError(
      `packCameUBO: maxSegments and maxNodes authorize ${totalBytes} combined bytes; ` +
      `the total allocation budget is ${MAX_PACKED_TOTAL_BYTES} bytes.`,
    );
  }
}

function resolvePackedCount(
  inputLength: number,
  limit: number,
  overflow: 'error' | 'truncate',
  label: 'segments' | 'nodes',
): number {
  if (inputLength <= limit) return inputLength;
  if (overflow !== 'truncate') {
    throw new RangeError(
      `packCameUBO: received ${inputLength} ${label} but the configured limit is ${limit}; ` +
      `pass { overflow: 'truncate' } only when discarding the tail is intentional.`,
    );
  }
  return limit;
}

function warnAboutTruncation(
  inputLength: number,
  limit: number,
  label: 'segments' | 'nodes',
): void {
  if (inputLength <= limit) return;
  console.warn(
    `packCameUBO: received ${inputLength} ${label} but the configured limit is ${limit}; ` +
    'the explicitly requested truncate policy discarded the excess tail.',
  );
}

function finiteF32(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
    throw new TypeError(`packCameUBO: ${label} must be a finite float32 value.`);
  }
  return value;
}

function positiveF32(value: unknown, label: string): number {
  const finite = finiteF32(value, label);
  if (!(finite > 0) || !(Math.fround(finite) > 0)) {
    throw new RangeError(`packCameUBO: ${label} must remain positive when stored as float32.`);
  }
  return finite;
}

function vector3(value: unknown, label: string): readonly [number, number, number] {
  if (value == null || typeof value !== 'object'
      || !('length' in value) || (value as ArrayLike<unknown>).length !== 3) {
    throw new TypeError(`packCameUBO: ${label} must contain exactly three values.`);
  }
  const vector = value as ArrayLike<unknown>;
  return [
    finiteF32(vector[0], `${label}[0]`),
    finiteF32(vector[1], `${label}[1]`),
    finiteF32(vector[2], `${label}[2]`),
  ];
}

function validateSegment(value: unknown, index: number): CameSegment {
  if (value == null || typeof value !== 'object') {
    throw new TypeError(`packCameUBO: segments[${index}] must be an object.`);
  }
  const segment = value as CameSegment;
  const startWorld = vector3(segment.startWorld, `segments[${index}].startWorld`);
  const endWorld = vector3(segment.endWorld, `segments[${index}].endWorld`);
  if (
    Math.fround(startWorld[0]) === Math.fround(endWorld[0])
    && Math.fround(startWorld[1]) === Math.fround(endWorld[1])
    && Math.fround(startWorld[2]) === Math.fround(endWorld[2])
  ) {
    throw new RangeError(
      `packCameUBO: segments[${index}] startWorld and endWorld must remain distinct ` +
      'when stored as float32.',
    );
  }
  return {
    startWorld,
    endWorld,
    railWidth: positiveF32(segment.railWidth, `segments[${index}].railWidth`),
    blockHeight: positiveF32(segment.blockHeight, `segments[${index}].blockHeight`),
    webThickness: positiveF32(segment.webThickness, `segments[${index}].webThickness`),
  };
}

function validateNode(value: unknown, index: number): CameNode {
  if (value == null || typeof value !== 'object') {
    throw new TypeError(`packCameUBO: nodes[${index}] must be an object.`);
  }
  const node = value as CameNode;
  return {
    position: vector3(node.position, `nodes[${index}].position`),
    radius: positiveF32(node.radius, `nodes[${index}].radius`),
  };
}

export function packCameUBO(
  segments: ReadonlyArray<CameSegment>,
  nodes: ReadonlyArray<CameNode>,
  opts?: CameUploadOptions,
): CamePackedUBO {
  if (!Array.isArray(segments)) {
    throw new TypeError('packCameUBO: segments must be an array.');
  }
  if (!Array.isArray(nodes)) {
    throw new TypeError('packCameUBO: nodes must be an array.');
  }
  if (opts !== undefined && (opts === null || typeof opts !== 'object')) {
    throw new TypeError('packCameUBO: options must be an object when supplied.');
  }
  if (opts !== undefined) {
    for (const key of Reflect.ownKeys(opts)) {
      if (typeof key !== 'string' || !OPTION_KEYS.has(key)) {
        throw new TypeError(`packCameUBO: unknown option ${String(key)}.`);
      }
    }
  }
  const overflow = opts?.overflow ?? 'error';
  if (overflow !== 'error' && overflow !== 'truncate') {
    throw new TypeError("packCameUBO: overflow must be either 'error' or 'truncate'.");
  }
  const maxSeg = allocationSafeCap(
    opts?.maxSegments, DEFAULT_MAX_SEGMENTS, SEGMENT_FLOATS, 'maxSegments',
  );
  const maxNode = allocationSafeCap(
    opts?.maxNodes, DEFAULT_MAX_NODES, NODE_FLOATS, 'maxNodes',
  );
  validateCombinedAllocationBudget(maxSeg, maxNode);
  const segCount = resolvePackedCount(segments.length, maxSeg, overflow, 'segments');
  const nodeCount = resolvePackedCount(nodes.length, maxNode, overflow, 'nodes');

  // Validate the complete retained prefix before allocating or writing either
  // output. A bad node must not leave a half-packed segment buffer as an
  // observable intermediate, and truncation never inspects the discarded tail.
  const checkedSegments = new Array<CameSegment>(segCount);
  for (let i = 0; i < segCount; i++) checkedSegments[i] = validateSegment(segments[i], i);
  const checkedNodes = new Array<CameNode>(nodeCount);
  for (let i = 0; i < nodeCount; i++) checkedNodes[i] = validateNode(nodes[i], i);

  let segBuf: Float32Array;
  let nodeBuf: Float32Array;
  try {
    segBuf = new Float32Array(segCount * SEGMENT_FLOATS);
    nodeBuf = new Float32Array(nodeCount * NODE_FLOATS);
  } catch (error) {
    throw new RangeError(
      `packCameUBO: unable to allocate ${segCount * SEGMENT_FLOATS * 4} segment bytes ` +
      `and ${nodeCount * NODE_FLOATS * 4} node bytes.`,
      { cause: error },
    );
  }
  for (let i = 0; i < segCount; i++) {
    const seg = checkedSegments[i]!;
    const base = i * SEGMENT_FLOATS;
    segBuf[base + 0] = seg.startWorld[0];
    segBuf[base + 1] = seg.startWorld[1];
    segBuf[base + 2] = seg.startWorld[2];
    segBuf[base + 3] = seg.railWidth;
    segBuf[base + 4] = seg.endWorld[0];
    segBuf[base + 5] = seg.endWorld[1];
    segBuf[base + 6] = seg.endWorld[2];
    segBuf[base + 7] = seg.blockHeight;
    segBuf[base + 8] = seg.webThickness;
  }

  for (let i = 0; i < nodeCount; i++) {
    const node = checkedNodes[i]!;
    const base = i * NODE_FLOATS;
    nodeBuf[base + 0] = node.position[0];
    nodeBuf[base + 1] = node.position[1];
    nodeBuf[base + 2] = node.position[2];
    nodeBuf[base + 3] = node.radius;
  }

  // Warning is a publication side effect: emit it only after validation,
  // allocation, and packing have all succeeded.
  if (overflow === 'truncate') {
    warnAboutTruncation(segments.length, maxSeg, 'segments');
    warnAboutTruncation(nodes.length, maxNode, 'nodes');
  }

  return {
    segments: segBuf,
    nodes: nodeBuf,
    segmentCount: segCount,
    nodeCount,
  };
}
