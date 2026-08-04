import type { SceneBVHBuffers } from '../restir/bvhTypes.js';

/** Three bindings preserve practical scene capacity while bringing every
 * scene-using pipeline under the WebGPU guaranteed eight-storage-buffer floor. */
export const SCENE_STORAGE_ARENA_SHARD_COUNT = 3 as const;
export const SCENE_STORAGE_ARENA_MAGIC = 0x31534156; // ASCII VSA1 little-endian
export const SCENE_STORAGE_ARENA_VERSION = 4 as const;
export const SCENE_STORAGE_ARENA_HEADER_WORDS = 64 as const;
/** Fingerprint of the fixed segment order, element strides, and header ABI. */
export const SCENE_STORAGE_ARENA_SCHEMA = 0x4f37b8a1;
export const SCENE_STORAGE_ARENA_EPOCH_WORD = 7 as const;
export const SCENE_STORAGE_ARENA_SCHEMA_WORD = 62 as const;
export const SCENE_STORAGE_ARENA_COMPATIBILITY_WORD = 63 as const;

export const SCENE_STORAGE_SEGMENTS = [
  'bvhNodes',
  'bvhIndex',
  'bvhPositions',
  'opticalTriangleIdentity',
  'emitters',
  'emitterCdf',
  'emitterAlias',
  'tlasNodes',
  'tlasInstanceIndices',
  'tlasBlasRoots',
  'tlasInstanceWorldToLocal',
  'tlasInstanceLocalToWorld',
  'mneeFacetDomains',
  'bvhNormals',
  'opticalInstanceBoundaryIdBasePlusOne',
] as const;
export type SceneStorageSegment = typeof SCENE_STORAGE_SEGMENTS[number];

/** Canonical CPU/WGSL element strides for every packed scene segment. */
export const SCENE_STORAGE_SEGMENT_STRIDE_BYTES = Object.freeze({
  bvhNodes: 32,
  bvhIndex: 16,
  bvhPositions: 16,
  opticalTriangleIdentity: 8,
  emitters: 80,
  emitterCdf: 4,
  emitterAlias: 16,
  tlasNodes: 32,
  tlasInstanceIndices: 4,
  tlasBlasRoots: 4,
  tlasInstanceWorldToLocal: 64,
  tlasInstanceLocalToWorld: 64,
  mneeFacetDomains: 32,
  bvhNormals: 16,
  opticalInstanceBoundaryIdBasePlusOne: 4,
} as const satisfies Readonly<Record<SceneStorageSegment, number>>);

/** Geometry, acceleration-instance data, and lighting are separate shards so
 * common mutations can replace one binding without repacking unrelated bytes. */
export const SCENE_STORAGE_SHARD_SEGMENTS = [
  [
    'bvhNodes',
    'bvhIndex',
    'bvhPositions',
    'bvhNormals',
    'opticalTriangleIdentity',
  ],
  [
    'tlasNodes',
    'tlasInstanceIndices',
    'tlasBlasRoots',
    'tlasInstanceWorldToLocal',
    'tlasInstanceLocalToWorld',
    'mneeFacetDomains',
    'opticalInstanceBoundaryIdBasePlusOne',
  ],
  ['emitters', 'emitterCdf', 'emitterAlias'],
] as const satisfies readonly (readonly SceneStorageSegment[])[];

const SEGMENT_INDEX = new Map<SceneStorageSegment, number>(
  SCENE_STORAGE_SEGMENTS.map((name, index) => [name, index]),
);

export interface SceneStorageSegmentSource {
  readonly data: ArrayBuffer;
  /** Logical element count used by WGSL bounds/length helpers. */
  readonly count: number;
}

interface SparseSourcePatch {
  readonly byteOffset: number;
  readonly data: ArrayBuffer;
}

interface SparseSourceState {
  readonly base: ArrayBuffer;
  /** Sorted, non-overlapping overlays over `base`; adjacent ranges are merged. */
  readonly patches: readonly SparseSourcePatch[];
  materialized?: ArrayBuffer;
}

const SOURCE_STATE: unique symbol = Symbol('scene-storage-source-state');
const SHARD_CACHE: unique symbol = Symbol('scene-storage-shard-cache');

type InternalSceneStorageSource = SceneStorageSegmentSource & {
  readonly [SOURCE_STATE]?: SparseSourceState;
};

type InternalSceneStoragePayload = SceneStorageArenaPayload & {
  readonly [SHARD_CACHE]: Map<number, SceneStorageArenaShard>;
};

export type SceneStorageSegmentSources = Readonly<
  Record<SceneStorageSegment, SceneStorageSegmentSource>
>;

export interface SceneStorageSegmentLayout {
  readonly shard: number;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly count: number;
}

export interface SceneStorageArenaShard {
  readonly index: number;
  readonly bytes: ArrayBuffer;
}

export interface SceneStorageArenaPayload {
  readonly shards: readonly [
    SceneStorageArenaShard,
    SceneStorageArenaShard,
    SceneStorageArenaShard,
  ];
  readonly segments: Readonly<Record<SceneStorageSegment, SceneStorageSegmentLayout>>;
  readonly sources: SceneStorageSegmentSources;
  /** Monotonic publication identity shared by all three shard headers. */
  readonly sceneEpoch: number;
  /** Collision-free host identity for retained geometry/TLAS content.
   * Lighting-only candidates must match this generation before publication. */
  readonly geometryGeneration: number;
  /** Low u32 of `geometryGeneration`, mirrored into packed GPU headers.
   * Host compatibility checks use the full safe-integer generation. */
  readonly compatibilityEpoch: number;
}

export interface SceneStorageArenaSourcePatch {
  readonly segment: SceneStorageSegment;
  readonly byteOffset: number;
  readonly data: ArrayBuffer;
}

export interface SceneStorageArenaLimitFailure {
  readonly shard: number;
  readonly actualBytes: number;
  readonly maxBytes: number;
}

export class SceneStorageArenaLimitError extends RangeError {
  readonly code = 'walkaround-hybrid.scene-storage-arena-limit' as const;
  readonly failures: readonly SceneStorageArenaLimitFailure[];
  readonly shardCount = SCENE_STORAGE_ARENA_SHARD_COUNT;
  readonly maxStorageBufferBindingSize: number;

  constructor(failures: readonly SceneStorageArenaLimitFailure[], maxBytes: number) {
    super(
      '[BvhBufferHost] packed scene storage exceeds the declared three-shard capacity: ' +
      failures.map((failure) =>
        'shard' + failure.shard + '=' + failure.actualBytes +
        ' bytes (max ' + failure.maxBytes + ')',
      ).join(', ') +
      '. Reduce scene geometry/instances/emitters or select an adapter with a ' +
      'larger maxStorageBufferBindingSize.',
    );
    this.name = 'SceneStorageArenaLimitError';
    this.failures = failures;
    this.maxStorageBufferBindingSize = maxBytes;
  }
}

function copyArrayBuffer(value: ArrayBuffer): ArrayBuffer {
  return value.slice(0);
}

function sourceState(value: SceneStorageSegmentSource): SparseSourceState | undefined {
  return (value as InternalSceneStorageSource)[SOURCE_STATE];
}

function sourceByteLength(value: SceneStorageSegmentSource): number {
  return sourceState(value)?.base.byteLength ?? value.data.byteLength;
}

function materializeSource(value: SceneStorageSegmentSource): ArrayBuffer {
  const state = sourceState(value);
  if (state == null) return value.data;
  if (state.patches.length === 0) return state.base;
  if (state.materialized != null) return state.materialized;
  const result = state.base.slice(0);
  for (const patch of state.patches) {
    new Uint8Array(result, patch.byteOffset, patch.data.byteLength)
      .set(new Uint8Array(patch.data));
  }
  state.materialized = result;
  return result;
}

function baseSource(
  data: ArrayBuffer,
  count: number,
  copy: boolean,
): SceneStorageSegmentSource {
  const base = copy ? copyArrayBuffer(data) : data;
  const state: SparseSourceState = { base, patches: [] };
  const value = {
    get data(): ArrayBuffer {
      return materializeSource(value);
    },
    count,
    [SOURCE_STATE]: state,
  } satisfies InternalSceneStorageSource;
  return value;
}

function coalesceSparsePatches(
  patches: readonly SparseSourcePatch[],
): readonly SparseSourcePatch[] {
  if (patches.length < 2) return patches;
  const result: SparseSourcePatch[] = [];
  for (const patch of patches) {
    const previous = result[result.length - 1];
    if (
      previous == null ||
      previous.byteOffset + previous.data.byteLength !== patch.byteOffset
    ) {
      result.push(patch);
      continue;
    }
    const joined = new Uint8Array(previous.data.byteLength + patch.data.byteLength);
    joined.set(new Uint8Array(previous.data));
    joined.set(new Uint8Array(patch.data), previous.data.byteLength);
    result[result.length - 1] = {
      byteOffset: previous.byteOffset,
      data: joined.buffer,
    };
  }
  return result;
}

/**
 * Overlay one immutable patch while retaining only still-visible bytes from
 * older overlays. The resulting list is sorted, non-overlapping, and coalesced;
 * repeated animation updates therefore replace history instead of chaining it.
 */
function overlaySparsePatch(
  current: readonly SparseSourcePatch[],
  incoming: SceneStorageArenaSourcePatch,
): readonly SparseSourcePatch[] {
  const start = incoming.byteOffset;
  const end = start + incoming.data.byteLength;
  const next: SparseSourcePatch[] = [];
  for (const patch of current) {
    const patchStart = patch.byteOffset;
    const patchEnd = patchStart + patch.data.byteLength;
    if (patchEnd <= start || patchStart >= end) {
      next.push(patch);
      continue;
    }
    if (patchStart < start) {
      next.push({
        byteOffset: patchStart,
        data: patch.data.slice(0, start - patchStart),
      });
    }
    if (patchEnd > end) {
      next.push({
        byteOffset: end,
        data: patch.data.slice(end - patchStart),
      });
    }
  }
  next.push({ byteOffset: start, data: incoming.data.slice(0) });
  next.sort((a, b) => a.byteOffset - b.byteOffset);
  return coalesceSparsePatches(next);
}

function sparseSource(
  current: SceneStorageSegmentSource,
  patches: readonly SceneStorageArenaSourcePatch[],
): SceneStorageSegmentSource {
  const currentState = sourceState(current) ?? {
    base: current.data,
    patches: [],
  };
  let normalized = currentState.patches;
  for (const patch of patches) normalized = overlaySparsePatch(normalized, patch);
  const state: SparseSourceState = {
    base: currentState.base,
    patches: normalized,
  };
  const value = {
    get data(): ArrayBuffer {
      return materializeSource(value);
    },
    count: current.count,
    [SOURCE_STATE]: state,
  } satisfies InternalSceneStorageSource;
  return value;
}

function sourceWordU32(value: SceneStorageSegmentSource, wordIndex: number): number {
  const byteOffset = wordIndex * 4;
  const state = sourceState(value);
  if (state != null) {
    for (const patch of state.patches) {
      if (patch.byteOffset > byteOffset) break;
      if (byteOffset + 4 <= patch.byteOffset + patch.data.byteLength) {
        return new Uint32Array(
          patch.data,
          byteOffset - patch.byteOffset,
          1,
        )[0]!;
      }
    }
    return new Uint32Array(state.base, byteOffset, 1)[0]!;
  }
  return new Uint32Array(value.data, byteOffset, 1)[0]!;
}

function source(
  handle: { readonly cpuData: ArrayBuffer; readonly count: number } | undefined,
): SceneStorageSegmentSource {
  if (handle == null) return baseSource(new ArrayBuffer(16), 0, false);
  return baseSource(handle.cpuData, handle.count, true);
}

export function emptySceneStorageSegmentSources(): SceneStorageSegmentSources {
  return Object.fromEntries(
    SCENE_STORAGE_SEGMENTS.map((name) => [
      name,
      baseSource(new ArrayBuffer(16), 0, false),
    ]),
  ) as unknown as SceneStorageSegmentSources;
}

export function sceneGeometryStorageSources(
  bvh: Pick<
    SceneBVHBuffers,
    | 'bvhNodes'
    | 'bvhIndex'
    | 'bvhPositions'
    | 'bvhNormals'
    | 'opticalTriangleIdentity'
    | 'opticalInstanceBoundaryIdBasePlusOne'
    | 'mneeFacetDomains'
    | 'bvhMode'
    | 'tlas'
  >,
): Pick<
  SceneStorageSegmentSources,
  | 'bvhNodes'
  | 'bvhIndex'
  | 'bvhPositions'
  | 'bvhNormals'
  | 'opticalTriangleIdentity'
  | 'tlasNodes'
  | 'tlasInstanceIndices'
  | 'tlasBlasRoots'
  | 'tlasInstanceWorldToLocal'
  | 'tlasInstanceLocalToWorld'
  | 'mneeFacetDomains'
  | 'opticalInstanceBoundaryIdBasePlusOne'
> {
  const tlas = bvh.bvhMode === 'tlas' ? bvh.tlas : undefined;
  return {
    bvhNodes: source(bvh.bvhNodes),
    bvhIndex: source(bvh.bvhIndex),
    bvhPositions: source(bvh.bvhPositions),
    bvhNormals: source(bvh.bvhNormals),
    opticalTriangleIdentity: source(bvh.opticalTriangleIdentity),
    tlasNodes: source(tlas?.nodes),
    tlasInstanceIndices: source(tlas?.instanceIndices),
    tlasBlasRoots: source(tlas?.blasRoots),
    tlasInstanceWorldToLocal: source(tlas?.worldToLocal),
    tlasInstanceLocalToWorld: source(tlas?.localToWorld),
    mneeFacetDomains: source(bvh.mneeFacetDomains),
    opticalInstanceBoundaryIdBasePlusOne: source(
      bvh.opticalInstanceBoundaryIdBasePlusOne,
    ),
  };
}

export function sceneLightingStorageSources(
  bvh: Pick<SceneBVHBuffers, 'emitters' | 'emitterCdf' | 'emitterAlias'>,
): Pick<SceneStorageSegmentSources, 'emitters' | 'emitterCdf' | 'emitterAlias'> {
  return {
    emitters: source(bvh.emitters),
    emitterCdf: source(bvh.emitterCdf),
    emitterAlias: source(bvh.emitterAlias),
  };
}

export function sceneStorageSegmentSources(
  bvh: Pick<
    SceneBVHBuffers,
    | 'bvhNodes'
    | 'bvhIndex'
    | 'bvhPositions'
    | 'emitters'
    | 'emitterCdf'
    | 'emitterAlias'
    | 'bvhNormals'
    | 'opticalTriangleIdentity'
    | 'opticalInstanceBoundaryIdBasePlusOne'
    | 'mneeFacetDomains'
    | 'bvhMode'
    | 'tlas'
  >,
): SceneStorageSegmentSources {
  return {
    ...sceneGeometryStorageSources(bvh),
    ...sceneLightingStorageSources(bvh),
  };
}

function alignWords(words: number): number {
  // Segment offsets are valid GPUBufferBinding offsets for GPU skinning, RC,
  // and ReGIR subrange bindings on the guaranteed 256-byte alignment floor.
  return (words + 63) & ~63;
}

export function nextSceneStorageArenaEpoch(epoch: number): number {
  const next = (epoch + 1) >>> 0;
  return next === 0 ? 1 : next;
}

function validateSource(name: SceneStorageSegment, value: SceneStorageSegmentSource): void {
  if (!Number.isSafeInteger(value.count) || value.count < 0) {
    throw new RangeError(
      '[sceneStorageArena] ' + name + ' count must be a non-negative safe integer.',
    );
  }
  const byteLength = sourceByteLength(value);
  if (value.count === 0) {
    if (byteLength !== 16) {
      throw new RangeError(
        '[sceneStorageArena] ' + name +
        ' count=0 requires the canonical 16-byte dummy payload; got ' +
        byteLength + ' bytes.',
      );
    }
    return;
  }
  const stride = SCENE_STORAGE_SEGMENT_STRIDE_BYTES[name];
  const expectedBytes = value.count * stride;
  if (!Number.isSafeInteger(expectedBytes) || byteLength !== expectedBytes) {
    throw new RangeError(
      '[sceneStorageArena] ' + name + ' count ' + value.count +
      ' requires exactly ' + expectedBytes + ' bytes at the ' + stride +
      '-byte ABI stride; got ' + byteLength + ' bytes.',
    );
  }
}

function validateGeneration(generation: number): void {
  if (!Number.isSafeInteger(generation) || generation <= 0) {
    throw new RangeError(
      '[sceneStorageArena] geometryGeneration must be a positive safe integer.',
    );
  }
}

function nextGeometryGeneration(generation: number): number {
  validateGeneration(generation);
  if (generation === Number.MAX_SAFE_INTEGER) {
    throw new RangeError(
      '[sceneStorageArena] geometryGeneration exhausted Number.MAX_SAFE_INTEGER.',
    );
  }
  return generation + 1;
}

function compatibilityWord(generation: number): number {
  validateGeneration(generation);
  return generation >>> 0;
}

function isGeometrySegment(segment: SceneStorageSegment): boolean {
  return segment !== 'emitters' && segment !== 'emitterCdf' && segment !== 'emitterAlias';
}

function validateSceneEpoch(sceneEpoch: number): void {
  if (!Number.isSafeInteger(sceneEpoch) || sceneEpoch <= 0 || sceneEpoch > 0xffff_ffff) {
    throw new RangeError('[sceneStorageArena] sceneEpoch must be a non-zero u32.');
  }
}

function layoutSceneStorageSources(
  sources: SceneStorageSegmentSources,
): Readonly<Record<SceneStorageSegment, SceneStorageSegmentLayout>> {
  const layouts = {} as Record<SceneStorageSegment, SceneStorageSegmentLayout>;
  SCENE_STORAGE_SHARD_SEGMENTS.forEach((names, shardIndex) => {
    let cursorWords: number = SCENE_STORAGE_ARENA_HEADER_WORDS;
    for (const name of names) {
      cursorWords = alignWords(cursorWords);
      const value = sources[name];
      const byteLength = sourceByteLength(value);
      layouts[name] = {
        shard: shardIndex,
        byteOffset: cursorWords * 4,
        byteLength,
        count: value.count,
      };
      cursorWords += Math.max(4, Math.ceil(byteLength / 4));
    }
  });
  return layouts;
}

function sceneStorageArenaShardByteLength(
  payload: SceneStorageArenaPayload,
  shardIndex: number,
): number {
  const names = SCENE_STORAGE_SHARD_SEGMENTS[shardIndex];
  if (names == null) throw new RangeError('invalid scene-storage shard index');
  let words: number = SCENE_STORAGE_ARENA_HEADER_WORDS;
  for (const name of names) {
    const layout = payload.segments[name];
    words = Math.max(
      words,
      layout.byteOffset / 4 + Math.max(4, Math.ceil(layout.byteLength / 4)),
    );
  }
  return alignWords(words) * 4;
}

export function sceneStorageArenaHeaderBytes(
  payload: SceneStorageArenaPayload,
  shardIndex: number,
): ArrayBuffer {
  const names = SCENE_STORAGE_SHARD_SEGMENTS[shardIndex];
  if (names == null) throw new RangeError('invalid scene-storage shard index');
  const bytes = new ArrayBuffer(SCENE_STORAGE_ARENA_HEADER_WORDS * 4);
  const header = new Uint32Array(bytes);
  header[0] = SCENE_STORAGE_ARENA_MAGIC;
  header[1] = SCENE_STORAGE_ARENA_VERSION;
  header[2] = shardIndex;
  header[3] = SCENE_STORAGE_ARENA_SHARD_COUNT;
  header[4] = sceneStorageArenaShardByteLength(payload, shardIndex) / 4;
  header[5] = names.length;
  header[6] = SCENE_STORAGE_ARENA_HEADER_WORDS;
  header[SCENE_STORAGE_ARENA_EPOCH_WORD] = payload.sceneEpoch;
  for (const name of SCENE_STORAGE_SEGMENTS) {
    const globalIndex = SEGMENT_INDEX.get(name)!;
    const layout = payload.segments[name];
    header[8 + globalIndex * 2] =
      layout.shard === shardIndex ? layout.byteOffset / 4 : 0;
    header[9 + globalIndex * 2] = layout.count;
  }
  header[SCENE_STORAGE_ARENA_SCHEMA_WORD] = SCENE_STORAGE_ARENA_SCHEMA;
  header[SCENE_STORAGE_ARENA_COMPATIBILITY_WORD] = payload.compatibilityEpoch;
  return bytes;
}

export function sceneStorageArenaShard(
  payload: SceneStorageArenaPayload,
  shardIndex: number,
): SceneStorageArenaShard {
  const internal = payload as InternalSceneStoragePayload;
  const cached = internal[SHARD_CACHE].get(shardIndex);
  if (cached != null) return cached;
  const names = SCENE_STORAGE_SHARD_SEGMENTS[shardIndex];
  if (names == null) throw new RangeError('invalid scene-storage shard index');
  const bytes = new ArrayBuffer(sceneStorageArenaShardByteLength(payload, shardIndex));
  new Uint8Array(bytes, 0, SCENE_STORAGE_ARENA_HEADER_WORDS * 4)
    .set(new Uint8Array(sceneStorageArenaHeaderBytes(payload, shardIndex)));
  for (const name of names) {
    const layout = payload.segments[name];
    new Uint8Array(bytes, layout.byteOffset, layout.byteLength)
      .set(new Uint8Array(materializeSource(payload.sources[name])));
  }
  const shard = { index: shardIndex, bytes };
  internal[SHARD_CACHE].set(shardIndex, shard);
  return shard;
}

function createSceneStorageArenaPayload(
  sources: SceneStorageSegmentSources,
  segments: Readonly<Record<SceneStorageSegment, SceneStorageSegmentLayout>>,
  sceneEpoch: number,
  geometryGeneration: number,
): SceneStorageArenaPayload {
  validateSceneEpoch(sceneEpoch);
  validateGeneration(geometryGeneration);
  const cache = new Map<number, SceneStorageArenaShard>();
  const payload: InternalSceneStoragePayload = {
    get shards(): SceneStorageArenaPayload['shards'] {
      return [
        sceneStorageArenaShard(payload, 0),
        sceneStorageArenaShard(payload, 1),
        sceneStorageArenaShard(payload, 2),
      ];
    },
    segments,
    sources,
    sceneEpoch,
    geometryGeneration,
    compatibilityEpoch: compatibilityWord(geometryGeneration),
    [SHARD_CACHE]: cache,
  };
  return payload;
}

/** Build an immutable, compacted CPU payload. Packed shards stay lazy so a
 * caller that uploads one replacement shard never materializes the others. */
export function buildSceneStorageArena(
  input: SceneStorageSegmentSources,
  sceneEpoch = 1,
  geometryGeneration = 1,
): SceneStorageArenaPayload {
  validateSceneEpoch(sceneEpoch);
  validateGeneration(geometryGeneration);
  const sources = Object.fromEntries(
    SCENE_STORAGE_SEGMENTS.map((name) => {
      const value = input[name];
      validateSource(name, value);
      return [
        name,
        baseSource(materializeSource(value), value.count, true),
      ];
    }),
  ) as unknown as SceneStorageSegmentSources;
  return createSceneStorageArenaPayload(
    sources,
    layoutSceneStorageSources(sources),
    sceneEpoch,
    geometryGeneration,
  );
}

/** Adopt retained internal sources and copy only actual replacements. */
export function retainSceneStorageArenaSources(
  payload: SceneStorageArenaPayload,
  replacements: Readonly<
    Partial<Record<SceneStorageSegment, SceneStorageSegmentSource>>
  >,
  sceneEpoch: number,
  geometryGeneration: number,
): SceneStorageArenaPayload {
  const sources = { ...payload.sources } as Record<
    SceneStorageSegment,
    SceneStorageSegmentSource
  >;
  for (const name of SCENE_STORAGE_SEGMENTS) {
    const replacement = replacements[name];
    if (replacement == null) continue;
    validateSource(name, replacement);
    sources[name] = sourceState(replacement) == null
      ? baseSource(replacement.data, replacement.count, true)
      : replacement;
  }
  return createSceneStorageArenaPayload(
    sources,
    layoutSceneStorageSources(sources),
    sceneEpoch,
    geometryGeneration,
  );
}

export function buildSceneStorageArenaForBvh(
  bvh: Parameters<typeof sceneStorageSegmentSources>[0],
): SceneStorageArenaPayload {
  return buildSceneStorageArena(sceneStorageSegmentSources(bvh));
}

export function replaceSceneStorageArenaSources(
  payload: SceneStorageArenaPayload,
  replacements: Readonly<Partial<Record<SceneStorageSegment, SceneStorageSegmentSource>>>,
  sceneEpoch = nextSceneStorageArenaEpoch(payload.sceneEpoch),
): SceneStorageArenaPayload {
  const changesGeometry = Object.keys(replacements)
    .some((name) => isGeometrySegment(name as SceneStorageSegment));
  return buildSceneStorageArena({
    ...payload.sources,
    ...replacements,
  }, sceneEpoch, changesGeometry
    ? nextGeometryGeneration(payload.geometryGeneration)
    : payload.geometryGeneration);
}

/**
 * Apply fixed-size dirty slices without rebuilding or copying unrelated source
 * buffers. Patches targeting one segment share one copy, and any number of
 * geometry/TLAS patches advance the collision-free host generation once.
 * Packed shard bytes are materialized lazily only for diagnostic consumers;
 * the live GPU path already owns resident arena buffers plus dirty copies.
 */
export function patchSceneStorageArenaSources(
  payload: SceneStorageArenaPayload,
  patches: readonly SceneStorageArenaSourcePatch[],
  sceneEpoch = nextSceneStorageArenaEpoch(payload.sceneEpoch),
): SceneStorageArenaPayload {
  if (patches.length === 0) return payload;
  if (!Number.isSafeInteger(sceneEpoch) || sceneEpoch <= 0 || sceneEpoch > 0xffff_ffff) {
    throw new RangeError('[sceneStorageArena] sceneEpoch must be a non-zero u32.');
  }
  const bySegment = new Map<SceneStorageSegment, SceneStorageArenaSourcePatch[]>();
  for (const patch of patches) {
    const current = payload.sources[patch.segment];
    if (
      patch.data.byteLength === 0 ||
      (patch.byteOffset & 3) !== 0 ||
      (patch.data.byteLength & 3) !== 0
    ) {
      throw new RangeError(
        '[sceneStorageArena] source patches must be non-empty and four-byte aligned.',
      );
    }
    if (
      patch.byteOffset < 0 ||
      patch.byteOffset + patch.data.byteLength > sourceByteLength(current)
    ) {
      throw new RangeError(
        '[sceneStorageArena] ' + patch.segment + ' patch [' + patch.byteOffset + ', ' +
        (patch.byteOffset + patch.data.byteLength) + ') exceeds ' +
        sourceByteLength(current) + ' bytes.',
      );
    }
    const entries = bySegment.get(patch.segment);
    if (entries == null) bySegment.set(patch.segment, [patch]);
    else entries.push(patch);
  }

  const sources = { ...payload.sources } as Record<
    SceneStorageSegment,
    SceneStorageSegmentSource
  >;
  for (const [segment, segmentPatches] of bySegment) {
    const current = payload.sources[segment];
    sources[segment] = sparseSource(current, segmentPatches);
  }
  const geometryGeneration = [...bySegment.keys()].some(isGeometrySegment)
    ? nextGeometryGeneration(payload.geometryGeneration)
    : payload.geometryGeneration;
  return createSceneStorageArenaPayload(
    sources,
    payload.segments,
    sceneEpoch,
    geometryGeneration,
  );
}

export function patchSceneStorageArenaSource(
  payload: SceneStorageArenaPayload,
  segment: SceneStorageSegment,
  byteOffset: number,
  data: ArrayBuffer,
  sceneEpoch = nextSceneStorageArenaEpoch(payload.sceneEpoch),
): SceneStorageArenaPayload {
  return patchSceneStorageArenaSources(
    payload,
    [{ segment, byteOffset, data }],
    sceneEpoch,
  );
}

export function maxSceneStorageArenaBindingBytes(
  limits: GPUSupportedLimits | null | undefined,
): number {
  const reported = limits as unknown as Record<string, unknown> | null | undefined;
  const readLimit = (name: string, fallback: number): number => {
    const value = reported?.[name];
    if (value == null) return fallback;
    const numeric = Number(value);
    if (!Number.isSafeInteger(numeric) || numeric <= 0) {
      throw new RangeError(
        '[sceneStorageArena] reported ' + name +
        ' must be a positive finite safe integer; got ' +
        Object.prototype.toString.call(value) + '.',
      );
    }
    return numeric;
  };
  const binding = readLimit('maxStorageBufferBindingSize', 128 * 1024 * 1024);
  const buffer = readLimit('maxBufferSize', 256 * 1024 * 1024);
  return Math.min(binding, buffer);
}

export function assertSceneStorageArenaFits(
  payload: SceneStorageArenaPayload,
  limits: GPUSupportedLimits | null | undefined,
  shardIndices: readonly number[] = [0, 1, 2],
): void {
  const maxBytes = maxSceneStorageArenaBindingBytes(limits);
  const failures = shardIndices
    .map((index) => sceneStorageArenaShard(payload, index))
    .filter((shard) => shard.bytes.byteLength > maxBytes)
    .map((shard) => ({
      shard: shard.index,
      actualBytes: shard.bytes.byteLength,
      maxBytes,
    }));
  if (failures.length > 0) throw new SceneStorageArenaLimitError(failures, maxBytes);
}

export function sceneStorageArenaSegmentU32(
  payload: SceneStorageArenaPayload,
  segment: SceneStorageSegment,
  wordIndex: number,
): number {
  const layout = payload.segments[segment];
  if (!Number.isSafeInteger(wordIndex) || wordIndex < 0 || wordIndex * 4 >= layout.byteLength) {
    throw new RangeError(
      '[sceneStorageArena] ' + segment + ' word index ' + wordIndex + ' is out of bounds.',
    );
  }
  return sourceWordU32(payload.sources[segment], wordIndex);
}

export function sceneStorageArenaSegmentF32(
  payload: SceneStorageArenaPayload,
  segment: SceneStorageSegment,
  wordIndex: number,
): number {
  return new Float32Array(
    Uint32Array.of(sceneStorageArenaSegmentU32(payload, segment, wordIndex)).buffer,
  )[0]!;
}
