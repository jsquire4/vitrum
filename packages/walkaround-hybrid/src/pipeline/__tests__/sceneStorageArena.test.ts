import { describe, expect, it } from 'vitest';
import {
  SCENE_STORAGE_ARENA_HEADER_WORDS,
  SCENE_STORAGE_ARENA_EPOCH_WORD,
  SCENE_STORAGE_ARENA_MAGIC,
  SCENE_STORAGE_ARENA_SCHEMA,
  SCENE_STORAGE_ARENA_SCHEMA_WORD,
  SCENE_STORAGE_ARENA_COMPATIBILITY_WORD,
  SCENE_STORAGE_ARENA_SHARD_COUNT,
  SCENE_STORAGE_ARENA_VERSION,
  SCENE_STORAGE_SEGMENT_STRIDE_BYTES,
  SCENE_STORAGE_SEGMENTS,
  SceneStorageArenaLimitError,
  assertSceneStorageArenaFits,
  buildSceneStorageArena,
  maxSceneStorageArenaBindingBytes,
  patchSceneStorageArenaSources,
  replaceSceneStorageArenaSources,
  sceneStorageArenaSegmentF32,
  sceneStorageArenaSegmentU32,
  type SceneStorageSegmentSources,
} from '../sceneStorageArena.js';

function sources(): SceneStorageSegmentSources {
  const result = {} as Record<string, { data: ArrayBuffer; count: number }>;
  SCENE_STORAGE_SEGMENTS.forEach((name, index) => {
    const count = index + 1;
    const words = new Uint32Array(
      SCENE_STORAGE_SEGMENT_STRIDE_BYTES[name] * count / 4,
    );
    for (let i = 0; i < words.length; i++) words[i] = (index + 1) * 1000 + i;
    result[name] = { data: words.buffer, count };
  });
  return result as unknown as SceneStorageSegmentSources;
}

describe('three-shard scene storage arena', () => {
  it('writes one versioned schema + epoch across 256-byte-aligned shards', () => {
    const payload = buildSceneStorageArena(sources(), 17);
    expect(payload.shards).toHaveLength(SCENE_STORAGE_ARENA_SHARD_COUNT);
    for (const shard of payload.shards) {
      const header = new Uint32Array(
        shard.bytes,
        0,
        SCENE_STORAGE_ARENA_HEADER_WORDS,
      );
      expect(header[0]).toBe(SCENE_STORAGE_ARENA_MAGIC);
      expect(header[1]).toBe(SCENE_STORAGE_ARENA_VERSION);
      expect(header[2]).toBe(shard.index);
      expect(header[3]).toBe(SCENE_STORAGE_ARENA_SHARD_COUNT);
      expect(header[4]! * 4).toBe(shard.bytes.byteLength);
      expect(header[SCENE_STORAGE_ARENA_EPOCH_WORD]).toBe(17);
      expect(header[SCENE_STORAGE_ARENA_SCHEMA_WORD]).toBe(SCENE_STORAGE_ARENA_SCHEMA);
      expect(header[SCENE_STORAGE_ARENA_COMPATIBILITY_WORD]).toBe(payload.compatibilityEpoch);
      SCENE_STORAGE_SEGMENTS.forEach((name, segmentIndex) => {
        expect(header[9 + segmentIndex * 2]).toBe(payload.segments[name].count);
      });
    }
    for (const name of SCENE_STORAGE_SEGMENTS) {
      expect(payload.segments[name].byteOffset % 256).toBe(0);
    }
  });

  it('round-trips every source word through the packed CPU loader oracle', () => {
    const input = sources();
    const payload = buildSceneStorageArena(input);
    for (const name of SCENE_STORAGE_SEGMENTS) {
      const expected = new Uint32Array(input[name].data);
      for (let word = 0; word < expected.length; word++) {
        expect(sceneStorageArenaSegmentU32(payload, name, word)).toBe(expected[word]);
      }
    }
  });

  it('bitcasts f32 lanes without numerical conversion', () => {
    const input = sources();
    const floats = new Float32Array([0, -1.25, 2.5, Number.POSITIVE_INFINITY]);
    const payload = buildSceneStorageArena({
      ...input,
      bvhPositions: { data: floats.buffer, count: 1 },
    });
    expect(sceneStorageArenaSegmentF32(payload, 'bvhPositions', 1)).toBe(-1.25);
    expect(sceneStorageArenaSegmentF32(payload, 'bvhPositions', 2)).toBe(2.5);
    expect(sceneStorageArenaSegmentF32(payload, 'bvhPositions', 3)).toBe(
      Number.POSITIVE_INFINITY,
    );
  });

  it('rebuilds only from immutable source snapshots', () => {
    const initial = buildSceneStorageArena(sources());
    const replacement = new Uint32Array([9, 8, 7, 6]);
    const next = replaceSceneStorageArenaSources(initial, {
      emitterCdf: { data: replacement.buffer, count: replacement.length },
    });
    expect(sceneStorageArenaSegmentU32(initial, 'emitterCdf', 0)).not.toBe(9);
    expect(sceneStorageArenaSegmentU32(next, 'emitterCdf', 0)).toBe(9);
    expect(next.segments.emitterCdf.count).toBe(4);
    expect(next.compatibilityEpoch).toBe(initial.compatibilityEpoch);
    expect(next.sceneEpoch).toBe(initial.sceneEpoch + 1);
  });

  it('advances the geometry generation for geometry but not lighting', () => {
    const initial = buildSceneStorageArena(sources());
    const lighting = replaceSceneStorageArenaSources(initial, {
      emitters: { data: new Uint32Array(20).fill(1).buffer, count: 1 },
    });
    const geometry = replaceSceneStorageArenaSources(initial, {
      bvhPositions: { data: new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8]).buffer, count: 2 },
    });
    expect(lighting.compatibilityEpoch).toBe(initial.compatibilityEpoch);
    expect(geometry.compatibilityEpoch).not.toBe(initial.compatibilityEpoch);
    expect(lighting.geometryGeneration).toBe(initial.geometryGeneration);
    expect(geometry.geometryGeneration).toBe(initial.geometryGeneration + 1);
  });

  it('advances generation for same-size geometry and ignores lighting bytes', () => {
    const input = sources();
    const initial = buildSceneStorageArena(input);
    const sameSizeGeometry = input.bvhPositions.data.slice(0);
    const geometryWords = new Uint32Array(sameSizeGeometry);
    geometryWords[0] = geometryWords[0]! ^ 0xffff_ffff;
    const geometry = replaceSceneStorageArenaSources(initial, {
      bvhPositions: {
        data: sameSizeGeometry,
        count: input.bvhPositions.count,
      },
    });
    const sameSizeLighting = input.emitters.data.slice(0);
    const lightingWords = new Uint32Array(sameSizeLighting);
    lightingWords[0] = lightingWords[0]! ^ 0xffff_ffff;
    const lighting = replaceSceneStorageArenaSources(initial, {
      emitters: { data: sameSizeLighting, count: input.emitters.count },
    });
    expect(geometry.compatibilityEpoch).not.toBe(initial.compatibilityEpoch);
    expect(lighting.compatibilityEpoch).toBe(initial.compatibilityEpoch);
  });

  it('batches dirty slices without reading or copying unrelated source buffers', () => {
    const initial = buildSceneStorageArena(sources(), 9, 41);
    const unrelated = SCENE_STORAGE_SEGMENTS.filter(
      (name) => name !== 'bvhPositions' && name !== 'bvhNormals',
    );
    const unrelatedObjects = new Map(
      unrelated.map((name) => [name, initial.sources[name]]),
    );
    let unrelatedDataReads = 0;
    for (const name of unrelated) {
      const source = initial.sources[name];
      const data = source.data;
      Object.defineProperty(source, 'data', {
        configurable: true,
        get: () => {
          unrelatedDataReads += 1;
          return data;
        },
      });
    }
    let dirtyDataReads = 0;
    for (const name of ['bvhPositions', 'bvhNormals'] as const) {
      const dirty = initial.sources[name];
      const descriptor = Object.getOwnPropertyDescriptor(dirty, 'data')!;
      Object.defineProperty(dirty, 'data', {
        configurable: true,
        get: () => {
          dirtyDataReads += 1;
          return descriptor.get!.call(dirty) as ArrayBuffer;
        },
      });
    }

    const patched = patchSceneStorageArenaSources(initial, [
      {
        segment: 'bvhPositions',
        byteOffset: 0,
        data: Uint32Array.of(11).buffer,
      },
      {
        segment: 'bvhPositions',
        byteOffset: 4,
        data: Uint32Array.of(22).buffer,
      },
      {
        segment: 'bvhNormals',
        byteOffset: 0,
        data: Uint32Array.of(33).buffer,
      },
    ], 10);

    expect(dirtyDataReads).toBe(0);
    expect(unrelatedDataReads).toBe(0);
    for (const name of unrelated) {
      expect(patched.sources[name]).toBe(unrelatedObjects.get(name));
    }
    expect(sceneStorageArenaSegmentU32(patched, 'bvhPositions', 0)).toBe(11);
    expect(sceneStorageArenaSegmentU32(patched, 'bvhPositions', 1)).toBe(22);
    expect(sceneStorageArenaSegmentU32(patched, 'bvhNormals', 0)).toBe(33);
    expect(dirtyDataReads).toBe(0);
    expect(patched.sources.bvhPositions).not.toBe(initial.sources.bvhPositions);
    expect(patched.sources.bvhNormals).not.toBe(initial.sources.bvhNormals);
    expect(patched.geometryGeneration).toBe(42);
    expect(patched.sceneEpoch).toBe(10);
  });

  it('bounds retained sparse state across repeated same-range animation updates', () => {
    let payload = buildSceneStorageArena(sources());
    for (let frame = 1; frame <= 2_000; frame += 1) {
      payload = patchSceneStorageArenaSources(payload, [{
        segment: 'bvhPositions',
        byteOffset: 4,
        data: Uint32Array.of(frame, frame ^ 0x55aa_55aa).buffer,
      }]);
    }

    const source = payload.sources.bvhPositions;
    const stateSymbol = Object.getOwnPropertySymbols(source).find(
      (symbol) => symbol.description === 'scene-storage-source-state',
    );
    expect(stateSymbol).toBeDefined();
    const state = (source as unknown as Record<symbol, {
      readonly patches: readonly { readonly data: ArrayBuffer }[];
    }>)[stateSymbol!];
    expect(state).toBeDefined();
    expect(state!.patches).toHaveLength(1);
    expect(state!.patches[0]!.data.byteLength).toBe(8);
    expect(sceneStorageArenaSegmentU32(payload, 'bvhPositions', 1)).toBe(2_000);
    expect(sceneStorageArenaSegmentU32(payload, 'bvhPositions', 2)).toBe(
      2_000 ^ 0x55aa_55aa,
    );
  });

  it('keeps the host generation collision-free across the u32 header wrap', () => {
    const initial = buildSceneStorageArena(sources(), 1, 0xffff_ffff);
    const next = replaceSceneStorageArenaSources(initial, {
      bvhPositions: initial.sources.bvhPositions,
    });
    expect(initial.compatibilityEpoch).toBe(0xffff_ffff);
    expect(next.compatibilityEpoch).toBe(0);
    expect(next.geometryGeneration).toBe(0x1_0000_0000);
    expect(next.geometryGeneration).not.toBe(initial.geometryGeneration);
  });

  it('enforces every count × WGSL ABI stride and only permits the canonical empty dummy', () => {
    const input = sources();
    for (const name of SCENE_STORAGE_SEGMENTS) {
      const stride = SCENE_STORAGE_SEGMENT_STRIDE_BYTES[name];
      expect(() => buildSceneStorageArena({
        ...input,
        [name]: { data: new ArrayBuffer(stride), count: 1 },
      })).not.toThrow();
      expect(() => buildSceneStorageArena({
        ...input,
        [name]: { data: new ArrayBuffer(stride + 4), count: 1 },
      })).toThrow(new RegExp(name + '.*requires exactly ' + stride + ' bytes'));
      expect(() => buildSceneStorageArena({
        ...input,
        [name]: { data: new ArrayBuffer(16), count: 0 },
      })).not.toThrow();
      expect(() => buildSceneStorageArena({
        ...input,
        [name]: { data: new ArrayBuffer(0), count: 0 },
      })).toThrow(new RegExp(name + '.*canonical 16-byte dummy'));
      expect(() => buildSceneStorageArena({
        ...input,
        [name]: { data: new ArrayBuffer(32), count: 0 },
      })).toThrow(new RegExp(name + '.*canonical 16-byte dummy'));
    }
  });

  it('accepts the exact per-binding capacity and rejects one byte beyond it', () => {
    const input = sources();
    const payload = buildSceneStorageArena(input);
    const max = Math.max(...payload.shards.map((shard) => shard.bytes.byteLength));
    expect(() => assertSceneStorageArenaFits(payload, {
      maxStorageBufferBindingSize: max,
      maxBufferSize: max,
    } as GPUSupportedLimits)).not.toThrow();
    try {
      assertSceneStorageArenaFits(payload, {
        maxStorageBufferBindingSize: max - 1,
        maxBufferSize: max,
      } as GPUSupportedLimits);
      throw new Error('expected SceneStorageArenaLimitError');
    } catch (error) {
      expect(error).toBeInstanceOf(SceneStorageArenaLimitError);
      expect((error as SceneStorageArenaLimitError).code).toBe(
        'walkaround-hybrid.scene-storage-arena-limit',
      );
      expect((error as SceneStorageArenaLimitError).failures.length).toBeGreaterThan(0);
    }
  });

  it.each([
    ['maxStorageBufferBindingSize', Number.NaN],
    ['maxStorageBufferBindingSize', Number.POSITIVE_INFINITY],
    ['maxStorageBufferBindingSize', 0],
    ['maxStorageBufferBindingSize', -1],
    ['maxStorageBufferBindingSize', 1.5],
    ['maxBufferSize', Number.NaN],
    ['maxBufferSize', Number.POSITIVE_INFINITY],
    ['maxBufferSize', 0],
    ['maxBufferSize', -1],
    ['maxBufferSize', 1.5],
  ])('rejects an invalid reported %s limit (%s)', (name, value) => {
    expect(() => maxSceneStorageArenaBindingBytes({
      maxStorageBufferBindingSize: 1024,
      maxBufferSize: 1024,
      [name]: value,
    } as unknown as GPUSupportedLimits)).toThrow(/positive finite safe integer/);
  });
});
