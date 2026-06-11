// gltfCompression.test.ts — GLTF-02: KHR_draco_mesh_compression +
// EXT_meshopt_compression via host-supplied decoder hooks.
//
// The package ships NO decoder, so these tests prove the HOOK CONTRACT with
// stub decoders: fixtures declare the extensions over trivial byte blobs, the
// stub hooks return known typed arrays, and we assert the decoded geometry
// lands in core primitives exactly as the equivalent uncompressed file would
// (including accessor normalization, morph interplay, and animation
// accessors). Real-decoder integration (draco3d / meshoptimizer) is a host
// concern — see README "Compressed geometry".

import { describe, it, expect } from 'vitest';
import { gltfToScene } from './gltfToScene.js';
import type { GltfJson } from './gltfTypes.js';
import type { DracoDecodeFn, DracoDecodeResult, MeshoptDecodeFn } from './compression.js';
import { sampleAnimationClip } from '@vitrum/core';
import type { MeshPrimitive, SkinnedMeshPrimitive } from '@vitrum/core';

// ────────────────────────────────────────────────────────────────────────────
// Fixture helpers (mirrors gltfAdapter.test.ts)
// ────────────────────────────────────────────────────────────────────────────

function f32Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

function u16Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 2);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setUint16(i * 2, v, true));
  return buf;
}

function u32Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setUint32(i * 4, v, true));
  return buf;
}

function concatBuffers(...bufs: ArrayBuffer[]): ArrayBuffer {
  const total = bufs.reduce((s, b) => s + b.byteLength, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const b of bufs) {
    out.set(new Uint8Array(b), off);
    off += b.byteLength;
  }
  return out.buffer;
}

/** Layout chunks sequentially in one buffer and emit matching bufferViews. */
function layoutBuffer(chunks: ArrayBuffer[]): {
  buffer: ArrayBuffer;
  bufferViews: { buffer: number; byteOffset: number; byteLength: number }[];
} {
  const bufferViews: { buffer: number; byteOffset: number; byteLength: number }[] = [];
  let off = 0;
  for (const c of chunks) {
    bufferViews.push({ buffer: 0, byteOffset: off, byteLength: c.byteLength });
    off += c.byteLength;
  }
  return { buffer: concatBuffers(...chunks), bufferViews };
}

// Reference triangle (XY plane, +Z normal).
const TRI_POSITIONS = [0, 0, 0, 1, 0, 0, 0, 1, 0];
const TRI_NORMALS = [0, 0, 1, 0, 0, 1, 0, 0, 1];
const TRI_UVS = [0, 0, 1, 0, 0, 1];
const TRI_INDICES = [0, 1, 2];
/** The "compressed" payload — a trivial sentinel blob the stub hooks receive. */
const COMPRESSED_BLOB = [0xde, 0xc0, 0xde, 0x01];

// ────────────────────────────────────────────────────────────────────────────
// Draco fixtures
// ────────────────────────────────────────────────────────────────────────────

interface DracoFixtureOpts {
  /** Add a TEXCOORD_0 accessor with this componentType (5123=u16, 5126=f32). */
  uvComponentType?: number;
  /** Mark UV accessor normalized. */
  uvNormalized?: boolean;
  /** Append a morph target whose POSITION-delta accessor has real (uncompressed) data. */
  morphDeltas?: number[];
  /** Give every accessor an uncompressed fallback bufferView with real data. */
  withFallback?: boolean;
  extensionsRequired?: string[];
}

/**
 * Triangle whose primitive carries KHR_draco_mesh_compression over a trivial
 * blob. Attribute/index accessors have NO bufferView (pure-compressed) unless
 * `withFallback` is set.
 */
function makeDracoGltf(opts: DracoFixtureOpts = {}): {
  gltf: GltfJson; buffers: Map<number, ArrayBuffer>;
} {
  const chunks: ArrayBuffer[] = [new Uint8Array(COMPRESSED_BLOB).buffer as ArrayBuffer];
  if (opts.morphDeltas) chunks.push(f32Buffer(opts.morphDeltas));
  if (opts.withFallback) {
    chunks.push(f32Buffer(TRI_POSITIONS), f32Buffer(TRI_NORMALS), u32Buffer(TRI_INDICES));
  }
  const { buffer, bufferViews } = layoutBuffer(chunks);

  const fb = (n: number) =>
    opts.withFallback ? { bufferView: (opts.morphDeltas ? 2 : 1) + n } : {};

  const accessors: NonNullable<GltfJson['accessors']> = [
    { componentType: 5126, count: 3, type: 'VEC3', ...fb(0) },            // 0 POSITION
    { componentType: 5126, count: 3, type: 'VEC3', ...fb(1) },            // 1 NORMAL
    { componentType: 5125, count: 3, type: 'SCALAR', ...fb(2) },          // 2 indices
  ];
  const attributes: Record<string, number> = { POSITION: 0, NORMAL: 1 };
  const dracoAttributes: Record<string, number> = { POSITION: 10, NORMAL: 11 };
  if (opts.uvComponentType !== undefined) {
    accessors.push({
      componentType: opts.uvComponentType,
      count: 3,
      type: 'VEC2',
      ...(opts.uvNormalized ? { normalized: true } : {}),
    });
    attributes['TEXCOORD_0'] = accessors.length - 1;
    dracoAttributes['TEXCOORD_0'] = 12;
  }
  if (opts.morphDeltas) {
    accessors.push({ bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' });
  }

  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      name: 'draco-tri',
      primitives: [{
        attributes,
        indices: 2,
        ...(opts.morphDeltas ? { targets: [{ POSITION: accessors.length - 1 }] } : {}),
        extensions: {
          KHR_draco_mesh_compression: { bufferView: 0, attributes: dracoAttributes },
        },
      }],
    }],
    accessors,
    bufferViews,
    buffers: [{ byteLength: buffer.byteLength }],
    extensionsUsed: ['KHR_draco_mesh_compression'],
    ...(opts.extensionsRequired ? { extensionsRequired: opts.extensionsRequired } : {}),
  };
  return { gltf, buffers: new Map([[0, buffer]]) };
}

/** The hook every happy-path Draco test uses: returns the reference triangle. */
const dracoTriHook = (
  _compressed?: Uint8Array,
  _ids?: Readonly<Record<string, number>>,
): DracoDecodeResult => ({
  attributes: {
    POSITION: new Float32Array(TRI_POSITIONS),
    NORMAL: new Float32Array(TRI_NORMALS),
  },
  indices: new Uint32Array(TRI_INDICES),
});

/** The same triangle as a plain uncompressed glTF (the equivalence reference). */
function makeUncompressedTriGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const { buffer, bufferViews } = layoutBuffer([
    f32Buffer(TRI_POSITIONS), f32Buffer(TRI_NORMALS), u32Buffer(TRI_INDICES),
  ]);
  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, indices: 2 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 2, componentType: 5125, count: 3, type: 'SCALAR' },
    ],
    bufferViews,
    buffers: [{ byteLength: buffer.byteLength }],
  };
  return { gltf, buffers: new Map([[0, buffer]]) };
}

// ────────────────────────────────────────────────────────────────────────────
// KHR_draco_mesh_compression
// ────────────────────────────────────────────────────────────────────────────

describe('KHR_draco_mesh_compression hooks (GLTF-02)', () => {
  it('stub-decoded geometry matches the equivalent uncompressed import exactly', async () => {
    const compressed = makeDracoGltf();
    const reference = makeUncompressedTriGltf();

    const a = await gltfToScene(compressed.gltf, {
      buffers: compressed.buffers, dracoDecode: dracoTriHook,
    });
    const b = await gltfToScene(reference.gltf, { buffers: reference.buffers });

    expect(a.scene.primitives).toHaveLength(1);
    const pa = a.scene.primitives[0] as MeshPrimitive;
    const pb = b.scene.primitives[0] as MeshPrimitive;
    expect(pa.kind).toBe('mesh');
    expect(Array.from(pa.positions)).toEqual(Array.from(pb.positions));
    expect(Array.from(pa.normals)).toEqual(Array.from(pb.normals));
    expect(Array.from(pa.indices!)).toEqual(Array.from(pb.indices!));
    expect(a.warnings.some(w => w.includes('SKIPPED'))).toBe(false);
  });

  it('hook receives the compressed bytes and the semantic → Draco-id map', async () => {
    const { gltf, buffers } = makeDracoGltf();
    let seenBytes: number[] = [];
    let seenIds: Record<string, number> = {};
    const hook: DracoDecodeFn = (compressed, attributeIds) => {
      seenBytes = Array.from(compressed);
      seenIds = { ...attributeIds };
      return dracoTriHook(compressed, attributeIds);
    };
    await gltfToScene(gltf, { buffers, dracoDecode: hook });
    expect(seenBytes).toEqual(COMPRESSED_BLOB);
    expect(seenIds).toEqual({ POSITION: 10, NORMAL: 11 });
  });

  it('async hooks are awaited', async () => {
    const { gltf, buffers } = makeDracoGltf();
    const hook: DracoDecodeFn = async (c, ids) => {
      await new Promise((r) => setTimeout(r, 1));
      return dracoTriHook(c, ids);
    };
    const { scene } = await gltfToScene(gltf, { buffers, dracoDecode: hook });
    expect(scene.primitives).toHaveLength(1);
    expect(Array.from((scene.primitives[0] as MeshPrimitive).positions)).toEqual(TRI_POSITIONS);
  });

  it('non-float attributes matching the accessor componentType go through standard normalization', async () => {
    // TEXCOORD_0 declared u16 normalized → decoder returns raw Uint16Array,
    // the adapter's accessor path divides by 65535.
    const { gltf, buffers } = makeDracoGltf({ uvComponentType: 5123, uvNormalized: true });
    const hook: DracoDecodeFn = (c, ids) => ({
      attributes: {
        ...dracoTriHook(c, ids).attributes,
        TEXCOORD_0: new Uint16Array([0, 0, 65535, 0, 0, 65535]),
      },
      indices: new Uint32Array(TRI_INDICES),
    });
    const { scene } = await gltfToScene(gltf, { buffers, dracoDecode: hook });
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.uvs).toBeDefined();
    expect(Array.from(prim.uvs!)).toEqual(TRI_UVS);
  });

  it('already-dequantized Float32Array is accepted for a non-float accessor', async () => {
    const { gltf, buffers } = makeDracoGltf({ uvComponentType: 5123, uvNormalized: true });
    const hook: DracoDecodeFn = (c, ids) => ({
      attributes: {
        ...dracoTriHook(c, ids).attributes,
        TEXCOORD_0: new Float32Array(TRI_UVS),
      },
      indices: new Uint32Array(TRI_INDICES),
    });
    const { scene, warnings } = await gltfToScene(gltf, { buffers, dracoDecode: hook });
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.uvs!)).toEqual(TRI_UVS);
    expect(warnings.some(w => w.includes('rejected'))).toBe(false);
  });

  it('typed-array/componentType mismatch on POSITION rejects the primitive with a warning', async () => {
    const { gltf, buffers } = makeDracoGltf();
    const hook: DracoDecodeFn = () => ({
      attributes: { POSITION: new Int16Array(9), NORMAL: new Float32Array(TRI_NORMALS) },
      indices: new Uint32Array(TRI_INDICES),
    });
    const { scene, warnings } = await gltfToScene(gltf, { buffers, dracoDecode: hook });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('Attribute rejected'))).toBe(true);
    expect(warnings.some(w => w.includes('SKIPPED'))).toBe(true);
  });

  it('element-count mismatch against the declared accessor rejects the attribute', async () => {
    const { gltf, buffers } = makeDracoGltf();
    const hook: DracoDecodeFn = () => ({
      attributes: {
        POSITION: new Float32Array(6), // accessor says count 3 × VEC3 = 9
        NORMAL: new Float32Array(TRI_NORMALS),
      },
      indices: new Uint32Array(TRI_INDICES),
    });
    const { scene, warnings } = await gltfToScene(gltf, { buffers, dracoDecode: hook });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('count × components'))).toBe(true);
  });

  it('index-count mismatch rejects the decoded indices', async () => {
    const { gltf, buffers } = makeDracoGltf();
    const hook: DracoDecodeFn = (c, ids) => ({
      ...dracoTriHook(c, ids),
      indices: new Uint32Array([0, 1, 2, 0, 2, 1]), // accessor declares count 3
    });
    const { scene, warnings } = await gltfToScene(gltf, { buffers, dracoDecode: hook });
    expect(warnings.some(w => w.includes('declares count=3'))).toBe(true);
    expect(scene.primitives).toHaveLength(0); // index accessor has no fallback
  });

  it('decoded POSITION composes with uncompressed morph-target deltas (skin promotion)', async () => {
    const deltas = [0, 0, 1, 0, 0, 1, 0, 0, 1];
    const { gltf, buffers } = makeDracoGltf({ morphDeltas: deltas });
    const { scene, warnings } = await gltfToScene(gltf, { buffers, dracoDecode: dracoTriHook });
    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(prim.kind).toBe('skinned-mesh'); // identity-skeleton promotion
    expect(Array.from(prim.positions)).toEqual(TRI_POSITIONS); // from the hook
    expect(prim.morphTargets).toHaveLength(1);
    expect(Array.from(prim.morphTargets![0]!)).toEqual(deltas); // from the buffer
    expect(warnings.some(w => w.includes('SKIPPED'))).toBe(false);
  });

  it('no hook + extensionsRequired + no fallback → throws a clear error', async () => {
    const { gltf, buffers } = makeDracoGltf({
      extensionsRequired: ['KHR_draco_mesh_compression'],
    });
    await expect(gltfToScene(gltf, { buffers })).rejects.toThrow(/extensionsRequired/);
  });

  it('no hook + optional + no fallback → warn + primitive skipped', async () => {
    const { gltf, buffers } = makeDracoGltf();
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('no opts.dracoDecode hook'))).toBe(true);
    expect(warnings.some(w => w.includes('SKIPPED'))).toBe(true);
  });

  it('no hook + optional + uncompressed fallback accessors → imports the fallback', async () => {
    const { gltf, buffers } = makeDracoGltf({ withFallback: true });
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.positions)).toEqual(TRI_POSITIONS);
    expect(Array.from(prim.indices!)).toEqual(TRI_INDICES);
    expect(warnings.some(w => w.includes('fallback accessors'))).toBe(true);
  });

  it('hook throw on an optional extension degrades to warn + skip', async () => {
    const { gltf, buffers } = makeDracoGltf();
    const hook: DracoDecodeFn = () => { throw new Error('boom'); };
    const { scene, warnings } = await gltfToScene(gltf, { buffers, dracoDecode: hook });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('dracoDecode hook failed') && w.includes('boom'))).toBe(true);
  });

  it('does not mutate the caller-supplied GltfJson', async () => {
    const { gltf, buffers } = makeDracoGltf();
    await gltfToScene(gltf, { buffers, dracoDecode: dracoTriHook });
    // Original primitive still carries the extension; accessors untouched.
    expect(gltf.meshes![0]!.primitives[0]!.extensions?.['KHR_draco_mesh_compression']).toBeDefined();
    expect(gltf.accessors![0]!.bufferView).toBeUndefined();
    expect(gltf.bufferViews).toHaveLength(1);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// EXT_meshopt_compression fixtures
// ────────────────────────────────────────────────────────────────────────────

interface MeshoptFixtureOpts {
  /** Put real fallback data in buffer 0 instead of a `fallback: true` stub. */
  realFallback?: boolean;
  extensionsRequired?: string[];
  /** Add an animation whose sampler input/output accessors sit on compressed views. */
  withAnimation?: boolean;
}

/**
 * Triangle whose POSITION (ATTRIBUTES view) and indices (INDICES view) sit on
 * EXT_meshopt_compression bufferViews. Buffer 0 = fallback (stub or real),
 * buffer 1 = "compressed" payload.
 */
function makeMeshoptGltf(opts: MeshoptFixtureOpts = {}): {
  gltf: GltfJson; buffers: Map<number, ArrayBuffer>;
} {
  const compressed = new Uint8Array(COMPRESSED_BLOB).buffer as ArrayBuffer;
  const buffers = new Map<number, ArrayBuffer>([[1, compressed]]);
  if (opts.realFallback) {
    buffers.set(0, concatBuffers(f32Buffer(TRI_POSITIONS), u16Buffer(TRI_INDICES)));
  }

  const meshoptExt = (byteStride: number, count: number, mode: string, filter?: string) => ({
    EXT_meshopt_compression: {
      buffer: 1, byteOffset: 0, byteLength: COMPRESSED_BLOB.length,
      byteStride, count, mode, ...(filter ? { filter } : {}),
    },
  });

  const bufferViews: NonNullable<GltfJson['bufferViews']> = [
    // 0: POSITION — 3 × vec3<f32>, stride 12.
    { buffer: 0, byteOffset: 0, byteLength: 36, byteStride: 12,
      extensions: meshoptExt(12, 3, 'ATTRIBUTES') },
    // 1: indices — 3 × u16.
    { buffer: 0, byteOffset: 36, byteLength: 6,
      extensions: meshoptExt(2, 3, 'TRIANGLES') },
  ];
  const accessors: NonNullable<GltfJson['accessors']> = [
    { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
    { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
  ];

  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ name: 'meshopt-tri', primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors,
    bufferViews,
    buffers: [
      {
        byteLength: opts.realFallback ? 42 : 0,
        ...(opts.realFallback
          ? {}
          : { extensions: { EXT_meshopt_compression: { fallback: true } } }),
      },
      { byteLength: COMPRESSED_BLOB.length },
    ],
    extensionsUsed: ['EXT_meshopt_compression'],
    ...(opts.extensionsRequired ? { extensionsRequired: opts.extensionsRequired } : {}),
  };

  if (opts.withAnimation) {
    // Sampler input (2 × f32 times) and output (2 × vec3 translations) on
    // compressed views — proves bufferView-level decode is transparent to the
    // animation reader.
    bufferViews.push(
      { buffer: 0, byteOffset: 42, byteLength: 8,
        extensions: meshoptExt(4, 2, 'ATTRIBUTES') },
      { buffer: 0, byteOffset: 50, byteLength: 24,
        extensions: meshoptExt(12, 2, 'ATTRIBUTES') },
    );
    accessors.push(
      { bufferView: 2, componentType: 5126, count: 2, type: 'SCALAR', min: [0], max: [1] },
      { bufferView: 3, componentType: 5126, count: 2, type: 'VEC3' },
    );
    gltf.animations = [{
      name: 'slide',
      samplers: [{ input: 2, output: 3 }],
      channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
    }];
  }
  return { gltf, buffers };
}

/** Stub meshopt decoder dispatching on (count, stride): returns reference bytes. */
const meshoptTriHook: MeshoptDecodeFn = (_compressed, count, byteStride) => {
  if (byteStride === 12 && count === 3) return new Uint8Array(f32Buffer(TRI_POSITIONS));
  if (byteStride === 2 && count === 3) return new Uint8Array(u16Buffer(TRI_INDICES));
  if (byteStride === 4 && count === 2) return new Uint8Array(f32Buffer([0, 1]));
  if (byteStride === 12 && count === 2) return new Uint8Array(f32Buffer([0, 0, 0, 2, 0, 0]));
  throw new Error(`unexpected decode request: count=${count} stride=${byteStride}`);
};

// ────────────────────────────────────────────────────────────────────────────
// EXT_meshopt_compression
// ────────────────────────────────────────────────────────────────────────────

describe('EXT_meshopt_compression hooks (GLTF-02)', () => {
  it('ATTRIBUTES + TRIANGLES bufferViews decode through the hook into core geometry', async () => {
    const { gltf, buffers } = makeMeshoptGltf();
    const calls: Array<{ bytes: number[]; count: number; stride: number; mode: string; filter: string }> = [];
    const hook: MeshoptDecodeFn = (c, count, stride, mode, filter) => {
      calls.push({ bytes: Array.from(c), count, stride, mode, filter });
      return meshoptTriHook(c, count, stride, mode, filter);
    };
    const { scene, warnings } = await gltfToScene(gltf, { buffers, meshoptDecode: hook });

    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.positions)).toEqual(TRI_POSITIONS);
    expect(Array.from(prim.indices!)).toEqual(TRI_INDICES);
    expect(warnings.some(w => w.includes('EXT_meshopt_compression'))).toBe(false);

    // Hook contract: compressed bytes + count/stride/mode/filter per view.
    expect(calls).toHaveLength(2);
    expect(calls[0]).toEqual({
      bytes: COMPRESSED_BLOB, count: 3, stride: 12, mode: 'ATTRIBUTES', filter: 'NONE',
    });
    expect(calls[1]).toEqual({
      bytes: COMPRESSED_BLOB, count: 3, stride: 2, mode: 'TRIANGLES', filter: 'NONE',
    });
  });

  it('decoded geometry matches the equivalent uncompressed import', async () => {
    const compressed = makeMeshoptGltf();
    const reference = makeUncompressedTriGltf();
    const a = await gltfToScene(compressed.gltf, {
      buffers: compressed.buffers, meshoptDecode: meshoptTriHook,
    });
    const b = await gltfToScene(reference.gltf, { buffers: reference.buffers });
    const pa = a.scene.primitives[0] as MeshPrimitive;
    const pb = b.scene.primitives[0] as MeshPrimitive;
    expect(Array.from(pa.positions)).toEqual(Array.from(pb.positions));
    expect(Array.from(pa.indices!)).toEqual(Array.from(pb.indices!));
  });

  it('animation sampler input/output accessors read decompressed data transparently', async () => {
    const { gltf, buffers } = makeMeshoptGltf({ withAnimation: true });
    const { animations, warnings } = await gltfToScene(gltf, { buffers, meshoptDecode: meshoptTriHook });
    expect(animations).toHaveLength(1);
    const clip = animations[0]!;
    expect(clip.duration).toBeCloseTo(1);
    const mid = sampleAnimationClip(clip, 0.5)[0]!;
    expect(Array.from(mid.value)).toEqual([1, 0, 0]);
    expect(warnings.some(w => w.includes('EXT_meshopt_compression'))).toBe(false);
  });

  it('no hook + real fallback buffer → imports the fallback with a warning', async () => {
    const { gltf, buffers } = makeMeshoptGltf({ realFallback: true });
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.positions)).toEqual(TRI_POSITIONS);
    expect(Array.from(prim.indices!)).toEqual(TRI_INDICES);
    expect(warnings.some(w =>
      w.includes('EXT_meshopt_compression') && w.includes('fallback'))).toBe(true);
  });

  it('no hook + fallback:true stub buffer → warn + primitive skipped', async () => {
    const { gltf, buffers } = makeMeshoptGltf(); // buffer 0 is a fallback stub
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('no uncompressed fallback buffer'))).toBe(true);
  });

  it('no hook + extensionsRequired → throws a clear error', async () => {
    const { gltf, buffers } = makeMeshoptGltf({
      extensionsRequired: ['EXT_meshopt_compression'],
    });
    await expect(gltfToScene(gltf, { buffers })).rejects.toThrow(/extensionsRequired/);
  });

  it('hook returning the wrong byte count warns and leaves the view unresolved (optional)', async () => {
    const { gltf, buffers } = makeMeshoptGltf();
    const hook: MeshoptDecodeFn = () => new Uint8Array(5); // never count × stride
    const { scene, warnings } = await gltfToScene(gltf, { buffers, meshoptDecode: hook });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('count × byteStride'))).toBe(true);
  });

  it('hook returning the wrong byte count throws when the extension is required', async () => {
    const { gltf, buffers } = makeMeshoptGltf({
      extensionsRequired: ['EXT_meshopt_compression'],
    });
    const hook: MeshoptDecodeFn = () => new Uint8Array(5);
    await expect(gltfToScene(gltf, { buffers, meshoptDecode: hook }))
      .rejects.toThrow(/count × byteStride/);
  });

  it('does not mutate the caller-supplied GltfJson', async () => {
    const { gltf, buffers } = makeMeshoptGltf();
    await gltfToScene(gltf, { buffers, meshoptDecode: meshoptTriHook });
    expect(gltf.bufferViews![0]!.extensions?.['EXT_meshopt_compression']).toBeDefined();
    expect(gltf.bufferViews![0]!.buffer).toBe(0);
    expect(gltf.buffers).toHaveLength(2);
  });
});
