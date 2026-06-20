// gltfCompression.test.ts — GLTF-02: KHR_draco_mesh_compression +
// EXT_meshopt_compression via host-supplied decoder hooks.
//
// The package ships NO runtime decoder, so most tests prove the HOOK CONTRACT
// with stub decoders. The final smoke tests also run real dev-only draco3d /
// meshoptimizer codecs through those same host hooks to prevent the hook shape
// from drifting away from actual decoder packages.

import { describe, it, expect } from 'vitest';
import { gltfToScene } from './gltfToScene.js';
import { GltfCompatibilityError, loadGltfForEngine } from './index.js';
import type { GltfJson } from './gltfTypes.js';
import type { DracoDecodeFn, DracoDecodeResult, MeshoptDecodeFn } from './compression.js';
import { sampleAnimationClip } from '@vitrum/core';
import type { MeshPrimitive, SkinnedMeshPrimitive } from '@vitrum/core';
import draco3d from 'draco3d';
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer';

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

function viewArrayBuffer(view: ArrayBufferView): ArrayBuffer {
  const copy = new Uint8Array(view.byteLength);
  copy.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
  return copy.buffer;
}

function expectArrayClose(actual: ArrayLike<number>, expected: readonly number[], digits = 5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]).toBeCloseTo(expected[i]!, digits);
  }
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

interface DracoInt8ArrayLike {
  GetValue(index: number): number;
}

interface DracoInt32ArrayLike {
  GetValue(index: number): number;
}

interface DracoFloat32ArrayLike {
  size(): number;
  GetValue(index: number): number;
}

interface DracoStatusLike {
  ok(): boolean;
  error_msg(): string;
}

interface DracoEncoderLike {
  SetSpeedOptions(encode: number, decode: number): void;
  EncodeMeshToDracoBuffer(mesh: unknown, out: DracoInt8ArrayLike): number;
}

interface DracoMeshBuilderLike {
  AddFacesToMesh(mesh: unknown, faceCount: number, indices: Uint32Array): void;
  AddFloatAttributeToMesh(
    mesh: unknown,
    attributeType: number,
    pointCount: number,
    componentCount: number,
    values: Float32Array,
  ): number;
}

interface DracoDecoderLike {
  DecodeBufferToMesh(buffer: unknown, mesh: unknown): DracoStatusLike;
  GetAttributeByUniqueId(mesh: unknown, uniqueId: number): unknown;
  GetAttributeFloatForAllPoints(mesh: unknown, attribute: unknown, out: DracoFloat32ArrayLike): void;
  GetFaceFromMesh(mesh: unknown, faceIndex: number, out: DracoInt32ArrayLike): void;
}

interface DracoMeshLike {
  num_faces(): number;
  num_points(): number;
}

interface DracoDecoderBufferLike {
  Init(data: Int8Array, byteLength: number): void;
}

interface DracoEncoderModuleLike {
  POSITION: number;
  NORMAL: number;
  Encoder: new () => DracoEncoderLike;
  MeshBuilder: new () => DracoMeshBuilderLike;
  Mesh: new () => unknown;
  DracoInt8Array: new () => DracoInt8ArrayLike;
  destroy(value: unknown): void;
}

interface DracoDecoderModuleLike {
  Decoder: new () => DracoDecoderLike;
  DecoderBuffer: new () => DracoDecoderBufferLike;
  Mesh: new () => DracoMeshLike;
  DracoFloat32Array: new () => DracoFloat32ArrayLike;
  DracoInt32Array: new () => DracoInt32ArrayLike;
  destroy(value: unknown): void;
}

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

async function makeRealDracoGltf(): Promise<{
  gltf: GltfJson;
  buffers: Map<number, ArrayBuffer>;
  decode: DracoDecodeFn;
}> {
  const encoderModule = await draco3d.createEncoderModule({}) as DracoEncoderModuleLike;
  const decoderModule = await draco3d.createDecoderModule({}) as DracoDecoderModuleLike;
  const builder = new encoderModule.MeshBuilder();
  const mesh = new encoderModule.Mesh();
  const encoder = new encoderModule.Encoder();
  const encoded = new encoderModule.DracoInt8Array();

  builder.AddFacesToMesh(mesh, 1, new Uint32Array(TRI_INDICES));
  const positionId = builder.AddFloatAttributeToMesh(
    mesh,
    encoderModule.POSITION,
    3,
    3,
    new Float32Array(TRI_POSITIONS),
  );
  const normalId = builder.AddFloatAttributeToMesh(
    mesh,
    encoderModule.NORMAL,
    3,
    3,
    new Float32Array(TRI_NORMALS),
  );
  encoder.SetSpeedOptions(10, 10);
  const encodedLength = encoder.EncodeMeshToDracoBuffer(mesh, encoded);
  if (encodedLength <= 0) {
    throw new Error('draco3d failed to encode the triangle fixture');
  }
  const compressed = new Int8Array(encodedLength);
  for (let i = 0; i < encodedLength; i += 1) compressed[i] = encoded.GetValue(i);
  encoderModule.destroy(encoded);
  encoderModule.destroy(encoder);
  encoderModule.destroy(mesh);
  encoderModule.destroy(builder);

  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      name: 'real-draco-tri',
      primitives: [{
        attributes: { POSITION: 0, NORMAL: 1 },
        indices: 2,
        extensions: {
          KHR_draco_mesh_compression: {
            bufferView: 0,
            attributes: { POSITION: positionId, NORMAL: normalId },
          },
        },
      }],
    }],
    accessors: [
      { componentType: 5126, count: 3, type: 'VEC3' },
      { componentType: 5126, count: 3, type: 'VEC3' },
      { componentType: 5125, count: 3, type: 'SCALAR' },
    ],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: compressed.byteLength }],
    buffers: [{ byteLength: compressed.byteLength }],
    extensionsUsed: ['KHR_draco_mesh_compression'],
    extensionsRequired: ['KHR_draco_mesh_compression'],
  };

  const decode: DracoDecodeFn = (bytes, attributeIds) => {
    const buffer = new decoderModule.DecoderBuffer();
    const decoder = new decoderModule.Decoder();
    const meshOut = new decoderModule.Mesh();
    const byteView = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    buffer.Init(byteView, byteView.byteLength);
    const status = decoder.DecodeBufferToMesh(buffer, meshOut);
    if (!status.ok()) {
      throw new Error(status.error_msg());
    }
    const readAttribute = (uniqueId: number): Float32Array => {
      const attr = decoder.GetAttributeByUniqueId(meshOut, uniqueId);
      const data = new decoderModule.DracoFloat32Array();
      decoder.GetAttributeFloatForAllPoints(meshOut, attr, data);
      const out = new Float32Array(data.size());
      for (let i = 0; i < out.length; i += 1) out[i] = data.GetValue(i);
      decoderModule.destroy(data);
      return out;
    };
    const face = new decoderModule.DracoInt32Array();
    const indices = new Uint32Array(meshOut.num_faces() * 3);
    for (let f = 0; f < meshOut.num_faces(); f += 1) {
      decoder.GetFaceFromMesh(meshOut, f, face);
      const base = f * 3;
      indices[base] = face.GetValue(0);
      indices[base + 1] = face.GetValue(1);
      indices[base + 2] = face.GetValue(2);
    }
    decoderModule.destroy(face);
    const result: DracoDecodeResult = {
      attributes: {
        POSITION: readAttribute(attributeIds.POSITION ?? positionId),
        NORMAL: readAttribute(attributeIds.NORMAL ?? normalId),
      },
      indices,
    };
    decoderModule.destroy(meshOut);
    decoderModule.destroy(decoder);
    decoderModule.destroy(buffer);
    return result;
  };

  return {
    gltf,
    buffers: new Map([[0, viewArrayBuffer(compressed)]]),
    decode,
  };
}

// ────────────────────────────────────────────────────────────────────────────
// KHR_draco_mesh_compression
// ────────────────────────────────────────────────────────────────────────────

describe('KHR_draco_mesh_compression hooks (GLTF-02)', () => {
  it('real draco3d encoded payload imports through the documented host hook', async () => {
    const { gltf, buffers, decode } = await makeRealDracoGltf();

    const result = await gltfToScene(gltf, { buffers, dracoDecode: decode });

    expect(result.warnings.some(w => w.includes('SKIPPED'))).toBe(false);
    expect(result.scene.primitives).toHaveLength(1);
    const primitive = result.scene.primitives[0] as MeshPrimitive;
    expect(primitive.kind).toBe('mesh');
    expectArrayClose(primitive.positions, TRI_POSITIONS, 4);
    expectArrayClose(primitive.normals, TRI_NORMALS, 4);
    expect(Array.from(primitive.indices!)).toEqual(TRI_INDICES);
  });

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
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers, dracoDecode: hook });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('Attribute rejected'))).toBe(true);
    expect(warnings.some(w => w.includes('SKIPPED'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'draco-attribute-component-type-mismatch',
        path: 'meshes[0].primitives[0].attributes.POSITION',
        extension: 'KHR_draco_mesh_compression',
        semantic: 'POSITION',
      }),
      expect.objectContaining({
        code: 'draco-geometry-unusable',
        path: 'meshes[0].primitives[0].extensions.KHR_draco_mesh_compression',
      }),
      expect.objectContaining({
        code: 'unresolved-compression',
        path: 'meshes[0].primitives[0].extensions.KHR_draco_mesh_compression',
      }),
    ]));
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
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers, dracoDecode: hook });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('count × components'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'draco-attribute-count-mismatch',
        path: 'meshes[0].primitives[0].attributes.POSITION',
        semantic: 'POSITION',
      }),
    ]));
  });

  it('classifies decoded Draco optional-attribute corruption as degraded in the one-call bridge', async () => {
    const { gltf, buffers } = makeDracoGltf();
    const hook: DracoDecodeFn = () => ({
      attributes: {
        POSITION: new Float32Array(TRI_POSITIONS),
        NORMAL: new Float32Array(6), // accessor says count 3 × VEC3 = 9
      },
      indices: new Uint32Array(TRI_INDICES),
    });

    const accepted = await loadGltfForEngine(gltf, {
      buffers,
      dracoDecode: hook,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
    });
    expect(accepted.asset.scene.primitives).toHaveLength(1);
    expect(accepted.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'draco-attribute-count-mismatch',
        path: 'meshes[0].primitives[0].attributes.NORMAL',
        semantic: 'NORMAL',
      }),
    ]));

    let error: unknown;
    try {
      await loadGltfForEngine(gltf, {
        buffers,
        dracoDecode: hook,
        backend: 'pt-webgl2',
        compatibilityMode: 'reject-degraded',
      });
    } catch (err) {
      error = err;
    }

    expect(error).toBeInstanceOf(GltfCompatibilityError);
    expect((error as GltfCompatibilityError).failures).toContain(
      'import:draco-attribute-count-mismatch=approximate at meshes[0].primitives[0].attributes.NORMAL',
    );
  });

  it('index-count mismatch rejects the decoded indices', async () => {
    const { gltf, buffers } = makeDracoGltf();
    const hook: DracoDecodeFn = (c, ids) => ({
      ...dracoTriHook(c, ids),
      indices: new Uint32Array([0, 1, 2, 0, 2, 1]), // accessor declares count 3
    });
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers, dracoDecode: hook });
    expect(warnings.some(w => w.includes('declares count=3'))).toBe(true);
    expect(scene.primitives).toHaveLength(0); // index accessor has no fallback
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'draco-index-count-mismatch',
        path: 'meshes[0].primitives[0].indices',
        accessorIndex: 2,
      }),
      expect.objectContaining({
        code: 'draco-geometry-unusable',
      }),
    ]));
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

  it('no hook + extensionsRequired + uncompressed fallback still throws', async () => {
    const { gltf, buffers } = makeDracoGltf({
      extensionsRequired: ['KHR_draco_mesh_compression'],
      withFallback: true,
    });
    await expect(gltfToScene(gltf, { buffers })).rejects.toThrow(
      /required Draco assets must decode the required extension/,
    );
  });

  it('no hook + optional + no fallback → warn + primitive skipped', async () => {
    const { gltf, buffers } = makeDracoGltf();
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('no opts.dracoDecode hook'))).toBe(true);
    expect(warnings.some(w => w.includes('SKIPPED'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'draco-decode-hook-missing',
        path: 'meshes[0].primitives[0].extensions.KHR_draco_mesh_compression',
        extension: 'KHR_draco_mesh_compression',
      }),
      expect.objectContaining({
        code: 'unresolved-compression',
      }),
    ]));
  });

  it('no hook + optional + uncompressed fallback accessors → imports the fallback', async () => {
    const { gltf, buffers } = makeDracoGltf({ withFallback: true });
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.positions)).toEqual(TRI_POSITIONS);
    expect(Array.from(prim.indices!)).toEqual(TRI_INDICES);
    expect(warnings.some(w => w.includes('fallback accessors'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'draco-fallback-accessors-used',
        path: 'meshes[0].primitives[0].extensions.KHR_draco_mesh_compression',
      }),
    ]));
    expect(diagnostics.some((diagnostic) => diagnostic.code === 'unresolved-compression')).toBe(false);
  });

  it('hook throw on an optional extension degrades to warn + skip', async () => {
    const { gltf, buffers } = makeDracoGltf();
    const hook: DracoDecodeFn = () => { throw new Error('boom'); };
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers, dracoDecode: hook });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('dracoDecode hook failed') && w.includes('boom'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'draco-decode-hook-failed',
        path: 'meshes[0].primitives[0].extensions.KHR_draco_mesh_compression',
      }),
    ]));
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

async function makeRealMeshoptGltf(): Promise<{
  gltf: GltfJson;
  buffers: Map<number, ArrayBuffer>;
  decode: MeshoptDecodeFn;
}> {
  await MeshoptEncoder.ready;
  await MeshoptDecoder.ready;
  const positionBytes = new Uint8Array(f32Buffer(TRI_POSITIONS));
  const indexBytes = new Uint8Array(u16Buffer(TRI_INDICES));
  const compressedPositions = MeshoptEncoder.encodeGltfBuffer(
    positionBytes,
    3,
    12,
    'ATTRIBUTES',
  );
  const compressedIndices = MeshoptEncoder.encodeGltfBuffer(
    indexBytes,
    3,
    2,
    'TRIANGLES',
  );
  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ name: 'real-meshopt-tri', primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    bufferViews: [
      {
        buffer: 0,
        byteOffset: 0,
        byteLength: 36,
        byteStride: 12,
        extensions: {
          EXT_meshopt_compression: {
            buffer: 1,
            byteOffset: 0,
            byteLength: compressedPositions.byteLength,
            byteStride: 12,
            count: 3,
            mode: 'ATTRIBUTES',
          },
        },
      },
      {
        buffer: 0,
        byteOffset: 36,
        byteLength: 6,
        extensions: {
          EXT_meshopt_compression: {
            buffer: 2,
            byteOffset: 0,
            byteLength: compressedIndices.byteLength,
            byteStride: 2,
            count: 3,
            mode: 'TRIANGLES',
          },
        },
      },
    ],
    buffers: [
      { byteLength: 0, extensions: { EXT_meshopt_compression: { fallback: true } } },
      { byteLength: compressedPositions.byteLength },
      { byteLength: compressedIndices.byteLength },
    ],
    extensionsUsed: ['EXT_meshopt_compression'],
    extensionsRequired: ['EXT_meshopt_compression'],
  };
  const decode: MeshoptDecodeFn = (compressed, count, byteStride, mode, filter) => {
    const target = new Uint8Array(count * byteStride);
    MeshoptDecoder.decodeGltfBuffer(
      target,
      count,
      byteStride,
      compressed,
      mode,
      filter === 'NONE' ? undefined : filter,
    );
    return target;
  };
  return {
    gltf,
    buffers: new Map([
      [1, viewArrayBuffer(compressedPositions)],
      [2, viewArrayBuffer(compressedIndices)],
    ]),
    decode,
  };
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
// EXT/KHR_meshopt_compression
// ────────────────────────────────────────────────────────────────────────────

describe('EXT/KHR_meshopt_compression hooks (GLTF-02)', () => {
  it('real meshoptimizer encoded payload imports through the documented host hook', async () => {
    const { gltf, buffers, decode } = await makeRealMeshoptGltf();

    const result = await gltfToScene(gltf, { buffers, meshoptDecode: decode });

    expect(result.warnings.some(w => w.includes('SKIPPED'))).toBe(false);
    expect(result.scene.primitives).toHaveLength(1);
    const primitive = result.scene.primitives[0] as MeshPrimitive;
    expect(Array.from(primitive.positions)).toEqual(TRI_POSITIONS);
    expect(Array.from(primitive.indices!)).toEqual(TRI_INDICES);
  });

  it('also accepts Khronos KHR_meshopt_compression assets through the meshopt hook', async () => {
    const { gltf, buffers } = makeMeshoptGltf({
      extensionsRequired: ['KHR_meshopt_compression'],
    });
    gltf.extensionsUsed = ['KHR_meshopt_compression'];
    for (const bufferView of gltf.bufferViews ?? []) {
      const ext = bufferView.extensions?.EXT_meshopt_compression;
      if (!ext) continue;
      bufferView.extensions = { KHR_meshopt_compression: ext };
    }
    gltf.buffers![0] = {
      byteLength: 0,
      extensions: { KHR_meshopt_compression: { fallback: true } },
    };

    const result = await gltfToScene(gltf, { buffers, meshoptDecode: meshoptTriHook });

    expect(result.warnings.some(w => w.includes('KHR_meshopt_compression'))).toBe(false);
    expect(result.scene.primitives).toHaveLength(1);
    const primitive = result.scene.primitives[0] as MeshPrimitive;
    expect(Array.from(primitive.positions)).toEqual(TRI_POSITIONS);
    expect(Array.from(primitive.indices!)).toEqual(TRI_INDICES);
  });

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
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.positions)).toEqual(TRI_POSITIONS);
    expect(Array.from(prim.indices!)).toEqual(TRI_INDICES);
    expect(warnings.some(w =>
      w.includes('EXT_meshopt_compression') && w.includes('fallback'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'meshopt-fallback-buffer-used',
        path: 'bufferViews[0].extensions.EXT_meshopt_compression',
        bufferViewIndex: 0,
      }),
      expect.objectContaining({
        code: 'meshopt-fallback-buffer-used',
        path: 'bufferViews[1].extensions.EXT_meshopt_compression',
        bufferViewIndex: 1,
      }),
    ]));
  });

  it('no hook + fallback:true stub buffer → warn + primitive skipped', async () => {
    const { gltf, buffers } = makeMeshoptGltf(); // buffer 0 is a fallback stub
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('no uncompressed fallback buffer'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'meshopt-decode-hook-missing',
        path: 'bufferViews[0].extensions.EXT_meshopt_compression',
        bufferViewIndex: 0,
      }),
      expect.objectContaining({
        code: 'meshopt-decode-hook-missing',
        path: 'bufferViews[1].extensions.EXT_meshopt_compression',
        bufferViewIndex: 1,
      }),
    ]));
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
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers, meshoptDecode: hook });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('count × byteStride'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'meshopt-decoded-byte-length-mismatch',
        path: 'bufferViews[0].extensions.EXT_meshopt_compression',
        bufferViewIndex: 0,
      }),
    ]));
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
