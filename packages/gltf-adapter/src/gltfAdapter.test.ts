// gltfAdapter.test.ts — Unit tests for @vitrum/gltf-adapter.
//
// All fixtures are built in-code (no network, no binary fixture files).
// Tests cover:
//   1.  Minimal triangle (positions, flat-normal generation, default material)
//   2.  Textured quad GLB built byte-by-byte
//   3.  Transmissive/IOR/volume material field mapping
//   4.  Node hierarchy with nested TRS transforms → correct world positions
//   5.  Multi-primitive mesh → multiple ScenePrimitives
//   6.  Sparse accessor handling
//   7.  Non-triangle mode warning + skip
//   8.  KHR_draco rejection warning
//   9.  Unknown extension warning
//   10. Missing buffer warning
//   11. alphaMode + alphaCutoff mapping
//   12. KHR_materials_sheen / clearcoat / iridescence / anisotropy / specular mapping
//   13. Skins → SkinnedMeshPrimitive (JOINTS_0 u8 + u16, WEIGHTS_0 float, inverseBindMatrices)
//   14. KHR_lights_punctual → SceneEmitter[] (point, spot, directional; world-transform applied)

import { describe, it, expect, vi } from 'vitest';
import { GltfImportError, gltfToScene } from './gltfToScene.js';
import { analyzeGltfAsset } from './featureReport.js';
import { GltfParseFailed } from './errors.js';
import { solveSkin, validateScene } from '@vitrum/core';
import type { GltfJson } from './gltfTypes.js';
import type {
  DirectionalEmitter,
  InstancedMeshPrimitive,
  MeshPrimitive,
  PointEmitter,
  SkinnedMeshPrimitive,
  SpotEmitter,
  TextureRef,
} from '@vitrum/core';

// ────────────────────────────────────────────────────────────────────────────
// Fixture helpers
// ────────────────────────────────────────────────────────────────────────────

/** Encode a Float32Array as a glTF buffer and return its ArrayBuffer. */
function f32Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

/** Encode a Uint16Array as a glTF buffer. */
function u16Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 2);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setUint16(i * 2, v, true));
  return buf;
}

/** Encode a Uint8Array as a glTF buffer. */
function u8Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setUint8(i, v));
  return buf;
}

/** Encode an Int8Array as a glTF buffer. */
function i8Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setInt8(i, v));
  return buf;
}

/** Encode a Uint32Array as a glTF buffer. */
function u32Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setUint32(i * 4, v, true));
  return buf;
}

/** Concatenate multiple ArrayBuffers into one. */
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

function textBuffer(text: string): ArrayBuffer {
  const encoded = new TextEncoder().encode(text);
  return encoded.buffer.slice(encoded.byteOffset, encoded.byteOffset + encoded.byteLength);
}

function transformPoint(m: Float32Array | undefined, x: number, y: number, z: number): [number, number, number] {
  if (!m) return [x, y, z];
  return [
    m[0]! * x + m[4]! * y + m[8]! * z + m[12]!,
    m[1]! * x + m[5]! * y + m[9]! * z + m[13]!,
    m[2]! * x + m[6]! * y + m[10]! * z + m[14]!,
  ];
}

// Triangle: 3 vertices at (0,0,0), (1,0,0), (0,1,0)
const TRIANGLE_POSITIONS = [0, 0, 0, 1, 0, 0, 0, 1, 0];
// 3 flat normals: cross((1,0,0)-(0,0,0), (0,1,0)-(0,0,0)) = (0,0,1)
const _TRIANGLE_FLAT_NORMAL = [0, 0, 1];
const TRIANGLE_NORMALS = [0, 0, 1, 0, 0, 1, 0, 0, 1];
const TRIANGLE_UVS = [0, 0, 1, 0, 0, 1];
const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47];

function makeMinimalTriangleGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const posBuf = f32Buffer(TRIANGLE_POSITIONS);
  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scenes: [{ nodes: [0] }],
    scene: 0,
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126 /* FLOAT */,
        count: 3,
        type: 'VEC3',
      },
    ],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength }],
    buffers: [{ byteLength: posBuf.byteLength }],
  };
  return { gltf, buffers: new Map([[0, posBuf]]) };
}

function appendF32Accessor(
  fixture: { gltf: GltfJson; buffers: Map<number, ArrayBuffer> },
  values: number[],
  type: NonNullable<GltfJson['accessors']>[number]['type'],
  count: number,
): number {
  const data = f32Buffer(values);
  const base = fixture.buffers.get(0) ?? new ArrayBuffer(0);
  const byteOffset = base.byteLength;
  const packed = concatBuffers(base, data);
  fixture.buffers.set(0, packed);

  fixture.gltf.bufferViews ??= [];
  const bufferView = fixture.gltf.bufferViews.length;
  fixture.gltf.bufferViews.push({
    buffer: 0,
    byteOffset,
    byteLength: data.byteLength,
  });

  fixture.gltf.accessors ??= [];
  const accessorIndex = fixture.gltf.accessors.length;
  fixture.gltf.accessors.push({
    bufferView,
    componentType: 5126,
    count,
    type,
  });

  fixture.gltf.buffers ??= [{ byteLength: 0 }];
  fixture.gltf.buffers[0] = {
    ...(fixture.gltf.buffers[0] ?? {}),
    byteLength: packed.byteLength,
  };
  return accessorIndex;
}

function makeNormalMappedTriangleGltf(
  authoredTangents?: number[],
): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const posBuf = f32Buffer(TRIANGLE_POSITIONS);
  const normalBuf = f32Buffer(TRIANGLE_NORMALS);
  const uvBuf = f32Buffer(TRIANGLE_UVS);
  const tangentBuf = authoredTangents ? f32Buffer(authoredTangents) : undefined;
  const imageBuf = u8Buffer(PNG_MAGIC);
  const packed = concatBuffers(
    posBuf,
    normalBuf,
    uvBuf,
    ...(tangentBuf ? [tangentBuf] : []),
    imageBuf,
  );

  const bufferViews: NonNullable<GltfJson['bufferViews']> = [];
  let offset = 0;
  for (const buf of [posBuf, normalBuf, uvBuf, ...(tangentBuf ? [tangentBuf] : []), imageBuf]) {
    bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: buf.byteLength });
    offset += buf.byteLength;
  }
  const imageBufferView = bufferViews.length - 1;

  const attributes: Record<string, number> = {
    POSITION: 0,
    NORMAL: 1,
    TEXCOORD_0: 2,
  };
  if (authoredTangents) attributes.TANGENT = 3;

  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scenes: [{ nodes: [0] }],
    scene: 0,
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes, material: 0 }] }],
    materials: [{ normalTexture: { index: 0 } }],
    textures: [{ source: 0 }],
    images: [{ bufferView: imageBufferView, mimeType: 'image/png' }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC2' },
      ...(authoredTangents
        ? [{ bufferView: 3, componentType: 5126, count: 3, type: 'VEC4' } as const]
        : []),
    ],
    bufferViews,
    buffers: [{ byteLength: packed.byteLength }],
  };
  return { gltf, buffers: new Map([[0, packed]]) };
}

function makeUv1NormalMappedTriangleGltf(opts: {
  normalTexCoord: number;
  includeUv0?: boolean;
}): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const posBuf = f32Buffer(TRIANGLE_POSITIONS);
  const normalBuf = f32Buffer(TRIANGLE_NORMALS);
  const uv1Buf = f32Buffer([
    0, 0,
    0, 1,
    1, 0,
  ]);
  const imageBuf = u8Buffer(PNG_MAGIC);
  const includeUv0 = opts.includeUv0 === true;
  const uv0Buf = includeUv0 ? f32Buffer(TRIANGLE_UVS) : undefined;
  const chunks = [posBuf, normalBuf, ...(uv0Buf ? [uv0Buf] : []), uv1Buf, imageBuf];
  const packed = concatBuffers(...chunks);
  const bufferViews: NonNullable<GltfJson['bufferViews']> = [];
  let offset = 0;
  for (const buf of chunks) {
    bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: buf.byteLength });
    offset += buf.byteLength;
  }
  const uv1Accessor = includeUv0 ? 3 : 2;
  const imageBufferView = bufferViews.length - 1;
  const attributes: Record<string, number> = {
    POSITION: 0,
    NORMAL: 1,
    TEXCOORD_1: uv1Accessor,
  };
  if (includeUv0) attributes.TEXCOORD_0 = 2;
  const accessors: NonNullable<GltfJson['accessors']> = [
    { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
    { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
    ...(includeUv0
      ? [{ bufferView: 2, componentType: 5126, count: 3, type: 'VEC2' } as const]
      : []),
    { bufferView: uv1Accessor, componentType: 5126, count: 3, type: 'VEC2' },
  ];
  return {
    gltf: {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes, material: 0 }] }],
      materials: [{ normalTexture: { index: 0, texCoord: opts.normalTexCoord } }],
      textures: [{ source: 0 }],
      images: [{ bufferView: imageBufferView, mimeType: 'image/png' }],
      accessors,
      bufferViews,
      buffers: [{ byteLength: packed.byteLength }],
    },
    buffers: new Map([[0, packed]]),
  };
}

// ────────────────────────────────────────────────────────────────────────────
// ArrayBuffer input parsing
// ────────────────────────────────────────────────────────────────────────────

describe('ArrayBuffer input parsing', () => {
  it('throws typed parse failures for malformed JSON buffers', async () => {
    await expect(gltfToScene(textBuffer('{ "asset": '))).rejects.toBeInstanceOf(GltfParseFailed);
    await expect(gltfToScene(textBuffer('{ "asset": '))).rejects.toMatchObject({
      code: 'GLTF_PARSE_FAILED',
      format: 'gltf-json',
      reason: 'json-parse-failed',
    });
  });

  it('throws typed parse failures for empty ArrayBuffer input', async () => {
    await expect(gltfToScene(new ArrayBuffer(0))).rejects.toBeInstanceOf(GltfParseFailed);
    await expect(gltfToScene(new ArrayBuffer(0))).rejects.toMatchObject({
      code: 'GLTF_PARSE_FAILED',
      format: 'gltf-json',
      reason: 'json-parse-failed',
    });
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 1 — Minimal triangle
// ────────────────────────────────────────────────────────────────────────────

describe('glTF version compatibility boundary', () => {
  it.each([
    { version: '2.7', minVersion: undefined },
    { version: '2.7', minVersion: '2.0' },
  ])('accepts forward-compatible glTF $version with minVersion $minVersion', async ({
    version,
    minVersion,
  }) => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.asset = { version, ...(minVersion !== undefined ? { minVersion } : {}) };

    await expect(gltfToScene(gltf, { buffers })).resolves.toMatchObject({
      scene: { primitives: [expect.any(Object)] },
    });
  });

  it('rejects a forward minor version that explicitly requires newer features', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.asset = { version: '2.7', minVersion: '2.1' };

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        code: 'unsupported-version',
        path: 'asset.minVersion',
      })],
    });
  });

  it.each([
    { asset: { version: '3.0' }, path: 'asset.version' },
    { asset: { version: '2' }, path: 'asset.version' },
    { asset: { version: '2.0', minVersion: '2.1' }, path: 'asset.minVersion' },
    { asset: { version: '2.7', minVersion: 'not-a-version' }, path: 'asset.minVersion' },
  ])('rejects unsupported or malformed version metadata at $path', async ({ asset, path }) => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.asset = asset;

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        code: 'unsupported-version',
        path,
      })],
    });
  });
});

describe('minimal triangle', () => {
  it('produces one MeshPrimitive with correct positions', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    const { scene } = await gltfToScene(gltf, { buffers });

    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.kind).toBe('mesh');
    expect(Array.from(prim.positions)).toEqual(TRIANGLE_POSITIONS);
  });

  it('generates vertex normals when NORMAL is absent', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.normals).toBeInstanceOf(Float32Array);
    // Flat normal for this triangle is [0,0,1] for each of 3 vertices
    expect(prim.normals[2]).toBeCloseTo(1, 5); // vertex 0, z
    expect(prim.normals[5]).toBeCloseTo(1, 5); // vertex 1, z
    expect(prim.normals[8]).toBeCloseTo(1, 5); // vertex 2, z
    expect(warnings.some(w => w.includes('vertex normals'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'generated-flat-normals',
      path: 'meshes[0].primitives[0].attributes.NORMAL',
    }));
  });

  it('uses the default glTF material when no material is specified', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    const { scene } = await gltfToScene(gltf, { buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.material.baseColor).toEqual([1, 1, 1]);
    expect(prim.material.metallic).toBe(1);
    expect(prim.material.roughness).toBe(1);
  });

  it('generates tangents from TEXCOORD_1 when a tangent-space map selects texCoord 1', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({ normalTexCoord: 1 });
    const { scene, diagnostics, warnings } = await gltfToScene(gltf, { buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.uvs).toBeUndefined();
    expect(Array.from(prim.uv1 ?? [])).toEqual([
      0, 0,
      0, 1,
      1, 0,
    ]);
    expect(prim.tangents).toBeInstanceOf(Float32Array);
    expect(prim.tangents).toHaveLength(12);
    expect(warnings.some((w) => w.includes('generated per-vertex tangents from POSITION/NORMAL/TEXCOORD_1'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'generated-tangents',
      path: 'meshes[0].primitives[0].attributes.TANGENT',
    }));
  });

  it('rejects a texCoord 2 tangent-space map when the primitive has no TEXCOORD_2 accessor', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({
      normalTexCoord: 2,
      includeUv0: true,
    });
    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'missing-material-texcoord',
        path: 'materials[0].normalTexture',
      })],
    });
  });

  it('preserves one high material UV set in its indexed core uvSets lane', async () => {
    const fixture = makeUv1NormalMappedTriangleGltf({
      normalTexCoord: 2,
      includeUv0: true,
    });
    const uv2 = appendF32Accessor(
      fixture,
      [
        0.25, 0.25,
        0.75, 0.25,
        0.25, 0.75,
      ],
      'VEC2',
      3,
    );
    fixture.gltf.meshes![0]!.primitives[0]!.attributes.TEXCOORD_2 = uv2;

    const { scene, warnings } = await gltfToScene(fixture.gltf, { buffers: fixture.buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect((prim.material.normalMap)?.texCoord).toBe(2);
    expect(Array.from(prim.uv1 ?? [])).toEqual([
      0, 0,
      0, 1,
      1, 0,
    ]);
    expect(Array.from(prim.uvSets?.[2] ?? [])).toEqual([
      0.25, 0.25,
      0.75, 0.25,
      0.25, 0.75,
    ]);
    expect(prim.tangents).toBeInstanceOf(Float32Array);
    expect(warnings.some((w) => w.includes('sampled with the wrong UV channel'))).toBe(false);
  });

  it('rejects a high-UV material map when the accessor is not VEC2', async () => {
    const fixture = makeUv1NormalMappedTriangleGltf({
      normalTexCoord: 2,
      includeUv0: true,
    });
    const uv2 = appendF32Accessor(
      fixture,
      [
        0.25, 0.25, 0,
        0.75, 0.25, 0,
        0.25, 0.75, 0,
      ],
      'VEC3',
      3,
    );
    fixture.gltf.meshes![0]!.primitives[0]!.attributes.TEXCOORD_2 = uv2;

    await expect(gltfToScene(fixture.gltf, { buffers: fixture.buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'invalid-primitive-attribute',
        path: 'meshes[0].primitives[0].attributes.TEXCOORD_2',
        message: expect.stringContaining('TEXCOORD_2 accessor must be VEC2'),
      })],
    });
  });

  it('preserves independent material-visible UV sets 1 and 2 without remapping', async () => {
    const fixture = makeUv1NormalMappedTriangleGltf({
      normalTexCoord: 2,
      includeUv0: true,
    });
    const uv2Values = [
      0.25, 0.25,
      0.75, 0.25,
      0.25, 0.75,
    ];
    const uv2 = appendF32Accessor(
      fixture,
      uv2Values,
      'VEC2',
      3,
    );
    fixture.gltf.meshes![0]!.primitives[0]!.attributes.TEXCOORD_2 = uv2;
    fixture.gltf.materials![0] = {
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0, texCoord: 1 },
      },
      normalTexture: { index: 0, texCoord: 2 },
    };

    const { scene, warnings } = await gltfToScene(fixture.gltf, { buffers: fixture.buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect((prim.material.baseColorMap)?.texCoord).toBe(1);
    expect((prim.material.normalMap)?.texCoord).toBe(2);
    expect(Array.from(prim.uvs ?? [])).toEqual(Array.from(new Float32Array(TRIANGLE_UVS)));
    expect(Array.from(prim.uv1 ?? [])).toEqual([
      0, 0,
      0, 1,
      1, 0,
    ]);
    expect(Array.from(prim.uvSets?.[2] ?? [])).toEqual(Array.from(new Float32Array(uv2Values)));
    expect(prim.tangents).toBeInstanceOf(Float32Array);
    expect(warnings.some((w) => w.includes('sampled with the wrong UV channel'))).toBe(false);
  });

  it('preserves two high material-visible UV sets in distinct indexed lanes', async () => {
    const fixture = makeUv1NormalMappedTriangleGltf({
      normalTexCoord: 3,
      includeUv0: true,
    });
    const uv2Values = [
      0.2, 0.2,
      0.8, 0.2,
      0.2, 0.8,
    ];
    const uv3Values = [
      0.1, 0.3,
      0.7, 0.3,
      0.1, 0.9,
    ];
    const uv2 = appendF32Accessor(fixture, uv2Values, 'VEC2', 3);
    const uv3 = appendF32Accessor(fixture, uv3Values, 'VEC2', 3);
    fixture.gltf.meshes![0]!.primitives[0]!.attributes.TEXCOORD_2 = uv2;
    fixture.gltf.meshes![0]!.primitives[0]!.attributes.TEXCOORD_3 = uv3;
    fixture.gltf.materials![0] = {
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0, texCoord: 2 },
      },
      normalTexture: { index: 0, texCoord: 3 },
    };

    const { scene, warnings } = await gltfToScene(fixture.gltf, { buffers: fixture.buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect((prim.material.baseColorMap)?.texCoord).toBe(2);
    expect((prim.material.normalMap)?.texCoord).toBe(3);
    expect(Array.from(prim.uvs ?? [])).toEqual(Array.from(new Float32Array(TRIANGLE_UVS)));
    expect(Array.from(prim.uv1 ?? [])).toEqual([0, 0, 0, 1, 1, 0]);
    expect(Array.from(prim.uvSets?.[2] ?? [])).toEqual(Array.from(new Float32Array(uv2Values)));
    expect(Array.from(prim.uvSets?.[3] ?? [])).toEqual(Array.from(new Float32Array(uv3Values)));
    expect(prim.tangents).toBeInstanceOf(Float32Array);
    expect(warnings.some((w) => w.includes('sampled with the wrong UV channel'))).toBe(false);
  });

  it('preserves sparse UV/color/morph lanes through fallback remap above the JS array-index ceiling', async () => {
    const nativeCeilingIndex = 0xffff_fffe;
    const ordinaryPropertyIndex = 0x1_0000_0001;
    const fixture = makeMinimalTriangleGltf();
    const uvValues = [
      0.2, 0.3,
      0.8, 0.3,
      0.2, 0.9,
    ];
    const colorValues = [
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ];
    const uvDelta = [
      0.1, 0.2,
      0.1, 0.2,
      0.1, 0.2,
    ];
    const uvAccessor = appendF32Accessor(fixture, uvValues, 'VEC2', 3);
    const colorAccessor = appendF32Accessor(fixture, colorValues, 'VEC3', 3);
    const morphAccessor = appendF32Accessor(fixture, uvDelta, 'VEC2', 3);
    const primitive = fixture.gltf.meshes![0]!.primitives[0]!;
    primitive.attributes[`TEXCOORD_${nativeCeilingIndex}`] = uvAccessor;
    primitive.attributes[`TEXCOORD_${ordinaryPropertyIndex}`] = uvAccessor;
    primitive.attributes[`COLOR_${nativeCeilingIndex}`] = colorAccessor;
    primitive.attributes[`COLOR_${ordinaryPropertyIndex}`] = colorAccessor;
    primitive.targets = [{
      [`TEXCOORD_${ordinaryPropertyIndex}`]: morphAccessor,
    }];
    primitive.mode = 1; // exercise point/line attribute + morph remapping
    fixture.gltf.meshes![0]!.weights = [0.5];

    const { scene } = await gltfToScene(fixture.gltf, { buffers: fixture.buffers });
    const imported = scene.primitives[0] as SkinnedMeshPrimitive;
    const remappedVertexCount = imported.positions.length / 3;

    expect(imported.kind).toBe('skinned-mesh');
    expect(imported.uvSets?.[nativeCeilingIndex]).toHaveLength(remappedVertexCount * 2);
    expect(imported.uvSets?.[ordinaryPropertyIndex]).toHaveLength(remappedVertexCount * 2);
    expect(imported.colorSets?.[nativeCeilingIndex]?.length).toBeGreaterThanOrEqual(
      remappedVertexCount * 3,
    );
    expect(imported.colorSets?.[ordinaryPropertyIndex]?.length).toBeGreaterThanOrEqual(
      remappedVertexCount * 3,
    );
    expect(imported.morphTargetUvSets?.[ordinaryPropertyIndex]).toHaveLength(1);
    expect(() => validateScene(scene)).not.toThrow();

    const solved = solveSkin(imported);
    expect(solved.uvSets?.[ordinaryPropertyIndex]).toHaveLength(remappedVertexCount * 2);
    expect(Object.keys(imported.uvSets ?? [])).toEqual(
      expect.arrayContaining([
        String(nativeCeilingIndex),
        String(ordinaryPropertyIndex),
      ]),
    );
  });

  it.each([
    ['TEXCOORD_02', 'attributes.TEXCOORD_02'],
    ['COLOR_00', 'attributes.COLOR_00'],
    ['TEXCOORD_9007199254740992', 'attributes.TEXCOORD_9007199254740992'],
    ['JOINTS_00', 'attributes.JOINTS_00'],
    ['WEIGHTS_9007199254740992', 'attributes.WEIGHTS_9007199254740992'],
    ['JOINTS_-1', 'attributes.JOINTS_-1'],
    ['WEIGHTS_1e2', 'attributes.WEIGHTS_1e2'],
    ['TEXCOORD_', 'attributes.TEXCOORD_'],
  ])('rejects noncanonical or unsafe reserved primitive semantic %s', async (semantic, path) => {
    const fixture = makeMinimalTriangleGltf();
    fixture.gltf.meshes![0]!.primitives[0]!.attributes[semantic] = 0;

    await expect(gltfToScene(fixture.gltf, { buffers: fixture.buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        code: 'ignored-primitive-attribute',
        path: `meshes[0].primitives[0].${path}`,
      })],
    });
  });

  it.each([
    'JOINTS_00',
    'WEIGHTS_9007199254740992',
    'JOINTS_-1',
    'WEIGHTS_1e2',
    'TEXCOORD_',
  ])('rejects malformed reserved semantic %s during feature inventory preflight', (semantic) => {
    const fixture = makeMinimalTriangleGltf();
    fixture.gltf.meshes![0]!.primitives[0]!.attributes[semantic] = 0;

    expect(() => analyzeGltfAsset(fixture.gltf)).toThrow(
      /not canonical|safe-integer semantic range|non-negative canonical integer/,
    );
  });

  it('rejects an unknown non-application primitive semantic during feature inventory preflight', () => {
    const fixture = makeMinimalTriangleGltf();
    fixture.gltf.meshes![0]!.primitives[0]!.attributes.CUSTOM_WEIGHT = 0;

    expect(() => analyzeGltfAsset(fixture.gltf)).toThrow(
      /unknown non-application primitive semantic "CUSTOM_WEIGHT"/,
    );
  });

  it('rejects a leading-zero TEXCOORD morph semantic before it can alias a canonical lane', async () => {
    const fixture = makeMinimalTriangleGltf();
    fixture.gltf.meshes![0]!.primitives[0]!.targets = [{ TEXCOORD_02: 0 }];
    fixture.gltf.meshes![0]!.weights = [0];

    await expect(gltfToScene(fixture.gltf, { buffers: fixture.buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        code: 'ignored-morph-target-attribute',
        path: 'meshes[0].primitives[0].targets[0].TEXCOORD_02',
      })],
    });
  });

  it('imports COLOR_0 vertex colors onto the core mesh primitive', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const colorBuf = f32Buffer([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const packed = concatBuffers(posBuf, colorBuf);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, COLOR_0: 1 } }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength, byteLength: colorBuf.byteLength },
      ],
      buffers: [{ byteLength: packed.byteLength }],
    };

    const { scene } = await gltfToScene(gltf, { buffers: new Map([[0, packed]]) });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.colors ?? [])).toEqual([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
  });

  it('rejects malformed TEXCOORD_0 and TEXCOORD_1 accessors', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const uv0Buf = f32Buffer([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]);
    const uv1Buf = f32Buffer([
      0, 0, 1,
      1, 0, 1,
      0, 1, 1,
    ]);
    const packed = concatBuffers(posBuf, uv0Buf, uv1Buf);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1, TEXCOORD_1: 2 } }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 2, componentType: 5126, count: 3, type: 'VEC3' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength, byteLength: uv0Buf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength + uv0Buf.byteLength, byteLength: uv1Buf.byteLength },
      ],
      buffers: [{ byteLength: packed.byteLength }],
    };

    await expect(gltfToScene(gltf, { buffers: new Map([[0, packed]]) })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'invalid-primitive-attribute',
        path: 'meshes[0].primitives[0].attributes.TEXCOORD_0',
        message: expect.stringContaining('TEXCOORD_0 accessor must be VEC2'),
      })],
    });
  });

  it('rejects malformed COLOR_0 accessors', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const colorBuf = f32Buffer([
      1, 0,
      0, 1,
      0, 0,
    ]);
    const packed = concatBuffers(posBuf, colorBuf);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, COLOR_0: 1 } }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength, byteLength: colorBuf.byteLength },
      ],
      buffers: [{ byteLength: packed.byteLength }],
    };

    await expect(gltfToScene(gltf, { buffers: new Map([[0, packed]]) })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'invalid-primitive-attribute',
        path: 'meshes[0].primitives[0].attributes.COLOR_0',
        message: expect.stringContaining('COLOR_0 accessor must be VEC3 or VEC4'),
      })],
    });
  });

  it('preserves every secondary vertex color set without degradation', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const color0Buf = f32Buffer([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const color1Buf = f32Buffer([
      0.25, 0.25, 0.25,
      0.50, 0.50, 0.50,
      0.75, 0.75, 0.75,
    ]);
    const packed = concatBuffers(posBuf, color0Buf, color1Buf);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, COLOR_0: 1, COLOR_1: 2 } }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 2, componentType: 5126, count: 3, type: 'VEC3' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength, byteLength: color0Buf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength + color0Buf.byteLength, byteLength: color1Buf.byteLength },
      ],
      buffers: [{ byteLength: packed.byteLength }],
    };

    const { scene, warnings } = await gltfToScene(gltf, { buffers: new Map([[0, packed]]) });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.colors ?? [])).toEqual([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    expect(Array.from(prim.colorSets?.[1] ?? [])).toEqual([
      0.25, 0.25, 0.25,
      0.50, 0.50, 0.50,
      0.75, 0.75, 0.75,
    ]);
    expect(warnings.some(w => w.includes('COLOR_1') && w.includes('ignored'))).toBe(false);
  });

  it('loads through application-specific primitive attributes with a structured degradation diagnostic', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.meshes![0]!.primitives[0]!.attributes._CUSTOM_WEIGHT = 0;

    const { diagnostics, scene, warnings } = await gltfToScene(gltf, { buffers });

    expect(scene.primitives).toHaveLength(1);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'ignored-primitive-attribute',
      path: 'meshes[0].primitives[0].attributes._CUSTOM_WEIGHT',
    }));
    expect(warnings.some((warning) => warning.includes('_CUSTOM_WEIGHT'))).toBe(true);
  });

  it('still rejects unknown non-application primitive semantics', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.meshes![0]!.primitives[0]!.attributes.CUSTOM_WEIGHT = 0;

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'ignored-primitive-attribute',
        path: 'meshes[0].primitives[0].attributes.CUSTOM_WEIGHT',
      })],
    });
  });

  it.each(['TEXCOORD_CUSTOM', 'COLOR_CUSTOM'])(
    'rejects malformed reserved primitive semantic %s',
    async (semantic) => {
      const { gltf, buffers } = makeMinimalTriangleGltf();
      gltf.meshes![0]!.primitives[0]!.attributes[semantic] = 0;

      await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
        name: 'GltfImportError',
        diagnostics: [expect.objectContaining({
          severity: 'error',
          code: 'ignored-primitive-attribute',
          path: `meshes[0].primitives[0].attributes.${semantic}`,
        })],
      });
    },
  );

  it('preserves and solves additive RGB/RGBA COLOR_n morph deltas end to end', async () => {
    const fixture = makeMinimalTriangleGltf();
    const baseColorAccessor = appendF32Accessor(
      fixture,
      [1, 0, 0, 0, 1, 0, 0, 0, 1],
      'VEC3',
      3,
    );
    const colorDeltaAccessor = appendF32Accessor(
      fixture,
      [-0.25, 0.25, 0, 0.25, -0.25, 0, 0, 0.25, -0.25],
      'VEC3',
      3,
    );
    const baseColor2Accessor = appendF32Accessor(
      fixture,
      [0.1, 0.2, 0.3, 1, 0.4, 0.5, 0.6, 0.8, 0.7, 0.8, 0.9, 0.6],
      'VEC4',
      3,
    );
    const color2DeltaAccessor = appendF32Accessor(
      fixture,
      [0.2, 0, -0.2, 0, -0.2, 0.2, 0, 0.2, 0, -0.2, 0.2, -0.2],
      'VEC4',
      3,
    );
    const primitive = fixture.gltf.meshes![0]!.primitives[0]!;
    primitive.attributes.COLOR_0 = baseColorAccessor;
    primitive.attributes.COLOR_2 = baseColor2Accessor;
    primitive.targets = [{
      COLOR_0: colorDeltaAccessor,
      COLOR_2: color2DeltaAccessor,
    }];
    fixture.gltf.meshes![0]!.weights = [0.5];

    const { diagnostics, scene } = await gltfToScene(fixture.gltf, {
      buffers: fixture.buffers,
    });
    const imported = scene.primitives[0] as SkinnedMeshPrimitive;

    expect(Array.from(imported.colors ?? [])).toEqual([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    expect(imported.morphTargetColors).toHaveLength(1);
    expect(imported.morphTargetColorSets?.[0]?.[0]).toBe(imported.morphTargetColors?.[0]);
    expect(Array.from(imported.morphTargetColorSets?.[2]?.[0] ?? [])).toEqual(
      Array.from(new Float32Array([
        0.2, 0, -0.2, 0,
        -0.2, 0.2, 0, 0.2,
        0, -0.2, 0.2, -0.2,
      ])),
    );
    expect(diagnostics.some((diagnostic) =>
      diagnostic.path.includes('COLOR_') && diagnostic.code === 'ignored-morph-target-attribute'
    )).toBe(false);

    const solved = solveSkin(imported);
    expect(Array.from(solved.colors ?? [])).toEqual(
      Array.from(new Float32Array([
        0.875, 0.125, 0,
        0.125, 0.875, 0,
        0, 0.125, 0.875,
      ])),
    );
    const solvedColor2 = solved.colorSets?.[2] ?? [];
    [
      0.2, 0.2, 0.2, 1,
      0.3, 0.6, 0.6, 0.9,
      0.7, 0.7, 1, 0.5,
    ].forEach((value, index) => {
      expect(solvedColor2[index]).toBeCloseTo(value);
    });
  });

  it('still rejects unknown non-application morph-target semantics', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.meshes![0]!.primitives[0]!.targets = [{ CUSTOM_DELTA: 0 }];
    gltf.meshes![0]!.weights = [0];

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'ignored-morph-target-attribute',
        path: 'meshes[0].primitives[0].targets[0].CUSTOM_DELTA',
      })],
    });
  });

  it('rejects an unknown non-application morph-target semantic during feature inventory preflight', () => {
    const fixture = makeMinimalTriangleGltf();
    fixture.gltf.meshes![0]!.primitives[0]!.targets = [{ CUSTOM_DELTA: 0 }];

    expect(() => analyzeGltfAsset(fixture.gltf)).toThrow(
      /unknown non-application morph-target semantic "CUSTOM_DELTA"/,
    );
  });

  it('scene has empty emitters and none environment', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    const { scene } = await gltfToScene(gltf, { buffers });
    expect(scene.emitters).toHaveLength(0);
    expect(scene.environment.kind).toBe('none');
  });

  it('imports EXT_mesh_gpu_instancing as a core InstancedMeshPrimitive', async () => {
    const fixture = makeMinimalTriangleGltf();
    const { gltf, buffers } = fixture;
    const translationAccessor = appendF32Accessor(
      fixture,
      [
        2, 0, 0,
        0, 3, 0,
      ],
      'VEC3',
      2,
    );
    gltf.extensionsUsed = ['EXT_mesh_gpu_instancing'];
    gltf.extensionsRequired = ['EXT_mesh_gpu_instancing'];
    gltf.nodes![0] = {
      ...gltf.nodes![0]!,
      translation: [10, 0, 0],
      extensions: {
        EXT_mesh_gpu_instancing: {
          attributes: {
            TRANSLATION: translationAccessor,
          },
        },
      },
    };

    const { scene, warnings, instancingBindings } = await gltfToScene(gltf, { buffers });

    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as InstancedMeshPrimitive;
    expect(prim.kind).toBe('instanced-mesh');
    expect('transform' in prim).toBe(false);
    expect(prim.instances).toHaveLength(2);
    expect(prim.instances[0]![12]).toBeCloseTo(12, 5);
    expect(prim.instances[0]![13]).toBeCloseTo(0, 5);
    expect(prim.instances[1]![12]).toBeCloseTo(10, 5);
    expect(prim.instances[1]![13]).toBeCloseTo(3, 5);
    expect(instancingBindings).toHaveLength(1);
    expect(instancingBindings?.[0]).toMatchObject({
      primitiveId: 'gltf-prim-0',
      nodeIndex: 0,
    });
    expect(instancingBindings?.[0]?.localInstanceTransforms).toHaveLength(2);
    expect(instancingBindings?.[0]?.localInstanceTransforms[0]![12]).toBeCloseTo(2, 5);
    expect(instancingBindings?.[0]?.localInstanceTransforms[1]![13]).toBeCloseTo(3, 5);
    expect(warnings.some((warning) => warning.includes('EXT_mesh_gpu_instancing'))).toBe(false);
  });

  it('ignores application-specific underscore instancing attributes without aborting standard transforms', async () => {
    const fixture = makeMinimalTriangleGltf();
    const { gltf, buffers } = fixture;
    const translationAccessor = appendF32Accessor(
      fixture,
      [2, 0, 0],
      'VEC3',
      1,
    );
    gltf.extensionsUsed = ['EXT_mesh_gpu_instancing'];
    gltf.nodes![0] = {
      ...gltf.nodes![0]!,
      extensions: {
        EXT_mesh_gpu_instancing: {
          attributes: {
            TRANSLATION: translationAccessor,
            _APPLICATION_ID: Number.NaN,
          },
        },
      },
    };

    const result = await gltfToScene(gltf, { buffers });
    const primitive = result.scene.primitives[0] as InstancedMeshPrimitive;

    expect(primitive.instances).toHaveLength(1);
    expect(primitive.instances[0]![12]).toBeCloseTo(2);
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'ignored-gpu-instancing-attribute',
      path: 'nodes[0].extensions.EXT_mesh_gpu_instancing.attributes._APPLICATION_ID',
    }));
  });

  it('imports spec-legal normalized BYTE instancing rotations', async () => {
    const fixture = makeMinimalTriangleGltf();
    const { gltf, buffers } = fixture;
    const rotations = i8Buffer([
      0, 0, 0, 127,
      0, 0, 90, 90,
    ]);
    const base = buffers.get(0)!;
    const byteOffset = base.byteLength;
    const packed = concatBuffers(base, rotations);
    buffers.set(0, packed);
    const bufferView = gltf.bufferViews!.length;
    gltf.bufferViews!.push({
      buffer: 0,
      byteOffset,
      byteLength: rotations.byteLength,
    });
    const rotationAccessor = gltf.accessors!.length;
    gltf.accessors!.push({
      bufferView,
      componentType: 5120,
      normalized: true,
      count: 2,
      type: 'VEC4',
    });
    gltf.buffers![0] = { byteLength: packed.byteLength };
    gltf.extensionsUsed = ['EXT_mesh_gpu_instancing'];
    gltf.extensionsRequired = ['EXT_mesh_gpu_instancing'];
    gltf.nodes![0] = {
      ...gltf.nodes![0]!,
      extensions: {
        EXT_mesh_gpu_instancing: {
          attributes: { ROTATION: rotationAccessor },
        },
      },
    };

    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as InstancedMeshPrimitive;
    expect(prim.instances).toHaveLength(2);
    expect(prim.instances[0]![0]).toBeCloseTo(1, 5);
    expect(prim.instances[1]![0]).toBeCloseTo(0, 4);
    expect(prim.instances[1]![1]).toBeCloseTo(1, 4);
    expect(prim.instances[1]![4]).toBeCloseTo(-1, 4);
    expect(prim.instances[1]![5]).toBeCloseTo(0, 4);
  });

  it('rejects EXT_mesh_gpu_instancing accessors that disagree', async () => {
    const fixture = makeMinimalTriangleGltf();
    const { gltf, buffers } = fixture;
    const translationAccessor = appendF32Accessor(
      fixture,
      [
        2, 0, 0,
        0, 3, 0,
      ],
      'VEC3',
      2,
    );
    const scaleAccessor = appendF32Accessor(
      fixture,
      [1, 1, 1],
      'VEC3',
      1,
    );
    gltf.extensionsUsed = ['EXT_mesh_gpu_instancing'];
    gltf.nodes![0] = {
      ...gltf.nodes![0]!,
      extensions: {
        EXT_mesh_gpu_instancing: {
          attributes: {
            TRANSLATION: translationAccessor,
            SCALE: scaleAccessor,
          },
        },
      },
    };

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'ignored-gpu-instancing',
        path: 'nodes[0].extensions.EXT_mesh_gpu_instancing.attributes.SCALE',
        message: expect.stringContaining('does not match instance count'),
      })],
    });
  });

  it('generates tangents for a normal-mapped primitive that omits TANGENT', async () => {
    const handle = { kind: 'decoded-normal' };
    const { gltf, buffers } = makeNormalMappedTriangleGltf();
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, {
      buffers,
      decodeImage: async () => handle,
    });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect((prim.material.normalMap as TextureRef).handle).toBe(handle);
    expect(prim.tangents).toBeInstanceOf(Float32Array);
    expect(Array.from(prim.tangents!)).toEqual([
      1, 0, 0, 1,
      1, 0, 0, 1,
      1, 0, 0, 1,
    ]);
    expect(warnings.some((w) => w.includes('generated per-vertex tangents'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'generated-tangents',
      path: 'meshes[0].primitives[0].attributes.TANGENT',
      message: expect.stringContaining('generated per-vertex tangents'),
    }));
  });

  it('rejects malformed authored TANGENT data', async () => {
    const { gltf, buffers } = makeNormalMappedTriangleGltf([
      1, 0, 0,
      1, 0, 0,
      1, 0, 0,
    ]);
    gltf.accessors![3] = {
      ...gltf.accessors![3]!,
      type: 'VEC3',
    };

    await expect(gltfToScene(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-normal' }),
    })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'invalid-primitive-attribute',
        path: 'meshes[0].primitives[0].attributes.TANGENT',
        message: expect.stringContaining('TANGENT accessor must be VEC4'),
      })],
    });
  });

  it('does not report clean generated tangents when every UV triangle is degenerate', async () => {
    const { gltf, buffers } = makeNormalMappedTriangleGltf();
    const source = new Uint8Array(buffers.get(0)!);
    const next = source.slice();
    const uvView = gltf.bufferViews![2]!;
    next.fill(0, uvView.byteOffset ?? 0, (uvView.byteOffset ?? 0) + uvView.byteLength);
    buffers.set(0, next.buffer);

    const { scene, diagnostics } = await gltfToScene(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-normal' }),
    });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.tangents).toBeUndefined();
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'tangent-generation-failed',
      path: 'meshes[0].primitives[0].attributes.TANGENT',
    }));
    expect(diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'generated-tangents',
      path: 'meshes[0].primitives[0].attributes.TANGENT',
    }));
  });

  it('rejects a normal map when tangent generation lacks TEXCOORD_0', async () => {
    const { gltf, buffers } = makeNormalMappedTriangleGltf();
    delete gltf.meshes![0]!.primitives[0]!.attributes.TEXCOORD_0;
    await expect(gltfToScene(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-normal' }),
    })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'missing-material-texcoord',
        path: 'materials[0].normalTexture',
      })],
    });
  });

  it('preserves authored tangents instead of regenerating them', async () => {
    const authored = [
      0, 1, 0, -1,
      0, 1, 0, -1,
      0, 1, 0, -1,
    ];
    const { gltf, buffers } = makeNormalMappedTriangleGltf(authored);
    const { scene, warnings } = await gltfToScene(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-normal' }),
    });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.tangents!)).toEqual(authored);
    expect(warnings.some((w) => w.includes('generated per-vertex tangents'))).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 2 — GLB container
// ────────────────────────────────────────────────────────────────────────────

function buildGlb(jsonObj: object, binChunk?: ArrayBuffer): ArrayBuffer {
  const jsonBytes = new TextEncoder().encode(JSON.stringify(jsonObj));
  // JSON chunk must be padded to 4 bytes with 0x20 (space).
  const jsonPadded = jsonBytes.length % 4 === 0 ? jsonBytes : (() => {
    const pad = 4 - (jsonBytes.length % 4);
    const out = new Uint8Array(jsonBytes.length + pad);
    out.set(jsonBytes);
    out.fill(0x20, jsonBytes.length);
    return out;
  })();

  let binChunkBytes: Uint8Array | undefined;
  if (binChunk) {
    const pad = binChunk.byteLength % 4 === 0 ? 0 : 4 - (binChunk.byteLength % 4);
    binChunkBytes = new Uint8Array(binChunk.byteLength + pad);
    binChunkBytes.set(new Uint8Array(binChunk));
  }

  const jsonChunkLen = jsonPadded.length;
  const binChunkLen = binChunkBytes ? binChunkBytes.length : 0;
  const totalLen = 12 + 8 + jsonChunkLen + (binChunkBytes ? 8 + binChunkLen : 0);

  const out = new ArrayBuffer(totalLen);
  const dv = new DataView(out);
  const u8 = new Uint8Array(out);

  // Header
  dv.setUint32(0, 0x46546c67, true); // magic
  dv.setUint32(4, 2, true);           // version
  dv.setUint32(8, totalLen, true);    // total length

  // JSON chunk
  dv.setUint32(12, jsonChunkLen, true);
  dv.setUint32(16, 0x4e4f534a, true); // JSON
  u8.set(jsonPadded, 20);

  // BIN chunk
  if (binChunkBytes) {
    const binStart = 20 + jsonChunkLen;
    dv.setUint32(binStart, binChunkLen, true);
    dv.setUint32(binStart + 4, 0x004e4942, true); // BIN\0
    u8.set(binChunkBytes, binStart + 8);
  }

  return out;
}

describe('GLB container', () => {
  it('parses a GLB with an inline binary chunk', async () => {
    const posData = f32Buffer(TRIANGLE_POSITIONS);
    const gltfJson: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: posData.byteLength }],
      buffers: [{ byteLength: posData.byteLength }],
    };
    const glb = buildGlb(gltfJson, posData);
    const { scene } = await gltfToScene(glb);

    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.positions)).toEqual(TRIANGLE_POSITIONS);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 3 — Material field mapping
// ────────────────────────────────────────────────────────────────────────────

describe('material field mapping', () => {
  type GltfMaterialItem = NonNullable<GltfJson['materials']>[number];
  function makeGltfWithMaterial(mat: GltfMaterialItem): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [mat],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteLength: posBuf.byteLength }],
      buffers: [{ byteLength: posBuf.byteLength }],
    };
    return { gltf, buffers: new Map([[0, posBuf]]) };
  }

  it('maps pbrMetallicRoughness scalar fields', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      pbrMetallicRoughness: {
        baseColorFactor: [0.8, 0.5, 0.2, 1.0],
        metallicFactor: 0.3,
        roughnessFactor: 0.7,
      },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.baseColor).toEqual([0.8, 0.5, 0.2]);
    expect(mat.metallic).toBeCloseTo(0.3);
    expect(mat.roughness).toBeCloseTo(0.7);
  });

  it('maps glTF doubleSided into the first-class material contract without degradation', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      name: 'leaf',
      doubleSided: true,
      pbrMetallicRoughness: {
        baseColorFactor: [0.2, 0.8, 0.3, 1],
      },
    });

    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers });

    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.doubleSided).toBe(true);
    expect(mat.extensions?.doubleSided).toBeUndefined();
    expect(diagnostics.some((diagnostic) => diagnostic.path === 'materials[0].doubleSided')).toBe(false);
    expect(warnings.some((warning) => warning.includes('doubleSided'))).toBe(false);

    const falseFixture = makeGltfWithMaterial({ doubleSided: false });
    const falseResult = await gltfToScene(falseFixture.gltf, { buffers: falseFixture.buffers });
    expect((falseResult.scene.primitives[0] as MeshPrimitive).material.doubleSided).toBe(false);
  });

  it('preserves KHR_texture_transform texCoord override on texture refs', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const uv1Buf = f32Buffer(TRIANGLE_UVS);
    const imageBuf = u8Buffer([0x89, 0x50, 0x4e, 0x47]);
    const totalBuf = concatBuffers(posBuf, uv1Buf, imageBuf);
    const handle = { kind: 'decoded-texture' };
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_1: 1 }, material: 0 }] }],
      materials: [{
        pbrMetallicRoughness: {
          baseColorTexture: {
            index: 0,
            texCoord: 0,
            extensions: {
              KHR_texture_transform: {
                texCoord: 1,
                offset: [0.25, 0.5],
                scale: [2, 3],
                rotation: 0.125,
              },
            },
          },
        },
      }],
      textures: [{ source: 0 }],
      images: [{ bufferView: 2, mimeType: 'image/png' }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength, byteLength: uv1Buf.byteLength },
        {
          buffer: 0,
          byteOffset: posBuf.byteLength + uv1Buf.byteLength,
          byteLength: imageBuf.byteLength,
        },
      ],
      buffers: [{ byteLength: totalBuf.byteLength }],
    };

    const { scene } = await gltfToScene(gltf, {
      buffers: new Map([[0, totalBuf]]),
      decodeImage: async (bytes, mimeType) => {
        expect(mimeType).toBe('image/png');
        expect(Array.from(bytes)).toEqual([0x89, 0x50, 0x4e, 0x47]);
        return handle;
      },
    });

    const mat = (scene.primitives[0] as MeshPrimitive).material;
    const ref = mat.baseColorMap as TextureRef;
    expect(ref.handle).toBe(handle);
    expect(ref.texCoord).toBe(1);
    expect(ref.transform?.offset).toEqual([0.25, 0.5]);
    expect(ref.transform?.scale).toEqual([2, 3]);
    expect(ref.transform?.rotation).toBeCloseTo(0.125);
  });

  it('keeps MASK baseColorTexture alpha on baseColorMap instead of inventing alphaMap', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const uvBuf = f32Buffer(TRIANGLE_UVS);
    const imageBuf = u8Buffer(PNG_MAGIC);
    const totalBuf = concatBuffers(posBuf, uvBuf, imageBuf);
    const handle = { kind: 'decoded-masked-base-color' };
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, material: 0 }] }],
      materials: [{
        alphaMode: 'MASK',
        alphaCutoff: 0.45,
        pbrMetallicRoughness: {
          baseColorTexture: { index: 0 },
        },
      }],
      textures: [{ source: 0 }],
      images: [{ bufferView: 2, mimeType: 'image/png' }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength, byteLength: uvBuf.byteLength },
        {
          buffer: 0,
          byteOffset: posBuf.byteLength + uvBuf.byteLength,
          byteLength: imageBuf.byteLength,
        },
      ],
      buffers: [{ byteLength: totalBuf.byteLength }],
    };

    const { scene } = await gltfToScene(gltf, {
      buffers: new Map([[0, totalBuf]]),
      decodeImage: async () => handle,
    });

    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.alphaMode).toBe('mask');
    expect(mat.alphaCutoff).toBeCloseTo(0.45);
    expect((mat.baseColorMap as TextureRef).handle).toBe(handle);
    expect(mat.alphaMap).toBeUndefined();
  });

  it('decodes baseColorTexture images embedded as data: URIs', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const uvBuf = f32Buffer(TRIANGLE_UVS);
    const vertexBuf = concatBuffers(posBuf, uvBuf);
    const handle = { kind: 'decoded-data-uri-texture' };
    const decodeImage = vi.fn(async (bytes: Uint8Array, mimeType: string) => {
      expect(Array.from(bytes)).toEqual([1, 2, 3]);
      expect(mimeType).toBe('image/png');
      return handle;
    });
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, material: 0 }] }],
      materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
      textures: [{ source: 0 }],
      images: [{ uri: 'data:image/png;base64,AQID' }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength, byteLength: uvBuf.byteLength },
      ],
      buffers: [{ byteLength: vertexBuf.byteLength }],
    };

    const { scene } = await gltfToScene(gltf, {
      buffers: new Map([[0, vertexBuf]]),
      decodeImage,
    });

    expect(decodeImage).toHaveBeenCalledOnce();
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    const ref = mat.baseColorMap as TextureRef;
    expect(ref.handle).toBe(handle);
  });

  it('rejects unresolved external URI images instead of handing them to decodeImage', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const decodeImage = vi.fn(async () => ({ kind: 'should-not-be-used' }));
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
      textures: [{ source: 0 }],
      images: [{ uri: 'textures/albedo.png', mimeType: 'image/png' }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteLength: posBuf.byteLength }],
      buffers: [{ byteLength: posBuf.byteLength }],
    };

    const importPromise = gltfToScene(gltf, {
      buffers: new Map([[0, posBuf]]),
      decodeImage,
    });

    await expect(importPromise).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: expect.arrayContaining([
        expect.objectContaining({
          severity: 'warning',
          code: 'external-image-uri',
          path: 'images[0].uri',
        }),
        expect.objectContaining({
          severity: 'error',
          code: 'material-texture-unresolved',
          path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        }),
      ]),
    });
    expect(decodeImage).not.toHaveBeenCalled();
  });

  it('maps emissiveFactor', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1] },
      emissiveFactor: [1.0, 0.5, 0.0],
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.emissive).toEqual([1.0, 0.5, 0.0]);
  });

  it('maps KHR_materials_emissive_strength', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      emissiveFactor: [1, 1, 1],
      extensions: { KHR_materials_emissive_strength: { emissiveStrength: 3.0 } },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.emissiveIntensity).toBeCloseTo(3.0);
  });

  it('maps KHR_materials_transmission + ior', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        KHR_materials_transmission: { transmissionFactor: 0.9 },
        KHR_materials_ior: { ior: 1.45 },
      },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.transmission).toBeCloseTo(0.9);
    expect(mat.ior).toBeCloseTo(1.45);
  });

  it('maps KHR_materials_volume (thickness + attenuationDistance + attenuationColor)', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        KHR_materials_volume: {
          thicknessFactor: 0.5,
          attenuationDistance: 2.0,
          attenuationColor: [0.9, 0.8, 0.7],
        },
      },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.thickness).toBeCloseTo(0.5);
    expect(mat.attenuationDistance).toBeCloseTo(2.0);
    expect(mat.attenuationColor).toEqual([0.9, 0.8, 0.7]);
  });

  it('maps alphaMode MASK + alphaCutoff', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      alphaMode: 'MASK',
      alphaCutoff: 0.3,
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.alphaMode).toBe('mask');
    expect(mat.alphaCutoff).toBeCloseTo(0.3);
  });

  it('maps alphaMode BLEND', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({ alphaMode: 'BLEND' });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.alphaMode).toBe('blend');
  });

  it('rejects an invalid alphaMode instead of silently treating it as opaque', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      alphaMode: 'INVALID' as never,
    });
    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [
        expect.objectContaining({
          severity: 'error',
          code: 'invalid-material-alpha-mode',
          path: 'materials[0].alphaMode',
        }),
      ],
    });
  });

  it('ignores baseColor alpha for opaque materials in conversion and feature reporting', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      pbrMetallicRoughness: { baseColorFactor: [0.8, 0.6, 0.4, 0.35] },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.baseColor).toEqual([0.8, 0.6, 0.4]);
    expect(mat.alphaMode).toBe('opaque');
    expect(mat.opacity).toBeUndefined();

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.materialFields).toContain('baseColor');
    expect(report.materials.materialFields).not.toContain('opacity');
  });

  it('maps and reports baseColor alpha as opacity for BLEND materials', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      alphaMode: 'BLEND',
      pbrMetallicRoughness: { baseColorFactor: [0.8, 0.6, 0.4, 0.35] },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.alphaMode).toBe('blend');
    expect(mat.opacity).toBeCloseTo(0.35);

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.materialFields).toContain('opacity');
    expect(report.materials.issuePaths['field:opacity']).toEqual([
      'materials[0].pbrMetallicRoughness.baseColorFactor[3]',
    ]);
  });

  it('ignores spec-gloss diffuse alpha for opaque materials in conversion and feature reporting', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        KHR_materials_pbrSpecularGlossiness: {
          diffuseFactor: [0.2, 0.4, 0.6, 0.25],
        },
      },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.baseColor).toEqual([0.2, 0.4, 0.6]);
    expect(mat.alphaMode).toBe('opaque');
    expect(mat.opacity).toBeUndefined();

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.materialFields).not.toContain('opacity');
  });

  it('maps KHR_materials_sheen', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        KHR_materials_sheen: {
          sheenColorFactor: [0.5, 0.3, 0.1],
          sheenRoughnessFactor: 0.4,
        },
      },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.sheen).toBe(1);
    expect(mat.sheenColor).toEqual([0.5, 0.3, 0.1]);
    expect(mat.sheenRoughness).toBeCloseTo(0.4);
  });

  it('maps KHR_materials_clearcoat', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        KHR_materials_clearcoat: {
          clearcoatFactor: 0.8,
          clearcoatRoughnessFactor: 0.1,
        },
      },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.clearcoat).toBeCloseTo(0.8);
    expect(mat.clearcoatRoughness).toBeCloseTo(0.1);
  });

  it('preserves authored clearcoat state when the factor is zero', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        KHR_materials_clearcoat: {
          clearcoatFactor: 0,
          clearcoatRoughnessFactor: 0.35,
        },
      },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.clearcoat).toBe(0);
    expect(mat.clearcoatRoughness).toBeCloseTo(0.35);
  });

  it('maps KHR_materials_iridescence', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        KHR_materials_iridescence: {
          iridescenceFactor: 0.7,
          iridescenceIor: 2.0,
          iridescenceThicknessMinimum: 200,
          iridescenceThicknessMaximum: 800,
        },
      },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.iridescence).toBeCloseTo(0.7);
    expect(mat.iridescenceIor).toBeCloseTo(2.0);
    expect(mat.iridescenceThicknessRange).toEqual([200, 800]);
  });

  it('preserves authored iridescence state when the factor is zero', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        KHR_materials_iridescence: {
          iridescenceFactor: 0,
          iridescenceIor: 1.8,
          iridescenceThicknessMinimum: 175,
          iridescenceThicknessMaximum: 625,
        },
      },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.iridescence).toBe(0);
    expect(mat.iridescenceIor).toBeCloseTo(1.8);
    expect(mat.iridescenceThicknessRange).toEqual([175, 625]);
  });

  it('maps KHR_materials_anisotropy', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        KHR_materials_anisotropy: {
          anisotropyStrength: 0.6,
          anisotropyRotation: 1.0,
        },
      },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.anisotropy).toBeCloseTo(0.6);
    expect(mat.anisotropyRotation).toBeCloseTo(1.0);
  });

  it('generates tangents for an anisotropy-mapped primitive that omits TANGENT', async () => {
    const handle = { kind: 'decoded-anisotropy' };
    const { gltf, buffers } = makeNormalMappedTriangleGltf();
    gltf.materials = [{
      extensions: {
        KHR_materials_anisotropy: {
          anisotropyStrength: 0.6,
          anisotropyRotation: 1.0,
          anisotropyTexture: { index: 0 },
        },
      },
    }];

    const { scene, warnings, diagnostics } = await gltfToScene(gltf, {
      buffers,
      decodeImage: async () => handle,
    });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect((prim.material.anisotropyMap as TextureRef).handle).toBe(handle);
    expect(prim.tangents).toBeInstanceOf(Float32Array);
    expect(Array.from(prim.tangents!)).toEqual([
      1, 0, 0, 1,
      1, 0, 0, 1,
      1, 0, 0, 1,
    ]);
    expect(warnings.some((w) => w.includes('generated per-vertex tangents'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'generated-tangents',
      path: 'meshes[0].primitives[0].attributes.TANGENT',
    }));
  });

  it('maps KHR_materials_specular', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        KHR_materials_specular: {
          specularFactor: 0.5,
          specularColorFactor: [0.9, 0.8, 0.7],
        },
      },
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.specularIntensity).toBeCloseTo(0.5);
    expect(mat.specularColor).toEqual([0.9, 0.8, 0.7]);
  });

  it('stores unknown extensions in material.extensions with a warning', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        MY_custom_extension: { foo: 42 },
      },
    });
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.includes('MY_custom_extension'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'unknown-material-extension',
      path: 'materials[0].extensions.MY_custom_extension',
      extensionName: 'MY_custom_extension',
      materialIndex: 0,
    }));
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect((mat.extensions as Record<string, unknown>)?.['MY_custom_extension']).toEqual({ foo: 42 });
  });

  it('rejects unknown required extensions instead of silently ignoring them', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      extensions: {
        MY_optional_extension: { foo: 42 },
      },
    });
    (gltf as GltfJson & { extensionsRequired: string[] }).extensionsRequired = [
      'KHR_materials_unlit',
      'AAA_required_extension',
    ];

    await expect(gltfToScene(gltf, { buffers })).rejects.toThrow(
      /extensionsRequired includes unsupported extension "AAA_required_extension"/,
    );
    await expect(gltfToScene(gltf, { buffers })).rejects.toBeInstanceOf(GltfImportError);
    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [{
        severity: 'error',
        code: 'unsupported-required-extension',
        path: 'extensionsRequired[1]',
        message: expect.stringContaining('AAA_required_extension'),
      }],
    });
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 4 — Node hierarchy with nested TRS transforms
// ────────────────────────────────────────────────────────────────────────────

describe('node hierarchy with nested TRS', () => {
  it('applies parent translation to child mesh world transform', async () => {
    // Parent: translate +5 on X. Child: no transform. Mesh at origin.
    const posBuf = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [
        { translation: [5, 0, 0], children: [1] },  // node 0: parent
        { mesh: 0 },                                   // node 1: child
      ],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteLength: posBuf.byteLength }],
      buffers: [{ byteLength: posBuf.byteLength }],
    };
    const { scene } = await gltfToScene(gltf, { buffers: new Map([[0, posBuf]]) });
    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;

    // World transform should have a +5 X translation in column 3 (index 12).
    expect(prim.transform).toBeDefined();
    // Column-major: col 3 starts at index 12 → translation [12]=tx, [13]=ty, [14]=tz.
    expect(prim.transform![12]).toBeCloseTo(5, 5);
    expect(prim.transform![13]).toBeCloseTo(0, 5);
    expect(prim.transform![14]).toBeCloseTo(0, 5);
  });

  it('concatenates multiple levels of translation', async () => {
    // Grandparent: +2 Y. Parent: +3 Z. Child has mesh.
    const posBuf = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [
        { translation: [0, 2, 0], children: [1] },  // grandparent
        { translation: [0, 0, 3], children: [2] },  // parent
        { mesh: 0 },                                   // child
      ],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteLength: posBuf.byteLength }],
      buffers: [{ byteLength: posBuf.byteLength }],
    };
    const { scene } = await gltfToScene(gltf, { buffers: new Map([[0, posBuf]]) });
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.transform![12]).toBeCloseTo(0, 5);  // X
    expect(prim.transform![13]).toBeCloseTo(2, 5);  // Y
    expect(prim.transform![14]).toBeCloseTo(3, 5);  // Z
  });

  it('rejects a reachable zero-scale transform at its exact glTF source path', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.nodes![0]!.scale = [1, 0, 1];

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [{
        severity: 'error',
        code: 'non-invertible-node-transform',
        path: 'nodes[0].scale',
      }],
    });
  });

  it('rejects a reachable singular matrix at its exact glTF source path', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.nodes![0]!.matrix = [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 1,
    ];

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [{
        severity: 'error',
        code: 'non-invertible-node-transform',
        path: 'nodes[0].matrix',
      }],
    });
  });

  it('rejects a Float32-noninvertible composed world transform before conversion', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.scenes![0]!.nodes = [0];
    gltf.nodes = [
      { scale: [1e-20, 1, 1], children: [1] },
      { mesh: 0, scale: [1e-20, 1, 1] },
    ];

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [{
        severity: 'error',
        code: 'non-invertible-node-transform',
        path: 'nodes[1]',
      }],
    });
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 5 — Multi-primitive mesh → multiple Scene primitives
// ────────────────────────────────────────────────────────────────────────────

describe('multi-primitive mesh', () => {
  it('produces N ScenePrimitives for N glTF mesh primitives', async () => {
    const posBuf = f32Buffer([...TRIANGLE_POSITIONS, ...TRIANGLE_POSITIONS]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{
        primitives: [
          { attributes: { POSITION: 0 } },
          { attributes: { POSITION: 1 } },
        ],
      }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 36 },
        { buffer: 0, byteOffset: 36, byteLength: 36 },
      ],
      buffers: [{ byteLength: posBuf.byteLength }],
    };
    const { scene } = await gltfToScene(gltf, { buffers: new Map([[0, posBuf]]) });
    expect(scene.primitives).toHaveLength(2);
    expect(scene.primitives[0]!.id).not.toBe(scene.primitives[1]!.id);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 6 — Sparse accessor handling
// ────────────────────────────────────────────────────────────────────────────

describe('sparse accessor', () => {
  it('applies sparse patch to base data', async () => {
    // Base: 4 VEC3 positions of [0,0,0].
    // Sparse: override index 2 with [9,8,7].
    const basePosBuf = f32Buffer([0,0,0, 0,0,0, 0,0,0, 0,0,0]);
    const sparseIdxBuf = u32Buffer([2]);
    const sparseValBuf = f32Buffer([9, 8, 7]);

    const totalBuf = concatBuffers(basePosBuf, sparseIdxBuf, sparseValBuf);

    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0 },
          indices: 1,
        }],
      }],
      accessors: [
        {
          // Sparse accessor with base bufferView
          bufferView: 0,
          componentType: 5126,
          count: 4,
          type: 'VEC3',
          sparse: {
            count: 1,
            indices: { bufferView: 1, componentType: 5125 /* UINT32 */ },
            values: { bufferView: 2 },
          },
        },
        // Dummy indices: [0,1,2] as UINT16 — 1 triangle
        {
          bufferView: 3,
          componentType: 5123 /* UNSIGNED_SHORT */,
          count: 3,
          type: 'SCALAR',
        },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: basePosBuf.byteLength },
        { buffer: 0, byteOffset: basePosBuf.byteLength, byteLength: sparseIdxBuf.byteLength },
        { buffer: 0, byteOffset: basePosBuf.byteLength + sparseIdxBuf.byteLength, byteLength: sparseValBuf.byteLength },
        { buffer: 0, byteOffset: basePosBuf.byteLength + sparseIdxBuf.byteLength + sparseValBuf.byteLength, byteLength: 6 },
      ],
      buffers: [{ byteLength: totalBuf.byteLength + 6 }],
    };

    // Append the index buffer (u16: 0,1,2)
    const finalBuf = concatBuffers(totalBuf, u16Buffer([0, 1, 2]));
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers: new Map([[0, finalBuf]]) });

    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    // Vertex 2 (positions 6,7,8) should have been patched to [9,8,7]
    expect(prim.positions[6]).toBeCloseTo(9);
    expect(prim.positions[7]).toBeCloseTo(8);
    expect(prim.positions[8]).toBeCloseTo(7);
    // Check that a sparse warning was emitted
    expect(warnings.some(w => w.includes('sparse'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'sparse-accessor-applied',
      path: 'accessors[0].sparse',
    }));
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 7 — Non-triangle mode warning + skip
// ────────────────────────────────────────────────────────────────────────────

describe('non-triangle primitive mode', () => {
  it('imports POINTS primitives as fallback-generated meshes and emits a warning', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{
        primitives: [
          { attributes: { POSITION: 0 }, mode: 0 /* POINTS */ },
          { attributes: { POSITION: 0 } /* TRIANGLES by default */ },
        ],
      }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteLength: posBuf.byteLength }],
      buffers: [{ byteLength: posBuf.byteLength }],
    };
    const { scene, warnings } = await gltfToScene(gltf, { buffers: new Map([[0, posBuf]]) });
    expect(scene.primitives).toHaveLength(2);
    expect(scene.primitives[0]?.kind).toBe('mesh');
    const points = scene.primitives[0] as MeshPrimitive;
    expect(points.positions.length).toBeGreaterThan(TRIANGLE_POSITIONS.length);
    expect(warnings.some(w => w.includes('POINTS') && w.includes('fallback-generated mesh'))).toBe(true);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 8 — Draco without a decode hook (GLTF-02: per-primitive resolution;
// deeper hook/fallback coverage lives in gltfCompression.test.ts)
// ────────────────────────────────────────────────────────────────────────────

describe('Draco without a decode hook', () => {
  it('warns per compressed primitive and uses the uncompressed fallback accessors', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    (gltf as GltfJson & { extensionsUsed: string[] }).extensionsUsed = ['KHR_draco_mesh_compression'];
    // The primitive carries the extension; its accessors still have
    // bufferViews → spec fallback geometry.
    gltf.meshes![0]!.primitives[0]!.extensions = {
      KHR_draco_mesh_compression: { bufferView: 0, attributes: { POSITION: 0 } },
    };
    const { diagnostics, scene, warnings } = await gltfToScene(gltf, {
      buffers,
      compressionDecoderPolicy: 'host-only',
    });
    expect(warnings.some(w => w.includes('fully validated uncompressed fallback accessors'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'draco-fallback-accessors-used',
      path: 'meshes[0].primitives[0].extensions.KHR_draco_mesh_compression',
    }));
    // Fallback accessors keep the primitive alive.
    expect(scene.primitives).toHaveLength(1);
  });

  it('emits no compression warning when extensionsUsed declares Draco but no primitive carries it', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    (gltf as GltfJson & { extensionsUsed: string[] }).extensionsUsed = ['KHR_draco_mesh_compression'];
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.includes('KHR_draco_mesh_compression'))).toBe(false);
    expect(scene.primitives).toHaveLength(1);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 9 — Animation + skin + camera warnings
// ────────────────────────────────────────────────────────────────────────────

describe('out-of-scope feature warnings', () => {
  it('preserves earlier material diagnostics when a later animation error rejects the import', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.materials = [{ extensions: { VENDOR_material_magic: { mode: 'hint' } } }];
    gltf.meshes![0]!.primitives[0]!.material = 0;
    gltf.animations = [{
      name: 'walk',
      channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
      samplers: [],
    }];
    const failure = await gltfToScene(gltf, { buffers }).then(
      () => undefined,
      (error: unknown) => error,
    );
    expect(failure).toBeInstanceOf(GltfImportError);
    const diagnostics = (failure as GltfImportError).diagnostics;
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        severity: 'warning',
        code: 'unknown-material-extension',
        path: 'materials[0].extensions.VENDOR_material_magic',
      }),
      expect.objectContaining({
        severity: 'error',
        code: 'missing-animation-sampler',
        path: 'animations[0].samplers[0]',
      }),
    ]));
    expect(diagnostics.filter((diagnostic) =>
      diagnostic.code === 'missing-animation-sampler')).toHaveLength(1);
  });

  it('rejects a skin binding whose primitive omits skin streams', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.skins = [{ joints: [0] }];
    gltf.nodes![0] = { ...gltf.nodes![0]!, skin: 0 };
    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'incomplete-skin-attributes',
        path: 'meshes[0].primitives[0].attributes.JOINTS_0',
      })],
    });
  });

  it('surfaces ignored cameras as structured import diagnostics', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.cameras = [{
      type: 'perspective',
      name: 'Hero Camera',
      perspective: { yfov: 0.7, znear: 0.1, zfar: 250, aspectRatio: 1.5 },
    }];
    gltf.nodes![0] = {
      ...gltf.nodes![0]!,
      name: 'Camera Rig',
      camera: 0,
      translation: [1, 2, 3],
    };
    const { warnings, diagnostics, cameras } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.includes('reported on result.cameras'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        severity: 'warning',
        code: 'ignored-camera',
        path: 'cameras[0]',
      }),
    ]));
    expect(cameras).toHaveLength(1);
    expect(cameras[0]).toEqual(expect.objectContaining({
      cameraIndex: 0,
      nodeIndex: 0,
      path: 'cameras[0]',
      nodePath: 'nodes[0]',
      type: 'perspective',
      name: 'Hero Camera',
      nodeName: 'Camera Rig',
      perspective: {
        yfov: 0.7,
        znear: 0.1,
        zfar: 250,
        aspectRatio: 1.5,
      },
    }));
    expect(cameras[0]!.worldMatrix[12]).toBeCloseTo(1);
    expect(cameras[0]!.worldMatrix[13]).toBeCloseTo(2);
    expect(cameras[0]!.worldMatrix[14]).toBeCloseTo(3);
  });

  it.each([
    {
      label: 'missing camera type',
      camera: {},
      path: 'cameras[0].type',
    },
    {
      label: 'unknown camera type',
      camera: { type: 'panoramic' },
      path: 'cameras[0].type',
    },
    {
      label: 'missing perspective object',
      camera: { type: 'perspective' },
      path: 'cameras[0].perspective',
    },
    {
      label: 'missing perspective yfov',
      camera: { type: 'perspective', perspective: { znear: 0.1 } },
      path: 'cameras[0].perspective.yfov',
    },
    {
      label: 'non-positive perspective yfov',
      camera: { type: 'perspective', perspective: { yfov: 0, znear: 0.1 } },
      path: 'cameras[0].perspective.yfov',
    },
    {
      label: 'non-positive perspective znear',
      camera: { type: 'perspective', perspective: { yfov: 0.7, znear: 0 } },
      path: 'cameras[0].perspective.znear',
    },
    {
      label: 'perspective zfar not beyond znear',
      camera: { type: 'perspective', perspective: { yfov: 0.7, znear: 1, zfar: 1 } },
      path: 'cameras[0].perspective.zfar',
    },
    {
      label: 'non-positive perspective aspect ratio',
      camera: { type: 'perspective', perspective: { yfov: 0.7, znear: 0.1, aspectRatio: -1 } },
      path: 'cameras[0].perspective.aspectRatio',
    },
    {
      label: 'non-finite perspective field',
      camera: { type: 'perspective', perspective: { yfov: Number.NaN, znear: 0.1 } },
      path: 'cameras[0].perspective.yfov',
    },
    {
      label: 'missing orthographic object',
      camera: { type: 'orthographic' },
      path: 'cameras[0].orthographic',
    },
    {
      label: 'zero orthographic magnification',
      camera: {
        type: 'orthographic',
        orthographic: { xmag: 0, ymag: 1, znear: 0, zfar: 10 },
      },
      path: 'cameras[0].orthographic.xmag',
    },
    {
      label: 'negative orthographic znear',
      camera: {
        type: 'orthographic',
        orthographic: { xmag: 1, ymag: 1, znear: -1, zfar: 10 },
      },
      path: 'cameras[0].orthographic.znear',
    },
    {
      label: 'orthographic zfar not beyond znear',
      camera: {
        type: 'orthographic',
        orthographic: { xmag: 1, ymag: 1, znear: 2, zfar: 1 },
      },
      path: 'cameras[0].orthographic.zfar',
    },
    {
      label: 'projection object inconsistent with type',
      camera: {
        type: 'perspective',
        perspective: { yfov: 0.7, znear: 0.1 },
        orthographic: { xmag: 1, ymag: 1, znear: 0, zfar: 10 },
      },
      path: 'cameras[0].orthographic',
    },
  ])('rejects reachable $label with a structured camera diagnostic', async ({ camera, path }) => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.cameras = [camera];
    gltf.nodes![0] = { ...gltf.nodes![0]!, camera: 0 };

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: expect.arrayContaining([
        expect.objectContaining({
          severity: 'error',
          code: 'invalid-camera',
          path,
        }),
      ]),
    });
  });

  it('does not reject malformed cameras outside the selected scene', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.cameras = [{ type: 'perspective' }];
    gltf.nodes!.push({ camera: 0 });

    const result = await gltfToScene(gltf, { buffers });

    expect(result.cameras).toEqual([]);
    expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'invalid-camera')).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 10 — Accessor indices: Uint16 and Uint32
// ────────────────────────────────────────────────────────────────────────────

describe('index buffer types', () => {
  it('reads Uint16 indices correctly', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const idxBuf = u16Buffer([0, 1, 2]);
    const totalBuf = concatBuffers(posBuf, idxBuf);

    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5123 /* UNSIGNED_SHORT */, count: 3, type: 'SCALAR' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength, byteLength: idxBuf.byteLength },
      ],
      buffers: [{ byteLength: totalBuf.byteLength }],
    };
    const { scene } = await gltfToScene(gltf, { buffers: new Map([[0, totalBuf]]) });
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.indices).toBeInstanceOf(Uint32Array);
    expect(Array.from(prim.indices!)).toEqual([0, 1, 2]);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 11 — normalTexture + occlusionTexture mapping (no real images, just null handles)
// ────────────────────────────────────────────────────────────────────────────

describe('texture info mapping', () => {
  it('rejects missing texture image indices with structured diagnostics', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({ normalTexCoord: 0, includeUv0: true });
    gltf.textures![0]!.source = 99;

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: expect.arrayContaining([
        expect.objectContaining({
          severity: 'warning',
        code: 'image-not-found',
        path: 'textures[0].source',
        textureIndex: 0,
        imageIndex: 99,
        }),
        expect.objectContaining({
        severity: 'error',
        code: 'material-texture-unresolved',
        path: 'materials[0].normalTexture',
        materialIndex: 0,
        textureIndex: 0,
        }),
      ]),
    });
  });

  it('rejects material texture infos that reference missing texture indices', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({ normalTexCoord: 0, includeUv0: true });
    gltf.materials![0]!.normalTexture = { index: 99 };

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'material-texture-not-found',
        path: 'materials[0].normalTexture.index',
        materialIndex: 0,
        textureIndex: 99,
      })],
    });
  });

  it('rejects missing embedded image bufferViews with structured diagnostics', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({ normalTexCoord: 0, includeUv0: true });
    gltf.images![0]!.bufferView = 99;

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: expect.arrayContaining([
        expect.objectContaining({
        severity: 'warning',
        code: 'image-buffer-view-not-found',
        path: 'images[0].bufferView',
        textureIndex: 0,
        imageIndex: 0,
        bufferViewIndex: 99,
        }),
        expect.objectContaining({ severity: 'error', code: 'material-texture-unresolved' }),
      ]),
    });
  });

  it('rejects unavailable embedded image buffers with structured diagnostics', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({ normalTexCoord: 0, includeUv0: true });
    const imageBufferView = gltf.images![0]!.bufferView!;
    gltf.bufferViews![imageBufferView] = { buffer: 99, byteOffset: 0, byteLength: 4 };

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: expect.arrayContaining([
        expect.objectContaining({
        severity: 'warning',
        code: 'image-buffer-unavailable',
        path: `bufferViews[${imageBufferView}].buffer`,
        textureIndex: 0,
        imageIndex: 0,
        bufferViewIndex: imageBufferView,
        bufferIndex: 99,
        }),
        expect.objectContaining({ severity: 'error', code: 'material-texture-unresolved' }),
      ]),
    });
  });

  it('rejects images without bufferView or URI with structured diagnostics', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({ normalTexCoord: 0, includeUv0: true });
    gltf.images![0] = { name: 'empty-image' };

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: expect.arrayContaining([
        expect.objectContaining({
        severity: 'warning',
        code: 'image-source-missing',
        path: 'images[0]',
        textureIndex: 0,
        imageIndex: 0,
        }),
        expect.objectContaining({ severity: 'error', code: 'material-texture-unresolved' }),
      ]),
    });
  });

  it('maps normalTexture scale', async () => {
    const { gltf, buffers } = makeNormalMappedTriangleGltf();
    gltf.materials![0]!.normalTexture = { index: 0, scale: 0.5 };
    const { scene } = await gltfToScene(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-normal' }),
    });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.normalMap).toBeDefined();
    expect(mat.normalScale).toBeCloseTo(0.5);
  });

  it('maps occlusionTexture strength to aoMapIntensity', async () => {
    const { gltf, buffers } = makeNormalMappedTriangleGltf();
    gltf.materials![0]!.occlusionTexture = { index: 0, strength: 0.75 };
    const { scene } = await gltfToScene(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-ao' }),
    });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.aoMap).toBeDefined();
    expect(mat.aoMapIntensity).toBeCloseTo(0.75);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 13 — Skins → SkinnedMeshPrimitive
//
// Fixture: 2-joint skin on a 1-triangle mesh (3 vertices, 4 influences/vertex).
// Two sub-tests:
//   A. JOINTS_0 as UNSIGNED_BYTE (5121) — common for ≤ 255 joints
//   B. JOINTS_0 as UNSIGNED_SHORT (5123) — required when joint count > 255
// ────────────────────────────────────────────────────────────────────────────

describe('skin → SkinnedMeshPrimitive', () => {
  /**
   * Build a minimal skinned glTF in-memory.
   * - 1 mesh, 1 primitive (3-vertex triangle)
   * - 2-joint skin; joint nodes at world positions (0,0,0) and (1,0,0)
   * - inverseBindMatrices: 2 × 16 floats (identity for both joints)
   * - JOINTS_0: 3 vertices × 4 u8 indices = [0,1,0,0, 0,1,0,0, 0,1,0,0]
   * - WEIGHTS_0: 3 vertices × 4 floats = [0.5,0.5,0,0, …]
   *
   * @param jointsComponentType 5121 = UNSIGNED_BYTE, 5123 = UNSIGNED_SHORT
   */
  function makeSkinnedGltf(
    jointsComponentType: 5121 | 5123,
    meshTranslation: [number, number, number] = [0, 0, 0],
  ): {
    gltf: GltfJson;
    buffers: Map<number, ArrayBuffer>;
  } {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);

    // inverseBindMatrices: 2 identity 4×4 matrices
    const ibm = new Array(32).fill(0);
    // joint 0 identity
    ibm[0] = 1; ibm[5] = 1; ibm[10] = 1; ibm[15] = 1;
    // joint 1 identity
    ibm[16] = 1; ibm[21] = 1; ibm[26] = 1; ibm[31] = 1;
    const ibmBuf = f32Buffer(ibm);

    // WEIGHTS_0: 3 verts × 4 weights [0.5, 0.5, 0, 0] each
    const weightsBuf = f32Buffer([
      0.5, 0.5, 0, 0,
      0.5, 0.5, 0, 0,
      0.5, 0.5, 0, 0,
    ]);

    // JOINTS_0: 3 verts × 4 indices [0, 1, 0, 0] each
    // encoded as u8 or u16
    const jointValues = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0];
    const jointsBuf = jointsComponentType === 5121
      ? u8Buffer(jointValues)
      : u16Buffer(jointValues);

    const totalBuf = concatBuffers(posBuf, ibmBuf, weightsBuf, jointsBuf);

    // Buffer view offsets
    const posOff = 0;
    const ibmOff = posBuf.byteLength;
    const wgtsOff = ibmOff + ibmBuf.byteLength;
    const jntsOff = wgtsOff + weightsBuf.byteLength;

    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [
        { mesh: 0, skin: 0, children: [1, 2], translation: meshTranslation }, // skinned mesh node
        { translation: [0, 0, 0] },                         // joint 0
        { translation: [1, 0, 0] },                         // joint 1
      ],
      meshes: [{
        primitives: [{
          attributes: {
            POSITION: 0,
            JOINTS_0: 3,
            WEIGHTS_0: 2,
          },
        }],
      }],
      skins: [{
        joints: [1, 2],
        inverseBindMatrices: 1,
      }],
      accessors: [
        // 0: POSITION VEC3 float
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        // 1: inverseBindMatrices MAT4 float (2 joints)
        { bufferView: 1, componentType: 5126, count: 2, type: 'MAT4' },
        // 2: WEIGHTS_0 VEC4 float
        { bufferView: 2, componentType: 5126, count: 3, type: 'VEC4' },
        // 3: JOINTS_0 VEC4 u8 or u16
        {
          bufferView: 3,
          componentType: jointsComponentType,
          count: 3,
          type: 'VEC4',
        },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: posOff, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: ibmOff, byteLength: ibmBuf.byteLength },
        { buffer: 0, byteOffset: wgtsOff, byteLength: weightsBuf.byteLength },
        { buffer: 0, byteOffset: jntsOff, byteLength: jointsBuf.byteLength },
      ],
      buffers: [{ byteLength: totalBuf.byteLength }],
    };

    return { gltf, buffers: new Map([[0, totalBuf]]) };
  }

  function makeSkinnedSecondaryInfluencesGltf(): {
    gltf: GltfJson;
    buffers: Map<number, ArrayBuffer>;
  } {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const ibm = new Array(96).fill(0);
    for (let joint = 0; joint < 6; joint++) {
      const base = joint * 16;
      ibm[base + 0] = 1;
      ibm[base + 5] = 1;
      ibm[base + 10] = 1;
      ibm[base + 15] = 1;
    }
    const ibmBuf = f32Buffer(ibm);
    const weights0 = [
      0.35, 0.25, 0.15, 0.05,
      0.35, 0.25, 0.15, 0.05,
      0.35, 0.25, 0.15, 0.05,
    ];
    const weights1 = [
      0.12, 0.04, 0.03, 0.01,
      0.12, 0.04, 0.03, 0.01,
      0.12, 0.04, 0.03, 0.01,
    ];
    const joints0 = [
      0, 1, 2, 3,
      0, 1, 2, 3,
      0, 1, 2, 3,
    ];
    const joints1 = [
      4, 5, 2, 1,
      4, 5, 2, 1,
      4, 5, 2, 1,
    ];
    const weights0Buf = f32Buffer(weights0);
    const joints0Buf = u8Buffer(joints0);
    const weights1Buf = f32Buffer(weights1);
    const joints1Buf = u8Buffer(joints1);
    const totalBuf = concatBuffers(posBuf, ibmBuf, weights0Buf, joints0Buf, weights1Buf, joints1Buf);
    const posOff = 0;
    const ibmOff = posBuf.byteLength;
    const weights0Off = ibmOff + ibmBuf.byteLength;
    const joints0Off = weights0Off + weights0Buf.byteLength;
    const weights1Off = joints0Off + joints0Buf.byteLength;
    const joints1Off = weights1Off + weights1Buf.byteLength;
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [
        { mesh: 0, skin: 0, children: [1, 2, 3, 4, 5, 6] },
        {},
        {},
        {},
        {},
        {},
        {},
      ],
      meshes: [{
        primitives: [{
          attributes: {
            POSITION: 0,
            WEIGHTS_0: 2,
            JOINTS_0: 3,
            WEIGHTS_1: 4,
            JOINTS_1: 5,
          },
        }],
      }],
      skins: [{ joints: [1, 2, 3, 4, 5, 6], inverseBindMatrices: 1 }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 6, type: 'MAT4' },
        { bufferView: 2, componentType: 5126, count: 3, type: 'VEC4' },
        { bufferView: 3, componentType: 5121, count: 3, type: 'VEC4' },
        { bufferView: 4, componentType: 5126, count: 3, type: 'VEC4' },
        { bufferView: 5, componentType: 5121, count: 3, type: 'VEC4' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: posOff, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: ibmOff, byteLength: ibmBuf.byteLength },
        { buffer: 0, byteOffset: weights0Off, byteLength: weights0Buf.byteLength },
        { buffer: 0, byteOffset: joints0Off, byteLength: joints0Buf.byteLength },
        { buffer: 0, byteOffset: weights1Off, byteLength: weights1Buf.byteLength },
        { buffer: 0, byteOffset: joints1Off, byteLength: joints1Buf.byteLength },
      ],
      buffers: [{ byteLength: totalBuf.byteLength }],
    };
    return { gltf, buffers: new Map([[0, totalBuf]]) };
  }

  it('emits kind:skinned-mesh when node has a skin (JOINTS_0 u8)', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121);
    const { scene } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0]!;
    expect(prim.kind).toBe('skinned-mesh');
  });

  it('imports a Float32-invertible tiny-scale skinned mesh transform', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121);
    gltf.nodes![0] = {
      ...gltf.nodes![0]!,
      scale: [1e-5, 1e-5, 1e-5],
    };

    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;

    expect(prim.kind).toBe('skinned-mesh');
    expect(prim.transform?.[0]).toBeCloseTo(1e-5, 10);
    expect(Array.from(prim.bones).every(Number.isFinite)).toBe(true);
  });

  it('skinIndices decoded correctly from UNSIGNED_BYTE JOINTS_0', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121);
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    // Each vertex: [0, 1, 0, 0]
    expect(Array.from(prim.skinIndices.slice(0, 4))).toEqual([0, 1, 0, 0]);
    expect(Array.from(prim.skinIndices.slice(4, 8))).toEqual([0, 1, 0, 0]);
    expect(Array.from(prim.skinIndices.slice(8, 12))).toEqual([0, 1, 0, 0]);
  });

  it('applies sparse patches to JOINTS_0 accessors', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121);
    const baseBuffer = buffers.get(0)!;
    const sparseIndices = u16Buffer([1]);
    const sparseValues = u8Buffer([1, 0, 0, 0]);
    const patchedBuffer = concatBuffers(baseBuffer, sparseIndices, sparseValues);
    const sparseIndicesView = gltf.bufferViews!.length;
    const sparseValuesView = sparseIndicesView + 1;
    gltf.bufferViews!.push(
      { buffer: 0, byteOffset: baseBuffer.byteLength, byteLength: sparseIndices.byteLength },
      {
        buffer: 0,
        byteOffset: baseBuffer.byteLength + sparseIndices.byteLength,
        byteLength: sparseValues.byteLength,
      },
    );
    gltf.buffers![0] = { byteLength: patchedBuffer.byteLength };
    gltf.accessors![3] = {
      ...gltf.accessors![3]!,
      sparse: {
        count: 1,
        indices: { bufferView: sparseIndicesView, componentType: 5123 },
        values: { bufferView: sparseValuesView },
      },
    };
    buffers.set(0, patchedBuffer);

    const { scene, diagnostics } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;

    expect(Array.from(prim.skinIndices.slice(0, 4))).toEqual([0, 1, 0, 0]);
    expect(Array.from(prim.skinIndices.slice(4, 8))).toEqual([1, 0, 0, 0]);
    expect(Array.from(prim.skinIndices.slice(8, 12))).toEqual([0, 1, 0, 0]);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'sparse-accessor-applied',
      path: 'accessors[3].sparse',
      accessorIndex: 3,
    }));
  });

  it('skinIndices decoded correctly from UNSIGNED_SHORT JOINTS_0', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5123);
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(prim.kind).toBe('skinned-mesh');
    expect(Array.from(prim.skinIndices.slice(0, 4))).toEqual([0, 1, 0, 0]);
  });

  it('skinWeights decoded correctly', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121);
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(prim.skinWeights[0]).toBeCloseTo(0.5);
    expect(prim.skinWeights[1]).toBeCloseTo(0.5);
    expect(prim.skinWeights[2]).toBeCloseTo(0);
    expect(prim.skinWeights[3]).toBeCloseTo(0);
  });

  it('normalizes a non-unit single WEIGHTS_0 set per vertex', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121);
    const weightsView = gltf.bufferViews![2]!;
    const weights = new Float32Array(
      buffers.get(weightsView.buffer)!,
      weightsView.byteOffset ?? 0,
      12,
    );
    weights.set([
      2, 1, 1, 0,
      2, 1, 1, 0,
      2, 1, 1, 0,
    ]);

    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(Array.from(prim.skinWeights.slice(0, 4))).toEqual([
      expect.closeTo(0.5, 6),
      expect.closeTo(0.25, 6),
      expect.closeTo(0.25, 6),
      expect.closeTo(0, 6),
    ]);
    expect(
      prim.skinWeights[0]! +
      prim.skinWeights[1]! +
      prim.skinWeights[2]! +
      prim.skinWeights[3]!,
    ).toBeCloseTo(1, 6);
  });

  it('rejects a vertex whose skin influence weight sum is zero', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121);
    const weightsView = gltf.bufferViews![2]!;
    const weights = new Float32Array(
      buffers.get(weightsView.buffer)!,
      weightsView.byteOffset ?? 0,
      12,
    );
    weights.fill(0, 0, 4);

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [
        expect.objectContaining({
          severity: 'error',
          code: 'unreadable-skin-weights',
          path: 'meshes[0].primitives[0].attributes.WEIGHTS_0',
        }),
      ],
    });
  });

  it('preserves every positive secondary skin influence without truncation', async () => {
    const { gltf, buffers } = makeSkinnedSecondaryInfluencesGltf();
    const { scene, diagnostics } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;

    expect(prim.kind).toBe('skinned-mesh');
    expect(prim.skinInfluencesPerVertex).toBe(6);
    expect(Array.from(prim.skinIndices.slice(0, 6))).toEqual([0, 1, 2, 4, 3, 5]);
    expect(Array.from(prim.skinWeights.slice(0, 6))).toEqual(expect.arrayContaining([
      expect.closeTo(0.35, 6),
      expect.closeTo(0.26, 6),
      expect.closeTo(0.18, 6),
      expect.closeTo(0.12, 6),
      expect.closeTo(0.05, 6),
      expect.closeTo(0.04, 6),
    ]));
    expect(diagnostics.some((diagnostic) => diagnostic.code === 'collapsed-skin-influence-sets')).toBe(false);

    const report = analyzeGltfAsset(gltf);
    expect(report.primitives.hasCollapsedSkinInfluenceSets).toBe(false);
    expect(report.primitives.issuePaths.collapsedSkinInfluenceSets).toBeUndefined();
  });

  it('boneInverses are the identity matrices for both joints', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121);
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    // boneInverses: 2 × 16 column-major f32. Identity has 1s at [0,5,10,15].
    expect(prim.boneInverses[0]).toBeCloseTo(1); // joint 0 [0,0]
    expect(prim.boneInverses[5]).toBeCloseTo(1); // joint 0 [1,1]
    expect(prim.boneInverses[10]).toBeCloseTo(1); // joint 0 [2,2]
    expect(prim.boneInverses[15]).toBeCloseTo(1); // joint 0 [3,3]
    expect(prim.boneInverses[16]).toBeCloseTo(1); // joint 1 [0,0]
    expect(prim.boneInverses[31]).toBeCloseTo(1); // joint 1 [3,3]
  });

  it('bones contains world transforms of joint nodes', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121);
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    // joint 0 is at [0,0,0] → identity → bones[0..15] column 3 = [0,0,0,1]
    expect(prim.bones[12]).toBeCloseTo(0); // translation x
    expect(prim.bones[13]).toBeCloseTo(0); // translation y
    expect(prim.bones[14]).toBeCloseTo(0); // translation z
    // joint 1 is at [1,0,0] → bones[16..31] column 3 translation x = 1
    expect(prim.bones[16 + 12]).toBeCloseTo(1); // translation x
    expect(prim.bones[16 + 13]).toBeCloseTo(0); // translation y
  });

  it('converts glTF joint worlds to mesh-local bones so skinned-node transform applies once', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121, [5, 0, 0]);
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;

    expect(prim.bindMatrix).toBeUndefined();
    expect(prim.bindMatrixInverse).toBeUndefined();
    expect(prim.transform?.[12]).toBeCloseTo(5);
    expect(prim.bones[12]).toBeCloseTo(0);
    expect(prim.bones[16 + 12]).toBeCloseTo(1);

    const solved = solveSkin(prim);
    expect(solved.positions[0]).toBeCloseTo(0.5);
    const world = transformPoint(prim.transform, solved.positions[0]!, solved.positions[1]!, solved.positions[2]!);
    expect(world[0]).toBeCloseTo(5.5);
    expect(world[1]).toBeCloseTo(0);
    expect(world[2]).toBeCloseTo(0);
  });

  it('keeps skinned output mesh-local under translated and scaled skinned nodes', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121, [6, 9, 12]);
    gltf.nodes![0]!.scale = [2, 3, 4];
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;

    expect(prim.bindMatrix).toBeUndefined();
    expect(prim.bindMatrixInverse).toBeUndefined();
    expect(prim.transform?.[0]).toBeCloseTo(2);
    expect(prim.transform?.[5]).toBeCloseTo(3);
    expect(prim.transform?.[10]).toBeCloseTo(4);
    expect(prim.transform?.[12]).toBeCloseTo(6);
    expect(prim.transform?.[13]).toBeCloseTo(9);
    expect(prim.transform?.[14]).toBeCloseTo(12);
    expect(prim.bones[12]).toBeCloseTo(0);
    expect(prim.bones[16 + 12]).toBeCloseTo(1);

    const solved = solveSkin(prim);
    expect(solved.positions[0]).toBeCloseTo(0.5);
    const world = transformPoint(prim.transform, solved.positions[0]!, solved.positions[1]!, solved.positions[2]!);
    expect(world[0]).toBeCloseTo(7);
    expect(world[1]).toBeCloseTo(9);
    expect(world[2]).toBeCloseTo(12);
  });

  it('skinIndices is Uint32Array', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121);
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(prim.skinIndices).toBeInstanceOf(Uint32Array);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 14 — KHR_lights_punctual → SceneEmitter[]
//
// Fixture: one point light, one spot light (with rotated parent node), one
// directional light. Asserts full field mapping including world-transformed
// position and direction.
// ────────────────────────────────────────────────────────────────────────────

describe('KHR_lights_punctual → SceneEmitter[]', () => {
  /**
   * Build a minimal glTF with KHR_lights_punctual containing:
   *   light 0 — point, intensity 100 cd, range 10, color [1,0.8,0.6]
   *   light 1 — spot, intensity 200 cd, innerConeAngle 0.2, outerConeAngle 0.5
   *             attached to a node translated to [2, 3, 4] and rotated 90° around Y
   *   light 2 — directional, intensity 50 lx, color [0.9, 0.9, 1]
   */
  function makeLightsGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);

    // Rotation 90° around Y: quaternion = [0, sin(π/4), 0, cos(π/4)]
    const s = Math.sin(Math.PI / 4);
    const c = Math.cos(Math.PI / 4);

    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0, 1, 2, 3] }],
      scene: 0,
      nodes: [
        // node 0: mesh (just so there's geometry)
        { mesh: 0 },
        // node 1: point light at [5, 0, 0]
        {
          translation: [5, 0, 0],
          extensions: { KHR_lights_punctual: { light: 0 } },
        },
        // node 2: spot light at [2, 3, 4], rotated 90° around Y
        //   local -Z after 90° Y rotation = [-1, 0, 0] in world space
        {
          translation: [2, 3, 4],
          rotation: [0, s, 0, c],
          extensions: { KHR_lights_punctual: { light: 1 } },
        },
        // node 3: directional light (no translation; direction from rotation)
        {
          extensions: { KHR_lights_punctual: { light: 2 } },
        },
      ],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteLength: posBuf.byteLength }],
      buffers: [{ byteLength: posBuf.byteLength }],
      extensionsUsed: ['KHR_lights_punctual'],
      extensions: {
        KHR_lights_punctual: {
          lights: [
            // light 0: point
            {
              type: 'point',
              color: [1, 0.8, 0.6],
              intensity: 100,
              range: 10,
            },
            // light 1: spot
            {
              type: 'spot',
              intensity: 200,
              spot: { innerConeAngle: 0.2, outerConeAngle: 0.5 },
            },
            // light 2: directional
            {
              type: 'directional',
              color: [0.9, 0.9, 1],
              intensity: 50,
            },
          ],
        },
      },
    };
    return { gltf, buffers: new Map([[0, posBuf]]) };
  }

  it('produces three emitters', async () => {
    const { gltf, buffers } = makeLightsGltf();
    const { scene } = await gltfToScene(gltf, { buffers });
    expect(scene.emitters).toHaveLength(3);
  });

  it('imports reachable node light payloads when extensionsUsed omits the redundant token', async () => {
    const { gltf, buffers } = makeLightsGltf();
    delete gltf.extensionsUsed;

    const { scene, diagnostics } = await gltfToScene(gltf, { buffers });

    expect(scene.emitters).toHaveLength(3);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'undeclared-punctual-light-extension',
      path: 'extensionsUsed',
    }));
  });

  it('validates malformed root light payloads only when a selected-scene node references them', async () => {
    const { gltf, buffers } = makeLightsGltf();
    gltf.extensions = { KHR_lights_punctual: {} };
    gltf.scenes = [{ nodes: [0] }, { nodes: [1] }];

    const selectedResult = await gltfToScene(gltf, { buffers });
    expect(selectedResult.scene.emitters).toEqual([]);

    gltf.scene = 1;
    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'missing-punctual-light',
        path: 'extensions.KHR_lights_punctual',
      })],
    });
  });

  it('point emitter: kind, position, color, intensity, distance, decay', async () => {
    const { gltf, buffers } = makeLightsGltf();
    const { scene } = await gltfToScene(gltf, { buffers });
    const point = scene.emitters.find(e => e.kind === 'point') as PointEmitter;
    expect(point).toBeDefined();
    expect(point.position[0]).toBeCloseTo(5);
    expect(point.position[1]).toBeCloseTo(0);
    expect(point.position[2]).toBeCloseTo(0);
    expect(point.color).toEqual([1, 0.8, 0.6]);
    expect(point.intensity).toBeCloseTo(100);
    expect(point.distance).toBeCloseTo(10);
    expect(point.decay).toBe(2);
  });

  it('imports a point emitter without imposing a world-scale direction threshold', async () => {
    const { gltf, buffers } = makeLightsGltf();
    gltf.nodes![1]!.scale = [1e-20, 1e-20, 1e-20];

    const { scene } = await gltfToScene(gltf, { buffers });

    const point = scene.emitters.find(e => e.kind === 'point') as PointEmitter;
    expect(point.position).toEqual([5, 0, 0]);
  });

  it('spot emitter: kind, position, angle, penumbra, intensity', async () => {
    const { gltf, buffers } = makeLightsGltf();
    const { scene } = await gltfToScene(gltf, { buffers });
    const spot = scene.emitters.find(e => e.kind === 'spot') as SpotEmitter;
    expect(spot).toBeDefined();
    expect(spot.position[0]).toBeCloseTo(2);
    expect(spot.position[1]).toBeCloseTo(3);
    expect(spot.position[2]).toBeCloseTo(4);
    // angle = outerConeAngle
    expect(spot.angle).toBeCloseTo(0.5);
    // penumbra = 1 - inner/outer = 1 - 0.2/0.5 = 0.6
    expect(spot.penumbra).toBeCloseTo(0.6, 4);
    expect(spot.intensity).toBeCloseTo(200);
    expect(spot.decay).toBe(2);
  });

  it('normalizes a spot direction from a finite subnormal-scale node basis', async () => {
    const { gltf, buffers } = makeLightsGltf();
    gltf.nodes![2]!.scale = [1e-20, 1e-20, 1e-20];

    const { scene } = await gltfToScene(gltf, { buffers });

    const spot = scene.emitters.find(e => e.kind === 'spot') as SpotEmitter;
    expect(spot.direction[0]).toBeCloseTo(-1, 4);
    expect(spot.direction[1]).toBeCloseTo(0, 4);
    expect(spot.direction[2]).toBeCloseTo(0, 4);
  });

  it('spot emitter: direction is world-space forward after 90° Y rotation', async () => {
    const { gltf, buffers } = makeLightsGltf();
    const { scene } = await gltfToScene(gltf, { buffers });
    const spot = scene.emitters.find(e => e.kind === 'spot') as SpotEmitter;
    // Node rotated 90° around Y: local -Z maps to [-1, 0, 0] in world space
    expect(spot.direction[0]).toBeCloseTo(-1, 4);
    expect(spot.direction[1]).toBeCloseTo(0, 4);
    expect(spot.direction[2]).toBeCloseTo(0, 4);
  });

  it('directional emitter: kind, color, intensity, direction', async () => {
    const { gltf, buffers } = makeLightsGltf();
    const { scene } = await gltfToScene(gltf, { buffers });
    const dir = scene.emitters.find(e => e.kind === 'directional') as DirectionalEmitter;
    expect(dir).toBeDefined();
    expect(dir.color[0]).toBeCloseTo(0.9);
    expect(dir.color[1]).toBeCloseTo(0.9);
    expect(dir.color[2]).toBeCloseTo(1);
    expect(dir.intensity).toBeCloseTo(50);
    // No rotation on node 3 → direction should point along +Z (default -Z local → +Z world,
    // then negated for "AT the light" convention → [0, 0, -1]).
    // Actually: local -Z = world -Z for identity node → dirX/Y/Z = (0,0,-1) → after negation = (0,0,1)
    // Wait: world Z column = [0,0,1] for identity. lzx = -(0)=0, lzy=-(0)=0, lzz=-(1)=-1.
    // direction = [-lzx, -lzy, -lzz] = [0, 0, 1].
    expect(dir.direction[2]).toBeCloseTo(1, 4);
  });

  it('does not warn about KHR_lights_punctual being unsupported', async () => {
    const { gltf, buffers } = makeLightsGltf();
    const { warnings } = await gltfToScene(gltf, { buffers });
    // The old "KHR_lights_punctual is present but NOT imported" warning must be gone.
    const unsupportedWarn = warnings.find(
      w => w.includes('KHR_lights_punctual') && w.includes('NOT imported'),
    );
    expect(unsupportedWarn).toBeUndefined();
  });

  it('skips unsupported punctual light types with a structured warning', async () => {
    const { gltf, buffers } = makeLightsGltf();
    const punctual = gltf.extensions!.KHR_lights_punctual as { lights: Array<{ type: string; name?: string }> };
    punctual.lights[2] = {
      type: 'tube',
      name: 'bad tube',
    };

    const result = await gltfToScene(gltf, { buffers });

    expect(result.scene.emitters).toHaveLength(2);
    expect(result.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        severity: 'warning',
        code: 'unsupported-punctual-light-type',
        path: 'extensions.KHR_lights_punctual.lights[2].type',
      }),
    ]));
  });

  it.each([
    ['missing light field', {}],
    ['non-integer light field', { light: 1.5 }],
    ['out-of-range light field', { light: 99 }],
  ])('rejects a malformed punctual light reference: %s', async (_label, payload) => {
    const { gltf, buffers } = makeLightsGltf();
    gltf.nodes![1]!.extensions = { KHR_lights_punctual: payload };

    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [expect.objectContaining({
        severity: 'error',
        code: 'missing-punctual-light',
        path: 'nodes[1].extensions.KHR_lights_punctual.light',
      })],
    });
  });
});
