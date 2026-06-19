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
import { solveSkin } from '@vitrum/core';
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

describe('minimal triangle', () => {
  it('produces one MeshPrimitive with correct positions', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    const { scene } = await gltfToScene(gltf, { buffers });

    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.kind).toBe('mesh');
    expect(Array.from(prim.positions)).toEqual(TRIANGLE_POSITIONS);
  });

  it('generates flat normals when NORMAL is absent', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.normals).toBeInstanceOf(Float32Array);
    // Flat normal for this triangle is [0,0,1] for each of 3 vertices
    expect(prim.normals[2]).toBeCloseTo(1, 5); // vertex 0, z
    expect(prim.normals[5]).toBeCloseTo(1, 5); // vertex 1, z
    expect(prim.normals[8]).toBeCloseTo(1, 5); // vertex 2, z
    expect(warnings.some(w => w.includes('flat normals'))).toBe(true);
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

  it('drops a texCoord 2 tangent-space map when the primitive has no TEXCOORD_2 accessor', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({
      normalTexCoord: 2,
      includeUv0: true,
    });
    const { scene, diagnostics, warnings } = await gltfToScene(gltf, { buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.tangents).toBeUndefined();
    expect(prim.material.normalMap).toBeUndefined();
    expect(warnings.some((w) => w.includes('normalMap') && w.includes('TEXCOORD_2'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'ignored-material-texcoord',
      path: 'meshes[0].primitives[0].material',
    }));
  });

  it('remaps one high material UV set into core uv1 when the accessor exists', async () => {
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

    const { scene, diagnostics, warnings } = await gltfToScene(fixture.gltf, { buffers: fixture.buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect((prim.material.normalMap as TextureRef | undefined)?.texCoord).toBe(1);
    expect(Array.from(prim.uv1 ?? [])).toEqual([
      0.25, 0.25,
      0.75, 0.25,
      0.25, 0.75,
    ]);
    expect(prim.tangents).toBeInstanceOf(Float32Array);
    expect(warnings.some((w) => w.includes('sampled with the wrong UV channel'))).toBe(false);
    expect(diagnostics.some((d) => d.code === 'ignored-material-texcoord')).toBe(false);
  });

  it('drops high-UV maps instead of remapping when texCoord 1 is also material-visible', async () => {
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
    fixture.gltf.materials![0] = {
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0, texCoord: 1 },
      },
      normalTexture: { index: 0, texCoord: 2 },
    };

    const { scene, diagnostics, warnings } = await gltfToScene(fixture.gltf, { buffers: fixture.buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect((prim.material.baseColorMap as TextureRef | undefined)?.texCoord).toBe(1);
    expect(prim.material.normalMap).toBeUndefined();
    expect(warnings.some((w) => w.includes('normalMap') && w.includes('already references TEXCOORD_1'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'ignored-material-texcoord',
      path: 'meshes[0].primitives[0].material',
    }));
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

  it('drops malformed COLOR_0 accessors instead of forwarding invalid vertex-color buffers', async () => {
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

    const { scene, diagnostics } = await gltfToScene(gltf, { buffers: new Map([[0, packed]]) });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.colors).toBeUndefined();
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'invalid-primitive-attribute',
      path: 'meshes[0].primitives[0].attributes.COLOR_0',
      message: expect.stringContaining('COLOR_0 accessor must be VEC3 or VEC4'),
    }));
  });

  it('warns when secondary vertex color sets are ignored', async () => {
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

    const { diagnostics, scene, warnings } = await gltfToScene(gltf, { buffers: new Map([[0, packed]]) });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.colors ?? [])).toEqual([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        severity: 'warning',
        code: 'ignored-vertex-color-set',
        path: 'meshes[0].primitives[0].attributes.COLOR_1',
      }),
    ]));
    expect(warnings.some(w => w.includes('COLOR_1') && w.includes('ignored'))).toBe(true);
  });

  it('surfaces unknown primitive attributes as structured diagnostics', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.meshes![0]!.primitives[0]!.attributes._CUSTOM_WEIGHT = 0;

    const { diagnostics, scene, warnings } = await gltfToScene(gltf, { buffers });

    expect(scene.primitives).toHaveLength(1);
    expect(warnings.some((warning) => warning.includes('_CUSTOM_WEIGHT') && warning.includes('ignored'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        severity: 'warning',
        code: 'ignored-primitive-attribute',
        path: 'meshes[0].primitives[0].attributes._CUSTOM_WEIGHT',
      }),
    ]));
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

  it('falls back to one mesh with a structured diagnostic when EXT_mesh_gpu_instancing accessors disagree', async () => {
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

    const { scene, diagnostics } = await gltfToScene(gltf, { buffers });

    expect(scene.primitives).toHaveLength(1);
    expect(scene.primitives[0]!.kind).toBe('mesh');
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'ignored-gpu-instancing',
      path: 'nodes[0].extensions.EXT_mesh_gpu_instancing.attributes.SCALE',
      message: expect.stringContaining('does not match instance count'),
    }));
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

  it('drops malformed authored TANGENT and regenerates when the material needs a tangent frame', async () => {
    const { gltf, buffers } = makeNormalMappedTriangleGltf([
      1, 0, 0,
      1, 0, 0,
      1, 0, 0,
    ]);
    gltf.accessors![3] = {
      ...gltf.accessors![3]!,
      type: 'VEC3',
    };

    const { scene, diagnostics } = await gltfToScene(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-normal' }),
    });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.tangents).toBeInstanceOf(Float32Array);
    expect(prim.tangents).toHaveLength(12);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'invalid-primitive-attribute',
      path: 'meshes[0].primitives[0].attributes.TANGENT',
      message: expect.stringContaining('TANGENT accessor must be VEC4'),
    }));
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'generated-tangents',
      path: 'meshes[0].primitives[0].attributes.TANGENT',
    }));
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

  it('emits a structured diagnostic when tangent generation lacks TEXCOORD_0', async () => {
    const { gltf, buffers } = makeNormalMappedTriangleGltf();
    delete gltf.meshes![0]!.primitives[0]!.attributes.TEXCOORD_0;
    const { scene, diagnostics } = await gltfToScene(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-normal' }),
    });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.tangents).toBeUndefined();
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'missing-tangent-texcoord',
      path: 'meshes[0].primitives[0].attributes.TEXCOORD_0',
      message: expect.stringContaining('has no TEXCOORD_0'),
    }));
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

  it('emits a structured diagnostic while preserving doubleSided materials', async () => {
    const { gltf, buffers } = makeGltfWithMaterial({
      name: 'leaf',
      doubleSided: true,
      pbrMetallicRoughness: {
        baseColorFactor: [0.2, 0.8, 0.3, 1],
      },
    });

    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers });

    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.extensions?.doubleSided).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        severity: 'warning',
        code: 'double-sided-material',
        path: 'materials[0].doubleSided',
      }),
    ]));
    expect(warnings.some((warning) =>
      warning.includes('Material "leaf" sets doubleSided=true') &&
      warning.includes('preserved at MaterialSpec.extensions.doubleSided'),
    )).toBe(true);
  });

  it('preserves KHR_texture_transform texCoord override on texture refs', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const imageBuf = u8Buffer([0x89, 0x50, 0x4e, 0x47]);
    const totalBuf = concatBuffers(posBuf, imageBuf);
    const handle = { kind: 'decoded-texture' };
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
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
      images: [{ bufferView: 1, mimeType: 'image/png' }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength, byteLength: imageBuf.byteLength },
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
    const imageBuf = u8Buffer(PNG_MAGIC);
    const totalBuf = concatBuffers(posBuf, imageBuf);
    const handle = { kind: 'decoded-masked-base-color' };
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{
        alphaMode: 'MASK',
        alphaCutoff: 0.45,
        pbrMetallicRoughness: {
          baseColorTexture: { index: 0 },
        },
      }],
      textures: [{ source: 0 }],
      images: [{ bufferView: 1, mimeType: 'image/png' }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength },
        { buffer: 0, byteOffset: posBuf.byteLength, byteLength: imageBuf.byteLength },
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
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
      textures: [{ source: 0 }],
      images: [{ uri: 'data:image/png;base64,AQID' }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteLength: posBuf.byteLength }],
      buffers: [{ byteLength: posBuf.byteLength }],
    };

    const { scene } = await gltfToScene(gltf, {
      buffers: new Map([[0, posBuf]]),
      decodeImage,
    });

    expect(decodeImage).toHaveBeenCalledOnce();
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    const ref = mat.baseColorMap as TextureRef;
    expect(ref.handle).toBe(handle);
  });

  it('warns and skips external URI images instead of handing them to decodeImage', async () => {
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

    const { scene, warnings, diagnostics } = await gltfToScene(gltf, {
      buffers: new Map([[0, posBuf]]),
      decodeImage,
    });

    expect(decodeImage).not.toHaveBeenCalled();
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.baseColorMap).toBeUndefined();
    expect(warnings.some((w) => w.includes('external image URIs'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'external-image-uri',
      path: 'images[0].uri',
      message: expect.stringContaining('external image URIs'),
    }));
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
    (gltf as GltfJson & { extensionsRequired: string[] }).extensionsRequired = ['VENDOR_required_extension'];

    await expect(gltfToScene(gltf, { buffers })).rejects.toThrow(
      /extensionsRequired includes unsupported extension "VENDOR_required_extension"/,
    );
    await expect(gltfToScene(gltf, { buffers })).rejects.toBeInstanceOf(GltfImportError);
    await expect(gltfToScene(gltf, { buffers })).rejects.toMatchObject({
      name: 'GltfImportError',
      diagnostics: [{
        severity: 'error',
        code: 'unsupported-required-extension',
        path: 'extensionsRequired[0]',
        message: expect.stringContaining('VENDOR_required_extension'),
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
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.includes('KHR_draco_mesh_compression'))).toBe(true);
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
  it('drops a malformed animation (no channels) with a warning', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.animations = [{ name: 'walk' }]; // no channels/samplers → not importable
    const { animations, diagnostics, warnings } = await gltfToScene(gltf, { buffers });
    expect(animations).toHaveLength(0);
    expect(warnings.some(w => w.includes('no importable channels'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        severity: 'warning',
        code: 'dropped-animation',
        path: 'animations[0]',
      }),
    ]));
  });

  it('warns about skins (rest pose; host drives the pose)', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.skins = [{ joints: [0] }];
    gltf.nodes![0] = { ...gltf.nodes![0]!, skin: 0 };
    const { warnings, diagnostics } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.toLowerCase().includes('skin'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        severity: 'warning',
        code: 'skin-rest-pose',
        path: 'skins[0]',
      }),
    ]));
  });

  it('surfaces ignored cameras as structured import diagnostics', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.cameras = [{ type: 'perspective' }];
    gltf.nodes![0] = { ...gltf.nodes![0]!, camera: 0 };
    const { warnings, diagnostics } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.includes('Camera nodes are present but ignored'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        severity: 'warning',
        code: 'ignored-camera',
        path: 'cameras[0]',
      }),
    ]));
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
  it('surfaces missing texture image indices as structured diagnostics', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({ normalTexCoord: 0, includeUv0: true });
    gltf.textures![0]!.source = 99;

    const { diagnostics, scene, warnings } = await gltfToScene(gltf, { buffers });

    expect((scene.primitives[0] as MeshPrimitive).material.normalMap).toBeUndefined();
    expect(warnings.some((warning) => warning.includes('missing image index 99'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'image-not-found',
        path: 'textures[0].source',
        textureIndex: 0,
        imageIndex: 99,
      }),
      expect.objectContaining({
        code: 'material-texture-unresolved',
        path: 'materials[0].normalTexture',
        materialIndex: 0,
        textureIndex: 0,
      }),
    ]));
  });

  it('surfaces material texture infos that reference missing texture indices', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({ normalTexCoord: 0, includeUv0: true });
    gltf.materials![0]!.normalTexture = { index: 99 };

    const { diagnostics, scene, warnings } = await gltfToScene(gltf, { buffers });

    expect((scene.primitives[0] as MeshPrimitive).material.normalMap).toBeUndefined();
    expect(warnings.some((warning) => warning.includes('missing texture index 99'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'material-texture-not-found',
        path: 'materials[0].normalTexture.index',
        materialIndex: 0,
        textureIndex: 99,
      }),
    ]));
  });

  it('surfaces missing embedded image bufferViews as structured diagnostics', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({ normalTexCoord: 0, includeUv0: true });
    gltf.images![0]!.bufferView = 99;

    const { diagnostics, scene, warnings } = await gltfToScene(gltf, { buffers });

    expect((scene.primitives[0] as MeshPrimitive).material.normalMap).toBeUndefined();
    expect(warnings.some((warning) => warning.includes('missing bufferView 99'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'image-buffer-view-not-found',
        path: 'images[0].bufferView',
        textureIndex: 0,
        imageIndex: 0,
        bufferViewIndex: 99,
      }),
    ]));
  });

  it('surfaces unavailable embedded image buffers as structured diagnostics', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({ normalTexCoord: 0, includeUv0: true });
    const imageBufferView = gltf.images![0]!.bufferView!;
    gltf.bufferViews![imageBufferView] = { buffer: 99, byteOffset: 0, byteLength: 4 };

    const { diagnostics, scene, warnings } = await gltfToScene(gltf, { buffers });

    expect((scene.primitives[0] as MeshPrimitive).material.normalMap).toBeUndefined();
    expect(warnings.some((warning) => warning.includes('unavailable buffer 99'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'image-buffer-unavailable',
        path: `bufferViews[${imageBufferView}].buffer`,
        textureIndex: 0,
        imageIndex: 0,
        bufferViewIndex: imageBufferView,
        bufferIndex: 99,
      }),
    ]));
  });

  it('surfaces images without bufferView or URI as structured diagnostics', async () => {
    const { gltf, buffers } = makeUv1NormalMappedTriangleGltf({ normalTexCoord: 0, includeUv0: true });
    gltf.images![0] = { name: 'empty-image' };

    const { diagnostics, scene, warnings } = await gltfToScene(gltf, { buffers });

    expect((scene.primitives[0] as MeshPrimitive).material.normalMap).toBeUndefined();
    expect(warnings.some((warning) => warning.includes('neither bufferView nor uri'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'image-source-missing',
        path: 'images[0]',
        textureIndex: 0,
        imageIndex: 0,
      }),
    ]));
  });

  it('maps normalTexture scale', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{
        normalTexture: { index: 0, scale: 0.5 },
        // textures/images arrays are absent → handle resolves to undefined → normalMap is undefined
      }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteLength: posBuf.byteLength }],
      buffers: [{ byteLength: posBuf.byteLength }],
    };
    const { scene } = await gltfToScene(gltf, { buffers: new Map([[0, posBuf]]) });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    // The texture index 0 has no entry in handleMap → normalMap should be undefined.
    // normalScale should be 0.5 from the material spec if the map resolves, else omitted.
    // (no textures array → handle is undefined → normalMap undefined)
    expect(mat.normalMap).toBeUndefined();
  });

  it('maps occlusionTexture strength to aoMapIntensity', async () => {
    const posBuf = f32Buffer(TRIANGLE_POSITIONS);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{
        occlusionTexture: { index: 0, strength: 0.75 },
      }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteLength: posBuf.byteLength }],
      buffers: [{ byteLength: posBuf.byteLength }],
    };
    const { scene } = await gltfToScene(gltf, { buffers: new Map([[0, posBuf]]) });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    // aoMapIntensity should be 0.75 even when the texture image can't be resolved
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

  it('emits kind:skinned-mesh when node has a skin (JOINTS_0 u8)', async () => {
    const { gltf, buffers } = makeSkinnedGltf(5121);
    const { scene } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0]!;
    expect(prim.kind).toBe('skinned-mesh');
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

  it('surfaces malformed punctual light references as structured diagnostics', async () => {
    const { gltf, buffers } = makeLightsGltf();
    gltf.nodes![1]!.extensions = { KHR_lights_punctual: { light: 99 } };
    const punctual = gltf.extensions!.KHR_lights_punctual as { lights: Array<{ type: string; name?: string }> };
    punctual.lights[2] = {
      type: 'tube',
      name: 'bad tube',
    };

    const { diagnostics, scene, warnings } = await gltfToScene(gltf, { buffers });

    expect(scene.emitters).toHaveLength(1);
    expect(warnings.some((warning) => warning.includes('light index 99'))).toBe(true);
    expect(warnings.some((warning) => warning.includes('unsupported type "tube"'))).toBe(true);
    expect(diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'missing-punctual-light',
        path: 'nodes[1].extensions.KHR_lights_punctual.light',
      }),
      expect.objectContaining({
        code: 'unsupported-punctual-light-type',
        path: 'extensions.KHR_lights_punctual.lights[2].type',
      }),
    ]));
  });
});
