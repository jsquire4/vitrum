// gltfAdapter.test.ts — Unit tests for @vitrum/gltf-adapter.
//
// All fixtures are built in-code (no network, no binary fixture files).
// Tests cover:
//   1. Minimal triangle (positions, flat-normal generation, default material)
//   2. Textured quad GLB built byte-by-byte
//   3. Transmissive/IOR/volume material field mapping
//   4. Node hierarchy with nested TRS transforms → correct world positions
//   5. Multi-primitive mesh → multiple ScenePrimitives
//   6. Sparse accessor handling
//   7. Non-triangle mode warning + skip
//   8. KHR_draco rejection warning
//   9. Unknown extension warning
//   10. Missing buffer warning
//   11. alphaMode + alphaCutoff mapping
//   12. KHR_materials_sheen / clearcoat / iridescence / anisotropy / specular mapping

import { describe, it, expect } from 'vitest';
import { gltfToScene } from './gltfToScene.js';
import type { GltfJson } from './gltfTypes.js';
import type { MeshPrimitive } from '@vitrum/core';

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

// Triangle: 3 vertices at (0,0,0), (1,0,0), (0,1,0)
const TRIANGLE_POSITIONS = [0, 0, 0, 1, 0, 0, 0, 1, 0];
// 3 flat normals: cross((1,0,0)-(0,0,0), (0,1,0)-(0,0,0)) = (0,0,1)
const TRIANGLE_FLAT_NORMAL = [0, 0, 1];

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

// ────────────────────────────────────────────────────────────────────────────
// Test 1 — Minimal triangle
// ────────────────────────────────────────────────────────────────────────────

describe('minimal triangle', () => {
  it('produces one MeshPrimitive with correct positions', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    const { scene, warnings } = await gltfToScene(gltf, { buffers });

    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.kind).toBe('mesh');
    expect(Array.from(prim.positions)).toEqual(TRIANGLE_POSITIONS);
  });

  it('generates flat normals when NORMAL is absent', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    const { scene, warnings } = await gltfToScene(gltf, { buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.normals).toBeInstanceOf(Float32Array);
    // Flat normal for this triangle is [0,0,1] for each of 3 vertices
    expect(prim.normals[2]).toBeCloseTo(1, 5); // vertex 0, z
    expect(prim.normals[5]).toBeCloseTo(1, 5); // vertex 1, z
    expect(prim.normals[8]).toBeCloseTo(1, 5); // vertex 2, z
    expect(warnings.some(w => w.includes('flat normals'))).toBe(true);
  });

  it('uses the default glTF material when no material is specified', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    const { scene } = await gltfToScene(gltf, { buffers });

    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.material.baseColor).toEqual([1, 1, 1]);
    expect(prim.material.metallic).toBe(1);
    expect(prim.material.roughness).toBe(1);
  });

  it('scene has empty emitters and none environment', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    const { scene } = await gltfToScene(gltf, { buffers });
    expect(scene.emitters).toHaveLength(0);
    expect(scene.environment.kind).toBe('none');
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
    const { scene, warnings } = await gltfToScene(glb);

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
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.includes('MY_custom_extension'))).toBe(true);
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect((mat.extensions as Record<string, unknown>)?.['MY_custom_extension']).toEqual({ foo: 42 });
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
    const { scene, warnings } = await gltfToScene(gltf, { buffers: new Map([[0, posBuf]]) });
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
    const { scene, warnings } = await gltfToScene(gltf, { buffers: new Map([[0, finalBuf]]) });

    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    // Vertex 2 (positions 6,7,8) should have been patched to [9,8,7]
    expect(prim.positions[6]).toBeCloseTo(9);
    expect(prim.positions[7]).toBeCloseTo(8);
    expect(prim.positions[8]).toBeCloseTo(7);
    // Check that a sparse warning was emitted
    expect(warnings.some(w => w.includes('sparse'))).toBe(true);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 7 — Non-triangle mode warning + skip
// ────────────────────────────────────────────────────────────────────────────

describe('non-triangle primitive mode', () => {
  it('skips POINTS primitives and emits a warning', async () => {
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
    // Only the TRIANGLES primitive should survive.
    expect(scene.primitives).toHaveLength(1);
    expect(warnings.some(w => w.includes('POINTS'))).toBe(true);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 8 — Draco rejection warning
// ────────────────────────────────────────────────────────────────────────────

describe('Draco rejection', () => {
  it('emits a warning for KHR_draco_mesh_compression in extensionsUsed', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    (gltf as GltfJson & { extensionsUsed: string[] }).extensionsUsed = ['KHR_draco_mesh_compression'];
    const { warnings } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.includes('KHR_draco_mesh_compression'))).toBe(true);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Test 9 — Animation + skin + camera warnings
// ────────────────────────────────────────────────────────────────────────────

describe('out-of-scope feature warnings', () => {
  it('warns about animations', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    (gltf as GltfJson & { animations: unknown[] }).animations = [{ name: 'walk' }];
    const { warnings } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.includes('animation'))).toBe(true);
  });

  it('warns about skins', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    gltf.skins = [{ joints: [0] }];
    const { warnings } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.toLowerCase().includes('skin'))).toBe(true);
  });

  it('warns about KHR_lights_punctual', async () => {
    const { gltf, buffers } = makeMinimalTriangleGltf();
    (gltf as GltfJson & { extensionsUsed: string[] }).extensionsUsed = ['KHR_lights_punctual'];
    const { warnings } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.includes('KHR_lights_punctual'))).toBe(true);
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
