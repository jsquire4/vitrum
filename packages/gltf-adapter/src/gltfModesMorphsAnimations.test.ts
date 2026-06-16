// gltfModesMorphsAnimations.test.ts — Tests for the GLTF-05 / GLTF-04 / GLTF-03
// closure slices:
//   GLTF-05  TRIANGLE_STRIP (5) / TRIANGLE_FAN (6) triangulation
//            (indexed + non-indexed, winding per glTF §3.7.2.1, degenerate skip)
//   GLTF-04  Morph targets (POSITION/NORMAL deltas, node/mesh weights,
//            identity-skin promotion for unskinned morphed meshes,
//            solveSkin end-to-end blend verification)
//   GLTF-03  Animations (LINEAR / STEP / CUBICSPLINE samplers, TRS + weights
//            channels, sampled-value verification via core sampleAnimationClip)
//
// All fixtures are built in-code (no network, no binary fixture files), same
// style as gltfAdapter.test.ts.

import { describe, it, expect } from 'vitest';
import { gltfToScene } from './gltfToScene.js';
import { animationNodeId } from './animations.js';
import { sequentialIndices, triangulateTopology } from './triangulation.js';
import type { GltfJson } from './gltfTypes.js';
import { sampleAnimationClip, solveSkin } from '@vitrum/core';
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

function u8Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setUint8(i, v));
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

// ────────────────────────────────────────────────────────────────────────────
// GLTF-05 — triangulateTopology unit behaviour
// ────────────────────────────────────────────────────────────────────────────

describe('triangulateTopology (GLTF-05 unit)', () => {
  it('TRIANGLE_STRIP alternates winding per glTF §3.7.2.1', () => {
    // 4-vertex strip → 2 triangles: even (0,1,2), odd (2,1,3).
    const out = triangulateTopology(new Uint32Array([0, 1, 2, 3]), 5);
    expect(Array.from(out)).toEqual([0, 1, 2, 2, 1, 3]);
  });

  it('TRIANGLE_FAN pivots on vertex 0 with spec ordering {v[i+1], v[i+2], v[0]}', () => {
    const out = triangulateTopology(new Uint32Array([0, 1, 2, 3, 4]), 6);
    expect(Array.from(out)).toEqual([1, 2, 0, 2, 3, 0, 3, 4, 0]);
  });

  it('skips degenerate strip triangles (repeated indices)', () => {
    // Strip with a stitching degenerate: 0,1,2,2,3,4
    // triangles: (0,1,2) ok; (2,1,2)✗; (2,2,3)✗→wait odd: (2,2,3)… compute:
    //   i=0 even (0,1,2) keep; i=1 odd (2,1,2) drop; i=2 even (2,2,3) drop;
    //   i=3 odd (3,2,4) keep.
    const out = triangulateTopology(new Uint32Array([0, 1, 2, 2, 3, 4]), 5);
    expect(Array.from(out)).toEqual([0, 1, 2, 3, 2, 4]);
  });

  it('returns empty for fewer than 3 indices', () => {
    expect(triangulateTopology(new Uint32Array([0, 1]), 5).length).toBe(0);
    expect(triangulateTopology(new Uint32Array([]), 6).length).toBe(0);
  });

  it('sequentialIndices produces 0..n-1', () => {
    expect(Array.from(sequentialIndices(4))).toEqual([0, 1, 2, 3]);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// GLTF-05 — strip / fan through gltfToScene (indexed + non-indexed)
// ────────────────────────────────────────────────────────────────────────────

// 4 vertices of a unit-square strip in the XY plane.
const STRIP_POSITIONS = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0];
// 5 vertices for a fan: pivot at origin + 4 rim points.
const FAN_POSITIONS = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, -1, 1, 0];

function makeModeGltf(
  mode: number,
  positions: number[],
  indices?: number[],
): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const chunks: ArrayBuffer[] = [f32Buffer(positions)];
  if (indices) chunks.push(u16Buffer(indices));
  const { buffer, bufferViews } = layoutBuffer(chunks);
  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scenes: [{ nodes: [0] }],
    scene: 0,
    nodes: [{ mesh: 0 }],
    meshes: [{
      primitives: [{
        attributes: { POSITION: 0 },
        mode,
        ...(indices ? { indices: 1 } : {}),
      }],
    }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: positions.length / 3, type: 'VEC3' },
      ...(indices
        ? [{ bufferView: 1, componentType: 5123 as const, count: indices.length, type: 'SCALAR' as const }]
        : []),
    ],
    bufferViews,
    buffers: [{ byteLength: buffer.byteLength }],
  };
  return { gltf, buffers: new Map([[0, buffer]]) };
}

describe('TRIANGLE_STRIP / TRIANGLE_FAN import (GLTF-05)', () => {
  it('non-indexed TRIANGLE_STRIP → indexed triangle list with alternating winding', async () => {
    const { gltf, buffers } = makeModeGltf(5, STRIP_POSITIONS);
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.kind).toBe('mesh');
    expect(Array.from(prim.indices!)).toEqual([0, 1, 2, 2, 1, 3]);
    // No "unsupported mode" warning.
    expect(warnings.some(w => w.includes('unsupported mode'))).toBe(false);
    // Both triangles face +Z (winding preserved): flat normals all +Z.
    expect(prim.normals[2]).toBeCloseTo(1, 5);
    expect(prim.normals[11]).toBeCloseTo(1, 5);
  });

  it('indexed TRIANGLE_STRIP → triangulated through the index buffer', async () => {
    // Reverse-order strip indices exercise the indexed path: 3,2,1,0.
    const { gltf, buffers } = makeModeGltf(5, STRIP_POSITIONS, [3, 2, 1, 0]);
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.indices!)).toEqual([3, 2, 1, 1, 2, 0]);
  });

  it('non-indexed TRIANGLE_FAN → pivot-on-vertex-0 triangle list', async () => {
    const { gltf, buffers } = makeModeGltf(6, FAN_POSITIONS);
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.indices!)).toEqual([1, 2, 0, 2, 3, 0, 3, 4, 0]);
  });

  it('indexed TRIANGLE_FAN with a degenerate rim repeat skips the degenerate', async () => {
    // Fan indices 0,1,2,2,3 → triangles (1,2,0), (2,2,0)✗drop, (2,3,0).
    const { gltf, buffers } = makeModeGltf(6, FAN_POSITIONS, [0, 1, 2, 2, 3]);
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.indices!)).toEqual([1, 2, 0, 2, 3, 0]);
  });

  it('strip with only degenerate triangles is skipped with a warning', async () => {
    const { gltf, buffers } = makeModeGltf(5, STRIP_POSITIONS, [1, 1, 1, 1]);
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(0);
    expect(warnings.some(w => w.includes('no non-degenerate triangles'))).toBe(true);
  });

  it('LINES (1) imports as fallback-generated mesh with a warning', async () => {
    const { gltf, buffers } = makeModeGltf(1, STRIP_POSITIONS);
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as MeshPrimitive;
    expect(prim.positions.length).toBeGreaterThan(STRIP_POSITIONS.length);
    expect(prim.indices?.length).toBeGreaterThan(0);
    expect(warnings.some(w => w.includes('LINES') && w.includes('fallback-generated mesh'))).toBe(true);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// GLTF-04 — morph targets
// ────────────────────────────────────────────────────────────────────────────

// Base triangle for morph fixtures.
const MORPH_BASE_POSITIONS = [0, 0, 0, 1, 0, 0, 0, 1, 0];
const MORPH_BASE_NORMALS = [0, 0, 1, 0, 0, 1, 0, 0, 1];

interface MorphTargetSpec {
  position?: number[]; // 9 floats (3 verts × 3)
  normal?: number[];   // 9 floats
  tangent?: number[];  // 9 floats (glTF morph target TANGENT is a VEC3 delta)
}

/**
 * Build a 1-triangle glTF with morph targets. Accessor/bufferView layout is
 * computed dynamically from the requested targets.
 */
function makeMorphGltf(opts: {
  targets: MorphTargetSpec[];
  meshWeights?: number[];
  nodeWeights?: number[];
}): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const chunks: ArrayBuffer[] = [f32Buffer(MORPH_BASE_POSITIONS), f32Buffer(MORPH_BASE_NORMALS)];
  const accessors: NonNullable<GltfJson['accessors']> = [
    { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }, // POSITION
    { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' }, // NORMAL
  ];
  const targets: Array<Record<string, number>> = [];

  for (const spec of opts.targets) {
    const target: Record<string, number> = {};
    if (spec.position) {
      target['POSITION'] = accessors.length;
      accessors.push({ bufferView: chunks.length, componentType: 5126, count: 3, type: 'VEC3' });
      chunks.push(f32Buffer(spec.position));
    }
    if (spec.normal) {
      target['NORMAL'] = accessors.length;
      accessors.push({ bufferView: chunks.length, componentType: 5126, count: 3, type: 'VEC3' });
      chunks.push(f32Buffer(spec.normal));
    }
    if (spec.tangent) {
      target['TANGENT'] = accessors.length;
      accessors.push({ bufferView: chunks.length, componentType: 5126, count: 3, type: 'VEC3' });
      chunks.push(f32Buffer(spec.tangent));
    }
    targets.push(target);
  }

  const { buffer, bufferViews } = layoutBuffer(chunks);
  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scenes: [{ nodes: [0] }],
    scene: 0,
    nodes: [{ mesh: 0, ...(opts.nodeWeights ? { weights: opts.nodeWeights } : {}) }],
    meshes: [{
      primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, targets }],
      ...(opts.meshWeights ? { weights: opts.meshWeights } : {}),
    }],
    accessors,
    bufferViews,
    buffers: [{ byteLength: buffer.byteLength }],
  };
  return { gltf, buffers: new Map([[0, buffer]]) };
}

describe('morph targets (GLTF-04)', () => {
  const POS_DELTA_1 = [0, 0, 1, 0, 0, 1, 0, 0, 1];      // +1 Z all verts
  const POS_DELTA_2 = [2, 0, 0, 2, 0, 0, 2, 0, 0];      // +2 X all verts

  it('promotes an unskinned morphed mesh to skinned-mesh with an identity skeleton', async () => {
    const { gltf, buffers } = makeMorphGltf({
      targets: [{ position: POS_DELTA_1 }],
      meshWeights: [0.5],
    });
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(scene.primitives).toHaveLength(1);
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(prim.kind).toBe('skinned-mesh');
    // Identity skeleton: 1 bone, all indices 0, weights [1,0,0,0] per vertex.
    expect(prim.bones.length).toBe(16);
    expect(prim.bones[0]).toBe(1);
    expect(prim.bones[15]).toBe(1);
    expect(Array.from(prim.skinIndices)).toEqual([0,0,0,0, 0,0,0,0, 0,0,0,0]);
    expect(Array.from(prim.skinWeights)).toEqual([1,0,0,0, 1,0,0,0, 1,0,0,0]);
    expect(prim.bindMatrix).toBeUndefined();
    expect(warnings.some(w => w.includes('synthesized identity skeleton'))).toBe(true);
    // Morph data imported.
    expect(prim.morphTargets).toHaveLength(1);
    expect(Array.from(prim.morphTargets![0]!)).toEqual(POS_DELTA_1);
    expect(Array.from(prim.morphWeights!)).toEqual([0.5]);
    expect(prim.morphTargetNormals).toBeUndefined();
  });

  it('position morph: solveSkin output matches hand-computed rest + w·Δ', async () => {
    const { gltf, buffers } = makeMorphGltf({
      targets: [{ position: POS_DELTA_1 }],
      meshWeights: [0.5],
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    const { positions } = solveSkin(prim);
    // Each vertex: restPos + 0.5 · (0,0,1) = restPos + (0,0,0.5)
    for (let v = 0; v < 3; v++) {
      expect(positions[v * 3 + 0]).toBeCloseTo(MORPH_BASE_POSITIONS[v * 3 + 0]!, 5);
      expect(positions[v * 3 + 1]).toBeCloseTo(MORPH_BASE_POSITIONS[v * 3 + 1]!, 5);
      expect(positions[v * 3 + 2]).toBeCloseTo(MORPH_BASE_POSITIONS[v * 3 + 2]! + 0.5, 5);
    }
  });

  it('normal morph: morphTargetNormals imported and blended by solveSkin (normalized)', async () => {
    // Normal delta tilts +Z normals toward +X: Δn = (1, 0, 0), weight 1
    // → blended (1, 0, 1) → normalized (1/√2, 0, 1/√2).
    const { gltf, buffers } = makeMorphGltf({
      targets: [{ position: [0,0,0, 0,0,0, 0,0,0], normal: [1,0,0, 1,0,0, 1,0,0] }],
      meshWeights: [1],
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(prim.morphTargetNormals).toHaveLength(1);
    const { normals } = solveSkin(prim);
    const inv = 1 / Math.SQRT2;
    for (let v = 0; v < 3; v++) {
      expect(normals[v * 3 + 0]).toBeCloseTo(inv, 5);
      expect(normals[v * 3 + 1]).toBeCloseTo(0, 5);
      expect(normals[v * 3 + 2]).toBeCloseTo(inv, 5);
    }
  });

  it('two-target weighted blend: node.weights overrides mesh.weights', async () => {
    const { gltf, buffers } = makeMorphGltf({
      targets: [{ position: POS_DELTA_1 }, { position: POS_DELTA_2 }],
      meshWeights: [9, 9],          // overridden
      nodeWeights: [0.25, 0.5],
    });
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(Array.from(prim.morphWeights!)).toEqual([0.25, 0.5]);
    const { positions } = solveSkin(prim);
    // restPos + 0.25·(0,0,1) + 0.5·(2,0,0) = restPos + (1, 0, 0.25)
    for (let v = 0; v < 3; v++) {
      expect(positions[v * 3 + 0]).toBeCloseTo(MORPH_BASE_POSITIONS[v * 3 + 0]! + 1, 5);
      expect(positions[v * 3 + 1]).toBeCloseTo(MORPH_BASE_POSITIONS[v * 3 + 1]!, 5);
      expect(positions[v * 3 + 2]).toBeCloseTo(MORPH_BASE_POSITIONS[v * 3 + 2]! + 0.25, 5);
    }
  });

  it('TANGENT morph deltas are preserved on the core primitive contract', async () => {
    const tangentDelta = [0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0];
    const { gltf, buffers } = makeMorphGltf({
      targets: [{ position: POS_DELTA_1, tangent: tangentDelta }],
      meshWeights: [1],
    });
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.includes('TANGENT deltas are ignored'))).toBe(false);
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(prim.morphTargets).toHaveLength(1);
    expect(prim.morphTargetTangents).toHaveLength(1);
    expect(Array.from(prim.morphTargetTangents![0]!)).toEqual(Array.from(new Float32Array(tangentDelta)));
  });

  it('morphWeights defaults to zeros when no weights are authored', async () => {
    const { gltf, buffers } = makeMorphGltf({ targets: [{ position: POS_DELTA_1 }] });
    const { scene } = await gltfToScene(gltf, { buffers });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(Array.from(prim.morphWeights!)).toEqual([0]);
    // Zero weights → solveSkin returns the rest pose.
    const { positions } = solveSkin(prim);
    for (let i = 0; i < 9; i++) {
      expect(positions[i]).toBeCloseTo(MORPH_BASE_POSITIONS[i]!, 5);
    }
  });

  it('sparse-accessor morph target: only the patched vertex moves', async () => {
    // Hand-build: target POSITION accessor with NO bufferView (zero base) and a
    // sparse patch moving vertex 1 by (0, 3, 0).
    const posBuf = f32Buffer(MORPH_BASE_POSITIONS);
    const sparseIdxBuf = u32Buffer([1]);
    const sparseValBuf = f32Buffer([0, 3, 0]);
    const { buffer, bufferViews } = layoutBuffer([posBuf, sparseIdxBuf, sparseValBuf]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [{ mesh: 0 }],
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0 },
          targets: [{ POSITION: 1 }],
        }],
        weights: [1],
      }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        {
          componentType: 5126, count: 3, type: 'VEC3',
          sparse: {
            count: 1,
            indices: { bufferView: 1, componentType: 5125 },
            values: { bufferView: 2 },
          },
        },
      ],
      bufferViews,
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const { scene } = await gltfToScene(gltf, { buffers: new Map([[0, buffer]]) });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    const delta = prim.morphTargets![0]!;
    expect(Array.from(delta)).toEqual([0, 0, 0, 0, 3, 0, 0, 0, 0]);
    const { positions } = solveSkin(prim);
    expect(positions[4]).toBeCloseTo(MORPH_BASE_POSITIONS[4]! + 3, 5); // vertex 1 Y
    expect(positions[0]).toBeCloseTo(0, 5);                            // vertex 0 untouched
  });

  it('skinned + morphed: morph pre-blend composes with LBS', async () => {
    // 2 joints: joint0 identity, joint1 translated +1 X; identity IBMs;
    // every vertex weighted 0.5/0.5 → skin adds (0.5, 0, 0) after the morph
    // pre-blend of +1 Z (weight 1).
    const posBuf = f32Buffer(MORPH_BASE_POSITIONS);
    const ibm = new Array(32).fill(0);
    ibm[0] = 1; ibm[5] = 1; ibm[10] = 1; ibm[15] = 1;
    ibm[16] = 1; ibm[21] = 1; ibm[26] = 1; ibm[31] = 1;
    const ibmBuf = f32Buffer(ibm);
    const weightsBuf = f32Buffer([0.5,0.5,0,0, 0.5,0.5,0,0, 0.5,0.5,0,0]);
    const jointsBuf = u8Buffer([0,1,0,0, 0,1,0,0, 0,1,0,0]);
    const morphBuf = f32Buffer([0,0,1, 0,0,1, 0,0,1]);
    const { buffer, bufferViews } = layoutBuffer([posBuf, ibmBuf, weightsBuf, jointsBuf, morphBuf]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      scene: 0,
      nodes: [
        { mesh: 0, skin: 0, children: [1, 2], weights: [1] },
        { translation: [0, 0, 0] }, // joint 0
        { translation: [1, 0, 0] }, // joint 1
      ],
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0, JOINTS_0: 3, WEIGHTS_0: 2 },
          targets: [{ POSITION: 4 }],
        }],
      }],
      skins: [{ joints: [1, 2], inverseBindMatrices: 1 }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 2, type: 'MAT4' },
        { bufferView: 2, componentType: 5126, count: 3, type: 'VEC4' },
        { bufferView: 3, componentType: 5121, count: 3, type: 'VEC4' },
        { bufferView: 4, componentType: 5126, count: 3, type: 'VEC3' },
      ],
      bufferViews,
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const { scene, warnings } = await gltfToScene(gltf, { buffers: new Map([[0, buffer]]) });
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(prim.kind).toBe('skinned-mesh');
    // Real skin (2 bones), morph fields attached, NOT the synthesized skeleton.
    expect(prim.bones.length).toBe(32);
    expect(prim.morphTargets).toHaveLength(1);
    expect(Array.from(prim.morphWeights!)).toEqual([1]);
    expect(warnings.some(w => w.includes('synthesized identity skeleton'))).toBe(false);
    const { positions } = solveSkin(prim);
    // morphedPos = rest + (0,0,1); skin = 0.5·I + 0.5·T(+1x) → +(0.5,0,0).
    for (let v = 0; v < 3; v++) {
      expect(positions[v * 3 + 0]).toBeCloseTo(MORPH_BASE_POSITIONS[v * 3 + 0]! + 0.5, 5);
      expect(positions[v * 3 + 1]).toBeCloseTo(MORPH_BASE_POSITIONS[v * 3 + 1]!, 5);
      expect(positions[v * 3 + 2]).toBeCloseTo(MORPH_BASE_POSITIONS[v * 3 + 2]! + 1, 5);
    }
  });
});

// ────────────────────────────────────────────────────────────────────────────
// GLTF-03 — animations
// ────────────────────────────────────────────────────────────────────────────

interface AnimChannelSpec {
  path: 'translation' | 'rotation' | 'scale' | 'weights';
  node?: number; // defaults to 0 (the mesh node)
  times: number[];
  values: number[];
  /** glTF accessor type for the output values. */
  outType: 'VEC3' | 'VEC4' | 'SCALAR';
  interpolation?: string;
}

/** Build a 1-triangle glTF with one animation made of the given channels. */
function makeAnimGltf(
  channels: AnimChannelSpec[],
  opts: { name?: string; morphTargets?: number[][] } = {},
): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const chunks: ArrayBuffer[] = [f32Buffer(MORPH_BASE_POSITIONS)];
  const accessors: NonNullable<GltfJson['accessors']> = [
    { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
  ];
  const targets: Array<Record<string, number>> = [];
  for (const deltas of opts.morphTargets ?? []) {
    targets.push({ POSITION: accessors.length });
    accessors.push({ bufferView: chunks.length, componentType: 5126, count: 3, type: 'VEC3' });
    chunks.push(f32Buffer(deltas));
  }

  const samplers: { input: number; output: number; interpolation?: string }[] = [];
  const gltfChannels: { sampler: number; target: { node?: number; path: string } }[] = [];
  for (const ch of channels) {
    const inputAcc = accessors.length;
    accessors.push({ bufferView: chunks.length, componentType: 5126, count: ch.times.length, type: 'SCALAR' });
    chunks.push(f32Buffer(ch.times));
    const compCount = ch.outType === 'VEC3' ? 3 : ch.outType === 'VEC4' ? 4 : 1;
    const outputAcc = accessors.length;
    accessors.push({
      bufferView: chunks.length, componentType: 5126,
      count: ch.values.length / compCount, type: ch.outType,
    });
    chunks.push(f32Buffer(ch.values));
    gltfChannels.push({
      sampler: samplers.length,
      target: { ...(ch.node === undefined ? { node: 0 } : { node: ch.node }), path: ch.path },
    });
    samplers.push({
      input: inputAcc, output: outputAcc,
      ...(ch.interpolation ? { interpolation: ch.interpolation } : {}),
    });
  }

  const { buffer, bufferViews } = layoutBuffer(chunks);
  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scenes: [{ nodes: [0] }],
    scene: 0,
    nodes: [{ mesh: 0 }],
    meshes: [{
      primitives: [{
        attributes: { POSITION: 0 },
        ...(targets.length > 0 ? { targets } : {}),
      }],
      ...(targets.length > 0 ? { weights: new Array(targets.length).fill(0) } : {}),
    }],
    accessors,
    bufferViews,
    buffers: [{ byteLength: buffer.byteLength }],
    animations: [{
      ...(opts.name !== undefined ? { name: opts.name } : {}),
      channels: gltfChannels,
      samplers,
    }],
  };
  return { gltf, buffers: new Map([[0, buffer]]) };
}

describe('animations (GLTF-03)', () => {
  it('LINEAR vec3 translation: sampled values match hand-computed lerp', async () => {
    const { gltf, buffers } = makeAnimGltf([{
      path: 'translation',
      times: [0, 1, 2],
      values: [0,0,0, 2,0,0, 2,4,0],
      outType: 'VEC3',
    }], { name: 'slide' });
    const { animations, animationTargets, warnings } = await gltfToScene(gltf, { buffers });
    expect(warnings.filter(w => w.toLowerCase().includes('animation'))).toHaveLength(0);
    expect(animations).toHaveLength(1);
    const clip = animations[0]!;
    expect(clip.name).toBe('slide');
    expect(clip.duration).toBeCloseTo(2);
    expect(clip.channels).toHaveLength(1);
    expect(clip.channels[0]!.target.node).toBe(animationNodeId(0));
    expect(clip.channels[0]!.target.path).toBe('translation');
    // Channel node id resolves to the imported primitive.
    expect(animationTargets[animationNodeId(0)]).toEqual(['gltf-prim-0']);

    const at05 = sampleAnimationClip(clip, 0.5)[0]!;
    expect(Array.from(at05.value)).toEqual([1, 0, 0]);
    const at15 = sampleAnimationClip(clip, 1.5)[0]!;
    expect(at15.value[0]).toBeCloseTo(2, 5);
    expect(at15.value[1]).toBeCloseTo(2, 5);
  });

  it('STEP scale holds the previous keyframe value', async () => {
    const { gltf, buffers } = makeAnimGltf([{
      path: 'scale',
      times: [0, 1],
      values: [1,1,1, 3,3,3],
      outType: 'VEC3',
      interpolation: 'STEP',
    }]);
    const { animations } = await gltfToScene(gltf, { buffers });
    const clip = animations[0]!;
    expect(clip.channels[0]!.sampler.interpolation).toBe('STEP');
    const before = sampleAnimationClip(clip, 0.99)[0]!;
    expect(Array.from(before.value)).toEqual([1, 1, 1]);
    const after = sampleAnimationClip(clip, 1)[0]!;
    expect(Array.from(after.value)).toEqual([3, 3, 3]);
  });

  it('CUBICSPLINE rotation: sampled quaternion is normalized and matches Hermite', async () => {
    // 2 keyframes, CUBICSPLINE → each keyframe = inTangent(4), value(4), outTangent(4).
    // kf0: value = identity (0,0,0,1), outTangent = (0,0,1,0)
    // kf1: value = 90° about Z (0,0,√½,√½), inTangent = (0,0,1,0)
    const h = Math.SQRT1_2;
    const { gltf, buffers } = makeAnimGltf([{
      path: 'rotation',
      times: [0, 1],
      values: [
        0,0,0,0,  0,0,0,1,  0,0,1,0,  // kf0: in, value, out
        0,0,1,0,  0,0,h,h,  0,0,0,0,  // kf1: in, value, out
      ],
      outType: 'VEC4',
      interpolation: 'CUBICSPLINE',
    }]);
    const { animations } = await gltfToScene(gltf, { buffers });
    const clip = animations[0]!;
    const q = sampleAnimationClip(clip, 0.5)[0]!.value;
    // Hand-computed Hermite at t=0.5 (dt=1): h00=.5 h10=.125 h01=.5 h11=-.125
    //   raw z = .125·1 + .5·√½ − .125·1 = .353553…, raw w = .5 + .5·√½ = .853553…
    //   |raw| = .923879…  → normalized z = sin(22.5°), w = cos(22.5°).
    const len = Math.hypot(q[0]!, q[1]!, q[2]!, q[3]!);
    expect(len).toBeCloseTo(1, 6); // core normalizes CUBICSPLINE rotation output
    expect(q[0]).toBeCloseTo(0, 6);
    expect(q[1]).toBeCloseTo(0, 6);
    expect(q[2]).toBeCloseTo(Math.sin(Math.PI / 8), 5);
    expect(q[3]).toBeCloseTo(Math.cos(Math.PI / 8), 5);
  });

  it('morph-weight animation: weights channel samples the per-target vector', async () => {
    const { gltf, buffers } = makeAnimGltf(
      [{
        path: 'weights',
        times: [0, 1],
        values: [0, 0, 1, 0.5], // 2 targets × 2 keyframes
        outType: 'SCALAR',
      }],
      { morphTargets: [[0,0,1, 0,0,1, 0,0,1], [2,0,0, 2,0,0, 2,0,0]] },
    );
    const { scene, animations } = await gltfToScene(gltf, { buffers });
    // The morphed mesh imports as a (identity-skin) skinned primitive.
    const prim = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(prim.morphTargets).toHaveLength(2);
    const clip = animations[0]!;
    const w = sampleAnimationClip(clip, 0.5)[0]!;
    expect(w.path).toBe('weights');
    expect(w.value).toHaveLength(2);
    expect(w.value[0]).toBeCloseTo(0.5, 5);
    expect(w.value[1]).toBeCloseTo(0.25, 5);
  });

  it('channel targeting a non-mesh (joint-like) node imports but has no primitive mapping', async () => {
    const { gltf, buffers } = makeAnimGltf([{
      path: 'translation',
      node: 1,
      times: [0, 1],
      values: [0,0,0, 1,0,0],
      outType: 'VEC3',
    }]);
    gltf.nodes!.push({ translation: [0, 0, 0] }); // node 1: empty (no mesh)
    gltf.scenes![0]!.nodes!.push(1);
    const { animations, animationTargets } = await gltfToScene(gltf, { buffers });
    expect(animations).toHaveLength(1);
    expect(animations[0]!.channels[0]!.target.node).toBe(animationNodeId(1));
    expect(animationTargets[animationNodeId(1)]).toBeUndefined();
    expect(animationTargets[animationNodeId(0)]).toEqual(['gltf-prim-0']);
  });

  it('unknown interpolation degrades to LINEAR with a warning; bad channels are skipped', async () => {
    const { gltf, buffers } = makeAnimGltf([
      {
        path: 'translation',
        times: [0, 1],
        values: [0,0,0, 1,0,0],
        outType: 'VEC3',
        interpolation: 'BEZIER', // not a glTF interpolation
      },
      {
        path: 'translation',
        node: 99, // nonexistent node
        times: [0, 1],
        values: [0,0,0, 1,0,0],
        outType: 'VEC3',
      },
    ]);
    const { animations, warnings } = await gltfToScene(gltf, { buffers });
    expect(warnings.some(w => w.includes('BEZIER'))).toBe(true);
    expect(warnings.some(w => w.includes('node 99'))).toBe(true);
    expect(animations).toHaveLength(1);
    expect(animations[0]!.channels).toHaveLength(1);
    expect(animations[0]!.channels[0]!.sampler.interpolation).toBe('LINEAR');
  });

  it('result.animations is empty (not undefined) when the glTF has no animations', async () => {
    const { gltf, buffers } = makeModeGltf(4, STRIP_POSITIONS, [0, 1, 2]);
    const { animations } = await gltfToScene(gltf, { buffers });
    expect(animations).toEqual([]);
  });
});
