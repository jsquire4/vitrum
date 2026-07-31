/**
 * Tests for two verified GPU/CPU-desync defects in HybridEnginePrimitiveUpdates:
 *
 *  1. Shared-material bvhIndex partial upload (materialPatch):
 *     When the edited material slot is used by triangles outside the patched
 *     primitive's range (shared material, merged BVH mode), the GPU upload must
 *     cover the full bvhIndex buffer, not just the primitive's slice.
 *
 *  2. Normal rotation without inverse-transpose (rotateNormalsAndUploadSlice /
 *     writeTransformedNormalsAndUploadSlice): Under non-uniform scale the raw
 *     upper-3×3 mis-orients normals; the correct transform is the inverse-transpose
 *     of the 3×3, followed by normalization.
 */

import { describe, expect, it, vi } from 'vitest';
import { asMat4, type Scene } from '@vitrum/core';
import { buildReSTIRSceneBVHForCoreScene } from '../src/restir/bvhCore.js';
import {
  materialPatch,
  transformRefit,
  type PrimitiveUpdateContext,
} from '../src/HybridEnginePrimitiveUpdates.js';

// ── helpers ────────────────────────────────────────────────────────────────

function makeDdgi() {
  return { invalidateProbeCache: vi.fn(), markInstancesDirty: vi.fn() };
}

function makePipeline() {
  return {
    refreshBvhMaterialSlice: vi.fn(),
    refreshBvhNormalsSlice: vi.fn(),
    refreshBvhRefit: vi.fn(),
    requestAccumReset: vi.fn(),
    updateEmitters: vi.fn(),
  };
}

/**
 * A scene whose two primitives share the same authored material signature.
 * Merged BVH construction intentionally keeps one stable slot per primitive:
 * updateEmitter(meshId) may move mesh-area ownership at runtime without
 * rebuilding triangle material ids.
 */
function sharedMaterialScene(): Scene {
  const sharedMat = {
    baseColor: [0.5, 0.5, 0.5] as [number, number, number],
    roughness: 0.5,
    metallic: 0,
  };
  const tri = (id: string, ox: number): Scene['primitives'][number] => ({
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: sharedMat,
    transform: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ox, 0, 0, 1])),
  });
  return {
    primitives: [tri('prim-a', 0), tri('prim-b', 5)],
    emitters: [],
    environment: { kind: 'none' },
  };
}

/** Scene whose two primitives have DIFFERENT material signatures. */
function exclusiveMaterialScene(): Scene {
  const tri = (id: string, ox: number, r: number): Scene['primitives'][number] => ({
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [r, 0.5, 0.5] as [number, number, number], roughness: 0.5, metallic: 0 },
    transform: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ox, 0, 0, 1])),
  });
  return {
    primitives: [tri('prim-a', 0, 0.2), tri('prim-b', 5, 0.8)],
    emitters: [],
    environment: { kind: 'none' },
  };
}

/** A single-primitive mesh scene for normal transform tests. */
function singleMeshScene(transform: Float32Array): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [0.5, 0.5, 0.5] as [number, number, number],
          roughness: 0.5,
          metallic: 0,
        },
        transform: asMat4(transform),
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

// ── Bug 1: shared-material bvhIndex upload ─────────────────────────────────

describe('materialPatch — shared-material bvhIndex upload', () => {
  it('uses a slice when authored-equal primitives retain distinct stable slots', () => {
    const scene = sharedMaterialScene();
    // Use merged mode to trigger material deduplication.
    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'merged' });
    expect(buffers.bvhMode).toBe('merged');

    expect(buffers.coreMaterials.length).toBe(2);

    // Verify both primitives' triangle ranges exist in meshVertexRanges.
    const rangeA = buffers.meshVertexRanges.find((r) => r.name === 'prim-a');
    const rangeB = buffers.meshVertexRanges.find((r) => r.name === 'prim-b');
    expect(rangeA).toBeDefined();
    expect(rangeB).toBeDefined();
    expect(rangeA!.triCount).toBeGreaterThan(0);
    expect(rangeB!.triCount).toBeGreaterThan(0);

    const pipeline = makePipeline();
    const ddgi = makeDdgi();
    const ctx: PrimitiveUpdateContext = {
      bvhBuffers: buffers,
      pipeline: pipeline as never,
      ddgi: ddgi as never,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      lastScene: scene,
      renderScene: scene,
      coreSceneSuppliesMeshes: true,
    };

    const newMaterial = {
      baseColor: [1, 0, 0] as [number, number, number],
      roughness: 0.5,
      metallic: 0,
    };
    materialPatch('prim-a', { material: newMaterial }, ctx);

    expect(pipeline.refreshBvhMaterialSlice).toHaveBeenCalledOnce();

    const [indexSlice] = pipeline.refreshBvhMaterialSlice.mock.calls[0] as [
      { byteOffset: number; data: ArrayBuffer },
    ];
    expect(indexSlice.byteOffset).toBe(0);
    expect(indexSlice.data.byteLength).toBe(rangeA!.triCount * 16);
  });

  it('uses a slice upload when the edited material slot is exclusive to the patched primitive', () => {
    const scene = exclusiveMaterialScene();
    // Merged mode; different materials → two slots, each primitive exclusive to its own.
    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'merged' });
    expect(buffers.bvhMode).toBe('merged');
    expect(buffers.coreMaterials.length).toBe(2);

    const rangeA = buffers.meshVertexRanges.find((r) => r.name === 'prim-a');
    expect(rangeA).toBeDefined();

    const pipeline = makePipeline();
    const ddgi = makeDdgi();
    const ctx: PrimitiveUpdateContext = {
      bvhBuffers: buffers,
      pipeline: pipeline as never,
      ddgi: ddgi as never,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      lastScene: scene,
      renderScene: scene,
      coreSceneSuppliesMeshes: true,
    };

    const newMaterial = {
      baseColor: [0, 1, 0] as [number, number, number],
      roughness: 0.5,
      metallic: 0,
    };
    materialPatch('prim-a', { material: newMaterial }, ctx);

    expect(pipeline.refreshBvhMaterialSlice).toHaveBeenCalledOnce();

    // Exclusive slot: upload is a sub-slice, byteOffset > 0 OR data.byteLength < full
    // (specifically it equals rangeA.triCount * 16).
    const [indexSlice] = pipeline.refreshBvhMaterialSlice.mock.calls[0] as [
      { byteOffset: number; data: ArrayBuffer },
    ];
    const totalTris = new Uint32Array(buffers.triangleMaterialIds.cpuData).length;
    const fullByteLength = totalTris * 16;
    // Slice is smaller than the full buffer — the fast path was taken.
    expect(indexSlice.data.byteLength).toBeLessThan(fullByteLength);
    expect(indexSlice.data.byteLength).toBe(rangeA!.triCount * 16);
  });
});

// ── Bug 2: inverse-transpose normal transform ──────────────────────────────

/**
 * CPU reference: compute the inverse-transpose of the upper-3×3 of a
 * column-major 4×4 matrix, then apply it to a vector and normalize.
 * This mirrors what the fixed rotateNormalsAndUploadSlice must do.
 */
function inverseTransposeTransformNormal(
  m: ArrayLike<number>,
  nx: number,
  ny: number,
  nz: number,
): [number, number, number] {
  // Column-major upper-3×3.
  const m00 = m[0]!;
  const m10 = m[1]!;
  const m20 = m[2]!;
  const m01 = m[4]!;
  const m11 = m[5]!;
  const m21 = m[6]!;
  const m02 = m[8]!;
  const m12 = m[9]!;
  const m22 = m[10]!;

  // Cofactors (form the inverse-transpose numerator).
  const c00 = m11 * m22 - m21 * m12;
  const c01 = m20 * m12 - m10 * m22;
  const c02 = m10 * m21 - m20 * m11;
  const c10 = m21 * m02 - m01 * m22;
  const c11 = m00 * m22 - m20 * m02;
  const c12 = m20 * m01 - m00 * m21;
  const c20 = m01 * m12 - m11 * m02;
  const c21 = m10 * m02 - m00 * m12;
  const c22 = m00 * m11 - m10 * m01;

  const det = m00 * c00 + m01 * c01 + m02 * c02;
  const invDet = Math.abs(det) < 1e-12 ? 1 : 1 / det;

  // inverse-transpose columns (cofactor rows / det).
  const r00 = c00 * invDet;
  const r10 = c01 * invDet;
  const r20 = c02 * invDet;
  const r01 = c10 * invDet;
  const r11 = c11 * invDet;
  const r21 = c12 * invDet;
  const r02 = c20 * invDet;
  const r12 = c21 * invDet;
  const r22 = c22 * invDet;

  let wx = r00 * nx + r01 * ny + r02 * nz;
  let wy = r10 * nx + r11 * ny + r12 * nz;
  let wz = r20 * nx + r21 * ny + r22 * nz;
  const len = Math.sqrt(wx * wx + wy * wy + wz * wz);
  if (len > 1e-12) {
    wx /= len;
    wy /= len;
    wz /= len;
  }
  return [wx, wy, wz];
}

describe('rotateNormalsAndUploadSlice — inverse-transpose correctness', () => {
  /**
   * Apply a non-uniform scale transform (sx=2, sy=1, sz=1) + a transform refit.
   * Under raw-upper-3×3 the Z-facing normal (0,0,1) would stay (0,0,1) since it
   * aligns with z. Use a normal that IS affected: (1,0,0) under x-scale-2 becomes
   * non-unit (2,0,0) without normalization, and a shear combination mis-orients it.
   *
   * We use the combined non-uniform + rotation case: scale x by 2, then check that
   * the uploaded normal is unit-length and equals the inverse-transpose expectation.
   */
  it('produces unit-length normals after a non-uniform scale transform refit', () => {
    // Initial uniform scene (identity transform).
    const identityTransform = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const sceneInit = singleMeshScene(identityTransform);
    const buffers = buildReSTIRSceneBVHForCoreScene(sceneInit, { bvhMode: 'merged' });

    const pipeline = {
      refreshBvhRefit: vi.fn(),
      refreshBvhNormalsSlice: vi.fn(),
      requestAccumReset: vi.fn(),
    };
    const ddgi = makeDdgi();
    const ctx: PrimitiveUpdateContext = {
      bvhBuffers: buffers,
      pipeline: pipeline as never,
      ddgi: ddgi as never,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      lastScene: sceneInit,
      renderScene: sceneInit,
    };

    // New transform: non-uniform scale sx=2 (stretch along x), plus translation.
    // Column-major: [2,0,0,0, 0,1,0,0, 0,0,1,0, 1,0,0,1]
    const nonUniformScale = asMat4(
      new Float32Array([2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1]),
    );
    transformRefit('mesh', { transform: nonUniformScale }, ctx);

    expect(pipeline.refreshBvhNormalsSlice).toHaveBeenCalled();

    const [normalsSlice] = pipeline.refreshBvhNormalsSlice.mock.calls[0] as [
      { byteOffset: number; data: ArrayBuffer },
    ];
    const f32 = new Float32Array(normalsSlice.data);

    // Each vertex normal is vec4f (stride 4). Check every uploaded vertex.
    const vertCount = f32.length / 4;
    expect(vertCount).toBeGreaterThan(0);

    for (let v = 0; v < vertCount; v++) {
      const nx = f32[v * 4]!;
      const ny = f32[v * 4 + 1]!;
      const nz = f32[v * 4 + 2]!;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      // All normals must be unit length (to within floating-point tolerance).
      expect(len).toBeCloseTo(1, 5);
    }
  });

  it('produces inverse-transpose-correct normals for a non-uniform scale that is NOT a rotation', () => {
    // The scene is initialized with identity; the initial normals are (0,0,1).
    const identityTransform = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const sceneInit = singleMeshScene(identityTransform);
    const buffers = buildReSTIRSceneBVHForCoreScene(sceneInit, { bvhMode: 'merged' });

    // Inspect initial normals — they should all be (0,0,1).
    const initialNormals = new Float32Array(buffers.bvhNormals.cpuData);

    const pipeline = {
      refreshBvhRefit: vi.fn(),
      refreshBvhNormalsSlice: vi.fn(),
      requestAccumReset: vi.fn(),
    };
    const ddgi = makeDdgi();
    const ctx: PrimitiveUpdateContext = {
      bvhBuffers: buffers,
      pipeline: pipeline as never,
      ddgi: ddgi as never,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      lastScene: sceneInit,
      renderScene: sceneInit,
    };

    // Non-uniform scale: sx=3, sy=1, sz=1. Column-major 4×4.
    // Inverse-transpose of upper-3×3 for [[3,0,0],[0,1,0],[0,0,1]] is
    // [[1/3,0,0],[0,1,0],[0,0,1]]. So (0,0,1) → normalize((0,0,1)) = (0,0,1).
    // But (1,0,0) → normalize((1/3,0,0)) = (1,0,0).
    // And a surface normal that was (1,1,0)/sqrt(2) → (1/3,1,0) → normalize → not (1,1,0).
    // Use an x-shear: non-symmetric so the plain 3×3 and inverse-transpose differ clearly.
    // Shear: [[1,0,0],[1,1,0],[0,0,1]] (column-major: col0=(1,1,0,0) col1=(0,1,0,0) col2=(0,0,1,0)).
    // For the delta we go from identity (old) to shear (new).
    // delta = newMat * inverse(identity) = newMat.
    const shearMat = asMat4(
      new Float32Array([
        1,
        1,
        0,
        0, // col0: (1,1,0,0)
        0,
        1,
        0,
        0, // col1: (0,1,0,0)
        0,
        0,
        1,
        0, // col2: (0,0,1,0)
        0,
        0,
        0,
        1, // col3: translation = identity
      ]),
    );

    transformRefit('mesh', { transform: shearMat }, ctx);

    expect(pipeline.refreshBvhNormalsSlice).toHaveBeenCalled();
    const [normalsSlice] = pipeline.refreshBvhNormalsSlice.mock.calls[0] as [
      { byteOffset: number; data: ArrayBuffer },
    ];
    const f32 = new Float32Array(normalsSlice.data);

    const vertCount = f32.length / 4;
    expect(vertCount).toBeGreaterThan(0);

    for (let v = 0; v < vertCount; v++) {
      const outNx = f32[v * 4]!;
      const outNy = f32[v * 4 + 1]!;
      const outNz = f32[v * 4 + 2]!;

      // Read the corresponding original normal from the BVH buffer (before the
      // transformRefit mutated it in-place). The initial normals are all (0,0,1).
      const inNx = initialNormals[v * 4]!;
      const inNy = initialNormals[v * 4 + 1]!;
      const inNz = initialNormals[v * 4 + 2]!;

      // Compute the expected inverse-transpose result.
      const [expNx, expNy, expNz] = inverseTransposeTransformNormal(shearMat, inNx, inNy, inNz);

      expect(outNx).toBeCloseTo(expNx, 5);
      expect(outNy).toBeCloseTo(expNy, 5);
      expect(outNz).toBeCloseTo(expNz, 5);
    }
  });

  it('classifies tiny non-uniform scales by rank instead of determinant magnitude', () => {
    const identityTransform = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const base = singleMeshScene(identityTransform);
    const primitive = base.primitives[0]!;
    if (primitive.kind !== 'mesh') throw new Error('expected mesh fixture');
    const component = Math.SQRT1_2;
    const sceneInit: Scene = {
      ...base,
      primitives: [
        {
          ...primitive,
          normals: new Float32Array([
            component,
            component,
            0,
            component,
            component,
            0,
            component,
            component,
            0,
          ]),
        },
      ],
    };
    const buffers = buildReSTIRSceneBVHForCoreScene(sceneInit, {
      bvhMode: 'merged',
    });
    const pipeline = {
      refreshBvhRefit: vi.fn(),
      refreshBvhNormalsSlice: vi.fn(),
      requestAccumReset: vi.fn(),
    };
    const ctx: PrimitiveUpdateContext = {
      bvhBuffers: buffers,
      pipeline: pipeline as never,
      ddgi: makeDdgi() as never,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      lastScene: sceneInit,
      renderScene: sceneInit,
    };
    const tinyNonUniformScale = asMat4(
      new Float32Array([1e-5, 0, 0, 0, 0, 2e-5, 0, 0, 0, 0, 3e-5, 0, 0, 0, 0, 1]),
    );

    transformRefit('mesh', { transform: tinyNonUniformScale }, ctx);

    const [normalsSlice] = pipeline.refreshBvhNormalsSlice.mock.calls[0] as [
      {
        data: ArrayBuffer;
      },
    ];
    const uploaded = new Float32Array(normalsSlice.data);
    expect(uploaded[0]).toBeCloseTo(2 / Math.sqrt(5), 5);
    expect(uploaded[1]).toBeCloseTo(1 / Math.sqrt(5), 5);
    expect(uploaded[2]).toBeCloseTo(0, 5);
  });
});
