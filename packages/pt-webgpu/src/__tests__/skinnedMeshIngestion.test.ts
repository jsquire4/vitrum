/**
 * skinnedMeshIngestion.test.ts — Item 1: solveSkin applied at ingestion.
 *
 * Verifies that:
 * 1. `buildPackedScene` applies CPU LBS and packs SOLVED positions (not rest pose)
 *    when the skinned-mesh carries bone data.
 * 2. A bones-only `updatePrimitive` patch on a skinned-mesh re-solves and routes
 *    through the geometry fast path (positions in the GPU buffer are updated).
 * 3. A skinned-mesh with zero bones (boneCount=0) is packed as rest-pose (no crash).
 *
 * The independent CPU expectation is built by running `solveSkin` directly,
 * mirroring how core's skinSolver tests construct a known-pose fixture.
 */

import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning, Scene, SkinnedMeshPrimitive } from '@vitrum/core';
import { solveSkin } from '@vitrum/core';
import { buildPackedScene, scenePackResultFromPacked } from '../scene/uploadSceneBuffers.js';
import { SceneMutationRouter } from '../sceneMutationRouter.js';
import type { MutationHost } from '../sceneMutationRouter.js';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';

// ─── helpers ──────────────────────────────────────────────────────────────────

/** Column-major identity 4×4. */
function ident4(): Float32Array {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

/** Column-major translate(tx, ty, tz). */
function translate4(tx: number, ty: number, tz: number): Float32Array {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    tx, ty, tz, 1,
  ]);
}

/**
 * Build a single-bone skinned-mesh primitive. All vertices are fully influenced
 * by bone 0 (weight = 1, index = 0). boneInverses = identity.
 */
function makeSkinnedPrim(opts: {
  id?: string;
  positions: Float32Array;
  normals: Float32Array;
  bonesMatrix: Float32Array;
}): SkinnedMeshPrimitive {
  const vCount = opts.positions.length / 3;
  const skinIndices = new Uint32Array(vCount * 4); // all bone 0
  const skinWeights = new Float32Array(vCount * 4);
  for (let i = 0; i < vCount; i++) skinWeights[i * 4] = 1.0;
  return {
    kind: 'skinned-mesh',
    id: opts.id ?? 'skinned',
    positions: opts.positions,
    normals: opts.normals,
    skinIndices,
    skinWeights,
    bones: opts.bonesMatrix,
    boneInverses: ident4(),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
  };
}

function makeScene(prim: SkinnedMeshPrimitive): Scene {
  return {
    primitives: [prim],
    emitters: [],
    environment: { kind: 'none' },
  };
}

// ─── buildPackedScene applies solveSkin ───────────────────────────────────────

describe('buildPackedScene — Item 1: skinned-mesh LBS at ingestion', () => {
  it('packs SOLVED positions (not rest pose) for a translated bone', () => {
    // Triangle at rest pose origin.
    const restPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const restNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const TX = 5, TY = -3, TZ = 2;
    const prim = makeSkinnedPrim({
      positions: restPositions,
      normals: restNormals,
      bonesMatrix: translate4(TX, TY, TZ),
    });

    // Independent CPU expectation via solveSkin directly.
    const expected = solveSkin(prim);

    const scene = makeScene(prim);
    const packed = buildPackedScene(scene, {});

    // packed.positions is vec4f-strided (4 floats per vertex).
    // Vertex 0 is at float indices 0..3 (x,y,z,w).
    expect(packed.positions[0]).toBeCloseTo(expected.positions[0]!, 4);
    expect(packed.positions[1]).toBeCloseTo(expected.positions[1]!, 4);
    expect(packed.positions[2]).toBeCloseTo(expected.positions[2]!, 4);
    // Vertex 1:
    expect(packed.positions[4]).toBeCloseTo(expected.positions[3]!, 4);
    expect(packed.positions[5]).toBeCloseTo(expected.positions[4]!, 4);
    expect(packed.positions[6]).toBeCloseTo(expected.positions[5]!, 4);

    // Confirm differ from rest pose (translation must be visible).
    expect(packed.positions[0]).toBeCloseTo(TX, 4);
    expect(packed.positions[1]).toBeCloseTo(TY, 4);
    expect(packed.positions[2]).toBeCloseTo(TZ, 4);
  });

  it('packs REST-POSE positions when boneCount = 0 (no crash)', () => {
    // A skinned-mesh with empty bones array — solveSkin is skipped, rest pose used.
    const restPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const restNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const vCount = restPositions.length / 3;
    const skinIndices = new Uint32Array(vCount * 4);
    const skinWeights = new Float32Array(vCount * 4);
    for (let i = 0; i < vCount; i++) skinWeights[i * 4] = 1.0;
    const prim: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 'no-bones',
      positions: restPositions,
      normals: restNormals,
      skinIndices,
      skinWeights,
      bones: new Float32Array(0),      // zero bones
      boneInverses: new Float32Array(0),
      material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    };
    const scene = makeScene(prim);
    // Must not throw.
    const packed = buildPackedScene(scene, {});
    // Should equal rest pose.
    expect(packed.positions[0]).toBeCloseTo(0, 4);
    expect(packed.positions[1]).toBeCloseTo(0, 4);
    expect(packed.positions[2]).toBeCloseTo(0, 4);
    expect(packed.positions[4]).toBeCloseTo(1, 4);
  });

  it('routes initial solveSkin failures through structured warnings when provided', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const restPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
      const restNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
      const prim: SkinnedMeshPrimitive = {
        ...makeSkinnedPrim({
          id: 'bad-bone-inverses',
          positions: restPositions,
          normals: restNormals,
          bonesMatrix: ident4(),
        }),
        boneInverses: new Float32Array(0),
      };
      const warnings: EngineWarning[] = [];

      const packed = buildPackedScene(makeScene(prim), {
        onWarning: (warning) => warnings.push(warning),
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      });

      expect(packed.positions[0]).toBeCloseTo(0, 4);
      expect(packed.positions[4]).toBeCloseTo(1, 4);
      expect(warn).not.toHaveBeenCalled();
      expect(warnings).toHaveLength(1);
      expect(warnings[0]!).toMatchObject({
        code: 'pt-webgpu.set-scene-skin-fallback',
        backend: 'pt-webgpu',
        phase: 'setScene',
        method: 'setScene',
        details: {
          primitiveId: 'bad-bone-inverses',
          fallback: 'rest-pose',
        },
      });
      expect(warnings[0]!.message).toContain('using rest pose');
      expect(String(warnings[0]!.raw)).toContain('boneInverses length');
    } finally {
      warn.mockRestore();
    }
  });

  it('morphTargets are applied before LBS when morphWeights are non-zero', () => {
    // Rest triangle in XY-plane; morph target shifts vertex 0 +2 on X.
    const restPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const restNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const restTangents = new Float32Array([
      1, 0, 0, 1,
      1, 0, 0, 1,
      1, 0, 0, 1,
    ]);
    // Morph delta: only vertex 0 shifts by (2, 0, 0).
    const morphDelta = new Float32Array([2, 0, 0, 0, 0, 0, 0, 0, 0]);
    const morphNormalDelta = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const morphTangentDelta = new Float32Array([0, 1, 0, 0, 0, 0, 0, 0, 0]);
    const vCount = restPositions.length / 3;
    const skinIndices = new Uint32Array(vCount * 4);
    const skinWeights = new Float32Array(vCount * 4);
    for (let i = 0; i < vCount; i++) skinWeights[i * 4] = 1.0;
    const prim: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 'morph-test',
      positions: restPositions,
      normals: restNormals,
      skinIndices,
      skinWeights,
      bones: ident4(),
      boneInverses: ident4(),
      tangents: restTangents,
      morphTargets: [morphDelta],
      morphTargetNormals: [morphNormalDelta],
      morphTargetTangents: [morphTangentDelta],
      morphWeights: new Float32Array([1.0]),
      material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    };

    // Independent expectation.
    const expected = solveSkin(prim);

    const scene = makeScene(prim);
    const packed = buildPackedScene(scene, {});

    // Vertex 0 should be at (2, 0, 0) = rest (0,0,0) + morph-delta (2,0,0).
    expect(packed.positions[0]).toBeCloseTo(expected.positions[0]!, 4); // 2
    expect(packed.positions[1]).toBeCloseTo(expected.positions[1]!, 4); // 0
    expect(packed.positions[2]).toBeCloseTo(expected.positions[2]!, 4); // 0
    expect(expected.tangents).toBeDefined();
    expect(packed.tangents[0]).toBeCloseTo(expected.tangents![0]!, 4);
    expect(packed.tangents[1]).toBeCloseTo(expected.tangents![1]!, 4);
    expect(packed.tangents[2]).toBeCloseTo(expected.tangents![2]!, 4);
    expect(packed.tangents[3]).toBeCloseTo(1, 4);
    expect(packed.tangents[1]).toBeGreaterThan(0);
  });
});

// ─── bones patch: updatePrimitive re-solves and routes geometry fast path ─────

describe('SceneMutationRouter — Item 1: bones patch re-solves skin', () => {
  function makeHostWithSkinnedScene(scene: Scene): {
    host: MutationHost;
    sceneRef: { current: Scene };
    positionsWriteCalls: Float32Array[];
    tangentsWriteCalls: Float32Array[];
  } {
    const packed = buildPackedScene(scene, {});
    const geoPack = scenePackResultFromPacked(packed);

    const positionsWriteCalls: Float32Array[] = [];
    const tangentsWriteCalls: Float32Array[] = [];

    const positionsBuffer = {
      size: Math.max(16, packed.positions.byteLength),
      destroy: vi.fn(),
    } as unknown as GPUBuffer;
    const tangentsBuffer = {
      size: Math.max(16, packed.tangents.byteLength),
      destroy: vi.fn(),
    } as unknown as GPUBuffer;

    const sceneBuffers: UploadedSceneBuffers = {
      ...packed,
      positionsBuffer,
      normalsBuffer: { size: Math.max(16, packed.normals.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      indicesBuffer: { size: Math.max(16, packed.indices.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      triMaterialIdsBuffer: { size: Math.max(16, packed.triMaterialIds.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      materialsBuffer: { size: Math.max(16, packed.materials.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      bvhNodesBuffer: { size: Math.max(16, packed.bvhNodes.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      analyticHeadersBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      analyticParamsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      analyticLocalToWorldBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      analyticWorldToLocalBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      environmentMapTexelsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      environmentMapCdfBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      pointLightsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      spotLightsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      rectAreaLightsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      meshAreaLightsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      tlasNodesBuffer: { size: Math.max(16, packed.tlasNodes.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      tlasInstanceIndicesBuffer: { size: Math.max(16, packed.tlasInstanceIndices.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      tlasBlasRootsBuffer: { size: Math.max(16, packed.tlasBlasRoots.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      tlasInstanceWorldToLocalBuffer: { size: Math.max(16, packed.tlasInstanceWorldToLocal.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      tlasInstanceLocalToWorldBuffer: { size: Math.max(16, packed.tlasInstanceLocalToWorld.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      lightTreeBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      uvsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      tangentsBuffer,
      colorsBuffer: { size: Math.max(16, packed.colors.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      materialTexDescriptorsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      materialTexture: {} as GPUTexture,
      materialTextureView: {} as GPUTextureView,
      materialTextureSampler: {} as GPUSampler,
      materialLinearTexture: {} as GPUTexture,
      materialLinearTextureView: {} as GPUTextureView,
      bvhNodeCount: Math.floor(packed.bvhNodes.length / 8),
      tlasNodeCount: Math.floor(packed.tlasNodes.length / 8),
      materialCount: 1,
      gpuMemoryBytes: () => ({ bufferBytes: 0, textureBytesByFormat: {} }),
      destroy: vi.fn(),
    } as unknown as UploadedSceneBuffers;

    const sceneRef = { current: scene };

    const host: MutationHost = {
      device: {
        queue: {
          writeBuffer: vi.fn((buf: unknown, _byteOffset: number, data: ArrayBuffer, srcOffset: number, length: number) => {
            if (buf === positionsBuffer) {
              positionsWriteCalls.push(new Float32Array(data, srcOffset, Math.floor(length / 4)));
            }
            if (buf === tangentsBuffer) {
              tangentsWriteCalls.push(new Float32Array(data, srcOffset, Math.floor(length / 4)));
            }
          }),
        },
      } as unknown as GPUDevice,
      assertLive: vi.fn(),
      getScene: () => sceneRef.current,
      setSceneState: vi.fn((s: Scene) => { sceneRef.current = s; }),
      getSceneBuffers: () => sceneBuffers,
      getGeoPack: () => geoPack,
      setGeoPack: vi.fn(),
      invalidateBindGroups: vi.fn(),
      supportedAnalyticShapes: () => new Set<string>(),
      cameraVisibleEmitters: () => false,
      repackScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
      setScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
      reset: vi.fn(),
    };

    return { host, sceneRef, positionsWriteCalls, tangentsWriteCalls };
  }

  it('bones-only patch re-solves skin and writes solved positions to GPU', () => {
    const restPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const restNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    // Start with identity bone (rest pose).
    const prim = makeSkinnedPrim({
      positions: restPositions,
      normals: restNormals,
      bonesMatrix: ident4(),
    });
    const scene = makeScene(prim);
    const { host, positionsWriteCalls } = makeHostWithSkinnedScene(scene);

    const router = new SceneMutationRouter(host);

    // Patch the bones to translate by (7, 0, 0).
    const newBones = translate4(7, 0, 0);
    router.updatePrimitive('skinned', { bones: newBones });

    // At least one write to the positions buffer must have occurred
    // (the geometry fast path re-uploads positions after re-solving).
    expect(positionsWriteCalls.length).toBeGreaterThan(0);

    // The written positions should be near (7, 0, 0), (8, 0, 0), (7, 1, 0)
    // in vec4f stride (4 floats per vertex).
    const written = positionsWriteCalls[positionsWriteCalls.length - 1]!;
    // Vertex 0 x-coordinate is at index 0 in the packed vec4f stream.
    expect(written[0]).toBeCloseTo(7, 3);
    expect(written[1]).toBeCloseTo(0, 3);
    expect(written[2]).toBeCloseTo(0, 3);
  });

  it('bones patch preserves bone matrices in scene state for future solves', () => {
    const restPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const restNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const prim = makeSkinnedPrim({
      positions: restPositions,
      normals: restNormals,
      bonesMatrix: ident4(),
    });
    const scene = makeScene(prim);
    const { host, sceneRef } = makeHostWithSkinnedScene(scene);

    const router = new SceneMutationRouter(host);
    const newBones = translate4(3, 0, 0);
    router.updatePrimitive('skinned', { bones: newBones });

    // The scene state should have the updated bones.
    const updatedPrim = sceneRef.current.primitives.find((p) => p.id === 'skinned');
    expect(updatedPrim?.kind).toBe('skinned-mesh');
    if (updatedPrim?.kind === 'skinned-mesh') {
      expect(updatedPrim.bones[12]).toBeCloseTo(3, 4); // tx column
    }
  });

  it('morphWeights patch re-solves morph tangent deltas into the tangent GPU buffer', () => {
    const restPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const restNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const restTangents = new Float32Array([
      1, 0, 0, 1,
      1, 0, 0, 1,
      1, 0, 0, 1,
    ]);
    const prim: SkinnedMeshPrimitive = {
      ...makeSkinnedPrim({
        positions: restPositions,
        normals: restNormals,
        bonesMatrix: ident4(),
      }),
      tangents: restTangents,
      morphTargets: [new Float32Array(restPositions.length)],
      morphTargetTangents: [new Float32Array([0, 1, 0, 0, 0, 0, 0, 0, 0])],
      morphWeights: new Float32Array([0]),
    };
    const scene = makeScene(prim);
    const { host, tangentsWriteCalls } = makeHostWithSkinnedScene(scene);

    const router = new SceneMutationRouter(host);
    router.updatePrimitive('skinned', { morphWeights: new Float32Array([1]) });

    expect(tangentsWriteCalls.length).toBeGreaterThan(0);
    const written = tangentsWriteCalls[tangentsWriteCalls.length - 1]!;
    const invSqrt2 = 1 / Math.sqrt(2);
    expect(written[0]).toBeCloseTo(invSqrt2, 4);
    expect(written[1]).toBeCloseTo(invSqrt2, 4);
    expect(written[2]).toBeCloseTo(0, 4);
    expect(written[3]).toBeCloseTo(1, 4);
  });
});
