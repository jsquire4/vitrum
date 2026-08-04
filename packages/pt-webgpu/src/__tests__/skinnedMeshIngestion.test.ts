/**
 * skinnedMeshIngestion.test.ts — Item 1: solveSkin applied at ingestion.
 *
 * Verifies that:
 * 1. `buildPackedScene` applies CPU LBS and packs SOLVED positions (not rest pose)
 *    when the skinned-mesh carries bone data.
 * 2. A bones-only `updatePrimitive` patch on a skinned-mesh re-solves and routes
 *    through the geometry fast path (positions in the GPU buffer are updated).
 * 3. Morph and arbitrary UV-set changes are solved and reach the GPU geometry path.
 *
 * The independent CPU expectation is built by running `solveSkin` directly,
 * mirroring how core's skinSolver tests construct a known-pose fixture.
 */

import { describe, expect, it, vi } from 'vitest';
import type { Scene, SkinnedMeshPrimitive } from '@vitrum/core';
import { solveSkin } from '@vitrum/core';
import * as core from '@vitrum/core';
import { buildPackedScene, scenePackResultFromPacked } from '../scene/uploadSceneBuffers.js';
import { SceneMutationRouter } from '../sceneMutationRouter.js';
import type { MutationHost } from '../sceneMutationRouter.js';
import { installGpuConstStubs } from './gpuStub.js';
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
/** Column-major scale(sx, sy, sz). */
function scale4(sx: number, sy: number, sz: number): Float32Array {
  return new Float32Array([
    sx, 0, 0, 0,
    0, sy, 0, 0,
    0, 0, sz, 0,
    0, 0, 0, 1,
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

  it('rejects an empty skinned-mesh bone palette at the canonical scene boundary', () => {
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
    expect(() => buildPackedScene(scene, {})).toThrow(/bones must not be empty/);
  });
  it('rejects morph-only skinned primitives without a bone palette', () => {
    const influenceWidth = 8;
    const skinIndices = new Uint32Array(3 * influenceWidth);
    const skinWeights = new Float32Array(3 * influenceWidth);
    for (let vertex = 0; vertex < 3; vertex += 1) {
      skinWeights[vertex * influenceWidth] = 1;
    }
    const prim: SkinnedMeshPrimitive = {
      ...makeSkinnedPrim({
        id: 'morph-only',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        bonesMatrix: ident4(),
      }),
      bones: new Float32Array(0),
      boneInverses: new Float32Array(0),
      skinIndices,
      skinWeights,
      skinInfluencesPerVertex: influenceWidth,
      uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
      morphTargets: [
        new Float32Array([2, 0, 0, 0, 0, 0, 0, 0, 0]),
      ],
      morphTargetUvs: [
        new Float32Array([0.25, 0.125, 0, 0, 0, 0]),
      ],
      morphWeights: new Float32Array([1]),
    };

    expect(() => buildPackedScene(makeScene(prim), {})).toThrow(
      /bones must not be empty/,
    );
  });


  it('rejects initial solveSkin failures instead of uploading a rest-pose fallback', () => {
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
      expect(() => buildPackedScene(makeScene(prim), {
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      })).toThrow(/solveSkin failed for primitive "bad-bone-inverses".*scene upload was rejected.*boneInverses length/);
      expect(warn).not.toHaveBeenCalled();
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
    const morphUvDelta = new Float32Array([0.2, 0.1, 0, 0, 0, 0]);
    const morphUv1Delta = new Float32Array([0, 0.25, 0, 0, 0, 0]);
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
      uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
      uv1: new Float32Array([0.5, 0.5, 1.5, 0.5, 0.5, 1.5]),
      tangents: restTangents,
      morphTargets: [morphDelta],
      morphTargetNormals: [morphNormalDelta],
      morphTargetTangents: [morphTangentDelta],
      morphTargetUvs: [morphUvDelta],
      morphTargetUv1s: [morphUv1Delta],
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
    expect(expected.uvs).toBeDefined();
    expect(expected.uv1).toBeDefined();
    expect(packed.uvs[0]).toBeCloseTo(expected.uvs![0]!, 4);
    expect(packed.uvs[1]).toBeCloseTo(expected.uvs![1]!, 4);
    expect(packed.uvs[2]).toBeCloseTo(expected.uv1![0]!, 4);
    expect(packed.uvs[3]).toBeCloseTo(expected.uv1![1]!, 4);
  });
});

// ─── bones patch: updatePrimitive re-solves and routes geometry fast path ─────

describe('SceneMutationRouter — Item 1: bones patch re-solves skin', () => {
  installGpuConstStubs();
  function makeHostWithSkinnedScene(scene: Scene): {
    host: MutationHost;
    sceneRef: { current: Scene };
    sceneBuffers: UploadedSceneBuffers;
    positionsWriteCalls: Float32Array[];
    tangentsWriteCalls: Float32Array[];
    normalsWriteCalls: Float32Array[];
    uvsWriteCalls: Float32Array[];
  } {
    const packed = buildPackedScene(scene, {});
    const geoPack = scenePackResultFromPacked(packed);

    const positionsWriteCalls: Float32Array[] = [];
    const tangentsWriteCalls: Float32Array[] = [];
    const normalsWriteCalls: Float32Array[] = [];
    const uvsWriteCalls: Float32Array[] = [];

    const testBuffer = (label: string, byteLength: number): GPUBuffer => ({
      label,
      size: Math.max(16, byteLength),
      destroy: vi.fn(),
    } as unknown as GPUBuffer);
    const positionsBuffer = testBuffer(
      'vitrum.pt-webgpu.scene.positions',
      packed.positions.byteLength,
    );
    const normalsBuffer = testBuffer(
      'vitrum.pt-webgpu.scene.normals',
      packed.normals.byteLength,
    );
    const tangentsBuffer = testBuffer(
      'vitrum.pt-webgpu.scene.tangents',
      packed.tangents.byteLength,
    );
    const uvsBuffer = testBuffer(
      'vitrum.pt-webgpu.scene.uvs',
      packed.uvs.byteLength,
    );

    const sceneBuffers: UploadedSceneBuffers = {
      ...packed,
      positionsBuffer,
      normalsBuffer,
      indicesBuffer: { size: Math.max(16, packed.indices.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      triMaterialIdsBuffer: { size: Math.max(16, packed.triMaterialIds.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      materialsBuffer: { size: Math.max(16, packed.materials.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      bvhNodesBuffer: { size: Math.max(16, packed.bvhNodes.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      cwbvhNodeBoundsBuffer: testBuffer('cwbvhNodeBounds', packed.cwbvhNodeBounds.byteLength),
      cwbvhChildBoundsPackedBuffer: testBuffer('cwbvhChildBoundsPacked', packed.cwbvhChildBoundsPacked.byteLength),
      cwbvhChildMetaBuffer: testBuffer('cwbvhChildMeta', packed.cwbvhChildMeta.byteLength),
      cwbvhChildCountBuffer: testBuffer('cwbvhChildCount', packed.cwbvhChildCount.byteLength),
      cwbvhTlasBlasRootsBuffer: testBuffer('cwbvhTlasBlasRoots', packed.cwbvhTlasBlasRoots.byteLength),
      analyticHeadersBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      analyticParamsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      analyticLocalToWorldBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      analyticWorldToLocalBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      environmentMapTexelsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      environmentMapCdfBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      directionalLightsBuffer: testBuffer('directionalLights', packed.directionalLightsData.byteLength),
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
      uvsBuffer,
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

    const bytesOf = (data: ArrayBufferView): Uint8Array => new Uint8Array(
      data.buffer,
      data.byteOffset,
      data.byteLength,
    ).slice();
    const gpuBytes = new Map<unknown, Uint8Array>([
      [positionsBuffer, bytesOf(packed.positions)],
      [normalsBuffer, bytesOf(packed.normals)],
      [tangentsBuffer, bytesOf(packed.tangents)],
      [uvsBuffer, bytesOf(packed.uvs)],
    ]);
    const writeBuffer = vi.fn((
      buffer: unknown,
      byteOffset: number,
      data: ArrayBuffer,
      sourceOffset: number,
      length: number,
    ) => {
      const previous = gpuBytes.get(buffer)
        ?? new Uint8Array(Number((buffer as { size?: number }).size ?? length));
      previous.set(new Uint8Array(data, sourceOffset, length), byteOffset);
      gpuBytes.set(buffer, previous);
    });
    const copyBufferToBuffer = vi.fn((
      source: unknown,
      sourceOffset: number,
      destination: unknown,
      destinationOffset: number,
      length: number,
    ) => {
      const sourceBytes = gpuBytes.get(source);
      if (sourceBytes == null) throw new Error('test source buffer was not uploaded');
      const destinationBytes = gpuBytes.get(destination)
        ?? new Uint8Array(Number((destination as { size?: number }).size ?? 0));
      destinationBytes.set(
        sourceBytes.subarray(sourceOffset, sourceOffset + length),
        destinationOffset,
      );
      gpuBytes.set(destination, destinationBytes);
      const snapshot = new Float32Array(destinationBytes.slice().buffer);
      if (destination === positionsBuffer) positionsWriteCalls.push(snapshot);
      if (destination === normalsBuffer) normalsWriteCalls.push(snapshot);
      if (destination === tangentsBuffer) tangentsWriteCalls.push(snapshot);
      if (destination === uvsBuffer) uvsWriteCalls.push(snapshot);
    });

    const host: MutationHost = {
      device: {
        createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
          label: desc.label,
          size: Number(desc.size),
          destroy: vi.fn(),
        })),
        createCommandEncoder: vi.fn(() => ({
          copyBufferToBuffer,
          finish: vi.fn(() => ({})),
        })),
        queue: { writeBuffer, submit: vi.fn() },
      } as unknown as GPUDevice,
      assertLive: vi.fn(),
      validatePrimitiveCandidate: vi.fn(),
      validateEmitterCandidate: vi.fn(),
      validateEnvironmentCandidate: vi.fn(),
      validateEmittersCandidate: vi.fn(),
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

    return {
      host,
      sceneRef,
      sceneBuffers,
      positionsWriteCalls,
      normalsWriteCalls,
      tangentsWriteCalls,
      uvsWriteCalls,
    };
  }

  it('does not solve skin for a non-emissive material-only patch', () => {
    const prim = makeSkinnedPrim({
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      bonesMatrix: ident4(),
    });
    const { host, positionsWriteCalls } = makeHostWithSkinnedScene(makeScene(prim));
    const solve = vi.spyOn(core, 'solveSkin');
    solve.mockClear();

    try {
      new SceneMutationRouter(host).updatePrimitive('skinned', {
        material: {
          baseColor: [0.5, 0.5, 0.5],
          roughness: 0.25,
          metallic: 0,
        },
      });

      expect(solve).not.toHaveBeenCalled();
      expect(positionsWriteCalls).toHaveLength(0);
    } finally {
      solve.mockRestore();
    }
  });

  it('uses posed geometry when a material patch creates an implicit mesh emitter', () => {
    const restPositions = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]);
    const prim = makeSkinnedPrim({
      positions: restPositions,
      normals: new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ]),
      bonesMatrix: translate4(5, 0, 0),
    });
    const { host, sceneRef, sceneBuffers } =
      makeHostWithSkinnedScene(makeScene(prim));

    new SceneMutationRouter(host).updatePrimitive('skinned', {
      material: {
        ...prim.material,
        emissive: [1, 0.5, 0.25],
        emissiveIntensity: 2,
      },
    });

    expect(sceneBuffers.meshAreaLightCount).toBe(1);
    expect(sceneBuffers.meshAreaLightsData).toHaveLength(28);
    // Mesh-area record vertices live at vec4 offsets 0, 4, and 8. The posed
    // triangle is translated +5 in X; rest-pose packing would be [0, 1, 0].
    expect(sceneBuffers.meshAreaLightsData[0]).toBeCloseTo(5, 5);
    expect(sceneBuffers.meshAreaLightsData[4]).toBeCloseTo(6, 5);
    expect(sceneBuffers.meshAreaLightsData[8]).toBeCloseTo(5, 5);

    // The host snapshot remains authored/rest-pose data for future skin solves.
    const stored = sceneRef.current.primitives[0];
    expect(stored?.kind).toBe('skinned-mesh');
    if (stored?.kind === 'skinned-mesh') {
      expect(stored.positions).toBe(restPositions);
      expect(stored.positions[0]).toBe(0);
    }
  });

  it('keeps every other posed skinned emitter solved during a global emitter rebuild', () => {
    const positions = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]);
    const normals = new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]);
    const patched = makeSkinnedPrim({
      id: 'patched',
      positions,
      normals,
      bonesMatrix: translate4(5, 0, 0),
    });
    const other = {
      ...makeSkinnedPrim({
        id: 'other',
        positions,
        normals,
        bonesMatrix: translate4(20, 0, 0),
      }),
      material: {
        baseColor: [0.5, 0.5, 0.5] as const,
        roughness: 0.5,
        metallic: 0,
        emissive: [0.25, 0.5, 1] as const,
        emissiveIntensity: 2,
      },
    };
    const scene: Scene = {
      primitives: [patched, other],
      emitters: [],
      environment: { kind: 'none' },
    };
    const { host, sceneBuffers } = makeHostWithSkinnedScene(scene);

    new SceneMutationRouter(host).updatePrimitive('patched', {
      material: {
        ...patched.material,
        emissive: [1, 0, 0],
        emissiveIntensity: 1,
      },
    });

    expect(sceneBuffers.meshAreaLightCount).toBe(2);
    expect(sceneBuffers.meshAreaLightsData).toHaveLength(56);
    expect(sceneBuffers.meshAreaLightsData[0]).toBeCloseTo(5, 5);
    // Second triangle begins at float 28 and must retain its +20 posed position.
    expect(sceneBuffers.meshAreaLightsData[28]).toBeCloseTo(20, 5);
    expect(sceneBuffers.meshAreaLightsData[32]).toBeCloseTo(21, 5);
    expect(sceneBuffers.meshAreaLightsData[36]).toBeCloseTo(20, 5);
  });

  it('rebuilds implicit emitter state when extensions.skipEmitter toggles', () => {
    const prim = {
      ...makeSkinnedPrim({
        positions: new Float32Array([
          0, 0, 0,
          1, 0, 0,
          0, 1, 0,
        ]),
        normals: new Float32Array([
          0, 0, 1,
          0, 0, 1,
          0, 0, 1,
        ]),
        bonesMatrix: translate4(7, 0, 0),
      }),
      material: {
        baseColor: [0.5, 0.5, 0.5] as const,
        roughness: 0.5,
        metallic: 0,
        emissive: [1, 1, 1] as const,
        emissiveIntensity: 1,
      },
    };
    const { host, sceneBuffers } =
      makeHostWithSkinnedScene(makeScene(prim));
    const router = new SceneMutationRouter(host);
    expect(sceneBuffers.meshAreaLightCount).toBe(1);

    router.updatePrimitive('skinned', {
      material: {
        ...prim.material,
        extensions: { skipEmitter: true },
      },
    });
    expect(sceneBuffers.meshAreaLightCount).toBe(0);
    expect(sceneBuffers.meshAreaLightsData).toHaveLength(0);

    router.updatePrimitive('skinned', {
      material: {
        ...prim.material,
        extensions: { skipEmitter: false },
      },
    });
    expect(sceneBuffers.meshAreaLightCount).toBe(1);
    expect(sceneBuffers.meshAreaLightsData).toHaveLength(28);
    expect(sceneBuffers.meshAreaLightsData[0]).toBeCloseTo(7, 5);
    expect(sceneBuffers.meshAreaLightsData[4]).toBeCloseTo(8, 5);
    expect(sceneBuffers.meshAreaLightsData[8]).toBeCloseTo(7, 5);
  });

  it('re-solves when skin weights or bind matrices change', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const vertexCount = positions.length / 3;
    const skinIndices = new Uint32Array(vertexCount * 4);
    const initialWeights = new Float32Array(vertexCount * 4);
    for (let vertex = 0; vertex < vertexCount; vertex += 1) {
      skinIndices[vertex * 4 + 1] = 1;
      initialWeights[vertex * 4] = 1;
    }
    const bones = new Float32Array(32);
    bones.set(ident4(), 0);
    bones.set(translate4(4, 0, 0), 16);
    const boneInverses = new Float32Array(32);
    boneInverses.set(ident4(), 0);
    boneInverses.set(ident4(), 16);
    const weightedPrimitive: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 'skinned',
      positions,
      normals,
      skinIndices,
      skinWeights: initialWeights,
      bones,
      boneInverses,
      material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    };
    const weightedHost = makeHostWithSkinnedScene(makeScene(weightedPrimitive));
    const nextWeights = new Float32Array(vertexCount * 4);
    for (let vertex = 0; vertex < vertexCount; vertex += 1) {
      nextWeights[vertex * 4 + 1] = 1;
    }

    new SceneMutationRouter(weightedHost.host).updatePrimitive('skinned', {
      skinWeights: nextWeights,
    });
    expect(weightedHost.positionsWriteCalls.at(-1)?.[0]).toBeCloseTo(4, 4);

    const bindPrimitive: SkinnedMeshPrimitive = {
      ...makeSkinnedPrim({
        positions,
        normals,
        bonesMatrix: scale4(2, 1, 1),
      }),
      bindMatrix: ident4(),
      bindMatrixInverse: ident4(),
    };
    const bindHost = makeHostWithSkinnedScene(makeScene(bindPrimitive));
    new SceneMutationRouter(bindHost.host).updatePrimitive('skinned', {
      bindMatrix: translate4(1, 0, 0),
      bindMatrixInverse: translate4(-1, 0, 0),
    });
    expect(bindHost.positionsWriteCalls.at(-1)?.[0]).toBeCloseTo(1, 4);
  });

  it('re-solves base and morph UV changes under active morph weights', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const prim: SkinnedMeshPrimitive = {
      ...makeSkinnedPrim({
        positions,
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        bonesMatrix: ident4(),
      }),
      uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
      morphTargets: [new Float32Array(positions.length)],
      morphTargetUvs: [
        new Float32Array([0.25, 0.125, 0, 0, 0, 0]),
      ],
      morphWeights: new Float32Array([1]),
    };
    const { host, sceneRef, uvsWriteCalls } =
      makeHostWithSkinnedScene(makeScene(prim));
    const router = new SceneMutationRouter(host);
    const nextBaseUvs = new Float32Array([0.5, 0.25, 1, 0, 0, 1]);

    router.updatePrimitive('skinned', { uvs: nextBaseUvs });
    expect(uvsWriteCalls.at(-1)?.[0]).toBeCloseTo(0.75, 4);
    expect(uvsWriteCalls.at(-1)?.[1]).toBeCloseTo(0.375, 4);

    const nextMorphUvs = [
      new Float32Array([0.5, -0.125, 0, 0, 0, 0]),
    ];
    router.updatePrimitive('skinned', { morphTargetUvs: nextMorphUvs });
    expect(uvsWriteCalls.at(-1)?.[0]).toBeCloseTo(1, 4);
    expect(uvsWriteCalls.at(-1)?.[1]).toBeCloseTo(0.125, 4);

    const stored = sceneRef.current.primitives[0];
    expect(stored?.kind).toBe('skinned-mesh');
    if (stored?.kind === 'skinned-mesh') {
      expect(stored.uvs).toBe(nextBaseUvs);
      expect(stored.morphTargetUvs).toBe(nextMorphUvs);
    }
  });

  it('applies a morph-weight patch on a valid one-bone primitive', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const prim: SkinnedMeshPrimitive = {
      ...makeSkinnedPrim({
        positions,
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        bonesMatrix: ident4(),
      }),
      uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
      morphTargets: [
        new Float32Array([3, 0, 0, 0, 0, 0, 0, 0, 0]),
      ],
      morphTargetUvs: [
        new Float32Array([0.5, 0.25, 0, 0, 0, 0]),
      ],
      morphWeights: new Float32Array([0]),
    };
    const { host, positionsWriteCalls, uvsWriteCalls } =
      makeHostWithSkinnedScene(makeScene(prim));

    new SceneMutationRouter(host).updatePrimitive('skinned', {
      morphWeights: new Float32Array([1]),
    });

    expect(positionsWriteCalls.at(-1)?.[0]).toBeCloseTo(3, 4);
    expect(uvsWriteCalls.at(-1)?.[0]).toBeCloseTo(0.5, 4);
    expect(uvsWriteCalls.at(-1)?.[1]).toBeCloseTo(0.25, 4);
  });

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

  it('repeated bone patches always solve from authored rest pose instead of compounding prior poses', () => {
    const restPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const restNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const prim = makeSkinnedPrim({
      positions: restPositions,
      normals: restNormals,
      bonesMatrix: ident4(),
    });
    const { host, sceneRef, positionsWriteCalls } = makeHostWithSkinnedScene(makeScene(prim));
    const router = new SceneMutationRouter(host);

    router.updatePrimitive('skinned', { bones: translate4(3, 0, 0) });
    expect(positionsWriteCalls.at(-1)?.[0]).toBeCloseTo(3, 4);
    router.updatePrimitive('skinned', { bones: translate4(7, 0, 0) });

    // A transient-pose scene would produce 10 here (3 + 7). The authored
    // scene must retain x=0 and the new solve must produce exactly x=7.
    expect(positionsWriteCalls.at(-1)?.[0]).toBeCloseTo(7, 4);
    const stored = sceneRef.current.primitives[0];
    expect(stored?.kind).toBe('skinned-mesh');
    if (stored?.kind === 'skinned-mesh') {
      expect(stored.positions).toBe(restPositions);
      expect(stored.positions[0]).toBeCloseTo(0, 4);
      expect(stored.bones[12]).toBeCloseTo(7, 4);
    }
  });

  it('treats explicit positions and normals patches as new authored rest data for later poses', () => {
    const restPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const restNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const prim = makeSkinnedPrim({
      positions: restPositions,
      normals: restNormals,
      bonesMatrix: translate4(2, 0, 0),
    });
    const {
      host,
      sceneRef,
      positionsWriteCalls,
      normalsWriteCalls,
    } = makeHostWithSkinnedScene(makeScene(prim));
    const router = new SceneMutationRouter(host);
    const editedRestPositions = new Float32Array([
      5, 0, 0,
      6, 0, 0,
      5, 1, 0,
    ]);
    const editedRestNormals = new Float32Array([
      0, 1, 0,
      0, 1, 0,
      0, 1, 0,
    ]);

    router.updatePrimitive('skinned', {
      positions: editedRestPositions,
      normals: editedRestNormals,
    });
    expect(positionsWriteCalls.at(-1)?.[0]).toBeCloseTo(7, 4);
    expect(normalsWriteCalls.at(-1)?.[0]).toBeCloseTo(0, 4);
    expect(normalsWriteCalls.at(-1)?.[1]).toBeCloseTo(1, 4);
    let stored = sceneRef.current.primitives[0];
    expect(stored?.kind).toBe('skinned-mesh');
    if (stored?.kind === 'skinned-mesh') {
      expect(stored.positions).toBe(editedRestPositions);
      expect(stored.normals).toBe(editedRestNormals);
      expect(stored.positions[0]).toBeCloseTo(5, 4);
    }

    router.updatePrimitive('skinned', { bones: translate4(3, 0, 0) });
    expect(positionsWriteCalls.at(-1)?.[0]).toBeCloseTo(8, 4);
    expect(normalsWriteCalls.at(-1)?.[1]).toBeCloseTo(1, 4);
    stored = sceneRef.current.primitives[0];
    if (stored?.kind === 'skinned-mesh') {
      expect(stored.positions).toBe(editedRestPositions);
      expect(stored.normals).toBe(editedRestNormals);
      expect(stored.bones[12]).toBeCloseTo(3, 4);
    }
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

  it('morphWeights patch re-solves tangent and UV deltas into the geometry buffers', () => {
    const restPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const restNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const restTangents = new Float32Array([
      1, 0, 0, 1,
      1, 0, 0, 1,
      1, 0, 0, 1,
    ]);
    const restUvs = new Float32Array([0, 0, 1, 0, 0, 1]);
    const restUv1 = new Float32Array([0.5, 0.5, 1.5, 0.5, 0.5, 1.5]);
    const prim: SkinnedMeshPrimitive = {
      ...makeSkinnedPrim({
        positions: restPositions,
        normals: restNormals,
        bonesMatrix: ident4(),
      }),
      tangents: restTangents,
      uvs: restUvs,
      uv1: restUv1,
      morphTargets: [new Float32Array(restPositions.length)],
      morphTargetTangents: [new Float32Array([0, 1, 0, 0, 0, 0, 0, 0, 0])],
      morphWeights: new Float32Array([0]),
      morphTargetUvs: [new Float32Array([0.25, 0.125, 0, 0, 0, 0])],
      morphTargetUv1s: [new Float32Array([0.2, -0.1, 0, 0, 0, 0])],
    };
    const scene = makeScene(prim);
    const { host, sceneRef, tangentsWriteCalls, uvsWriteCalls } = makeHostWithSkinnedScene(scene);

    const router = new SceneMutationRouter(host);
    router.updatePrimitive('skinned', { morphWeights: new Float32Array([1]) });

    expect(tangentsWriteCalls.length).toBeGreaterThan(0);
    const written = tangentsWriteCalls[tangentsWriteCalls.length - 1]!;
    const invSqrt2 = 1 / Math.sqrt(2);
    expect(written[0]).toBeCloseTo(invSqrt2, 4);
    expect(written[1]).toBeCloseTo(invSqrt2, 4);
    expect(written[2]).toBeCloseTo(0, 4);
    expect(written[3]).toBeCloseTo(1, 4);
    expect(uvsWriteCalls.length).toBeGreaterThan(0);
    const writtenUvs = uvsWriteCalls[uvsWriteCalls.length - 1]!;
    expect(writtenUvs[0]).toBeCloseTo(0.25, 4);
    expect(writtenUvs[1]).toBeCloseTo(0.125, 4);
    expect(writtenUvs[2]).toBeCloseTo(0.7, 4);
    expect(writtenUvs[3]).toBeCloseTo(0.4, 4);
    const stored = sceneRef.current.primitives[0];
    expect(stored?.kind).toBe('skinned-mesh');
    if (stored?.kind === 'skinned-mesh') expect(stored.uvs).toBe(restUvs);
  });
});
