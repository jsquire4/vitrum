import { describe, expect, it } from 'vitest';
import {
  fingerprintBuffer,
  fingerprintBufferExact,
  fingerprintBuffers,
  fingerprintBuffersExact,
  fingerprintPackedSceneBvhState,
  isTlasOnlyVersionBump,
  packedSceneBvhStateEqual,
  type PackedSceneBvhFingerprintState,
} from '../bufferFingerprint.js';
import {
  MATERIAL_FLAG_DOUBLE_SIDED,
  MATERIAL_FLAG_IS_GLASS,
  packMaterials,
} from '../materialEntry.js';

function packedSceneState(
  materialEntries: ArrayBuffer | ArrayBufferView = packMaterials([{}]),
): PackedSceneBvhFingerprintState {
  return {
    bvhNodes: new Float32Array([0, 0, 0, 1, 1, 1, 0, 0]),
    positions: new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
    indices: new Uint32Array([0, 1, 2]),
    normals: new Float32Array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]),
    triMaterialId: new Uint32Array([0]),
    materialEntries,
  };
}

describe('bufferFingerprint', () => {
  it('changes when same-length buffer content changes', () => {
    const a = new Uint8Array(32);
    const b = new Uint8Array(32);
    a[0] = 1;
    b[0] = 2;
    expect(fingerprintBuffer(a.buffer)).not.toBe(fingerprintBuffer(b.buffer));
  });

  it('fingerprintBuffers combines multiple parts', () => {
    const x = new Uint8Array([1, 2, 3]);
    const y = new Uint8Array([4, 5, 6]);
    const combined = fingerprintBuffers(x.buffer, y.buffer);
    expect(combined).not.toBe(fingerprintBuffer(x.buffer));
    expect(combined).not.toBe(fingerprintBuffer(y.buffer));
  });

  it('exact fingerprint catches an unsampled interior byte in a large buffer', () => {
    const a = new Uint8Array(1024 * 1024);
    const b = new Uint8Array(1024 * 1024);
    b[1] = 1;

    expect(fingerprintBuffer(a.buffer)).toBe(fingerprintBuffer(b.buffer));
    expect(fingerprintBufferExact(a.buffer)).not.toBe(fingerprintBufferExact(b.buffer));
  });

  it('fingerprintBuffersExact combines exact per-buffer fingerprints', () => {
    const a = new Uint8Array(1024 * 1024);
    const b = new Uint8Array(1024 * 1024);
    b[1] = 1;

    expect(fingerprintBuffers(a.buffer)).toBe(fingerprintBuffers(b.buffer));
    expect(fingerprintBuffersExact(a.buffer)).not.toBe(fingerprintBuffersExact(b.buffer));
  });

  it('packed SceneBvh fingerprint covers every published byte buffer', () => {
    const baseline = packedSceneState();
    const baselineFingerprint = fingerprintPackedSceneBvhState(baseline);
    const mutations: ReadonlyArray<readonly [string, PackedSceneBvhFingerprintState]> = [
      ['bvhNodes', { ...baseline, bvhNodes: new Float32Array([0, 0, 0, 2, 1, 1, 0, 0]) }],
      ['positions', { ...baseline, positions: new Float32Array([0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0]) }],
      ['indices', { ...baseline, indices: new Uint32Array([0, 2, 1]) }],
      ['normals', { ...baseline, normals: new Float32Array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]) }],
      ['triMaterialId', { ...baseline, triMaterialId: new Uint32Array([1]) }],
      ['materialEntries', {
        ...baseline,
        materialEntries: packMaterials([{ baseColor: [0.5, 1, 1] }]),
      }],
    ];
    for (const [label, state] of mutations) {
      expect(
        fingerprintPackedSceneBvhState(state),
        `${label} must participate in the packed-state fingerprint`,
      ).not.toBe(baselineFingerprint);
    }
  });

  it('packed SceneBvh fingerprint includes true-u32 material flags', () => {
    const opaque = packedSceneState(packMaterials([{ flags: 0 }]));
    const glass = packedSceneState(packMaterials([{ flags: MATERIAL_FLAG_IS_GLASS }]));
    const doubleSided = packedSceneState(packMaterials([{ flags: MATERIAL_FLAG_DOUBLE_SIDED }]));
    const values = new Set([
      fingerprintPackedSceneBvhState(opaque),
      fingerprintPackedSceneBvhState(glass),
      fingerprintPackedSceneBvhState(doubleSided),
    ]);
    expect(values.size).toBe(3);
  });

  it('fresh and incrementally written canonical packed states fingerprint identically', () => {
    const flags = MATERIAL_FLAG_IS_GLASS | MATERIAL_FLAG_DOUBLE_SIDED;
    const fresh = packMaterials([{
      baseColor: [0.25, 0.5, 0.75],
      roughness: 0.375,
      metalness: 0.625,
      emissive: [1, 2, 3],
      ior: 1.7,
      transmission: 0.8,
      attenuationDistance: 2.5,
      thickness: 0.125,
      attenuationColor: [0.9, 0.8, 0.7],
      flags,
    }]);
    const incremental = packMaterials([{}]);
    incremental.set(fresh.subarray(0, 15), 0);
    new Uint32Array(incremental.buffer)[15] = flags;

    expect(new Uint8Array(incremental.buffer)).toEqual(new Uint8Array(fresh.buffer));
    expect(fingerprintPackedSceneBvhState(packedSceneState(incremental)))
      .toBe(fingerprintPackedSceneBvhState(packedSceneState(fresh)));
  });

  it('does not trust a 32-bit fingerprint collision in a rebuild-skip comparison', () => {
    // Deterministic equal-length FNV-1a collision for fingerprintBufferExact.
    const collidingA = new Uint8Array([164, 191, 1, 0, 57]);
    const collidingB = new Uint8Array([110, 130, 2, 0, 27]);
    expect(fingerprintBufferExact(collidingA))
      .toBe(fingerprintBufferExact(collidingB));

    const a = packedSceneState();
    const b = packedSceneState();
    const stateA = { ...a, positions: collidingA };
    const stateB = { ...b, positions: collidingB };
    expect(fingerprintPackedSceneBvhState(stateA))
      .toBe(fingerprintPackedSceneBvhState(stateB));
    expect(packedSceneBvhStateEqual(stateA, stateB)).toBe(false);
  });

  it('covers and exactly compares ordered raw-material behavior signatures', () => {
    const a = { ...packedSceneState(), materialSignatures: ['maps=object:h1'] };
    const b = { ...packedSceneState(), materialSignatures: ['maps=object:h2'] };
    expect(fingerprintPackedSceneBvhState(a))
      .not.toBe(fingerprintPackedSceneBvhState(b));
    expect(packedSceneBvhStateEqual(a, b)).toBe(false);
  });

  it('isTlasOnlyVersionBump detects transform-only TLAS bumps', () => {
    expect(isTlasOnlyVersionBump(10, 11, { blasContentVersion: 10, tlasContentVersion: 10 })).toBe(true);
    expect(isTlasOnlyVersionBump(10, 10, { blasContentVersion: 10, tlasContentVersion: 10 })).toBe(false);
    expect(isTlasOnlyVersionBump(11, 12, { blasContentVersion: 10, tlasContentVersion: 10 })).toBe(false);
  });
});
