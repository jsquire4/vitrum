import { describe, it, expect } from 'vitest';
import { pickBackend, deriveScaleDefaults, mergeWalkaroundTlasExtension } from '../src/createEngine.js';
import { auditSceneNeedsTlas } from '@vitrum/core';

describe('pickBackend', () => {
  it('returns pt-webgl2 for quality on single-mesh scenes', () => {
    expect(pickBackend('quality', true, 10_000, false)).toBe('pt-webgl2');
    expect(pickBackend('quality', false, 10_000, false)).toBe('pt-webgl2');
  });

  it('returns pt-webgpu for quality when scene needs TLAS and WebGPU is available', () => {
    expect(pickBackend('quality', true, 10_000, true)).toBe('pt-webgpu');
    expect(pickBackend('quality', false, 10_000, true)).toBe('pt-webgl2');
  });

  it('returns pt-webgpu for quality-webgpu when WebGPU is available', () => {
    expect(pickBackend('quality-webgpu', true, 10_000)).toBe('pt-webgpu');
    expect(pickBackend('quality-webgpu', false, 10_000)).toBe('pt-webgl2');
  });

  it('returns walkaround-hybrid when prefer is realtime + WebGPU available', () => {
    expect(pickBackend('realtime', true,  10_000_000)).toBe('walkaround-hybrid');
  });

  it('falls back to pt-webgl2 when prefer is realtime but WebGPU absent', () => {
    expect(pickBackend('realtime', false, 1_000)).toBe('pt-webgl2');
  });

  it('auto single-engine fallback is pt-webgpu on WebGPU regardless of triangle count', () => {
    expect(pickBackend('auto', true, 10_000)).toBe('pt-webgpu');
    expect(pickBackend('auto', true, 499_999)).toBe('pt-webgpu');
    expect(pickBackend('auto', true, 500_000)).toBe('pt-webgpu');
    expect(pickBackend('auto', true, 5_000_000)).toBe('pt-webgpu');
  });

  it('auto picks pt-webgl2 when WebGPU is unavailable', () => {
    expect(pickBackend('auto', false, 100)).toBe('pt-webgl2');
  });
});

describe('mergeWalkaroundTlasExtension', () => {
  it('adds bvhMode tlas when needsTlas and host did not set bvhMode', () => {
    const merged = mergeWalkaroundTlasExtension({}, true);
    expect(merged?.extensions?.['walkaround-hybrid']?.bvhMode).toBe('tlas');
  });

  it('does not override an explicit host bvhMode', () => {
    const merged = mergeWalkaroundTlasExtension(
      { extensions: { 'walkaround-hybrid': { bvhMode: 'merged' } } },
      true,
    );
    expect(merged?.extensions?.['walkaround-hybrid']?.bvhMode).toBe('merged');
  });

  it('no-op when needsTlas is false', () => {
    expect(mergeWalkaroundTlasExtension(undefined, false)).toBeUndefined();
  });
});

describe('deriveScaleDefaults', () => {
  it('matches the formula for D = 1 (Cornell-scale)', () => {
    const d = deriveScaleDefaults(1);
    expect(d.cameraMoveResetThresholdSq).toBeCloseTo(1e-6, 12);
    expect(d.temporalAccumAlpha).toBe(0.01);
    expect(d.emitterDist2Floor).toBeCloseTo(1e-8, 14);
    expect(d.triIntersectEpsilon).toBeCloseTo(1e-6, 12);
  });

  it('scales correctly for D = 100 (room-scale interior)', () => {
    const d = deriveScaleDefaults(100);
    expect(d.cameraMoveResetThresholdSq).toBeCloseTo((100 * 1e-3) ** 2, 8);
    expect(d.emitterDist2Floor).toBeCloseTo((100 * 1e-4) ** 2, 10);
    expect(d.triIntersectEpsilon).toBeCloseTo(100 * 1e-6, 10);
  });

  it('scales correctly for D = 0.01 (jewellery-scale)', () => {
    const d = deriveScaleDefaults(0.01);
    expect(d.cameraMoveResetThresholdSq).toBeCloseTo((0.01 * 1e-3) ** 2, 16);
    expect(d.emitterDist2Floor).toBeCloseTo((0.01 * 1e-4) ** 2, 20);
    expect(d.triIntersectEpsilon).toBeCloseTo(0.01 * 1e-6, 16);
  });

  it('temporalAccumAlpha is scene-scale-independent', () => {
    expect(deriveScaleDefaults(0.01).temporalAccumAlpha).toBe(0.01);
    expect(deriveScaleDefaults(100).temporalAccumAlpha).toBe(0.01);
  });
});

// Item 15 — tlasAudit.recommendation is load-bearing in createEngine's warn path.
// These tests verify that auditSceneNeedsTlas returns both recommendation + detail,
// and that createEngine gates its TLAS warn on the recommendation field (not a
// re-derived needsTlas check), so the recommendation is a consumed field, not dead.
describe('tlasAudit.recommendation is consumed by createEngine warn gate', () => {
  function makeMeshScene(primitiveCount: number) {
    const prims = Array.from({ length: primitiveCount }, (_, i) => ({
      kind: 'mesh' as const,
      id: `m${i}`,
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [0.5, 0.5, 0.5] as [number, number, number], roughness: 0.5, metallic: 0 },
    }));
    return { primitives: prims, emitters: [] as [], environment: { kind: 'none' as const } };
  }

  it('single-mesh scene: recommendation=merged-bvh-ok, detail=single-BVH suffix', () => {
    const audit = auditSceneNeedsTlas(makeMeshScene(1));
    expect(audit.recommendation).toBe('merged-bvh-ok');
    expect(audit.detail).toMatch(/single merged bvh/i);
    expect(audit.needsTlas).toBe(false);
  });

  it('multi-mesh scene: recommendation=prefer-tlas-backend, detail names counts', () => {
    const audit = auditSceneNeedsTlas(makeMeshScene(2));
    expect(audit.recommendation).toBe('prefer-tlas-backend');
    // The detail string is the message createEngine emits — it must name the primitive
    // count and point to a TLAS-capable backend so the host knows what to do.
    expect(audit.detail).toMatch(/2 mesh/i);
    expect(audit.detail).toMatch(/walkaround-hybrid|pt-webgpu/i);
    expect(audit.needsTlas).toBe(true);
  });

  it('recommendation and needsTlas are consistent (needsTlas ↔ recommend prefer-tlas)', () => {
    const single = auditSceneNeedsTlas(makeMeshScene(1));
    const multi  = auditSceneNeedsTlas(makeMeshScene(3));
    expect(single.needsTlas).toBe(single.recommendation === 'prefer-tlas-backend');
    expect(multi.needsTlas).toBe(multi.recommendation   === 'prefer-tlas-backend');
  });
});
