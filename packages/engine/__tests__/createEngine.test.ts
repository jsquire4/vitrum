import { describe, it, expect } from 'vitest';
import { pickBackend, deriveScaleDefaults, mergeWalkaroundTlasExtension } from '../src/createEngine.js';

describe('pickBackend', () => {
  it('returns pt-webgl for quality on single-mesh scenes', () => {
    expect(pickBackend('quality', true, 10_000, false)).toBe('pt-webgl');
    expect(pickBackend('quality', false, 10_000, false)).toBe('pt-webgl');
  });

  it('returns pt-webgpu for quality when scene needs TLAS and WebGPU is available', () => {
    expect(pickBackend('quality', true, 10_000, true)).toBe('pt-webgpu');
    expect(pickBackend('quality', false, 10_000, true)).toBe('pt-webgl');
  });

  it('returns pt-webgpu for quality-webgpu when WebGPU is available', () => {
    expect(pickBackend('quality-webgpu', true, 10_000)).toBe('pt-webgpu');
    expect(pickBackend('quality-webgpu', false, 10_000)).toBe('pt-webgl');
  });

  it('returns walkaround-hybrid when prefer is realtime + WebGPU available', () => {
    expect(pickBackend('realtime', true,  10_000_000)).toBe('walkaround-hybrid');
  });

  it('falls back to pt-webgl when prefer is realtime but WebGPU absent', () => {
    expect(pickBackend('realtime', false, 1_000)).toBe('pt-webgl');
  });

  it('auto picks walkaround-hybrid for small scenes on WebGPU', () => {
    expect(pickBackend('auto', true, 10_000)).toBe('walkaround-hybrid');
    expect(pickBackend('auto', true, 499_999)).toBe('walkaround-hybrid');
  });

  it('auto picks pt-webgpu above the triangle budget when WebGPU is available', () => {
    expect(pickBackend('auto', true, 500_000)).toBe('pt-webgpu');
    expect(pickBackend('auto', true, 5_000_000)).toBe('pt-webgpu');
  });

  it('auto picks pt-webgl when WebGPU is unavailable', () => {
    expect(pickBackend('auto', false, 100)).toBe('pt-webgl');
  });

  it('auto selects pt-webgpu only above the realtime triangle budget', () => {
    expect(pickBackend('auto', true, 10_000)).toBe('walkaround-hybrid');
    expect(pickBackend('auto', true, 600_000)).toBe('pt-webgpu');
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
