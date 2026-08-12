import { describe, expect, it } from 'vitest';
import {
  applyExperimentalPreset,
  experimentalPreset,
  EXPERIMENTAL_PRESET_IDS,
  isExperimentalPresetId,
} from '../experimentalPresets.js';
import type { CreateEngineOptions } from '../createEngineInternals.js';

describe('experimentalPreset', () => {
  it('names only legal full-tier research combos', () => {
    expect([...EXPERIMENTAL_PRESET_IDS]).toEqual([
      'spectral-bdpt',
      'manifold-caustics',
      'photon-map-caustics',
      'one-edge-gris',
      'walkaround-nrc',
      'walkaround-ppg',
      'walkaround-rc',
    ]);
    expect(isExperimentalPresetId('spectral-bdpt')).toBe(true);
    expect(isExperimentalPresetId('lite-bdpt')).toBe(false);
  });

  it('expands spectral-bdpt onto both path tracers without lite or reuse flags', () => {
    const bag = experimentalPreset('spectral-bdpt');
    expect(bag.prefer).toBe('quality');
    expect(bag.advancedByBackend['pt-webgpu']).toEqual({ spectral: true, bdpt: true });
    expect(bag.advancedByBackend['pt-webgl2']).toEqual({ spectral: true, bdpt: true });
    expect(bag.advancedByBackend['pt-webgpu']).not.toHaveProperty('traceTier');
    expect(bag.advancedByBackend['pt-webgpu']).not.toHaveProperty('oneEdgeReconnectionReuse');
    expect(bag.advancedByBackend['pt-webgpu']).not.toHaveProperty('causticStrategy');
  });

  it('keeps MNEE / SPPM / GRIS on quality-webgpu only', () => {
    expect(experimentalPreset('manifold-caustics')).toEqual({
      prefer: 'quality-webgpu',
      advancedByBackend: { 'pt-webgpu': { causticStrategy: 'manifold-nee' } },
    });
    expect(experimentalPreset('photon-map-caustics').advancedByBackend['pt-webgpu'])
      .toEqual({ causticStrategy: 'photon-map' });
    expect(experimentalPreset('one-edge-gris').advancedByBackend['pt-webgpu'])
      .toEqual({ oneEdgeReconnectionReuse: true });
    expect(experimentalPreset('manifold-caustics').advancedByBackend['pt-webgl2'])
      .toBeUndefined();
  });

  it('keeps walkaround research layers on the realtime engine', () => {
    expect(experimentalPreset('walkaround-nrc')).toEqual({
      prefer: 'realtime',
      advancedByBackend: { 'walkaround-hybrid': { nrcEnabled: true } },
    });
    expect(experimentalPreset('walkaround-ppg').advancedByBackend['walkaround-hybrid'])
      .toEqual({ ppgEnabled: true });
    expect(experimentalPreset('walkaround-rc').advancedByBackend['walkaround-hybrid'])
      .toEqual({ rcEnabled: true });
  });
});

describe('applyExperimentalPreset', () => {
  const canvas = { getContext() { return null; }, width: 1, height: 1 } as unknown as HTMLCanvasElement;
  const scene = { primitives: [], emitters: [], environment: { kind: 'none' as const } };

  it('is a no-op when the host omits experimentalPreset', () => {
    const opts: CreateEngineOptions = { canvas, scene, prefer: 'auto' };
    expect(applyExperimentalPreset(opts)).toBe(opts);
  });

  it('fills prefer from the preset when the host omitted it', () => {
    const resolved = applyExperimentalPreset({
      canvas,
      scene,
      experimentalPreset: 'one-edge-gris',
    });
    expect(resolved.prefer).toBe('quality-webgpu');
    expect(resolved.advancedByBackend).toEqual({
      'pt-webgpu': { oneEdgeReconnectionReuse: true },
    });
    expect(resolved).not.toHaveProperty('experimentalPreset');
  });

  it('lets host prefer and per-backend keys win', () => {
    const resolved = applyExperimentalPreset({
      canvas,
      scene,
      experimentalPreset: 'spectral-bdpt',
      prefer: 'quality-webgpu',
      advancedByBackend: {
        'pt-webgpu': { maxBounces: 6 },
      },
    });
    expect(resolved.prefer).toBe('quality-webgpu');
    expect(resolved.advancedByBackend).toEqual({
      'pt-webgpu': { spectral: true, bdpt: true, maxBounces: 6 },
      'pt-webgl2': { spectral: true, bdpt: true },
    });
  });
});
