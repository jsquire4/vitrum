import { afterEach, describe, expect, it, vi } from 'vitest';
import type { EngineError, EngineWarning, Scene } from '@vitrum/core';
import { SceneBvh } from '@vitrum/shared-bvh';
import { DDGI } from '../DDGI.js';
import { ProbeUpdatePass } from '../probeUpdatePass.js';

function makeBoxScene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'box',
      positions: new Float32Array([
        -1, -1, -1,
         1, -1, -1,
        -1,  1, -1,
         1,  1,  1,
      ]),
      normals: new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ]),
      indices: new Uint32Array([0, 1, 2, 1, 3, 2]),
      material: { baseColor: [1, 1, 1], roughness: 1, metallic: 0 },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('DDGI structured error reporting', () => {
  it('routes missing-device skips through the structured warning sink', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const warnings: EngineWarning[] = [];
    const errors: EngineError[] = [];

    const ddgi = new DDGI({
      onWarning: (warning) => warnings.push(warning),
      onError: (error) => errors.push(error),
    });
    await ddgi.updateFrame({ enabled: true });

    expect(warnSpy).not.toHaveBeenCalled();
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatchObject({
      code: 'walkaround-hybrid.ddgi-missing-device',
      backend: 'walkaround-hybrid',
      phase: 'renderFrame',
      method: 'DDGI.updateFrame',
      details: { fallback: 'skip-ddgi-frame' },
    });
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatchObject({
      kind: 'render',
      message: '[DDGI] updateFrame called without device; skipping.',
      fatal: false,
    });
    ddgi.dispose();
  });

  it('reports GPU init failure through the structured error sink', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    vi.spyOn(ProbeUpdatePass.prototype, 'init').mockResolvedValue(false);
    const received: EngineError[] = [];

    const ddgi = new DDGI({ onError: (error) => received.push(error) });
    await ddgi.updateFrame({
      coreScene: makeBoxScene(),
      device: {} as unknown as GPUDevice,
      enabled: true,
    });

    expect(received).toContainEqual({
      kind: 'render',
      message: '[DDGI] GPU init failed — DDGI compute disabled (scene still renders without indirect).',
      fatal: false,
    });
    expect(ddgi.state()).toBe('failed');
    expect(warnSpy).toHaveBeenCalled();
    ddgi.dispose();
  });

  it('routes GPU init disabled fallback through the structured warning sink', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    vi.spyOn(ProbeUpdatePass.prototype, 'init').mockResolvedValue(false);
    const warnings: EngineWarning[] = [];
    const errors: EngineError[] = [];

    const ddgi = new DDGI({
      onWarning: (warning) => warnings.push(warning),
      onError: (error) => errors.push(error),
    });
    await ddgi.updateFrame({
      coreScene: makeBoxScene(),
      device: {} as unknown as GPUDevice,
      enabled: true,
    });

    expect(warnSpy).not.toHaveBeenCalled();
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatchObject({
      code: 'walkaround-hybrid.ddgi-init-disabled',
      backend: 'walkaround-hybrid',
      phase: 'renderFrame',
      method: 'DDGI.updateFrame',
      details: { fallback: 'disable-ddgi-compute' },
    });
    expect(errors).toContainEqual({
      kind: 'render',
      message: '[DDGI] GPU init failed — DDGI compute disabled (scene still renders without indirect).',
      fatal: false,
    });
    expect(ddgi.state()).toBe('failed');
    ddgi.dispose();
  });

  it('reports BVH update failures and skips the probe dispatch for that frame', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    vi.spyOn(ProbeUpdatePass.prototype, 'init').mockResolvedValue(true);
    const runFrameSpy = vi.spyOn(ProbeUpdatePass.prototype, 'runFrame').mockResolvedValue();
    const thrown = new Error('bvh exploded');
    vi.spyOn(SceneBvh.prototype, 'updateFromCore').mockImplementation(() => {
      throw thrown;
    });
    const received: EngineError[] = [];

    const ddgi = new DDGI({ onError: (error) => received.push(error) });
    await ddgi.updateFrame({
      coreScene: makeBoxScene(),
      device: {} as unknown as GPUDevice,
      enabled: true,
    });

    expect(received).toHaveLength(1);
    expect(received[0]).toMatchObject({
      kind: 'render',
      message: '[DDGI] BVH update failed: bvh exploded',
      fatal: false,
      raw: thrown,
    });
    expect(runFrameSpy).not.toHaveBeenCalled();
    expect(errorSpy).toHaveBeenCalled();
    ddgi.dispose();
  });

  it('reports runFrame failures and does not advertise ready from a failed probe frame', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    vi.spyOn(ProbeUpdatePass.prototype, 'init').mockResolvedValue(true);
    const thrown = new Error('probe dispatch failed');
    vi.spyOn(ProbeUpdatePass.prototype, 'runFrame').mockRejectedValue(thrown);
    const received: EngineError[] = [];

    const ddgi = new DDGI({ onError: (error) => received.push(error) });
    ddgi.setProbeUpdateDivisor(1);
    await ddgi.updateFrame({
      coreScene: makeBoxScene(),
      device: {} as unknown as GPUDevice,
      enabled: true,
    });

    expect(received).toHaveLength(1);
    expect(received[0]).toMatchObject({
      kind: 'render',
      message: '[DDGI] runFrame error: probe dispatch failed',
      fatal: false,
      raw: thrown,
    });
    expect(ddgi.state()).toBe('initializing');
    expect(errorSpy).toHaveBeenCalled();
    ddgi.dispose();
  });
});
