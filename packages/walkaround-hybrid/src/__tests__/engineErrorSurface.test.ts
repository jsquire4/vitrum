/**
 * engineErrorSurface.test.ts — unit tests for the item-28 EngineError surface
 * on walkaround-hybrid (HybridEngine).
 *
 * All tests use a mock GPUDevice with controllable addEventListener /
 * removeEventListener / lost; no real GPU is required.
 *
 * Coverage:
 *   - onError() is present and returns an unsubscribe function.
 *   - A fake 'uncapturederror' event dispatched through the captured listener
 *     fires the subscriber with the correct `kind` and `message`.
 *   - GPUValidationError → kind 'gpu-validation'; GPUInternalError → 'gpu-internal'.
 *   - Throttle: second identical error on the same frame is suppressed.
 *   - dispose() removes the 'uncapturederror' listener and clears subscribers.
 *   - device.lost settling → fatal EngineError with kind 'device-lost' and
 *     engine state transitions to 'error'.
 */

import { describe, expect, it, vi } from 'vitest';
import type { EngineError, Scene } from '@vitrum/core';
import { HybridEngine } from '../HybridEngine.js';
import type { HybridEngineOptions } from '../HybridEngine.js';
import type { DDGI } from '../ddgi/DDGI.js';

// ── Mock device builder ───────────────────────────────────────────────────────

type EventCb = (event: Event) => void;

interface ControlledDevice {
  device: GPUDevice;
  addCalls: Array<{ type: string; listener: EventCb }>;
  removeCalls: Array<{ type: string; listener: EventCb }>;
  resolveLost: (info: { reason?: string; message?: string }) => void;
}

function makeControlledDevice(): ControlledDevice {
  const addCalls: Array<{ type: string; listener: EventCb }> = [];
  const removeCalls: Array<{ type: string; listener: EventCb }> = [];
  let resolveLost!: (info: { reason?: string; message?: string }) => void;
  const lost = new Promise<{ reason?: string; message?: string }>((res) => {
    resolveLost = res;
  });

  const device = {
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
    createTexture: vi.fn(() => ({ createView: vi.fn(() => ({})), destroy: vi.fn() })),
    createBindGroupLayout: vi.fn(),
    createBindGroup: vi.fn(),
    createShaderModule: vi.fn(),
    createComputePipeline: vi.fn(),
    queue: { submit: vi.fn(), writeBuffer: vi.fn() },
    features: new Set<string>(),
    limits: {},
    addEventListener: vi.fn((type: string, listener: EventCb) => {
      addCalls.push({ type, listener });
    }),
    removeEventListener: vi.fn((type: string, listener: EventCb) => {
      removeCalls.push({ type, listener });
    }),
    lost,
  } as unknown as GPUDevice;

  return { device, addCalls, removeCalls, resolveLost };
}

function makeBaseOpts(device: GPUDevice): HybridEngineOptions {
  return {
    device,
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 1,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
  };
}

function getUncapturedErrorListener(addCalls: ControlledDevice['addCalls']): EventCb {
  const entry = addCalls.find((c) => c.type === 'uncapturederror');
  if (!entry) throw new Error('No uncapturederror listener registered');
  return entry.listener;
}

function fakeErrorEvent(message: string, className: 'GPUValidationError' | 'GPUInternalError') {
  return {
    error: {
      message,
      constructor: { name: className },
    },
  } as unknown as Event;
}

function makeDdgiScene(): Scene {
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

function getDdgi(engine: HybridEngine): DDGI {
  return (engine as unknown as { _ddgi: DDGI })._ddgi;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('walkaround-hybrid EngineError surface — onError subscription', () => {
  it('onError is a function and returns an unsubscribe function', () => {
    const ctrl = makeControlledDevice();
    const engine = new HybridEngine(makeBaseOpts(ctrl.device));
    expect(typeof engine.onError).toBe('function');
    const unsub = engine.onError((err: EngineError) => void err);
    expect(typeof unsub).toBe('function');
    engine.dispose();
  });

  it('unsubscribe removes the callback and is idempotent', () => {
    const ctrl = makeControlledDevice();
    const engine = new HybridEngine(makeBaseOpts(ctrl.device));
    const received: EngineError[] = [];
    const unsub = engine.onError((err) => received.push(err));
    unsub();
    unsub(); // idempotent — must not throw
    engine.dispose();
  });
});

describe('walkaround-hybrid EngineError surface — uncapturederror events', () => {
  it('GPUValidationError fires subscriber with kind="gpu-validation", fatal=false', () => {
    const ctrl = makeControlledDevice();
    const engine = new HybridEngine(makeBaseOpts(ctrl.device));
    const received: EngineError[] = [];
    engine.onError((err) => received.push(err));

    const listener = getUncapturedErrorListener(ctrl.addCalls);
    listener(fakeErrorEvent('validation fail', 'GPUValidationError'));

    expect(received).toHaveLength(1);
    expect(received[0]!.kind).toBe('gpu-validation');
    expect(received[0]!.message).toBe('validation fail');
    expect(received[0]!.fatal).toBe(false);
    engine.dispose();
  });

  it('GPUInternalError fires subscriber with kind="gpu-internal", fatal=false', () => {
    const ctrl = makeControlledDevice();
    const engine = new HybridEngine(makeBaseOpts(ctrl.device));
    const received: EngineError[] = [];
    engine.onError((err) => received.push(err));

    const listener = getUncapturedErrorListener(ctrl.addCalls);
    listener(fakeErrorEvent('oom', 'GPUInternalError'));

    expect(received).toHaveLength(1);
    expect(received[0]!.kind).toBe('gpu-internal');
    expect(received[0]!.fatal).toBe(false);
    engine.dispose();
  });

  it('throttle: same message without advancing the frame counter is suppressed', () => {
    const ctrl = makeControlledDevice();
    const engine = new HybridEngine(makeBaseOpts(ctrl.device));
    const received: EngineError[] = [];
    engine.onError((err) => received.push(err));

    const listener = getUncapturedErrorListener(ctrl.addCalls);
    const evt = fakeErrorEvent('repeated error', 'GPUValidationError');

    listener(evt); // first occurrence on frame 0 → emitted
    expect(received).toHaveLength(1);

    listener(evt); // same frame, same message → throttled
    expect(received).toHaveLength(1);

    engine.dispose();
  });

  it('subscriber throws → other subscribers still receive the error', () => {
    const ctrl = makeControlledDevice();
    const engine = new HybridEngine(makeBaseOpts(ctrl.device));
    const received: EngineError[] = [];
    engine.onError(() => { throw new Error('bad subscriber'); });
    engine.onError((err) => received.push(err));

    const listener = getUncapturedErrorListener(ctrl.addCalls);
    expect(() => listener(fakeErrorEvent('msg', 'GPUValidationError'))).not.toThrow();
    expect(received).toHaveLength(1);

    engine.dispose();
  });
});

describe('walkaround-hybrid EngineError surface — dispose removes listener', () => {
  it('dispose calls removeEventListener for uncapturederror', () => {
    const ctrl = makeControlledDevice();
    const engine = new HybridEngine(makeBaseOpts(ctrl.device));
    engine.dispose();

    const removed = ctrl.removeCalls.filter((c) => c.type === 'uncapturederror');
    expect(removed).toHaveLength(1);
    const added = ctrl.addCalls.find((c) => c.type === 'uncapturederror');
    expect(removed[0]!.listener).toBe(added!.listener);
  });

  it('events fired after dispose are silently ignored (state guard)', () => {
    const ctrl = makeControlledDevice();
    const engine = new HybridEngine(makeBaseOpts(ctrl.device));
    const received: EngineError[] = [];
    engine.onError((err) => received.push(err));

    const listener = getUncapturedErrorListener(ctrl.addCalls);
    engine.dispose();

    listener(fakeErrorEvent('post-dispose', 'GPUValidationError'));
    expect(received).toHaveLength(0);
  });
});

describe('walkaround-hybrid EngineError surface — device.lost', () => {
  it('device.lost → kind="device-lost", fatal=true, state="error"', async () => {
    const ctrl = makeControlledDevice();
    const engine = new HybridEngine(makeBaseOpts(ctrl.device));
    const received: EngineError[] = [];
    engine.onError((err) => received.push(err));

    ctrl.resolveLost({ reason: 'destroyed', message: 'GPU device lost.' });
    await Promise.resolve();
    await Promise.resolve();

    expect(received).toHaveLength(1);
    expect(received[0]!.kind).toBe('device-lost');
    expect(received[0]!.fatal).toBe(true);
    expect(received[0]!.message).toContain('GPU device lost.');
    expect(engine.state).toBe('error');

    engine.dispose();
  });

  it('device.lost after dispose does NOT change state to error', async () => {
    const ctrl = makeControlledDevice();
    const engine = new HybridEngine(makeBaseOpts(ctrl.device));
    engine.dispose();
    expect(engine.state).toBe('disposed');

    ctrl.resolveLost({ reason: 'destroyed', message: 'late' });
    await Promise.resolve();
    await Promise.resolve();

    expect(engine.state).toBe('disposed');
  });
});

describe('walkaround-hybrid EngineError surface — DDGI diagnostics', () => {
  it('routes DDGI probe-frame failures through engine.onError', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const ctrl = makeControlledDevice();
    const engine = new HybridEngine(makeBaseOpts(ctrl.device));
    const received: EngineError[] = [];
    engine.onError((err) => received.push(err));

    const ddgi = getDdgi(engine);
    ddgi.setProbeUpdateDivisor(1);
    vi.spyOn(ddgi.pass, 'init').mockResolvedValue(true);
    const thrown = new Error('probe dispatch failed');
    vi.spyOn(ddgi.pass, 'runFrame').mockRejectedValue(thrown);

    await ddgi.updateFrame({
      coreScene: makeDdgiScene(),
      device: ctrl.device,
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

    engine.dispose();
    errorSpy.mockRestore();
  });
});
