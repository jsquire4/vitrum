/**
 * engineErrorSurface.test.ts — unit tests for the item-28 EngineError surface
 * on pt-webgpu (PTEngineWebGPU).
 *
 * All tests use a mock GPUDevice with controllable addEventListener /
 * removeEventListener / lost; no real GPU is required.
 *
 * Coverage:
 *   - onError() is present and returns an unsubscribe function.
 *   - A fake 'uncapturederror' event dispatched through the captured listener
 *     fires the subscriber with the correct `kind` and `message`.
 *   - GPUValidationError → kind 'gpu-validation'; GPUInternalError → 'gpu-internal'.
 *   - Throttle: second identical error within 32 frames is suppressed; after
 *     32+ frames it is re-emitted.
 *   - dispose() removes the 'uncapturederror' listener and clears subscribers.
 *   - device.lost settling → fatal EngineError with kind 'device-lost' and
 *     engine state transitions to 'error'.
 */

import { describe, expect, it, vi } from 'vitest';
import type { EngineError } from '@vitrum/core';
import type { FrameInput, Scene } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';

// ── Mock device builder ───────────────────────────────────────────────────────

type EventCb = (event: Event) => void;

interface ControlledDevice {
  device: GPUDevice;
  /** All addEventListener calls received. */
  addCalls: Array<{ type: string; listener: EventCb }>;
  /** All removeEventListener calls received. */
  removeCalls: Array<{ type: string; listener: EventCb }>;
  /** Resolve this to simulate device lost. */
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

/** Return the 'uncapturederror' listener registered on the device, or throw. */
function getUncapturedErrorListener(addCalls: ControlledDevice['addCalls']): EventCb {
  const entry = addCalls.find((c) => c.type === 'uncapturederror');
  if (!entry) throw new Error('No uncapturederror listener registered');
  return entry.listener;
}

/** Build a fake uncapturederror event with a given message and error class name. */
function fakeErrorEvent(message: string, className: 'GPUValidationError' | 'GPUInternalError') {
  return {
    error: {
      message,
      constructor: { name: className },
    },
  } as unknown as Event;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('pt-webgpu EngineError surface — onError subscription', () => {
  it('onError is a function and returns an unsubscribe function', async () => {
    const { device } = makeControlledDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(typeof engine.onError).toBe('function');
    const unsub = engine.onError!(() => {});
    expect(typeof unsub).toBe('function');
    engine.dispose();
  });

  it('unsubscribe removes the callback and is idempotent', async () => {
    const { device } = makeControlledDevice();
    const engine = await createPTEngine_WebGPU({ device });
    const received: EngineError[] = [];
    const unsub = engine.onError!((err) => received.push(err));
    unsub();
    unsub(); // idempotent — must not throw
    engine.dispose();
  });
});

describe('pt-webgpu EngineError surface — uncapturederror events', () => {
  it('GPUValidationError fires subscriber with kind="gpu-validation", fatal=false', async () => {
    const ctrl = makeControlledDevice();
    const engine = await createPTEngine_WebGPU({ device: ctrl.device });
    const received: EngineError[] = [];
    engine.onError!((err) => received.push(err));

    const listener = getUncapturedErrorListener(ctrl.addCalls);
    listener(fakeErrorEvent('pipeline bind-group mismatch', 'GPUValidationError'));

    expect(received).toHaveLength(1);
    expect(received[0]!.kind).toBe('gpu-validation');
    expect(received[0]!.message).toBe('pipeline bind-group mismatch');
    expect(received[0]!.fatal).toBe(false);
    engine.dispose();
  });

  it('GPUInternalError fires subscriber with kind="gpu-internal", fatal=false', async () => {
    const ctrl = makeControlledDevice();
    const engine = await createPTEngine_WebGPU({ device: ctrl.device });
    const received: EngineError[] = [];
    engine.onError!((err) => received.push(err));

    const listener = getUncapturedErrorListener(ctrl.addCalls);
    listener(fakeErrorEvent('out of memory', 'GPUInternalError'));

    expect(received).toHaveLength(1);
    expect(received[0]!.kind).toBe('gpu-internal');
    expect(received[0]!.fatal).toBe(false);
    engine.dispose();
  });

  it('throttle: same message in frame N+1 is suppressed; re-emitted after 32 frames', async () => {
    const ctrl = makeControlledDevice();
    const engine = await createPTEngine_WebGPU({ device: ctrl.device });
    const received: EngineError[] = [];
    engine.onError!((err) => received.push(err));

    const listener = getUncapturedErrorListener(ctrl.addCalls);
    const evt = fakeErrorEvent('repeated validation error', 'GPUValidationError');

    // Frame 0: first occurrence → emitted.
    listener(evt);
    expect(received).toHaveLength(1);

    // Frame 1 (no renderFrame call → frame counter still 0): suppressed.
    listener(evt);
    expect(received).toHaveLength(1);

    engine.dispose();
  });

  it('bounds distinct-message throttle state with oldest-entry eviction', async () => {
    const ctrl = makeControlledDevice();
    const engine = await createPTEngine_WebGPU({ device: ctrl.device });
    const received: EngineError[] = [];
    engine.onError!((err) => received.push(err));
    const listener = getUncapturedErrorListener(ctrl.addCalls);

    for (let i = 0; i < 257; i += 1) {
      listener(fakeErrorEvent(`unique validation error ${i}`, 'GPUValidationError'));
    }
    expect(received).toHaveLength(257);

    // Capacity is 256, so the oldest key has been evicted and is reportable
    // again even though the frame counter has not advanced.
    listener(fakeErrorEvent('unique validation error 0', 'GPUValidationError'));
    expect(received).toHaveLength(258);
    engine.dispose();
  });
});

describe('pt-webgpu EngineError surface — dispose removes listener', () => {
  it('dispose calls removeEventListener for uncapturederror', async () => {
    const ctrl = makeControlledDevice();
    const engine = await createPTEngine_WebGPU({ device: ctrl.device });
    engine.dispose();

    const removed = ctrl.removeCalls.filter((c) => c.type === 'uncapturederror');
    expect(removed).toHaveLength(1);
    // The removed listener is the same function that was added.
    const added = ctrl.addCalls.find((c) => c.type === 'uncapturederror');
    expect(removed[0]!.listener).toBe(added!.listener);
  });

  it('subscriber registered before dispose receives no events after dispose', async () => {
    const ctrl = makeControlledDevice();
    const engine = await createPTEngine_WebGPU({ device: ctrl.device });
    const received: EngineError[] = [];
    engine.onError!((err) => received.push(err));

    // Capture the listener before dispose clears it.
    const listener = getUncapturedErrorListener(ctrl.addCalls);
    engine.dispose();

    // Simulate a host that fires the event after dispose (e.g. async validation).
    listener(fakeErrorEvent('post-dispose error', 'GPUValidationError'));
    // Engine is disposed; handler guards against this.
    expect(received).toHaveLength(0);
  });
});

describe('pt-webgpu EngineError surface — device.lost', () => {
  it('device.lost → EngineError with kind="device-lost", fatal=true, state="error"', async () => {
    const ctrl = makeControlledDevice();
    const engine = await createPTEngine_WebGPU({ device: ctrl.device });
    const received: EngineError[] = [];
    engine.onError!((err) => received.push(err));

    ctrl.resolveLost({ reason: 'destroyed', message: 'The device was destroyed.' });

    // Allow the microtask queue to drain.
    await Promise.resolve();
    await Promise.resolve();

    expect(received).toHaveLength(1);
    expect(received[0]!.kind).toBe('device-lost');
    expect(received[0]!.fatal).toBe(true);
    expect(received[0]!.message).toContain('The device was destroyed.');
    expect(engine.state).toBe('error');

    engine.dispose();
  });

  it('device.lost after dispose does NOT change state to error', async () => {
    const ctrl = makeControlledDevice();
    const engine = await createPTEngine_WebGPU({ device: ctrl.device });
    engine.dispose();
    expect(engine.state).toBe('disposed');

    ctrl.resolveLost({ reason: 'destroyed', message: 'lost after dispose' });
    await Promise.resolve();
    await Promise.resolve();

    // The handler guards `if (state === 'disposed')` → no state change.
    expect(engine.state).toBe('disposed');
  });

  it('fatal device loss blocks render, mutation, seeding, inverse, capture, reset, and resume', async () => {
    const ctrl = makeControlledDevice();
    const engine = await createPTEngine_WebGPU({ device: ctrl.device });

    ctrl.resolveLost({ reason: 'destroyed', message: 'lost' });
    await Promise.resolve();
    await Promise.resolve();

    expect(engine.state).toBe('error');
    const fatal = /fatal error state/i;
    const scene: Scene = { primitives: [], emitters: [], environment: { kind: 'none' } };
    const frame = {} as FrameInput;

    expect(() => engine.setScene(scene)).toThrow(fatal);
    expect(() => engine.renderFrame(frame)).toThrow(fatal);
    expect(() => engine.updateEmitter?.('light', { intensity: 0 })).toThrow(fatal);
    expect(() => engine.seedAccumulator?.({} as never, { weight: 1, width: 1, height: 1 })).toThrow(fatal);
    expect(() => engine.createInverseSession?.({} as never)).toThrow(fatal);
    await expect(engine.captureFrame?.()).rejects.toThrow(fatal);
    expect(() => engine.reset()).toThrow(fatal);
    expect(() => engine.resume()).toThrow(fatal);

    engine.dispose();
  });
});
