import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';

function makeStubDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

describe('pt-webgpu telemetry subscriptions', () => {
  it('allows subscribe/unsubscribe for onFrame and onProgress', async () => {
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice() });
    const onFrame = vi.fn();
    const onProgress = vi.fn();
    const offFrame = engine.onFrame?.(onFrame);
    const offProgress = engine.onProgress?.(onProgress);
    expect(typeof offFrame).toBe('function');
    expect(typeof offProgress).toBe('function');
    offFrame?.();
    offProgress?.();
  });

  it('clears telemetry subscribers on dispose and keeps unsubscribe idempotent', async () => {
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice() });
    const offFrame = engine.onFrame?.(() => {});
    const offProgress = engine.onProgress?.(() => {});
    engine.dispose();
    expect(() => offFrame?.()).not.toThrow();
    expect(() => offProgress?.()).not.toThrow();
  });
});
