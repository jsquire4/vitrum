import { describe, expect, it, vi } from 'vitest';
import { asMat4, type Engine, type FrameInput, type Scene, type SceneEmitter } from '@vitrum/core';
import { ProgressiveHandoffCoordinator } from '../progressiveHandoff.js';
import { stubEngine } from './fixtures/stubEngine.js';

function directional(intensity = 1): SceneEmitter {
  return {
    kind: 'directional',
    id: 'sun',
    color: [1, 1, 1],
    intensity,
    direction: [0, 1, 0],
  };
}

function scene(environment: Scene['environment'] = { kind: 'none' }): Scene {
  return {
    primitives: [],
    emitters: [directional()],
    environment,
  };
}

function frameInput(): FrameInput {
  return {
    viewMatrix: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
    projMatrix: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
    cameraPosition: [0, 0, 0],
  } as unknown as FrameInput;
}

function pair(overrides: Partial<Engine> = {}) {
  const realtime = { ...stubEngine(), ...overrides };
  const converged = { ...stubEngine(), ...overrides };
  return { realtime, converged };
}

describe('ProgressiveHandoffCoordinator contract hardening', () => {
  it.each([
    ['stillFramesBeforeHandoff', Number.NaN],
    ['stillFramesBeforeHandoff', Number.POSITIVE_INFINITY],
    ['stillFramesBeforeHandoff', 1.5],
    ['stillFramesBeforeHandoff', 0],
    ['convergedDisplaySamples', Number.NaN],
    ['convergedDisplaySamples', 0],
    ['cameraEpsilon', Number.NaN],
    ['cameraEpsilon', -1],
    ['seedWeight', Number.POSITIVE_INFINITY],
    ['seedWeight', -1],
    ['controllerDeltaSeconds', Number.NaN],
  ])('rejects invalid %s=%s before any frame can run', (key, value) => {
    const { realtime, converged } = pair();
    expect(() => new ProgressiveHandoffCoordinator({
      realtime,
      converged,
      [key]: value,
    })).toThrow();
  });

  it('rejects non-finite and malformed camera payloads before rendering either engine', () => {
    const { realtime, converged } = pair();
    const coordinator = new ProgressiveHandoffCoordinator({ realtime, converged });
    const invalid = frameInput();
    (invalid.viewMatrix as Float32Array)[7] = Number.NaN;

    expect(() => coordinator.frame(invalid)).toThrow(/viewMatrix\[7\] must be finite/);
    expect(realtime.renderFrame).not.toHaveBeenCalled();
    expect(converged.renderFrame).not.toHaveBeenCalled();

    const malformed = { ...frameInput(), cameraPosition: [0, 0] } as unknown as FrameInput;
    expect(() => coordinator.frame(malformed)).toThrow(/cameraPosition must contain exactly 3 values/);
  });

  it('forwards emitter patches to both engines and retains the patched authoritative scene', () => {
    const updateEmitterRealtime = vi.fn();
    const updateEmitterConverged = vi.fn();
    const realtime = { ...stubEngine(), updateEmitter: updateEmitterRealtime } as Engine;
    const converged = { ...stubEngine(), updateEmitter: updateEmitterConverged } as Engine;
    const coordinator = new ProgressiveHandoffCoordinator({ realtime, converged, scene: scene() });

    coordinator.updateEmitter('sun', { intensity: 4 });

    expect(updateEmitterRealtime).toHaveBeenCalledWith('sun', { intensity: 4 });
    expect(updateEmitterConverged).toHaveBeenCalledWith('sun', { intensity: 4 });
    expect(coordinator.getScene()?.emitters[0]?.intensity).toBe(4);
  });

  it('uses a full-scene fallback when either engine lacks the emitter fast path', () => {
    const updateEmitter = vi.fn();
    const realtime = { ...stubEngine(), updateEmitter } as Engine;
    const converged = stubEngine();
    const coordinator = new ProgressiveHandoffCoordinator({ realtime, converged, scene: scene() });

    coordinator.updateEmitter('sun', { intensity: 7 });

    expect(updateEmitter).not.toHaveBeenCalled();
    expect(realtime.setScene).toHaveBeenCalledTimes(1);
    expect(converged.setScene).toHaveBeenCalledTimes(1);
    const next = vi.mocked(realtime.setScene).mock.calls[0]![0];
    expect(next.emitters[0]?.intensity).toBe(7);
    expect(converged.setScene).toHaveBeenCalledWith(next);
  });

  it('normalizes a null environment and synchronizes it through full-scene fallback', () => {
    const initial = scene({ kind: 'hdri', hdri: {}, intensity: 2 });
    const realtime = { ...stubEngine(), updateEnvironment: vi.fn() } as Engine;
    const converged = stubEngine();
    const coordinator = new ProgressiveHandoffCoordinator({ realtime, converged, scene: initial });

    coordinator.updateEnvironment(null);

    expect(realtime.updateEnvironment).not.toHaveBeenCalled();
    const next = vi.mocked(realtime.setScene).mock.calls[0]![0];
    expect(next.environment).toEqual({ kind: 'none' });
    expect(converged.setScene).toHaveBeenCalledWith(next);
    expect(coordinator.getScene()?.environment).toEqual({ kind: 'none' });
  });

  it('preflights paired runtime-lighting support before mutating either engine', () => {
    const updateLighting = vi.fn();
    const realtime = { ...stubEngine(), updateLighting } as Engine;
    const converged = stubEngine();
    const coordinator = new ProgressiveHandoffCoordinator({ realtime, converged, scene: scene() });

    expect(() => coordinator.updateLighting({ environmentIntensity: 2 }))
      .toThrow(/both engines must implement updateLighting/);
    expect(updateLighting).not.toHaveBeenCalled();
    expect(coordinator.synchronizationError).toBeNull();
  });

  it('enters terminal synchronization state when paired runtime lighting splits', () => {
    const realtimeUpdate = vi.fn();
    const convergedUpdate = vi.fn(() => { throw new Error('converged lighting rejected'); });
    const realtime = { ...stubEngine(), updateLighting: realtimeUpdate } as Engine;
    const converged = { ...stubEngine(), updateLighting: convergedUpdate } as Engine;
    const coordinator = new ProgressiveHandoffCoordinator({ realtime, converged, scene: scene() });
    const update = { environmentIntensity: 2 };

    let thrown: unknown;
    try {
      coordinator.updateLighting(update);
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(AggregateError);
    expect(realtimeUpdate).toHaveBeenCalledWith(update);
    expect(convergedUpdate).toHaveBeenCalledWith(update);
    expect(coordinator.synchronizationError).toBe(thrown);
    expect(() => coordinator.frame(frameInput())).toThrow(thrown as AggregateError);
  });

  it('restores the previous scene on both engines when a paired setScene fails', () => {
    const previous = scene();
    const next = scene({ kind: 'hdri', hdri: { replacement: true }, intensity: 1 });
    const realtime = stubEngine();
    const convergedSetScene = vi.fn((candidate: Scene) => {
      if (candidate === next) throw new Error('converged rejected scene');
    });
    const converged = { ...stubEngine(), setScene: convergedSetScene } as Engine;
    const coordinator = new ProgressiveHandoffCoordinator({ realtime, converged, scene: previous });

    expect(() => coordinator.setScene(next)).toThrow('converged rejected scene');
    expect(realtime.setScene).toHaveBeenNthCalledWith(1, next);
    expect(realtime.setScene).toHaveBeenNthCalledWith(2, previous);
    expect(convergedSetScene).toHaveBeenNthCalledWith(1, next);
    expect(convergedSetScene).toHaveBeenNthCalledWith(2, previous);
    expect(coordinator.getScene()).toBe(previous);
  });
});
