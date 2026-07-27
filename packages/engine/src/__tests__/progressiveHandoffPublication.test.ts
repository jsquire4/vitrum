import { describe, expect, it, vi } from 'vitest';
import { asMat4, type Engine, type FrameInput, type FrameOutput } from '@vitrum/core';

import {
  ProgressiveHandoffCoordinator,
  type ProgressiveHandoffController,
} from '../progressiveHandoff.js';

function frameInput(x: number): FrameInput {
  return {
    viewMatrix: asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      -x, 0, 0, 1,
    ])),
    projMatrix: asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ])),
    cameraPosition: [x, 0, 0],
    viewport: { width: 16, height: 16, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
  };
}

function rendered(samplesAccumulated = 1): FrameOutput {
  return {
    kind: 'rendered',
    samplesAccumulated,
    isConverged: false,
    primaryRadiance: {},
  } as FrameOutput;
}

function stubEngine(source: object) {
  const renderFrame = vi.fn(() => rendered());
  const reset = vi.fn();
  const seedAccumulator = vi.fn();
  const getProgressiveSeedTexture = vi.fn(() => ({
    texture: {} as never,
    width: 16,
    height: 16,
  }));
  const getPresentationSource = vi.fn(() => source as never);
  const engine = {
    renderFrame,
    reset,
    seedAccumulator,
    getProgressiveSeedTexture,
    getPresentationSource,
  } as unknown as Engine;
  return {
    engine,
    renderFrame,
    reset,
    seedAccumulator,
    getProgressiveSeedTexture,
    getPresentationSource,
  };
}

describe('ProgressiveHandoffCoordinator frame-state publication', () => {
  it('keeps the last successful realtime state and camera snapshot when realtime rendering fails', () => {
    const realtimeSource = { id: 'realtime' };
    const rt = stubEngine(realtimeSource);
    const cv = stubEngine({ id: 'converged' });
    const coordinator = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      stillFramesBeforeHandoff: 3,
    });

    expect(coordinator.frame(frameInput(0)).phase).toBe('realtime');
    rt.renderFrame.mockImplementationOnce(() => { throw new Error('realtime failed'); });
    expect(() => coordinator.frame(frameInput(1))).toThrow('realtime failed');

    expect(coordinator.phase).toBe('realtime');
    expect(coordinator.stillFrames).toBe(0);
    expect(coordinator.getPresentationSource()).toBe(realtimeSource);
    // The failed x=1 attempt did not publish #prev, so x=0 is still relative to
    // the last successful frame and advances the still counter.
    expect(coordinator.frame(frameInput(0))).toMatchObject({
      phase: 'settling',
      stillFrames: 1,
    });
  });

  it('retries the first handoff reset after reset failure without publishing handoff state', () => {
    const rt = stubEngine({ id: 'realtime' });
    const cv = stubEngine({ id: 'converged' });
    cv.reset.mockImplementationOnce(() => { throw new Error('reset failed'); });
    const coordinator = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      stillFramesBeforeHandoff: 1,
    });

    coordinator.frame(frameInput(0));
    expect(() => coordinator.frame(frameInput(0))).toThrow('reset failed');
    expect(coordinator.phase).toBe('realtime');
    expect(coordinator.stillFrames).toBe(0);
    expect(cv.renderFrame).not.toHaveBeenCalled();

    expect(coordinator.frame(frameInput(0)).phase).toBe('converging');
    expect(cv.reset).toHaveBeenCalledTimes(2);
  });

  it('retries reset and seeding when first-handoff seeding fails', () => {
    const rt = stubEngine({ id: 'realtime' });
    const cv = stubEngine({ id: 'converged' });
    cv.seedAccumulator.mockImplementationOnce(() => { throw new Error('seed failed'); });
    const coordinator = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      stillFramesBeforeHandoff: 1,
      seedFromRealtime: true,
    });

    coordinator.frame(frameInput(0));
    expect(() => coordinator.frame(frameInput(0))).toThrow('seed failed');
    expect(coordinator.phase).toBe('realtime');
    expect(coordinator.stillFrames).toBe(0);
    expect(cv.renderFrame).not.toHaveBeenCalled();

    expect(coordinator.frame(frameInput(0)).phase).toBe('converging');
    expect(cv.reset).toHaveBeenCalledTimes(2);
    expect(cv.seedAccumulator).toHaveBeenCalledTimes(2);
  });

  it('retries reset after a first-handoff converged render failure', () => {
    const rt = stubEngine({ id: 'realtime' });
    const cv = stubEngine({ id: 'converged' });
    cv.renderFrame.mockImplementationOnce(() => { throw new Error('converged failed'); });
    const coordinator = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      stillFramesBeforeHandoff: 1,
    });

    coordinator.frame(frameInput(0));
    expect(() => coordinator.frame(frameInput(0))).toThrow('converged failed');
    expect(coordinator.phase).toBe('realtime');
    expect(coordinator.stillFrames).toBe(0);

    expect(coordinator.frame(frameInput(0)).phase).toBe('converging');
    expect(cv.reset).toHaveBeenCalledTimes(2);
  });

  it('does not publish preroll when the behind render succeeds but the displayed realtime render fails', () => {
    const realtimeSource = { id: 'realtime' };
    const rt = stubEngine(realtimeSource);
    const cv = stubEngine({ id: 'converged' });
    const coordinator = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      stillFramesBeforeHandoff: 1,
      settleBehindRealtime: true,
      convergedDisplaySamples: 64,
    });

    coordinator.frame(frameInput(0));
    rt.renderFrame.mockImplementationOnce(() => { throw new Error('display render failed'); });
    expect(() => coordinator.frame(frameInput(0))).toThrow('display render failed');

    expect(cv.renderFrame).toHaveBeenCalledTimes(1);
    expect(coordinator.phase).toBe('realtime');
    expect(coordinator.stillFrames).toBe(0);
    expect(coordinator.getPresentationSource()).toBe(realtimeSource);

    expect(coordinator.frame(frameInput(0)).phase).toBe('prerolling');
    expect(cv.reset).toHaveBeenCalledTimes(2);
    expect(cv.renderFrame).toHaveBeenCalledTimes(2);
  });

  it('uses controller-published state as the rollback baseline for a failed frame', () => {
    const rt = stubEngine({ id: 'realtime' });
    const cv = stubEngine({ id: 'converged' });
    let resetOnAdvance = false;
    const controller: ProgressiveHandoffController = {
      animations: [{}],
      advance(_delta, options) {
        if (resetOnAdvance) options?.engine?.reset?.();
      },
    };
    const coordinator = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      stillFramesBeforeHandoff: 1,
      controller,
    });

    coordinator.frame(frameInput(0));
    expect(coordinator.frame(frameInput(0)).phase).toBe('converging');

    resetOnAdvance = true;
    rt.renderFrame.mockImplementationOnce(() => { throw new Error('post-controller render failed'); });
    expect(() => coordinator.frame(frameInput(0))).toThrow('post-controller render failed');
    expect(coordinator.phase).toBe('realtime');
    expect(coordinator.stillFrames).toBe(0);
  });
});
