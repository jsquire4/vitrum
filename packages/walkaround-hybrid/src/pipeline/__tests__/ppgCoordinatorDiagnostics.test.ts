import { describe, expect, it, vi } from 'vitest';
import type { EngineError, EngineWarning } from '@vitrum/core';
import { PPGCoordinator } from '../PPGCoordinator.js';
import type { FrameResources } from '../resourceManager.js';

type PPGCoordinatorInternals = {
  _enabled: boolean;
  _maxSpatialCells?: number;
  _maxDTreeNodesPerCell?: number;
  _sceneAABB: {
    min: [number, number, number];
    max: [number, number, number];
  };
  _reportTrainingReadbackFailure(raw: unknown): void;
};

function makeCoordinator(
  warnings: EngineWarning[] = [],
  errors: EngineError[] = [],
): PPGCoordinator {
  const coordinator = new PPGCoordinator({} as GPUDevice, {
    onWarning: (warning) => warnings.push(warning),
    onError: (error) => errors.push(error),
  });
  const internals = coordinator as unknown as PPGCoordinatorInternals;
  internals._enabled = true;
  internals._maxSpatialCells = 8;
  internals._maxDTreeNodesPerCell = 4;
  internals._sceneAABB = { min: [0, 0, 0], max: [1, 1, 1] };
  return coordinator;
}

function makeSnapshot(overrides: Partial<Parameters<PPGCoordinator['importSTree']>[0]> = {}): Parameters<PPGCoordinator['importSTree']>[0] {
  return {
    maxSpatialCells: 8,
    maxDTreeNodesPerCell: 4,
    sTreeBuf: new Float32Array(0),
    dTreeBuf: new Float32Array(0),
    dTreeOffsets: new Uint32Array(0),
    sceneBoundsMin: [0, 0, 0],
    sceneBoundsMax: [1, 1, 1],
    ...overrides,
  };
}

describe('PPGCoordinator diagnostics', () => {
  it('routes maxSpatialCells import mismatch through structured warnings', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const warnings: EngineWarning[] = [];
    const coordinator = makeCoordinator(warnings);

    const ok = coordinator.importSTree(
      makeSnapshot({ maxSpatialCells: 16 }),
      {} as FrameResources,
    );

    expect(ok).toBe(false);
    expect(warnSpy).not.toHaveBeenCalled();
    expect(warnings).toEqual([
      expect.objectContaining({
        code: 'walkaround-hybrid.ppg-import-max-spatial-cells-mismatch',
        backend: 'walkaround-hybrid',
        phase: 'lifecycle',
        method: 'importGIState',
        details: {
          snapshotMaxSpatialCells: 16,
          liveMaxSpatialCells: 8,
          fallback: 'cold PPG restart',
        },
      }),
    ]);
    expect(warnings[0]?.message).toContain('maxSpatialCells mismatch');
    warnSpy.mockRestore();
  });

  it('routes scene-bounds import mismatch through structured warnings', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const warnings: EngineWarning[] = [];
    const coordinator = makeCoordinator(warnings);

    const ok = coordinator.importSTree(
      makeSnapshot({ sceneBoundsMin: [2, 0, 0], sceneBoundsMax: [3, 1, 1] }),
      {} as FrameResources,
    );

    expect(ok).toBe(false);
    expect(warnSpy).not.toHaveBeenCalled();
    expect(warnings).toEqual([
      expect.objectContaining({
        code: 'walkaround-hybrid.ppg-import-scene-bounds-mismatch',
        details: {
          snapshotSceneBounds: { min: [2, 0, 0], max: [3, 1, 1] },
          liveSceneBounds: { min: [0, 0, 0], max: [1, 1, 1] },
          epsilon: 1e-3,
          fallback: 'cold PPG restart',
        },
      }),
    ]);
    expect(warnings[0]?.message).toContain('scene-bounds mismatch');
    warnSpy.mockRestore();
  });

  it('reports training readback failures as deduped non-fatal EngineErrors', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const errors: EngineError[] = [];
    const coordinator = makeCoordinator([], errors) as unknown as PPGCoordinatorInternals;
    const raw = new Error('mapAsync failed');

    coordinator._reportTrainingReadbackFailure(raw);
    coordinator._reportTrainingReadbackFailure(raw);

    expect(warnSpy).not.toHaveBeenCalled();
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatchObject({
      kind: 'render',
      fatal: false,
      raw,
    });
    expect(errors[0]?.message).toContain('training refine readback failed');
    expect(errors[0]?.message).toContain('mapAsync failed');
    warnSpy.mockRestore();
  });
});
