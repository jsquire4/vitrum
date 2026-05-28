import { describe, expect, it } from 'vitest';
import {
  fingerprintHybridPipelineRebuildKey,
  HYBRID_FRAME_SKIP_OUTPUT,
} from '../HybridEngineFrameOrchestrator.js';

describe('HybridEngineFrameOrchestrator', () => {
  it('fingerprintHybridPipelineRebuildKey is stable', () => {
    expect(fingerprintHybridPipelineRebuildKey(null)).toBe('__null');
    expect(fingerprintHybridPipelineRebuildKey(42)).toBe('__n:42');
    expect(fingerprintHybridPipelineRebuildKey('scene-v2')).toBe('__s:scene-v2');
  });

  it('HYBRID_FRAME_SKIP_OUTPUT is skipped kind', () => {
    expect(HYBRID_FRAME_SKIP_OUTPUT.kind).toBe('skipped');
    expect(HYBRID_FRAME_SKIP_OUTPUT.samplesAccumulated).toBe(0);
  });
});
