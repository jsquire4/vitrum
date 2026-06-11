/**
 * B3a (2026-05-19) — atrous DIRECT/INDIRECT sigmas as host-overridable
 * `HybridEngineOptions` fields. Pre-B3a these were hardcoded
 * `ATROUS_DIRECT_SIGMAS` / `ATROUS_INDIRECT_SIGMAS` constants in
 * `bindGroupBuilders.ts:178,195`. Hosts on non-Cornell scene scales
 * couldn't override the depth-tolerance (sigmaZ in world units).
 *
 * These tests are structural pins:
 *   1. The option fields are tuples `[sigmaN, sigmaZ, sigmaC]`.
 *   2. The PipelineFrameInputs interface carries them.
 *   3. The DenoiserDispatchContext threads the direct sigmas.
 *   4. The exported defaults match the prior hardcoded constants
 *      (so omitting the option preserves Cornell behaviour).
 */

import { describe, it, expect } from 'vitest';
import {
  ATROUS_DIRECT_SIGMAS,
  ATROUS_INDIRECT_SIGMAS,
  type AtrousSigmas,
} from '../src/pipeline/bindGroupBuilders.js';
import type { HybridEngineOptions } from '../src/HybridEngineOptions.js';
import type { PipelineFrameFilter } from '../src/pipeline/WalkaroundGPUPipeline.js';
import type { DenoiserDispatchContext } from '../src/pipeline/denoisers/index.js';

describe('atrous sigmas override (B3a)', () => {
  it('exports the legacy direct defaults [128, 5, 0.05]', () => {
    const def: Readonly<AtrousSigmas> = ATROUS_DIRECT_SIGMAS;
    expect(def.sigmaN).toBe(128.0);
    expect(def.sigmaZ).toBe(5.0);
    expect(def.sigmaC).toBe(0.05);
  });

  it('exports the legacy indirect defaults [32, 20, 0.5]', () => {
    const def: Readonly<AtrousSigmas> = ATROUS_INDIRECT_SIGMAS;
    expect(def.sigmaN).toBe(32.0);
    expect(def.sigmaZ).toBe(20.0);
    expect(def.sigmaC).toBe(0.5);
  });

  it('HybridEngineOptions carries optional atrous{Direct,Indirect}Sigmas tuples', () => {
    // Structural check via the option-object literal type — this fails to
    // compile if the field is missing or has a non-tuple type.
    const opts: HybridEngineOptions = {
      device: {} as GPUDevice,
      width: 64, height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1.0,
      skyTint: [1, 1, 1],
      skyIrradiance: 1.0,
      atrousDirectSigmas:   [256.0, 10.0, 0.1] as const,
      atrousIndirectSigmas: [64.0, 40.0, 1.0] as const,
    };
    expect(opts.atrousDirectSigmas?.[0]).toBe(256.0);
    expect(opts.atrousIndirectSigmas?.[1]).toBe(40.0);
  });

  it('PipelineFrameInputs.filter carries atrous{Direct,Indirect}Sigmas (required, not optional)', () => {
    // Structural check via a constructed filter sub-object literal. The
    // required-ness means the per-frame builder from HybridEngine MUST include both.
    const filter: Pick<PipelineFrameFilter, 'atrousDirectSigmas' | 'atrousIndirectSigmas'> = {
      atrousDirectSigmas:   [128.0, 5.0, 0.05],
      atrousIndirectSigmas: [32.0, 20.0, 0.5],
    };
    expect(filter.atrousDirectSigmas.length).toBe(3);
    expect(filter.atrousIndirectSigmas.length).toBe(3);
  });

  it('DenoiserDispatchContext threads atrousDirectSigmas to denoisers', () => {
    // Structural check on the dispatch-time context shape so a future edit
    // that drops the field fails this test instead of silently re-falling-
    // back to the hardcoded ATROUS_DIRECT_SIGMAS default.
    const ctx: Pick<DenoiserDispatchContext, 'atrousDirectSigmas'> = {
      atrousDirectSigmas: [128.0, 5.0, 0.05],
    };
    expect(ctx.atrousDirectSigmas[0]).toBe(128.0);
  });
});
