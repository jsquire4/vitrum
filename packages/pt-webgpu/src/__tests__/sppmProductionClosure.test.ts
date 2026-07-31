import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  SPPM_GROUP3_BINDINGS_WGSL,
  SPPM_PHOTON_PASS_WGSL,
} from '../wgsl/pathTrace/sppmBindings.wgsl.js';

const INDEX_SOURCE = readFileSync(new URL('../index.ts', import.meta.url), 'utf8');
const GPU_RESOURCES_SOURCE = readFileSync(
  new URL('../gpuResources.ts', import.meta.url),
  'utf8',
);
const VALIDATION_SOURCE = readFileSync(
  new URL('../ptWebgpuValidation.ts', import.meta.url),
  'utf8',
);
const MUTATION_ROUTER_SOURCE = readFileSync(
  new URL('../sceneMutationRouter.ts', import.meta.url),
  'utf8',
);

function section(source: string, start: string, end: string): string {
  const startIndex = source.indexOf(start);
  const endIndex = source.indexOf(end, startIndex + start.length);
  expect(startIndex, `missing section start: ${start}`).toBeGreaterThanOrEqual(0);
  expect(endIndex, `missing section end: ${end}`).toBeGreaterThan(startIndex);
  return source.slice(startIndex, endIndex);
}

describe('SPPM production pipeline closure', () => {
  it('proves SPPM readiness before packing caustic mode or building bind groups', () => {
    const frameSetup = section(
      INDEX_SOURCE,
      '#ensurePerFrameResources(',
      '  #ensureParamsPerFrame(\n    gpu:',
    );
    const sppmCall = frameSetup.indexOf('this.#ensureSppmPerFrame');
    const paramsCall = frameSetup.indexOf('this.#ensureParamsPerFrame');
    const bindGroupCall = frameSetup.indexOf('gpu.buildBindGroups');
    expect(sppmCall).toBeGreaterThanOrEqual(0);
    expect(sppmCall).toBeLessThan(paramsCall);
    expect(paramsCall).toBeLessThan(bindGroupCall);

    const readiness = section(
      INDEX_SOURCE,
      '#ensureSppmPerFrame(',
      'Per-frame ReSTIR-PT reuse setup',
    );
    const bufferCheck = readiness.indexOf('gpu.ensureSppmBuffers(true)');
    const pixelCheck = readiness.indexOf('gpu.ensureSppmPixelStatsBuffer');
    const pipelineCheck = readiness.indexOf('gpu.sppm.sppmPhotonPipeline == null');
    const statsWrite = readiness.indexOf('gpu.writeSppmStats');
    expect(bufferCheck).toBeLessThan(pixelCheck);
    expect(pixelCheck).toBeLessThan(pipelineCheck);
    expect(pipelineCheck).toBeLessThan(statsWrite);
    expect(readiness).toContain('if (!sppmBuffersOk)');
    expect(readiness).toContain('if (!sppmPixelStatsOk || gpu.sppm.sppmPixelStatsBuffer == null)');
    expect(readiness.match(/throw new Error/g)?.length).toBe(3);
  });

  it('rejects only the unsupported lite tier rather than suppressing peer estimators', () => {
    const validation = section(
      VALIDATION_SOURCE,
      "if (traceTier === 'lite' && opts.causticStrategy",
      "if (traceTier === 'lite' && opts.lightTreeImportanceSampling",
    );
    expect(validation).toContain("traceTier === 'lite'");
    expect(validation).toContain("opts.causticStrategy !== 'none'");
    expect(validation).not.toContain('opts.spectral === true');
    expect(validation).not.toContain('opts.bdpt === true');
    expect(validation).not.toContain('opts.restirPtReuse === true');
    expect(validation.match(/throw new Error/g)?.length).toBe(1);
  });

  it('clears bucket heads, emits photons, then launches the eye megakernel', () => {
    const passes = section(
      INDEX_SOURCE,
      '#encodePathTracePasses(',
      '#handleDenoiserPresentationFailure(',
    );
    const clear = passes.indexOf('encoder.clearBuffer(gpu.sppm.sppmCellCountersBuffer)');
    const photonPass = passes.indexOf("label: 'vitrum.pt-webgpu.sppm.photonPass'");
    const photonDispatch = passes.indexOf('photonPass.dispatchWorkgroups(SPPM_WORKGROUP_COUNT');
    const eyePass = passes.indexOf("label: 'vitrum.pt-webgpu.pathTrace.pass'");
    expect(clear).toBeGreaterThanOrEqual(0);
    expect(clear).toBeLessThan(photonPass);
    expect(photonPass).toBeLessThan(photonDispatch);
    expect(photonDispatch).toBeLessThan(eyePass);
  });

  it('resets progressive pixel state on allocation, resize, camera reset, and scene reset', () => {
    const reset = section(INDEX_SOURCE, 'reset(): void {', 'Progressive walkaround→PT handoff');
    expect(reset).toContain('this.#gpu.clearTemporalBuffers()');
    expect(reset).toContain('this.#samplesAccumulated = 0');

    const temporalClear = section(
      GPU_RESOURCES_SOURCE,
      'clearTemporalBuffers(): void {',
      '(Re)allocate the accum + aux textures',
    );
    expect(temporalClear).toContain('this.#sppm.sppmPixelStatsWidth !== 0');
    expect(temporalClear).toContain('buffers.add(this.#sppm.sppmPixelStatsBuffer)');
    expect(temporalClear).toContain('encoder.clearBuffer(buffer)');

    const allocation = section(
      GPU_RESOURCES_SOURCE,
      'ensureSppmPixelStatsBuffer(',
      '  #disposeSppmResources(): void {',
    );
    expect(allocation).toContain('enc.clearBuffer(candidate)');
    expect(allocation.indexOf('enc.clearBuffer(candidate)')).toBeLessThan(
      allocation.indexOf('this.#sppm.sppmPixelStatsBuffer = candidate'),
    );
    expect(allocation).toContain("fallback: 'throw'");
  });

  it('refreshes scene-centered launch bounds transactionally after geometry publication', () => {
    const commit = section(
      MUTATION_ROUTER_SOURCE,
      '#commitPreparedMutation(',
      'Add one whole primitive',
    );
    const publishPack = commit.indexOf('host.setGeoPack(nextGeoPack)');
    const refresh = commit.indexOf('host.refreshSceneGeometryStats?.()');
    const rollback = commit.indexOf('rollbackSceneGeometryStats?.()');
    expect(publishPack).toBeLessThan(refresh);
    expect(refresh).toBeLessThan(rollback);

    const callback = section(
      INDEX_SOURCE,
      'refreshSceneGeometryStats: () => {',
      'setScene: (scene) => this.setScene(scene)',
    );
    expect(callback).toContain('this.#computeSppmSceneStats()');
    expect(callback).toContain('this.#sppmR0 = previous.r0');
    expect(callback).toContain('this.#sppmSceneExtent = previous.extent');
    expect(callback).toContain('this.#sppmSceneCenter = previous.center');
    expect(callback).toContain('this.#gpu.sppm.sppmBuffersReady = previous.buffersReady');
  });

  it('defines explicit zero-work exits and no legacy reservoir estimator path', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (availableLightCount == 0u) { return; }');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain(
      'if (!(r0 > 0.0) || nPhotons == 0u) { return; }',
    );
    expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain('r0 <= 1e-9');
    expect(SPPM_GROUP3_BINDINGS_WGSL).toContain('atomicExchange(&sppmCellCounters[cellIdx]');
    for (const legacy of [
      'reservoirXi',
      'cellSampleScale',
      'streaming-window',
      'counter/stored',
      'insertion-normalised',
    ]) {
      expect(SPPM_GROUP3_BINDINGS_WGSL).not.toContain(legacy);
    }
  });
});
