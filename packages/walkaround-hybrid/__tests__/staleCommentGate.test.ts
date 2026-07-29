import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

function readSource(relative: string): string {
  return readFileSync(new URL(`../src/${relative}`, import.meta.url), 'utf8');
}

function readWorkspace(relative: string): string {
  return readFileSync(new URL(`../../../${relative}`, import.meta.url), 'utf8');
}

describe('Road D5 stale-comment gates', () => {
  it('documents generalized reservoir metadata as live reuse inputs', () => {
    const frameResources = readSource('pipeline/frameResources/createRestirGIFrameResources.ts');
    const resourceManager = readSource('pipeline/resourceManager.ts');

    expect(frameResources).not.toContain('written-but-unread');
    expect(frameResources).not.toContain('will read');
    expect(frameResources).toContain('sole live 28-u32 generalized-reuse ABI');
    expect(frameResources).toContain('canonical temporal/spatial');
    expect(resourceManager).toContain('generalized reconnection-shift metadata');
    expect(resourceManager).toContain('read by temporal/spatial passes and shade');
  });

  it('keeps RC cascade dispatch verification status current', () => {
    const cascadeDispatch = readWorkspace('packages/walkaround-rc/src/cascadeDispatch.ts');

    expect(cascadeDispatch).not.toContain('not verified');
    expect(cascadeDispatch).toContain('PLUS behavioral');
    expect(cascadeDispatch).toContain('GPU smoke');
  });

  it('keeps H38 fork-era DDGI/GI comments reconciled to the raw WebGPU path', () => {
    const risGi = readSource('shaders/risGi.wgsl.ts');
    const risGiNrc = readSource('shaders/risGiNrc.wgsl.ts');
    const probeUpdatePass = readSource('ddgi/probeUpdatePass.ts');
    const probeGrid = readSource('ddgi/probeGrid.ts');

    for (const source of [risGi, risGiNrc, probeUpdatePass, probeGrid]) {
      expect(source).not.toContain('path-traced fork');
      expect(source).not.toContain('three.js binding system');
      expect(source).not.toContain('three/webgpu');
    }
    expect(probeGrid).toContain('raw WebGPU bind groups');
  });

  it('keeps NRC bias and empty-record comments aligned with current code', () => {
    const risGiNrc = readSource('shaders/risGiNrc.wgsl.ts');
    const nrcSubsystem = readSource('neural/nrc/nrcSubsystem.ts');

    expect(risGiNrc).not.toContain('centroid-p');
    expect(risGiNrc).toContain('receiver-lobe/material');
    expect(nrcSubsystem).not.toContain('checking whether the first');
    expect(nrcSubsystem).toContain('const unpacked = unpackRecords(');
    expect(nrcSubsystem).toContain('if (unpacked.filled === 0) return;');
  });

  it('does not retain the retired GRIS selector or draft G-buffer bindings', () => {
    const pipeline = readSource('pipeline/WalkaroundGPUPipeline.ts');
    const temporalPass = readSource('pipeline/passes/TemporalGIReservoirPass.ts');
    const spatialPass = readSource('pipeline/passes/SpatialGIReservoirPass.ts');
    const bindingTable = readSource('pipeline/bindGroupDescriptors.ts');

    expect(pipeline).not.toContain('_grisReuseStructural');
    expect(temporalPass).not.toContain('_grisEnabled');
    expect(spatialPass).not.toContain('_grisEnabled');
    expect(spatialPass).not.toContain('GRIS gate');
    expect(bindingTable).not.toContain('gDepth/gNormal/gAlbedo/gRough/motionVec');
    expect(bindingTable).toContain('frame layout intentionally begins at binding 5');
  });
});
