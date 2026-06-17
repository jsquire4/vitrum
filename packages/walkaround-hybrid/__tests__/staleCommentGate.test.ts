import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

function readSource(relative: string): string {
  return readFileSync(new URL(`../src/${relative}`, import.meta.url), 'utf8');
}

function readWorkspace(relative: string): string {
  return readFileSync(new URL(`../../../${relative}`, import.meta.url), 'utf8');
}

describe('Road D5 stale-comment gates', () => {
  it('documents GRIS reservoir cache fields as live reuse inputs', () => {
    const frameResources = readSource('pipeline/frameResources/createRestirGIFrameResources.ts');
    const resourceManager = readSource('pipeline/resourceManager.ts');

    expect(frameResources).not.toContain('written-but-unread');
    expect(frameResources).not.toContain('will read');
    expect(frameResources).toContain('read by the GRIS reuse');
    expect(resourceManager).toContain('READ by the GRIS reuse variants');
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
    expect(nrcSubsystem).toContain('scanning the entire encoded-');
  });
});
