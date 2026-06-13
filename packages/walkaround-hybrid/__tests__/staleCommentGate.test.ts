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
});
