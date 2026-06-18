import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const environmentSourcePath = fileURLToPath(new URL('../scene/environment.ts', import.meta.url));

describe('SceneEnvironment API docs', () => {
  it('do not describe walkaround HDRI transport as a pre-env-pillar tint-only path', () => {
    const source = readFileSync(environmentSourcePath, 'utf8');

    expect(source).not.toContain('Wave-4 env pillar');
    expect(source).not.toContain('full directional IBL is not yet complete');
    expect(source).not.toContain('DDGI probe misses still use a procedural gradient');
    expect(source).not.toContain('contributes only as an approximate tint');

    expect(source).toContain('walkaround-hybrid` — IMPLEMENTED');
    expect(source).toContain('envRadiance');
    expect(source).toContain('DDGI probe updates receive the same');
  });
});
