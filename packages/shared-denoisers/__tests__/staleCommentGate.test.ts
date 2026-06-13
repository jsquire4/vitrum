import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

describe('Road D5 stale-comment gates', () => {
  it('keeps atrous-variance entry-point names from implying Schied SVGF', () => {
    const source = readFileSync(new URL('../src/wgsl/atrousVariance.wgsl.ts', import.meta.url), 'utf8');

    expect(source).toContain('NOT a Schied 2017 SVGF implementation');
    expect(source).toContain('legacy-named compute entry points');
  });
});
