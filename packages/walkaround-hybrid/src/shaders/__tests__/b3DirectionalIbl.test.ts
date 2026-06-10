/**
 * B3 (road-to-100) — directional IBL shader-composition gate.
 *
 * Verifies the env-sample module composes cleanly into ris/risGi/shade: the
 * directional helpers are present exactly once, the sky-miss / GI-escape sites
 * call envRadiance, and the env bindings + EnvParams struct appear once per
 * composed pass (a duplicate would be a naga redefinition error). This is the
 * vitest proxy; the runtime naga compile is the pre-push T1 GPU smoke.
 */
import { describe, it, expect } from 'vitest';
import { composeWgsl } from '../../pipeline/wgslComposer.js';
import { WGSL_MODULES } from '../../pipeline/wgslModules.js';
import { RIS_MODULE } from '../ris.wgsl.js';
import { RIS_GI_MODULE } from '../risGi.wgsl.js';
import { SHADE_MODULE } from '../shade.wgsl.js';

function countOccurrences(haystack: string, needle: string): number {
  let n = 0;
  let i = haystack.indexOf(needle);
  while (i >= 0) { n += 1; i = haystack.indexOf(needle, i + needle.length); }
  return n;
}

describe('B3 — directional IBL composes into ris/risGi/shade', () => {
  for (const [name, mod] of [
    ['ris', RIS_MODULE],
    ['risGi', RIS_GI_MODULE],
    ['shade', SHADE_MODULE],
  ] as const) {
    it(`${name} composes with the env helpers present exactly once`, () => {
      const src = composeWgsl(mod, WGSL_MODULES);
      // The env helpers + bindings appear EXACTLY once (no duplicate-symbol /
      // duplicate-binding naga error).
      expect(countOccurrences(src, 'fn envRadiance(')).toBe(1);
      expect(countOccurrences(src, 'fn envImportanceSample(')).toBe(1);
      expect(countOccurrences(src, 'struct EnvParams {')).toBe(1);
      expect(countOccurrences(src, '@group(1) @binding(15) var env_map')).toBe(1);
      expect(countOccurrences(src, '@group(1) @binding(19) var<uniform> envParams')).toBe(1);
      // The helpers reference symbols that MUST be in scope by composition order.
      expect(src).toContain('fn safe_normalize');
      expect(src).toContain('struct WalkaroundUBO');
    });
  }

  it('ris sky-miss + risGi GI-escape + shade sky-miss call envRadiance', () => {
    const ris = composeWgsl(RIS_MODULE, WGSL_MODULES);
    const risGi = composeWgsl(RIS_GI_MODULE, WGSL_MODULES);
    const shade = composeWgsl(SHADE_MODULE, WGSL_MODULES);
    expect(ris).toContain('envRadiance(primaryRay.direction)');
    expect(risGi).toContain('Lo = envRadiance(wi)');
    expect(shade).toContain('envRadiance(primaryRay.direction)');
    // The old scalar-only sky-miss must be gone from these sites.
    expect(ris).not.toContain('let skyColor = ubo.skyTint * ubo.skyIrradiance;');
  });
});
