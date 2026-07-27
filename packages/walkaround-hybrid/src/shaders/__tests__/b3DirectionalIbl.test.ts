/**
 * B3 (road-to-100) — directional IBL shader-composition gate.
 *
 * Verifies the env-sample module composes cleanly into ris/risGi/shade: the
 * directional helpers are present exactly once, the sky-miss / GI-escape sites
 * call envRadiance, and the env bindings + EnvParams struct appear once per
 * composed pass (a duplicate would be a naga redefinition error). This is the
 * vitest proxy; the runtime naga compile is the pre-push T1 GPU smoke.
 *
 * Wave 4: added assertions for the env DI NEE candidate call sites:
 *   - ris calls envImportanceSample (M_ENV loop) and ENV_SAMPLE_SENTINEL
 *   - shade's lo_direct handles ENV_SAMPLE_SENTINEL (envDirFromXi + envRadiance)
 *   - temporal/spatial reuse calls restir_di_compute_phat_xi (not the old wrapper)
 *   - risGiNrc now calls envRadiance in its GI-escape path (parity with risGi)
 *
 * Wave 4: 2026-06-10 — RENDER-CHANGING for HDRI scenes (env as DI NEE candidate,
 * NRC env parity). A/B pending V28-B.
 */
import { describe, it, expect } from 'vitest';
import { composeWgsl } from '../../pipeline/wgslComposer.js';
import { WGSL_MODULES } from '../../pipeline/wgslModules.js';
import { RIS_MODULE } from '../ris.wgsl.js';
import { RIS_GI_MODULE } from '../risGi.wgsl.js';
import { SHADE_MODULE } from '../shade.wgsl.js';
import { TEMPORAL_MODULE } from '../temporal.wgsl.js';
import { SPATIAL_MODULE } from '../spatial.wgsl.js';
import { buildRisGiNrcModule } from '../risGiNrc.wgsl.js';

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

// ── Wave 4 — env DI NEE candidate structural gates ──────────────────────────
describe('Wave 4 — env DI NEE candidate call sites', () => {
  it('ris calls envImportanceSample in the M_ENV loop', () => {
    const src = composeWgsl(RIS_MODULE, WGSL_MODULES);
    // The M_ENV constant must be declared.
    expect(src).toContain('const M_ENV');
    // The importance-sample call must be present (env DI NEE candidate loop).
    expect(src).toContain('envImportanceSample(');
    // The sentinel constant must be in scope.
    expect(src).toContain('ENV_SAMPLE_SENTINEL');
    // envDirToXi must be called to encode the sampled direction.
    expect(src).toContain('envDirToXi(');
    // The env candidate must use the xi-aware pHat helper.
    expect(src).toContain('restir_di_compute_phat_xi(');
  });

  it('ris finalizes DI reservoirs with selected xi and support-family M', () => {
    const src = composeWgsl(RIS_MODULE, WGSL_MODULES);
    expect(src).toContain('var mAreaSupport = 0u;');
    expect(src).toContain('var mEnvSupport = 0u;');
    expect(src).toContain('mAreaSupport = mAreaSupport + 1u;');
    expect(src).toContain('mEnvSupport = mEnvSupport + 1u;');
    expect(src).toContain('let pHatZ = restir_di_compute_phat_xi(lid, r.xi, surf);');
    expect(src).toContain('r.areaM = mAreaSupport;');
    expect(src).toContain('r.envM = mEnvSupport;');
    expect(src).toContain('r.M = mAreaSupport + mEnvSupport;');
    expect(countOccurrences(src, 'restir_di_compute_phat_from_surface(')).toBe(1);
  });

  it('shade lo_direct handles ENV_SAMPLE_SENTINEL via envDirFromXi + envRadiance', () => {
    const src = composeWgsl(SHADE_MODULE, WGSL_MODULES);
    // The sentinel branch must be present in lo_direct.
    expect(src).toContain('ENV_SAMPLE_SENTINEL');
    // Decode direction from xi.
    expect(src).toContain('envDirFromXi(');
    // Evaluate env radiance.
    expect(src).toContain('envColor');
    // Finite emitters must shade the selected reservoir sample, not a fresh xi.
    expect(src).toContain('let ls = sampleEmitterPoint(e, r.xi);');
    expect(src).not.toContain('sampleEmitterPoint(e, rand2');
  });

  it('temporal and spatial reuse call restir_di_compute_phat_xi (not the old wrapper)', () => {
    const temporal = composeWgsl(TEMPORAL_MODULE, WGSL_MODULES);
    const spatial  = composeWgsl(SPATIAL_MODULE, WGSL_MODULES);
    // Both must use the xi-aware variant so env-sentinel reservoirs survive reuse.
    expect(temporal).toContain('restir_di_compute_phat_xi(');
    expect(spatial).toContain('restir_di_compute_phat_xi(');
    // The env helpers must be present via restirPHat → environmentSample.
    expect(countOccurrences(temporal, 'fn envRadiance(')).toBe(1);
    expect(countOccurrences(spatial,  'fn envRadiance(')).toBe(1);
  });

  it('envImportanceSample has at least one call site in the ris source (zero-call-site regression guard)', () => {
    // This test is the structural WGSL zero-call-site guard: if a future refactor
    // removes the M_ENV loop without removing the import, this catches it.
    const risSrc = composeWgsl(RIS_MODULE, WGSL_MODULES);
    // We already assert the import presence above; here we assert it's USED.
    const importCount = countOccurrences(risSrc, 'fn envImportanceSample(');
    const callCount   = countOccurrences(risSrc, 'envImportanceSample(');
    // callCount includes the definition; subtract it to get call-only count.
    expect(callCount - importCount).toBeGreaterThanOrEqual(1);
  });

  it('risGiNrc GI-escape calls envRadiance (parity with risGi)', () => {
    // buildRisGiNrcModule requires a config; use minimal valid values.
    // NrcQueryWgslOptions = NrcEncodeWgslOptions + { width, outWidth, hidden }.
    // recordCap is a WGSL uniform (not a TS compile-time option).
    const nrcMod = buildRisGiNrcModule({
      levels: 4,
      featuresPerEntry: 4,
      oneBlobBins: 4,
      width: 16,
      outWidth: 3,
      hidden: 1,
    });
    const src = composeWgsl(nrcMod, WGSL_MODULES);
    // The GI-escape sky-miss must use envRadiance (not the old scalar fallback).
    expect(src).toContain('Lo = envRadiance(wi)');
    // The old scalar-only path must be absent (Lo = ubo.skyTint * ubo.skyIrradiance).
    // We check for the assignment form used in the NRC body before Wave 4.
    expect(src).not.toContain('Lo = ubo.skyTint * ubo.skyIrradiance;');
    // env bindings must be present exactly once.
    expect(countOccurrences(src, 'fn envRadiance(')).toBe(1);
  });
});
