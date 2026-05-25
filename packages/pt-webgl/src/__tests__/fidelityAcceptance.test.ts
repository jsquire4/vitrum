import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';

const enabled = process.env['VITRUM_PTWEBGL_FIDELITY_ACCEPTANCE'] === '1';
const runIfEnabled = enabled ? describe : describe.skip;

runIfEnabled('pt-webgl fidelity acceptance (env-gated)', () => {
  function readMetrics(): {
    readonly requiredScenarios: readonly string[];
    readonly minPsnr: number;
    readonly matchedRequired: Readonly<Record<string, readonly string[]>>;
    readonly missingRequired: readonly string[];
    readonly failingScenarios: readonly string[];
    readonly allRequiredPresent: boolean;
    readonly allPassing: boolean;
    readonly results: readonly {
      readonly scenarioId: string;
      readonly psnr: number;
      readonly minPsnr: number;
      readonly meanAbsDelta: number;
      readonly pass: boolean;
    }[];
  } {
    const path = process.env['VITRUM_PTWEBGL_FIDELITY_METRICS'];
    if (!path) {
      throw new Error(
        'VITRUM_PTWEBGL_FIDELITY_ACCEPTANCE=1 requires ' +
          'VITRUM_PTWEBGL_FIDELITY_METRICS=<json file> produced by tools/benchmark-runner.',
      );
    }
    const parsed = JSON.parse(readFileSync(path, 'utf8')) as {
      requiredScenarios?: string[];
      minPsnr?: number;
      matchedRequired?: Record<string, string[]>;
      missingRequired?: string[];
      failingScenarios?: string[];
      allRequiredPresent?: boolean;
      allPassing?: boolean;
      results?: Array<{
        scenarioId?: string;
        psnr?: number;
        minPsnr?: number;
        meanAbsDelta?: number;
        pass?: boolean;
      }>;
    };
    if (
      !Array.isArray(parsed.requiredScenarios) ||
      typeof parsed.minPsnr !== 'number' ||
      parsed.matchedRequired == null ||
      typeof parsed.matchedRequired !== 'object' ||
      Array.isArray(parsed.matchedRequired) ||
      !Array.isArray(parsed.missingRequired) ||
      !Array.isArray(parsed.failingScenarios) ||
      typeof parsed.allRequiredPresent !== 'boolean' ||
      typeof parsed.allPassing !== 'boolean' ||
      !Array.isArray(parsed.results)
    ) {
      throw new Error(
        `Invalid PT-WebGL fidelity metrics JSON at ${path}.`,
      );
    }
    const globalMinPsnr = parsed.minPsnr;
    return {
      requiredScenarios: parsed.requiredScenarios,
      minPsnr: globalMinPsnr,
      matchedRequired: parsed.matchedRequired,
      missingRequired: parsed.missingRequired,
      failingScenarios: parsed.failingScenarios,
      allRequiredPresent: parsed.allRequiredPresent,
      allPassing: parsed.allPassing,
      results: parsed.results.map((r) => ({
        scenarioId: String(r.scenarioId ?? ''),
        psnr: typeof r.psnr === 'number' ? r.psnr : Number.NaN,
        minPsnr: typeof r.minPsnr === 'number' ? r.minPsnr : globalMinPsnr,
        meanAbsDelta: typeof r.meanAbsDelta === 'number' ? r.meanAbsDelta : Number.NaN,
        pass: r.pass === true,
      })),
    };
  }

  it('includes all required scenarios', () => {
    const metrics = readMetrics();
    expect(metrics.missingRequired).toEqual([]);
    expect(metrics.allRequiredPresent).toBe(true);
    for (const required of metrics.requiredScenarios) {
      const matched = metrics.matchedRequired[required] ?? [];
      expect(Array.isArray(matched)).toBe(true);
      expect(matched.length).toBeGreaterThan(0);
      for (const id of matched) {
        expect(id === required || id.startsWith(`${required}.`) || id.startsWith(`${required}-`)).toBe(true);
      }
    }
  });

  it('reports finite numeric metrics for each scenario', () => {
    const metrics = readMetrics();
    for (const row of metrics.results) {
      expect(row.scenarioId.length).toBeGreaterThan(0);
      expect(Number.isFinite(row.psnr)).toBe(true);
      expect(Number.isFinite(row.minPsnr)).toBe(true);
      expect(Number.isFinite(row.meanAbsDelta)).toBe(true);
      expect(row.psnr).toBeGreaterThanOrEqual(0);
      expect(row.minPsnr).toBeGreaterThan(0);
      expect(row.meanAbsDelta).toBeGreaterThanOrEqual(0);
    }
  });

  it('meets configured PSNR threshold for all scenarios', () => {
    const metrics = readMetrics();
    expect(metrics.failingScenarios).toEqual([]);
    expect(metrics.allPassing).toBe(true);
    for (const row of metrics.results) {
      expect(row.psnr).toBeGreaterThanOrEqual(row.minPsnr);
      expect(row.pass).toBe(true);
    }
  });
});

