import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const COMMITTED_METRICS_PATH = fileURLToPath(
  new URL('../../../../tools/benchmark-runner/results/acceptance/ptwgpu-tlas-real.json', import.meta.url),
);

describe('pt-webgpu TLAS promotion acceptance', () => {
  function readMetrics(): {
    readonly schemaVersion?: string;
    readonly tlasVsLegacyMeanAbs: number;
    readonly tlasVsLegacyP95Abs: number;
    readonly tlasVsLegacyMaxAbs: number;
    readonly nanPixelCount: number;
    readonly orderedStats?: boolean;
    readonly pass?: {
      readonly mean?: boolean;
      readonly p95?: boolean;
      readonly peak?: boolean;
      readonly overall?: boolean;
    };
  } {
    const path = process.env['VITRUM_PTWGPU_TLAS_METRICS'] ?? COMMITTED_METRICS_PATH;
    const parsed = JSON.parse(readFileSync(path, 'utf8')) as {
      schemaVersion?: string;
      tlasVsLegacyMeanAbs?: number;
      tlasVsLegacyP95Abs?: number;
      tlasVsLegacyMaxAbs?: number;
      nanPixelCount?: number;
      orderedStats?: boolean;
      pass?: {
        mean?: boolean;
        p95?: boolean;
        peak?: boolean;
        overall?: boolean;
      };
    };
    if (
      typeof parsed.tlasVsLegacyMeanAbs !== 'number' ||
      typeof parsed.tlasVsLegacyP95Abs !== 'number' ||
      typeof parsed.tlasVsLegacyMaxAbs !== 'number' ||
      typeof parsed.nanPixelCount !== 'number'
    ) {
      throw new Error(
        `Invalid PT-WebGPU TLAS metrics JSON at ${path}. ` +
          'Expected numeric tlasVsLegacyMeanAbs, tlasVsLegacyP95Abs, tlasVsLegacyMaxAbs, and nanPixelCount.',
      );
    }
    return {
      ...(typeof parsed.schemaVersion === 'string' ? { schemaVersion: parsed.schemaVersion } : {}),
      tlasVsLegacyMeanAbs: parsed.tlasVsLegacyMeanAbs,
      tlasVsLegacyP95Abs: parsed.tlasVsLegacyP95Abs,
      tlasVsLegacyMaxAbs: parsed.tlasVsLegacyMaxAbs,
      nanPixelCount: parsed.nanPixelCount,
      ...(parsed.orderedStats !== undefined ? { orderedStats: parsed.orderedStats } : {}),
      ...(parsed.pass !== undefined ? { pass: parsed.pass } : {}),
    };
  }

  it('TLAS path remains visually close to legacy path on parity scenes', () => {
    const metrics = readMetrics();
    const maxDelta = Number(process.env['VITRUM_PTWGPU_TLAS_MAX_DELTA'] ?? '0.02');
    const maxP95 = Number(process.env['VITRUM_PTWGPU_TLAS_MAX_P95_DELTA'] ?? '0.06');
    const maxPeak = Number(process.env['VITRUM_PTWGPU_TLAS_MAX_PEAK_DELTA'] ?? '0.2');
    expect(metrics.tlasVsLegacyMeanAbs, 'mean abs delta exceeds VITRUM_PTWGPU_TLAS_MAX_DELTA').toBeLessThanOrEqual(maxDelta);
    expect(metrics.tlasVsLegacyP95Abs, 'p95 abs delta exceeds VITRUM_PTWGPU_TLAS_MAX_P95_DELTA').toBeLessThanOrEqual(maxP95);
    expect(metrics.tlasVsLegacyMaxAbs, 'peak abs delta exceeds VITRUM_PTWGPU_TLAS_MAX_PEAK_DELTA').toBeLessThanOrEqual(maxPeak);
  });

  it('reports finite non-negative TLAS metric fields', () => {
    const metrics = readMetrics();
    if (metrics.schemaVersion !== undefined) {
      expect(metrics.schemaVersion).toBe('ptwgpu-tlas-metrics-2026-05-25');
    }
    expect(Number.isFinite(metrics.tlasVsLegacyMeanAbs)).toBe(true);
    expect(Number.isFinite(metrics.tlasVsLegacyP95Abs)).toBe(true);
    expect(Number.isFinite(metrics.tlasVsLegacyMaxAbs)).toBe(true);
    expect(metrics.tlasVsLegacyMeanAbs).toBeGreaterThanOrEqual(0);
    expect(metrics.tlasVsLegacyP95Abs).toBeGreaterThanOrEqual(0);
    expect(metrics.tlasVsLegacyMaxAbs).toBeGreaterThanOrEqual(0);
    expect(metrics.tlasVsLegacyMeanAbs).toBeLessThanOrEqual(metrics.tlasVsLegacyP95Abs);
    expect(metrics.tlasVsLegacyP95Abs).toBeLessThanOrEqual(metrics.tlasVsLegacyMaxAbs);
    if (metrics.orderedStats !== undefined) {
      expect(metrics.orderedStats).toBe(true);
    }
    if (metrics.pass != null) {
      expect(metrics.pass.mean).toBe(true);
      expect(metrics.pass.p95).toBe(true);
      expect(metrics.pass.peak).toBe(true);
      expect(metrics.pass.overall).toBe(true);
    }
  });

  it('TLAS capture has no NaN/Inf artifacts', () => {
    const metrics = readMetrics();
    expect(metrics.nanPixelCount).toBe(0);
  });
});
