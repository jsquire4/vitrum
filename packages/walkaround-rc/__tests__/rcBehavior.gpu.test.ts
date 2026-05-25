import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';

const RC_BEHAVIOR_ENABLED =
  typeof process !== 'undefined' &&
  process.env != null &&
  process.env['VITRUM_RC_BEHAVIOR_ACCEPTANCE'] === '1';

describe.skipIf(!RC_BEHAVIOR_ENABLED)('walkaround-rc behavior acceptance (GPU-gated)', () => {
  function readMetrics(): {
    readonly indirectEnergyDelta: number;
    readonly nanPixelCount: number;
  } {
    const path = process.env['VITRUM_RC_BEHAVIOR_METRICS'];
    if (!path) {
      throw new Error(
        'VITRUM_RC_BEHAVIOR_ACCEPTANCE=1 requires VITRUM_RC_BEHAVIOR_METRICS=<json file> ' +
        'produced by tools/benchmark-runner.',
      );
    }
    const parsed = JSON.parse(readFileSync(path, 'utf8')) as {
      indirectEnergyDelta?: unknown;
      nanPixelCount?: unknown;
    };
    if (
      typeof parsed.indirectEnergyDelta !== 'number' ||
      typeof parsed.nanPixelCount !== 'number'
    ) {
      throw new Error(
        `Invalid RC behavior metrics JSON at ${path}. ` +
        'Expected numeric indirectEnergyDelta and nanPixelCount.',
      );
    }
    return {
      indirectEnergyDelta: parsed.indirectEnergyDelta,
      nanPixelCount: parsed.nanPixelCount,
    };
  }

  it('produces non-zero indirect contribution without NaN/Inf artifacts', () => {
    const metrics = readMetrics();
    expect(metrics.indirectEnergyDelta).toBeGreaterThan(0.001);
    expect(metrics.nanPixelCount).toBe(0);
  });
});

