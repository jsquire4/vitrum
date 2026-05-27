/**
 * Sprint 10c / 14 — BDPT vs layered-BSDF metrics gate (opt-in).
 *
 *   VITRUM_BDPT_LAYERED_ACCEPTANCE=1
 *   VITRUM_BDPT_LAYERED_METRICS=<json from benchmark:bdpt-layered-mechanical>
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';

const ENABLED =
  typeof process !== 'undefined' &&
  process.env != null &&
  process.env['VITRUM_BDPT_LAYERED_ACCEPTANCE'] === '1';

describe.skipIf(!ENABLED)('BDPT vs layered-BSDF acceptance metrics', () => {
  function readMetrics(): { bdptDeltaMean: number } {
    const path = process.env['VITRUM_BDPT_LAYERED_METRICS'];
    if (!path) {
      throw new Error(
        'VITRUM_BDPT_LAYERED_ACCEPTANCE=1 requires VITRUM_BDPT_LAYERED_METRICS=<json file>',
      );
    }
    const parsed = JSON.parse(readFileSync(path, 'utf8')) as { bdptDeltaMean?: unknown };
    if (typeof parsed.bdptDeltaMean !== 'number') {
      throw new Error(`Invalid metrics JSON at ${path}`);
    }
    return { bdptDeltaMean: parsed.bdptDeltaMean };
  }

  it('layered vs BDPT captures differ beyond read noise (harness contract)', () => {
    expect(readMetrics().bdptDeltaMean).toBeGreaterThan(0.005);
  });
});

describe('BDPT layered acceptance gate', () => {
  it('is opt-in only', () => {
    expect(ENABLED).toBe(process.env['VITRUM_BDPT_LAYERED_ACCEPTANCE'] === '1');
  });
});
