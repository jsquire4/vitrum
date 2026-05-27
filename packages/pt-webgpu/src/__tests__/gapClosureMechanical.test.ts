import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const repoRoot = resolve(import.meta.dirname, '../../../..');
const baselineDir = resolve(repoRoot, 'tools/reference-renders/baseline');

/** pt-webgpu gap-closure rows with mechanical + baseline evidence (WG-D2 CI). */
const MECHANICAL_GAP_SCENARIOS = [
  'ptwgpu-parity-material-fields',
  'rfe03-layered-front-back',
  'rfe07-11-sss-mixed-panels',
  'rfe08-13-spectral-payload',
  'rfe14-thinfilm-angle-shift',
  'rfe09-bridge-global-cmf',
] as const;

describe('gap-closure mechanical baselines (WG-D2)', () => {
  for (const scenarioId of MECHANICAL_GAP_SCENARIOS) {
    it(`${scenarioId} has a committed baseline PNG`, () => {
      const path = resolve(baselineDir, `${scenarioId}.png`);
      expect(existsSync(path), `missing ${path}`).toBe(true);
      const buf = readFileSync(path);
      expect(buf.length).toBeGreaterThan(100);
    });
  }
});
