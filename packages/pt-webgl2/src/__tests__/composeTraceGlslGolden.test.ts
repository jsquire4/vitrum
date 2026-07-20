// T3-D byte-identity golden: the composed WebGL2 trace fragment body is the
// load-bearing artifact. The composeTraceGlsl split (renderMain.glsl.ts +
// uniformManifest.ts extraction) MUST NOT perturb a single byte of the composed
// string for any representative feature set. This test snapshots the sha256 +
// length of the composed GLSL for the three feature sets the composer branches on
// (bdpt on/off is the only JS-compose-time branch). Captured on the pre-refactor
// code; if any assembly-order or whitespace drift occurs, these hashes fail.

import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { composeTraceGlsl } from '../glsl/composeTraceGlsl.js';
import {
  DEFAULT_TRACE_FEATURES,
  type TraceFeatures,
} from '../featureTypes.js';

function sha(s: string): string {
  return createHash('sha256').update(s, 'utf8').digest('hex');
}

const FEATURE_SETS: Record<string, TraceFeatures> = {
  default: DEFAULT_TRACE_FEATURES,
  bdptOn: { ...DEFAULT_TRACE_FEATURES, bdpt: true },
  bdptOff: { ...DEFAULT_TRACE_FEATURES, bdpt: false },
  sobol: { ...DEFAULT_TRACE_FEATURES, randomType: 1 },
  orthographic: { ...DEFAULT_TRACE_FEATURES, cameraType: 1 },
  dof: { ...DEFAULT_TRACE_FEATURES, dof: true },
};

// Byte-identity goldens captured 2026-07-20 on the pre-T3-D-split source.
// length + sha256(utf8) of composeTraceGlsl(features) for each set.
const GOLDENS: Record<string, { length: number; sha256: string }> = {
  default: { length: 204466, sha256: '575bb96d6a9e1161194ed1f652d1e289d07170f9bf1fcd32b9bd728672bc77aa' },
  bdptOn: { length: 233704, sha256: '170f99e69244bad1eb33c51bfb858833659865e989fb5ad270cbde8d49b89c88' },
  bdptOff: { length: 204466, sha256: '575bb96d6a9e1161194ed1f652d1e289d07170f9bf1fcd32b9bd728672bc77aa' },
  sobol: { length: 204466, sha256: '575bb96d6a9e1161194ed1f652d1e289d07170f9bf1fcd32b9bd728672bc77aa' },
  orthographic: { length: 204466, sha256: '575bb96d6a9e1161194ed1f652d1e289d07170f9bf1fcd32b9bd728672bc77aa' },
  dof: { length: 204466, sha256: '575bb96d6a9e1161194ed1f652d1e289d07170f9bf1fcd32b9bd728672bc77aa' },
};

describe('composeTraceGlsl byte-identity golden (T3-D)', () => {
  for (const [name, features] of Object.entries(FEATURE_SETS)) {
    it(`composed GLSL for "${name}" matches the pinned golden`, () => {
      const composed = composeTraceGlsl(features);
      const golden = GOLDENS[name]!;
      // Self-seed on first run: if the golden is empty, print the values so the
      // maintainer can paste them in. This keeps the test honest — an empty
      // golden fails until pinned.
      if (golden.sha256 === '') {
        console.warn(`GOLDEN[${name}] = { length: ${composed.length}, sha256: '${sha(composed)}' }`);
      }
      expect(composed.length).toBe(golden.length);
      expect(sha(composed)).toBe(golden.sha256);
    });
  }

  it('bdptOn differs from bdptOff (the only compose-time branch is real)', () => {
    expect(composeTraceGlsl(FEATURE_SETS.bdptOn!)).not.toBe(
      composeTraceGlsl(FEATURE_SETS.bdptOff!),
    );
  });
});
