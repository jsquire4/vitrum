// T3-D byte-identity golden: the composed WebGL2 trace fragment body is the
// load-bearing artifact. The composeTraceGlsl split (renderMain.glsl.ts +
// uniformManifest.ts extraction) MUST NOT perturb a single byte of the composed
// string for any representative feature set. This test snapshots the sha256 +
// length of the composed GLSL for representative feature sets the composer branches on.
// Captured on the pre-refactor
// code; if any assembly-order or whitespace drift occurs, these hashes fail.

import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { composeTraceGlsl } from '../glsl/composeTraceGlsl.js';
import { DEFAULT_TRACE_FEATURES, type TraceFeatures } from '../featureTypes.js';

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

// Byte-identity goldens updated 2026-07-24 after bounded general BDPT gained an
// eight-entry medium stack, exact majorant-ratio Beer transport, HG phase PDFs,
// zero-preserving light-power selection, and both BSDF tiers gained exact-zero
// discrete roughness plus cutoff-free authored-positive GGX/HG support. The
// current capture additionally pins stable volume-particle HG inversion,
// zero-vector TIR rejection, strict LIFO participating-medium exits, complete
// medium SurfaceRecord initialization, and finite-light NEE measure ownership.
// The 2026-07-27 closure removes unreachable global-medium Jakob and flat-
// shading fork lanes, and pins the corrected punctual-spot back-axis convention.
// The 2026-07-28 closure makes camera-surface depth zero-based for light maps
// and removes a duplicate diffuse-transmission attenuation. It also removes
// the permanently-false internal liteMode lane so every compiled BSDF has one
// full-fidelity behavior rather than an unreachable experimental branch, and
// drops two zero-consumer BDPT octahedral-direction helpers.
// The compiler-surface closure also deletes unexposed background/debug/stained-
// glass/stratified branches, fixed-only cache-key dimensions, and the orphaned
// heroWeightFromRgb alias.
// C41 adds source-rectangle offsets to mapped-material atlas sampling so small
// maps can share split RGBA8 LDR and RGBA16F radiance-atlas layers without
// changing authored filter/wrap behavior.
// The 2026-07-29 radiometric closure aligns analytic area lights with the core
// one-sided half-extent contract in NEE, forward-hit MIS, and BDPT.
// The affine-containment closure replaces orthogonal-only independent axis
// projection with the full Gram solve for sheared analytic-light bases. Its
// compacted shared source delta is exactly +1337 bytes in every variant.
// length + sha256(utf8) of composeTraceGlsl(features) for each set.
const GOLDENS: Record<string, { length: number; sha256: string }> = {
  default: {
    length: 139197,
    sha256: '4b3b94191a708de8d23f174dd4d89d7c050691a0cd7ebc6fde2304780516f0b6',
  },
  bdptOn: {
    length: 184015,
    sha256: 'a241092f4703e44ee9aef29d16795d90c99680552fc47515bccbe593bd1c1e02',
  },
  bdptOff: {
    length: 139197,
    sha256: '4b3b94191a708de8d23f174dd4d89d7c050691a0cd7ebc6fde2304780516f0b6',
  },
  sobol: {
    length: 145069,
    sha256: '19b62e98126c24f9a72df3048b1c1991a7583ab56916cfae16a3ffba9331ebc6',
  },
  orthographic: {
    length: 139197,
    sha256: '4b3b94191a708de8d23f174dd4d89d7c050691a0cd7ebc6fde2304780516f0b6',
  },
  dof: {
    length: 139197,
    sha256: '4b3b94191a708de8d23f174dd4d89d7c050691a0cd7ebc6fde2304780516f0b6',
  },
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
        console.warn(
          `GOLDEN[${name}] = { length: ${composed.length}, sha256: '${sha(composed)}' }`,
        );
      }
      expect(composed.length).toBe(golden.length);
      expect(sha(composed)).toBe(golden.sha256);
    });
  }

  it('bdptOn differs from bdptOff (the compose-time branch is real)', () => {
    expect(composeTraceGlsl(FEATURE_SETS.bdptOn!)).not.toBe(
      composeTraceGlsl(FEATURE_SETS.bdptOff!),
    );
  });
});
