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
// The 2026-07-29 audit closure adds RGB thin-film evaluation in the default
// renderer, exact signed HG inversion shared by both volume samplers, the phase
// factor missing from SSS throughput, and a cancellation-safe aligned HG density
// at the shared ±0.999999 numerical cap.
// The dispersion closure evaluates each material's packed two-term Cauchy B
// coefficient directly in nm², anchors authored IOR at the Fraunhofer d line,
// and removes the obsolete scene-global three-term Cauchy uniforms. That
// semantic replacement is exactly -337 bytes in every representative variant.
// The bounded-BDPT agreement closure gives terminal NEE full ownership when no
// continuation can run, enforces one eye-plus-light path-depth budget, and
// preserves a light vertex's RGB throughput when a delayed reverse-density
// patch updates its row-2 w lane.
// length + sha256(utf8) of composeTraceGlsl(features) for each set.
const GOLDENS: Record<string, { length: number; sha256: string }> = {
  default: {
    length: 139235,
    sha256: '06c9902d5d7ec324ab885f09ac4794d7a5046b14fa3f383d09b4dd7bd834ad1f',
  },
  bdptOn: {
    length: 184194,
    sha256: 'e82db65f4d826df58407415195b84a5d2733bb969df03ffa8b924171e0a9e7de',
  },
  bdptOff: {
    length: 139235,
    sha256: '06c9902d5d7ec324ab885f09ac4794d7a5046b14fa3f383d09b4dd7bd834ad1f',
  },
  sobol: {
    length: 145107,
    sha256: 'acec1c2160d3b8f4735de9d24c7e60579b70dfe3d29e41f2e499b0003b4f7557',
  },
  orthographic: {
    length: 139235,
    sha256: '06c9902d5d7ec324ab885f09ac4794d7a5046b14fa3f383d09b4dd7bd834ad1f',
  },
  dof: {
    length: 139235,
    sha256: '06c9902d5d7ec324ab885f09ac4794d7a5046b14fa3f383d09b4dd7bd834ad1f',
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
