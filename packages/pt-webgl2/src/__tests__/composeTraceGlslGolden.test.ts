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
// The final completeness pass crosses delta vertices without terminating the
// BDPT density recurrence, gives shadow-disabled finite lights exclusive NEE
// ownership, and makes mesh-emitter sampling and forward visibility obey the
// material's authored sidedness. The common transport delta is +669 bytes;
// enabling BDPT adds its recurrence changes for a +1633-byte total delta.
// The render-target closure folds main-path samples into a shader-readable
// progressive history, initializes fog-miss distance, and packs the two
// environment inverse-CDF lookups into one sampler.
// Bump-map finite differences now derive their independent UV steps from each
// packed source rectangle rather than assuming a fixed or atlas-wide extent.
// The 2026-07-30 numerical-safety closure makes transport distances,
// normalization, MIS products, directional cones, BVH slabs, BDPT records, and
// finite-light visibility well-defined across every accepted scene scale.
// The final frame-boundary closure adds pre-mutation camera/transport
// representability, scale-stable thin-lens refocusing, exact max-float
// infinite-distance classification, staged environment radiance products, and
// finite-f32 exposure with an RGBA16F final-write ceiling.
// The represented environment-emitter PMF now reads the packed CDF texture
// through EquirectHdrInfo.distributionWeights, matching the declared GLSL field.
// length + sha256(utf8) of composeTraceGlsl(features) for each set.
const GOLDENS: Record<string, { length: number; sha256: string }> = {
  default: {
    length: 236986,
    sha256: 'e78d484646cbac1a5656b50cfadd18d297a43b524e9eb50ebfba21cebe68d944',
  },
  bdptOn: {
    length: 307565,
    sha256: '49e49b5b0e48697b7d3576a8add8cd017573220f2fe108a613d6d546c73350d1',
  },
  bdptOff: {
    length: 236986,
    sha256: 'e78d484646cbac1a5656b50cfadd18d297a43b524e9eb50ebfba21cebe68d944',
  },
  sobol: {
    length: 242858,
    sha256: '150ac91e3bf35ba4bb3c41c76bbe8a0e4a185155b0580630e50c6f7c22782480',
  },
  orthographic: {
    length: 236986,
    sha256: 'e78d484646cbac1a5656b50cfadd18d297a43b524e9eb50ebfba21cebe68d944',
  },
  dof: {
    length: 236986,
    sha256: 'e78d484646cbac1a5656b50cfadd18d297a43b524e9eb50ebfba21cebe68d944',
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
