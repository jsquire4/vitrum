/**
 * Byte-identity composed-WGSL goldens for the T4 (group A) shader dedup wave.
 *
 * The complexity-sweep 2026-07-20 plan (WAVE T4, group A) refactors several
 * `shaders/*.wgsl.ts` modules to remove hand-cloned duplication (risGiNrc≡risGi
 * body, temporalGi/spatialGi shared preamble, shadingTerms/transparentOit/regir
 * pure-math). The ABSOLUTE constraint is that the *composed* WGSL string emitted
 * by `composeWgsl(MODULE, registry)` for every affected pipeline is
 * BYTE-FOR-BYTE unchanged.
 *
 * Per CLAUDE.md, byte-identity string goldens are necessary but NOT sufficient
 * (a shared bug can survive them, and naga is the real compile gate) — but they
 * ARE the primary tripwire for the accidental-byte-drift class this wave risks.
 * These goldens are captured against the PRE-refactor composed output (sha256 +
 * length) and MUST stay green through the refactor.
 *
 * Registry construction mirrors `pipelineCompiler.compilePipelines`: the sole
 * live generalized-reuse reservoir and GI pass roots are composed directly.
 * The NRC gi-ris module is built from a representative config.
 */

import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';

import { composeWgsl } from '../pipeline/wgslComposer.js';
import { WGSL_MODULES } from '../pipeline/wgslModules.js';
import { RIS_GI_MODULE } from '../shaders/risGi.wgsl.js';
import { buildRisGiNrcModule, type RisGiNrcConfig } from '../shaders/risGiNrc.wgsl.js';
import { TEMPORAL_GI_MODULE } from '../shaders/temporalGi.wgsl.js';
import { SPATIAL_GI_MODULE } from '../shaders/spatialGi.wgsl.js';
import { SHADE_MODULE } from '../shaders/shade.wgsl.js';
import { TRANSPARENT_OIT_MODULE } from '../shaders/transparentOit.wgsl.js';
import { REGIR_MODULE, REGIR_BUILD_MODULE } from '../shaders/regir.wgsl.js';
import type { WgslModule } from '../wgslTypes.js';

// Representative NRC config (mirrors nrcQueryHarness.ts CFG shape; the exact
// values only affect the NRC helper prefix sizes, not the shared body bytes).
const NRC_CFG: RisGiNrcConfig = {
  levels: 4,
  featuresPerEntry: 2,
  oneBlobBins: 6,
  width: 16,
  outWidth: 3,
  hidden: 2,
};

function canonicalRegistry(): ReadonlyMap<string, WgslModule> {
  return new Map(WGSL_MODULES);
}

function digest(code: string): { sha256: string; length: number } {
  return { sha256: createHash('sha256').update(code, 'utf8').digest('hex'), length: code.length };
}

// Composed-output goldens. Originally captured 2026-07-20 before T4 group-A.
// Any change here = a byte-drift regression and MUST be investigated, not
// blindly re-pinned.
//
// T4 group-B refresh (2026-07-20): the barycentric+Beer-Lambert glass-tint
// decode in `surfaceTextures.wgsl.ts` was hoisted into a shared
// `_bvhBeerTintFactor` helper (D8-5) — a naga-validated (shader-gate 40/40),
// semantically-identical WGSL restructure that shrinks every pipeline including
// surfaceTextures by exactly 841 bytes. The seven surfaceTextures-bearing
// goldens (risGi{off,gris}, risGiNrc{off,gris}, shade{off,gris},
// transparentOit.off) are re-pinned below to the post-hoist bytes. The offset-
// ABI single-sourcing and the alpha-mask walk-wrapper templating (also T4
// group-B) are byte-identical → they did NOT move these goldens. temporalGi /
// spatialGi / regir do not include surfaceTextures and are unchanged.
//
// NRC hardening refresh (2026-07-21): finite-positive reservoir acceptance,
// bounded diagnostics, and shared reservoir representability guards changed the
// composed bytes intentionally. These values were captured from the fully
// composed modules after the NRC semantic and shader-validation gates passed.
// The RIS-GI-only values also include a 2026-07-21 correction of its stale
// Phase-0 GRIS metadata comment; executable WGSL is unchanged by that repin.
// Packed-scene-arena refresh (2026-07-22): scene storage was consolidated into
// three portable arenas, and all affected call sites now use the value-return
// loader surface. Semantic composition, binding-shape, and shader-validation
// tests were audited before these intentionally changed bytes were re-pinned.
// Renderer-correctness refresh (2026-07-27): the shared PCG pixel hash now
// consumes its salt, PrimarySurface carries the glass domain into canonical DI
// p-hat evaluation, glass direct lighting retains reflection-only BRDF
// families, and finite-sun samples are frame-scrambled for temporal
// integration. The production shader gate compiled 72/72 modules and created
// 45/45 pipelines before these intentional semantic bytes were re-pinned.
// Inert-resource/GRIS-layout refresh (2026-07-27): RIS-GI dropped its unread
// gNormalDepth binding, ReGIR dropped its zero-read emitter buffer, and the
// GRIS reservoir removed two never-consumed cached scalars (30 → 28 u32).
// The production shader gate compiled all 74 modules before this repin.
// DDGI packed-state refresh (2026-07-27): relocation/classification moved into
// the irradiance atlas's reserved rgba16float texel, the standalone sampled
// state binding was removed, and DDGI sampling now uses the dedicated linear
// filtering sampler. The production shader gate compiled 79/79 modules and
// created 51/51 pipelines before these intentional bytes were re-pinned.
// Whitespace hygiene removed 10 nonsemantic trailing-space bytes from the
// shared NRC suffix after that gate; only the two NRC compositions changed.
// Default-material energy refresh (2026-07-28): the thin-film F0 marker now
// occupies disjoint [2,3], reflection eval/sample/pdf share a bounded
// continuous roughness, and default clearcoat/sheen lobes remain active at
// authored roughness zero. The shipped walkaround/RC shader gate compiled all
// 31 compositions before this intentional semantic repin.
// ReSTIR-DI temporal/spatial closure (2026-07-28): PrimarySurface gained
// stable correspondence keys, DI reuse gained finite generalized-Talbot
// helpers with area/environment support accounting, and the temporal/spatial
// kernels now perform recast rejection and real ping-pong reuse. The shipped
// walkaround/RC shader gate compiled all 31 compositions before this repin.
// Extreme-density hardening evaluates the generalized-Talbot denominator with
// max-normalized log2 terms, preserving ratios when finite M*pHat products
// exceed f32. The same 31/31 shipped compositions passed the shader gate.
// Transparent visibility overflow now preserves explicit-transmission
// ownership, and transparent OIT bounds its cosine-importance division before
// rgba16float storage. The affected compositions were re-pinned after focused
// shader and numerical-oracle gates.
// Generalized-reuse closure (2026-07-28): the compact execution ABI and
// duplicate OFF/GRIS shader roots were retired, the canonical producer/reuse
// path now uses complete log-domain transformed-density MIS, and the orphaned
// scalar-Jacobian module was removed. These goldens pin the sole live roots.
// ReSTIR-DI overflow-ratio closure (2026-07-28): temporal/spatial reuse now
// shares a maximum log weight across candidates before weighted-reservoir
// sampling, preserving relative probability when multiple finite pHat*W
// products exceed f32. The production shader gate passed before this repin.
// Primary-visibility source reconciliation then removed shade's stale
// fallback/G-buffer prose; executable WGSL is unchanged by that comment-only
// repin.
// Layered-extension energy closure (2026-07-28): KHR clearcoat now attenuates
// every lower layer with its authored-normal Fresnel weight, and KHR sheen
// attenuates the base with the Estevez-Kulla directional-albedo fit before the
// clearcoat layer is applied. The shader gate compiled 78/78 modules, including
// all 29 shipped walkaround/RC compositions, before this intentional repin.
// Tier-1 semantic closure (2026-07-28): GI producer/reuse targets now retain
// rich receiver lobes, material-atlas F0 obeys KHR_specular's absolute domain,
// IOR=0 keeps its infinite-IOR meaning, DDGI feedback uses the canonical
// relocated/visibility-aware sampler, and checkerboard history requires a
// valid motion payload. The shipped walkaround/RC gate compiled 29/29 roots
// before this intentional repin.
// Multi-UV tangent-frame closure (2026-07-28): authored glTF tangents now
// override the derivative frame only for TEXCOORD_0; maps selecting UV1+
// retain the frame derived from their exact authored lane. The production
// shader gate compiled 78/78 modules (29/29 shipped walkaround/RC roots)
// before this intentional semantic repin.
// Executable-surface cleanup (2026-07-28): removed unreferenced compatibility
// GGX evaluators, environment/GRIS/material-atlas wrappers, the duplicated NRC
// one-blob helper, and the retired compact GI finaliser path. The canonical GI
// producer/reuse roots and every live rich-material evaluator remain composed.
// Canonical-GI prose reconciliation (2026-07-28): reservoirGi's obsolete
// "compact vs GRIS" finalisation comment was updated to describe the sole live
// generalized-reuse path. This adds exactly 43 comment bytes to every root
// below; executable WGSL is unchanged.
// Emitter-distance regularization (2026-07-29): transparent OIT and the NRC
// independent teacher now use the shared runtime emitterDist2Floor geometry
// helper, matching the live opaque shading and RIS estimators.
// Final executable-surface hygiene (2026-07-29): unused scalar-alpha/boolean
// shadow wrappers, the bare GI reservoir update, and the shader-side arena
// validation chain were removed. Live textured/transmittance/metadata paths
// remain pinned by focused semantic tests and the shader compiler gate.
// Final renderer audit closure (2026-07-29): fractional metallic surfaces keep
// their dielectric diffuse GI share, retired UBO controls are explicit zero ABI
// pads, the scene arena name exposes BVH ownership to validation adapters, NRC
// termination remains finite in f32, and unread environment/ReGIR resources
// were removed. Focused estimator, UBO-layout, binding-limit, and composition
// tests pin each intentional semantic/ABI change represented by these bytes.
// KHR punctual-range + SMS hygiene (2026-07-29): every walkaround punctual
// route uses the unsquared KHR range window, and the SMS multiplicity helper
// returns only its consumed weight rather than carrying three unread fields.
// Final completeness closure (2026-07-29): runtime-sized light tables, explicit
// material-presence metadata, atlas alias validation, and one-/two-sided
// emissive transport intentionally changed these shared roots. The semantic
// unit oracles and full shader gate are required alongside these byte pins.
// Transport/material ABI closure (2026-07-30): compact material ABI v2,
// authored atlas addressing, ordered material/instance-owned glass media,
// canonical Beer attenuation, and finite DDGI miss observations intentionally
// changed the reviewed shared roots below.
// Finite-f32 environment closure (2026-07-30): all scalar/HDRI environment
// products now route through one fail-closed scaling helper. Compact material
// decode and DDGI runtime-light hardening are included in these live-source
// digests; focused semantic tests and the shader gate accompany this repin.
// Material-atlas radiance closure (2026-07-30): emissive/light-map products
// now fail dark when an opaque GPU source yields negative or non-finite
// radiance. The semantic atlas preflight suite and the 78/78 WGSL compile gate
// passed before these six material-atlas-bearing roots were re-pinned.
// Native-transmission closure (2026-08-03): the reusable GI paths now carry
// RGB recast tint and continuous material shares, while full-resolution camera
// glass owns bounded refractive suffix/direct/volume transport. Stable medium
// identity, unlit parity, and stochastic alpha-blend metadata are included in
// these source-frozen roots; semantic tests and the 78/78 shader gate passed.
// Numeric/material-sample closure (2026-08-03): atlas reads now carry explicit
// validity, metadata and address arithmetic fail closed before integer casts,
// tangent/normal/bump construction is scale safe, and DDGI/world-distance
// medium walks use exact represented state. Focused semantic suites and the
// safe shader gate accompany this intentional source-freeze refresh.
// Walkaround albedo/metal honesty (2026-08-12): unmapped base color is the
// atlas disabled-meta float RGB, and shade's conductor-class gate uses sampled
// metalness >= 0.5 instead of the packed metallic>0 bit.
const GOLDENS: Record<string, { sha256: string; length: number }> = {
  risGi: { sha256: '649c049fb81780e665ededc48940111e5b6749f118acc0b4d2fbaa7e9e52d775', length: 441575 },
  risGiNrc: { sha256: 'a700b282d1bcb4393639cbd6495bcd36b375968ab019e9b786c8102fe7adef65', length: 541037 },
  temporalGi: { sha256: 'a178ca390bfef2ffedf5578631d1094188d21603fd09614c86dfbea6ba9a4728', length: 368254 },
  spatialGi: { sha256: '4b50fa5c631a081730b1615e25b0eb8687d1211d48542faa2e0ec5cfcdc61a3d', length: 368218 },
  shade: { sha256: '5a715a8a8951b944f0e82278e842914f1117f92cd685d3c0a1e1b16697c668cf', length: 684292 },
  transparentOit: { sha256: '77fac7c53b31e08c9d9660d3cedcaa348b13a056478c557d8f74bfab369abf9b', length: 402581 },
  regir: { sha256: 'f44877d88d86b470d7906fd6e35bf2d162c0b190b62cf0f7f8453a17c0e7580a', length: 217345 },
  regirBuild: { sha256: '2e2ccb0bb0f4dbde8455648be937f075854aef49ef80941461393c26f46559ae', length: 218992 },
};

interface Case { name: string; code: () => string; }

const CASES: Case[] = [
  { name: 'risGi', code: () => composeWgsl(RIS_GI_MODULE, canonicalRegistry()) },
  { name: 'risGiNrc', code: () => composeWgsl(buildRisGiNrcModule(NRC_CFG), canonicalRegistry()) },
  { name: 'temporalGi', code: () => composeWgsl(TEMPORAL_GI_MODULE, canonicalRegistry()) },
  { name: 'spatialGi', code: () => composeWgsl(SPATIAL_GI_MODULE, canonicalRegistry()) },
  { name: 'shade', code: () => composeWgsl(SHADE_MODULE, canonicalRegistry()) },
  { name: 'transparentOit', code: () => composeWgsl(TRANSPARENT_OIT_MODULE, canonicalRegistry()) },
  { name: 'regir', code: () => composeWgsl(REGIR_MODULE, canonicalRegistry()) },
  { name: 'regirBuild', code: () => composeWgsl(REGIR_BUILD_MODULE, canonicalRegistry()) },
];

describe('T4 group-A composed-WGSL byte-identity goldens', () => {
  for (const c of CASES) {
    it(`${c.name} composes deterministically and matches its byte golden`, () => {
      const code = c.code();
      // Determinism: same (root, registry) → identical bytes.
      expect(c.code()).toBe(code);
      const d = digest(code);
      const golden = GOLDENS[c.name];
      // If the golden is not yet captured, surface the current digest so it can
      // be pinned. Once pinned, the equality below guards byte-identity.
      expect(golden, `no golden for ${c.name}; current=${JSON.stringify(d)}`).toBeDefined();
      expect(
        d,
        `${c.name} current=${JSON.stringify(d)}`,
      ).toEqual(golden);
    });
  }
});
