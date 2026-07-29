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
const GOLDENS: Record<string, { sha256: string; length: number }> = {
  risGi: { sha256: 'cbc0a9dadd9349d8d068da25c34e9100a47a26641a2708e0fec7d97c0ea45b83', length: 263466 },
  risGiNrc: { sha256: 'c53a5fbabb952589439f3ef33d7794514518720ade055e0691c66f31904177ac', length: 314792 },
  temporalGi: { sha256: '16beaf9ce227871a395e85e3a4a85c9d0900b4e7f1ae0ec1cfae73890137a4b2', length: 230458 },
  spatialGi: { sha256: '6c3342acdaed4c8be4d4fe73a6e06c218773b0823427ba383bc869385f172b59', length: 230965 },
  shade: { sha256: 'f4743596b9f834497be54e250dc7f7eca5f95fbea5a047222b2fecfc7cae4581', length: 386203 },
  transparentOit: { sha256: '9726b22caf92fcef182128075e9c9a24e11959dfa34464648789e0284c6063b3', length: 258233 },
  regir: { sha256: '2208cc7a3509a8e705c8013fbf4c7024bf63f9d04760e8915774b50cbce9fe83', length: 138788 },
  regirBuild: { sha256: '1d56e95eba15a878f92ff0936a67402e0c2002f9654c5bd4f2c5d60e82cdca58', length: 137985 },
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
