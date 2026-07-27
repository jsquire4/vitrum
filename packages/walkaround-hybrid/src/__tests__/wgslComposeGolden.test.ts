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
 * Registry construction mirrors `pipelineCompiler.compilePipelines`: the
 * `reservoirGi` module is overridden per-grisOn (grisCache true/false), and the
 * GI passes are composed with the matching reservoirGi flavour. The NRC gi-ris
 * module is built from a representative config.
 */

import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';

import { composeWgsl } from '../pipeline/wgslComposer.js';
import { WGSL_MODULES } from '../pipeline/wgslModules.js';
import { buildReservoirGiModule } from '../shaders/reservoirGi.wgsl.js';
import { RIS_GI_MODULE } from '../shaders/risGi.wgsl.js';
import { buildRisGiNrcModule, type RisGiNrcConfig } from '../shaders/risGiNrc.wgsl.js';
import { TEMPORAL_GI_MODULE, TEMPORAL_GI_GRIS_MODULE } from '../shaders/temporalGi.wgsl.js';
import { SPATIAL_GI_MODULE, SPATIAL_GI_GRIS_MODULE } from '../shaders/spatialGi.wgsl.js';
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

function registryFor(grisOn: boolean): ReadonlyMap<string, WgslModule> {
  const m = new Map(WGSL_MODULES);
  m.set('reservoirGi', buildReservoirGiModule({ grisCache: grisOn }));
  return m;
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
const GOLDENS: Record<string, { sha256: string; length: number }> = {
  'risGi.off': { sha256: '5c5a2dcbf55f1558678eeece027290c089bf7481a08086ef88afb7eb4454a6f5', length: 267228 },
  'risGi.gris': { sha256: 'a11f917c7ef94b44443ffa767269dc4febbe9a4fc2decaf2e453c40d42b2d295', length: 267701 },
  'risGiNrc.off': { sha256: '27700174c1499469a6795618c565696b729b162632661c96f8e562ca4e2f3824', length: 315824 },
  'risGiNrc.gris': { sha256: '0a7f288d4cd9201da88f4651ab2da83cd82b1061f390605018be462c41800d8d', length: 316297 },
  'temporalGi.off': { sha256: '0ee86f6ea4f1bc504c0dc6bb238c7307a3f1d437c5d0a659374482ed775608c5', length: 191226 },
  'temporalGi.gris': { sha256: '7f122efca4b7f7a868875c0cf9a51f190570a063d5bd65cf95d5cdb58334daee', length: 226876 },
  'spatialGi.off': { sha256: '210af3c6fcca83805e12c193a146bd68e8a0115a04842ac0b473b77dcb78826a', length: 189881 },
  'spatialGi.gris': { sha256: 'f0e30abc585591e3a638b548bc53c23bb96d6613f551b1999b4a0c8916351482', length: 226619 },
  'shade.off': { sha256: '9e14afbdd4c78531256ca1b53009e6eb2ba5c0aa355c6404399b18d69c0f3d8a', length: 380946 },
  'shade.gris': { sha256: '797a7adb8c7fe888f7658e473eb068706358b0b0161da37cb56b55478d62fca7', length: 381419 },
  'transparentOit.off': { sha256: '8e62b4de402234b88469cb3d47b31b315d312feb42b03d2d352edd28486b4ad6', length: 253768 },
  'regir': { sha256: '8f17da4b79e10934995c7e9ee37212c4b1560c0469bb57b76ae39cf0e951c3c2', length: 135978 },
  'regirBuild': { sha256: '613d6758efb7f5617f78b4bbc7ed5a1978d5b1e6cef7c3f5972b470a468b9953', length: 135175 },
};

interface Case { name: string; code: () => string; }

const CASES: Case[] = [
  // GI passes are grisOn-sensitive via the reservoirGi override. Both flavours
  // are composed the way pipelineCompiler emits them.
  { name: 'risGi.off', code: () => composeWgsl(RIS_GI_MODULE, registryFor(false)) },
  { name: 'risGi.gris', code: () => composeWgsl(RIS_GI_MODULE, registryFor(true)) },
  { name: 'risGiNrc.off', code: () => composeWgsl(buildRisGiNrcModule(NRC_CFG), registryFor(false)) },
  { name: 'risGiNrc.gris', code: () => composeWgsl(buildRisGiNrcModule(NRC_CFG), registryFor(true)) },
  { name: 'temporalGi.off', code: () => composeWgsl(TEMPORAL_GI_MODULE, registryFor(false)) },
  { name: 'temporalGi.gris', code: () => composeWgsl(TEMPORAL_GI_GRIS_MODULE, registryFor(true)) },
  { name: 'spatialGi.off', code: () => composeWgsl(SPATIAL_GI_MODULE, registryFor(false)) },
  { name: 'spatialGi.gris', code: () => composeWgsl(SPATIAL_GI_GRIS_MODULE, registryFor(true)) },
  { name: 'shade.off', code: () => composeWgsl(SHADE_MODULE, registryFor(false)) },
  { name: 'shade.gris', code: () => composeWgsl(SHADE_MODULE, registryFor(true)) },
  { name: 'transparentOit.off', code: () => composeWgsl(TRANSPARENT_OIT_MODULE, registryFor(false)) },
  { name: 'regir', code: () => composeWgsl(REGIR_MODULE, registryFor(false)) },
  { name: 'regirBuild', code: () => composeWgsl(REGIR_BUILD_MODULE, registryFor(false)) },
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
