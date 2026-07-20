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
const GOLDENS: Record<string, { sha256: string; length: number }> = {
  'risGi.off': { sha256: 'f61f5bd325e262aa9af0a6701c3358ea455ebcf9a8f67e9d6645f4b70744ce0d', length: 236025 },
  'risGi.gris': { sha256: 'fd2677a8f960022d028ba48e87b2968336b598d3c9dd036e268e13bbcb62c7de', length: 236849 },
  'risGiNrc.off': { sha256: 'b59012fa9bee90ddc8b10e8a19ae6a3d1caff150dce3629da6bafe5a62e077fd', length: 246007 },
  'risGiNrc.gris': { sha256: 'd3349eb6bcfe9af810adb95302362ade2ef8d525d39503fd34f48bff68315535', length: 246831 },
  'temporalGi.off': { sha256: '06cdd76582960e6f6c5ee6d24148ece2b5153f729a253a3ef854f666915b855f', length: 171856 },
  'temporalGi.gris': { sha256: 'd8a0ab37ec3beee120a8840a519566032957207c1a96fdf9c5ecc081722a1d01', length: 180938 },
  'spatialGi.off': { sha256: '19abca7efdb5d20f49c8661014d694be0b382bc7c2c64fd4f5908429b74c0ef7', length: 170679 },
  'spatialGi.gris': { sha256: '57f54f32aee0c9a07b298c3ccf0423850afc63e7b4aebd7cb852146fd8752de2', length: 182343 },
  'shade.off': { sha256: '968d3256a6581b12787d1b9c5c136e0abf734b22f986de97c9e87ed7cf9ec9bf', length: 282739 },
  'shade.gris': { sha256: '628c2374daf1bc32293a964b1b9f8f9761b3074e5cc13e4093c05a162605d7cd', length: 283563 },
  'transparentOit.off': { sha256: '62c7651ee11a3c3eef2431d9bed357238f27132ab37a59983704789afcc5b783', length: 208520 },
  'regir': { sha256: '170c46264184fc7c4ce67aa73750a3e54e49db3ea2d41644ded36735995c9fb6', length: 127198 },
  'regirBuild': { sha256: 'a24cd5b396f3a05667c6c2b562cc3eac5eecd954466f2e703fafdb12b6320ee5', length: 126593 },
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
      expect(d).toEqual(golden);
    });
  }
});
