import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { composeWgsl } from '../../pipeline/wgslComposer.js';
import { WGSL_MODULES } from '../../pipeline/wgslModules.js';
import { SHADE_MODULE } from '../shade.wgsl.js';
import { RIS_GI_MODULE } from '../risGi.wgsl.js';
import { MATERIAL_ATLAS_WGSL } from '../materialAtlas.wgsl.js';
import { SURFACE_TEXTURES_WGSL } from '../surfaceTextures.wgsl.js';
import { makeProbeUpdateRaysWGSL } from '../../ddgi/wgsl/probeUpdateRays.wgsl.js';

/**
 * Byte-identity goldens for the T4-2 62-texel material-atlas decode ABI dedup
 * (2026-07-20). Guards the composed WGSL against drift after:
 *   - the offset-const block was single-sourced from @vitrum/shared-bvh
 *     (byte-identical — shade / DDGI / RC unchanged),
 *   - the two textured first-hit alpha-mask walk-wrappers were templated
 *     (byte-identical — MATERIAL_ATLAS_WGSL unchanged),
 *   - transparent visibility now uses one ordered, instance-owned medium walk
 *     with actual entry/exit Beer distance (SURFACE_TEXTURES_WGSL refreshed).
 *
 * Packed-scene-arena refresh (2026-07-22): scene storage moved behind the
 * three-arena value-return loader ABI. The semantic binding/traversal suites
 * and shader gates were audited before the intentionally changed bytes below
 * were re-pinned. Any future edit that perturbs these strings must likewise be
 * investigated before re-pinning.
 *
 * DDGI packed-state refresh (2026-07-27): the standalone probe-state texture
 * binding was replaced by one reserved irradiance-atlas texel per probe. The
 * production shader gate compiled 79/79 modules and created 51/51 pipelines
 * before the composed and probe-ray digests below were re-pinned.
 *
 * Default-material energy refresh (2026-07-28): the absolute thin-film F0
 * marker moved from the colliding [1,2] range to disjoint [2,3], and the GGX
 * reflection eval/sample/pdf paths regained one shared bounded continuous
 * roughness. The walkaround/RC shader gate compiled all shipped
 * compositions before these intentional semantic bytes were re-pinned.
 *
 * ReSTIR-DI closure refresh (2026-07-28): stable temporal correspondence,
 * generalized area/environment support, real spatial ping-pong, and
 * max-normalized log2 Talbot denominators intentionally changed shared
 * composed bytes.
 *
 * Generalized-reuse closure (2026-07-28): the dead opticalIor member was
 * removed from RestirDIMaterialPayload, the compact GI execution roots were
 * retired, and the shipped 29-root walkaround/RC gate compiled 29/29.
 *
 * Layered-extension energy closure (2026-07-28): the shared GGX module now
 * attenuates lower layers for KHR clearcoat and sheen. Both composed roots
 * below were re-pinned after the 78-module shader gate passed.
 *
 * Tier-1 material/GI closure (2026-07-28): absolute KHR specular F0, the
 * preserved IOR=0 endpoint, rich ReSTIR-GI targets, and canonical DDGI feedback
 * intentionally changed these fragments. The shipped walkaround/RC gate
 * compiled 29/29 roots before this repin.
 *
 * Multi-UV tangent-frame closure (2026-07-28): authored tangents are UV0-only,
 * so main and DDGI material decoders retain derivative frames for UV1+. The
 * shader gate compiled 78/78 roots before these semantic bytes were re-pinned.
 *
 * Executable-surface cleanup (2026-07-28): the unused boolean alpha-shadow
 * wrapper and compatibility-only shared WGSL helpers were removed after their
 * call sites had already migrated to the canonical transmittance/rich-material
 * paths. These goldens pin the resulting live fragment and compositions.
 *
 * Canonical-GI prose reconciliation (2026-07-28): a documentation-only update
 * in the shared reservoirGi fragment adds 43 bytes to both composed roots
 * below. Material-atlas and surface-texture fragment bytes are unchanged.
 *
 * Final renderer audit closure (2026-07-29): fractional-metal GI, explicit
 * retired-control ABI pads, validation-visible BVH arena ownership, and inert
 * resource removal intentionally changed composed roots without changing the
 * material-atlas or surface-texture fragments pinned above.
 *
 * KHR punctual-range closure (2026-07-29): every walkaround punctual-light
 * route now shares the unsquared KHR range window. Shade and DDGI probe-update
 * bytes intentionally changed; RIS-GI already consumed the canonical helper.
 *
 * Final completeness closure (2026-07-29): material-presence flags and
 * one-/two-sided emissive transport changed the atlas decoder and all roots
 * that consume it. Semantic sidedness/alias tests and the shader gate remain
 * mandatory; these hashes only pin the reviewed composed bytes.
 *
 * Transport/material ABI closure (2026-07-30): compact material ABI v2,
 * authored atlas addressing, ordered glass-medium ownership, finite DDGI miss
 * observations, and canonical Beer attenuation intentionally changed these
 * reviewed fragments and roots.
 *
 * Finite-f32 environment and runtime-light closure (2026-07-30) changed the
 * shared composed roots and DDGI probe shader intentionally. These values are
 * captured from the live generators and remain paired with semantic tests.
 *
 * Material-atlas radiance closure (2026-07-30) adds a finite/non-negative
 * fail-dark boundary to emissive/light-map shader products. Focused
 * radiometric tests and the complete 78-module WGSL compiler gate passed
 * before the fragment and its two composed consumers were re-pinned.
 *
 * Native-transmission closure (2026-08-03) adds stable material identities,
 * stochastic alpha-blend metadata, continuous opaque/transmitted sharing,
 * unlit transport, and bounded full-resolution camera-prefix glass. The live
 * semantic transport tests and the complete shader gate were reviewed before
 * this source-freeze repin.
 *
 * Numeric/material-sample closure (2026-08-03) makes atlas availability and
 * sample validity explicit, checks metadata/address arithmetic before casts,
 * and carries the same fail-closed semantics through DDGI and ordered medium
 * visibility. These bytes are paired with the focused atlas, DDGI, and
 * transport tests rather than standing alone as correctness evidence.
 */
function sha(s: string): string {
  return createHash('sha256').update(s, 'utf8').digest('hex');
}

describe('material-atlas decode ABI composed byte identity', () => {
  it('pins the MATERIAL_ATLAS_WGSL fragment (offset ABI + alpha-mask walkers)', () => {
    expect({ length: MATERIAL_ATLAS_WGSL.length, sha256: sha(MATERIAL_ATLAS_WGSL) }).toEqual({
      length: 93146,
      sha256: 'c730dee87fd263d0f1b6aa9c84df9f3b9ad9ae96b6976a6abd10b13b46775485',
    });
  });

  it('pins the SURFACE_TEXTURES_WGSL fragment (ordered medium visibility)', () => {
    expect({ length: SURFACE_TEXTURES_WGSL.length, sha256: sha(SURFACE_TEXTURES_WGSL) }).toEqual({
      length: 35721,
      sha256: '9a04a6b50d81f262def22e632e7a31b96555dc1a144b2537b47a91c726cf94d8',
    });
  });

  it('pins the composed shade + risGi pipelines', () => {
    const shade = composeWgsl(SHADE_MODULE, WGSL_MODULES);
    const risGi = composeWgsl(RIS_GI_MODULE, WGSL_MODULES);
    const shadeDigest = { length: shade.length, sha256: sha(shade) };
    expect(shadeDigest, `shade current=${JSON.stringify(shadeDigest)}`).toEqual({
      length: 683389,
      sha256: '7b9501e410a837bbf2772150f92d0220b4440a4fdf6a886c55d0b3b08906690d',
    });
    const risGiDigest = { length: risGi.length, sha256: sha(risGi) };
    expect(risGiDigest, `risGi current=${JSON.stringify(risGiDigest)}`).toEqual({
      length: 440872,
      sha256: '7f10a8e8af925e16524f7ffe085d89105f684b5e940639938bbe00aee63ee937',
    });
  });

  it('pins probeUpdateRays for representative maxMaterials (DDGI offset ABI)', () => {
    const cases: Array<[number, number, string]> = [
      [1, 278354, '4c9f1991ed13702b7a8e5b134f09672c7685e43b82e3cfe74d345f48c7a134cd'],
      [8, 278354, 'b51d55bef6a9a792ff4d3029a83c1b285c80d1f3d49acece99bc55cd62f8a384'],
      [64, 278356, 'c00b0a89f1917d3153e646301461f7e949ad083ab59d9be832aa0fadf470ba76'],
    ];
    const current = cases.map(([m]) => {
      const wgsl = makeProbeUpdateRaysWGSL(m);
      return { m, length: wgsl.length, sha256: sha(wgsl) };
    });
    expect(
      current,
      `probeUpdateRays current=${JSON.stringify(current)}`,
    ).toEqual(cases.map(([m, length, sha256]) => ({ m, length, sha256 })));
  });
});
