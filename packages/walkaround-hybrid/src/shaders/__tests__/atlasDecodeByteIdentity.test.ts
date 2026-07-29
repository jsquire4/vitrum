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
 *   - the barycentric+Beer-tint decode was hoisted into `_bvhBeerTintFactor`
 *     (SURFACE_TEXTURES_WGSL golden-refreshed; naga-validated via shader-gate).
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
 */
function sha(s: string): string {
  return createHash('sha256').update(s, 'utf8').digest('hex');
}

describe('material-atlas decode ABI composed byte identity', () => {
  it('pins the MATERIAL_ATLAS_WGSL fragment (offset ABI + alpha-mask walkers)', () => {
    expect({ length: MATERIAL_ATLAS_WGSL.length, sha256: sha(MATERIAL_ATLAS_WGSL) }).toEqual({
      length: 54493,
      sha256: '427a90f73f786fee62b0cabc01b65e3622226d7c1a17e7838c44ca94d338f02d',
    });
  });

  it('pins the SURFACE_TEXTURES_WGSL fragment (Beer-tint helper)', () => {
    expect({ length: SURFACE_TEXTURES_WGSL.length, sha256: sha(SURFACE_TEXTURES_WGSL) }).toEqual({
      length: 23985,
      sha256: 'a304ad3c99c3d43513c3228518808ed87da570c4abd16212c90af6a3b24661c4',
    });
  });

  it('pins the composed shade + risGi pipelines', () => {
    const shade = composeWgsl(SHADE_MODULE, WGSL_MODULES);
    const risGi = composeWgsl(RIS_GI_MODULE, WGSL_MODULES);
    const shadeDigest = { length: shade.length, sha256: sha(shade) };
    expect(shadeDigest, `shade current=${JSON.stringify(shadeDigest)}`).toEqual({
      length: 382477,
      sha256: '384f0530b75f8efb7185981ded062bd1eeae491a7b43929ef1ac82ee90223e91',
    });
    const risGiDigest = { length: risGi.length, sha256: sha(risGi) };
    expect(risGiDigest, `risGi current=${JSON.stringify(risGiDigest)}`).toEqual({
      length: 259742,
      sha256: '0a0596c8a6ecbb9fe0d0a952124409d521fb98c0276034160d77abba50fe042e',
    });
  });

  it('pins probeUpdateRays for representative maxMaterials (DDGI offset ABI)', () => {
    const cases: Array<[number, number, string]> = [
      [1, 154559, '810074423d04c33eb06b477844e315e1c8e535186f7f584591aff97c41071def'],
      [8, 154559, 'e6f51266b25a6a0860d797b936ea50a47688cbf44505cc8aa0e2bcce51589098'],
      [64, 154560, 'e3bad56cfcce53d41a382cea24a4595a9d8073a3e88507deeec7f65a7d1164a4'],
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
