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
 * Any future edit that perturbs these strings must consciously re-pin here.
 */
function sha(s: string): string {
  return createHash('sha256').update(s, 'utf8').digest('hex');
}

describe('material-atlas decode ABI composed byte identity', () => {
  it('pins the MATERIAL_ATLAS_WGSL fragment (offset ABI + alpha-mask walkers)', () => {
    expect({ length: MATERIAL_ATLAS_WGSL.length, sha256: sha(MATERIAL_ATLAS_WGSL) }).toEqual({
      length: 41802,
      sha256: '183ff5ff1d03a9cba1a6d4fbb4ba12a73585423bcfec66c63fb38f9b6f2f2899',
    });
  });

  it('pins the SURFACE_TEXTURES_WGSL fragment (Beer-tint helper)', () => {
    expect({ length: SURFACE_TEXTURES_WGSL.length, sha256: sha(SURFACE_TEXTURES_WGSL) }).toEqual({
      length: 22809,
      sha256: '041157d1854cf98940190087cc9e00a43c67a5793519629419b45d08576c4128',
    });
  });

  it('pins the composed shade + risGi pipelines', () => {
    const shade = composeWgsl(SHADE_MODULE, WGSL_MODULES);
    const risGi = composeWgsl(RIS_GI_MODULE, WGSL_MODULES);
    expect({ length: shade.length, sha256: sha(shade) }).toEqual({
      length: 283563,
      sha256: '628c2374daf1bc32293a964b1b9f8f9761b3074e5cc13e4093c05a162605d7cd',
    });
    expect({ length: risGi.length, sha256: sha(risGi) }).toEqual({
      length: 236849,
      sha256: 'fd2677a8f960022d028ba48e87b2968336b598d3c9dd036e268e13bbcb62c7de',
    });
  });

  it('pins probeUpdateRays for representative maxMaterials (DDGI offset ABI)', () => {
    const cases: Array<[number, number, string]> = [
      [1, 130509, 'bc4d6fcd5af057a7a67340f9353d1c2b4882a2ed65e5f929c792947687032ee4'],
      [8, 130509, '3fc2223ac3de327483e7cf535e4e8764b1d4158e44f54a12948e734a4c62043c'],
      [64, 130510, 'ca797b0144ed8924be943856a79b6e77fdbe826c93331980bab7f8dc88607e8e'],
    ];
    for (const [m, length, sha256] of cases) {
      const wgsl = makeProbeUpdateRaysWGSL(m);
      expect({ m, length: wgsl.length, sha256: sha(wgsl) }).toEqual({ m, length, sha256 });
    }
  });
});
