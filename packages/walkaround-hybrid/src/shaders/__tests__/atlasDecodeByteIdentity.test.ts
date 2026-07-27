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
 */
function sha(s: string): string {
  return createHash('sha256').update(s, 'utf8').digest('hex');
}

describe('material-atlas decode ABI composed byte identity', () => {
  it('pins the MATERIAL_ATLAS_WGSL fragment (offset ABI + alpha-mask walkers)', () => {
    expect({ length: MATERIAL_ATLAS_WGSL.length, sha256: sha(MATERIAL_ATLAS_WGSL) }).toEqual({
      length: 53288,
      sha256: 'a89da77e54b7fbc140a11eb4d2f4ecad46bcd72c447a239b4c54c1488cd88c71',
    });
  });

  it('pins the SURFACE_TEXTURES_WGSL fragment (Beer-tint helper)', () => {
    expect({ length: SURFACE_TEXTURES_WGSL.length, sha256: sha(SURFACE_TEXTURES_WGSL) }).toEqual({
      length: 23959,
      sha256: '1c7a3c9016a875e7318b195a6b1b4228fca737f3526b3445d545a822d29a01f4',
    });
  });

  it('pins the composed shade + risGi pipelines', () => {
    const shade = composeWgsl(SHADE_MODULE, WGSL_MODULES);
    const risGi = composeWgsl(RIS_GI_MODULE, WGSL_MODULES);
    const shadeDigest = { length: shade.length, sha256: sha(shade) };
    expect(shadeDigest, `shade current=${JSON.stringify(shadeDigest)}`).toEqual({
      length: 381419,
      sha256: '797a7adb8c7fe888f7658e473eb068706358b0b0161da37cb56b55478d62fca7',
    });
    const risGiDigest = { length: risGi.length, sha256: sha(risGi) };
    expect(risGiDigest, `risGi current=${JSON.stringify(risGiDigest)}`).toEqual({
      length: 267701,
      sha256: 'a11f917c7ef94b44443ffa767269dc4febbe9a4fc2decaf2e453c40d42b2d295',
    });
  });

  it('pins probeUpdateRays for representative maxMaterials (DDGI offset ABI)', () => {
    const cases: Array<[number, number, string]> = [
      [1, 149638, '0da0392fefe56d0bcb97f6696ca56c83f7266cc9bca28e8a4a59f9bf123220f2'],
      [8, 149638, 'ed687be1afe12d6dc23d5fe61f6004975391b5e26460ba54d9465dd18fc3e2eb'],
      [64, 149639, 'c86ed4e93865c5ca4ae7bf210bd1837e9c1a90a4e493f3bd16e7f68289502086'],
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
