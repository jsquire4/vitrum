import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { PROBE_RAY_CAST_WGSL } from '../src/wgsl/probeRayCast.wgsl.js';

/**
 * Byte-identity golden for the assembled RC probe-ray-cast WGSL.
 *
 * The material-atlas decode helpers and PBR BRDF lobes were extracted into
 * `rcMaterialAtlas.wgsl.ts` / `rcBrdf.wgsl.ts` and re-composed at their original
 * insertion points. The current golden also includes the intentional shared
 * producer/receiver octahedral-stratification fragment, so any later shader
 * composition drift remains explicit. The current capture also includes the
 * shared-BVH value-return loader seam and eight-binding packed scene arena; its
 * traversal, material-atlas, light-evaluation, and binding contracts are pinned
 * independently by the semantic package tests before this hash is updated. The
 * shared any-hit overflow guard is fail-closed in both its canonical and derived
 * shadow-predicate forms so malformed trees cannot leak light. The C71 material
 * remediation additionally pins scalar/mapped specular F0, thin-film
 * reflectance, the authored-IOR sentinel, and opaque-receiver layered/volume
 * attenuation below so this golden cannot silently bless those semantics. U10
 * additionally completes a transmissive suffix under the scene-diagonal and
 * interface-budget bound; `glassTransportClosure.test.ts` pins that behavior
 * before this composed-shader identity is updated. The 2026-07-29 audit aligns
 * RC's equirectangular V coordinate with the main renderer's top-to-bottom map
 * convention.
 */
describe('PROBE_RAY_CAST_WGSL byte identity', () => {
  it('retains the C71 specular, IOR, and layered-volume transport semantics', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('var color = vec3f(0.04);');
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let scalar = rcSampleSpecularMeta(triIndex);',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let dielectricF0 = max(mat.specular.rgb, vec3f(0.0)) *',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let transportIor = select(max(scalarIor, 1.0), 1e6, scalarIor == 0.0);',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'out.specular = vec4f(film.reflectance, 1.0);',
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain(
      'out.specular = vec4f(vec3f(1.0) + film.reflectance, 1.0);',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      ') * probeMat.layerTransmission;',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'radiance = rcApplyHomogeneousVolumeSingleScatter(',
    );
  });

  it('matches the captured length + sha256 golden', () => {
    const length = PROBE_RAY_CAST_WGSL.length;
    const sha256 = createHash('sha256').update(PROBE_RAY_CAST_WGSL, 'utf8').digest('hex');
    expect({ length, sha256 }).toEqual({
      length: 141100,
      sha256: '2a59e369ee48e6a4c5e1c65e300a89524a16ac0419a7344fcc95c13e7dcb289f',
    });
  });
});
