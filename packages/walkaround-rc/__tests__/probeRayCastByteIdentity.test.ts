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
 * convention. The final completeness pass also pins the absent-layer identity,
 * the shared emitter flag layout (cast-shadow plus two-sided emission), and
 * backface emission admission. The transport-closure refresh replaces
 * unordered tint accumulation with an ordered material/instance-owned medium
 * walk, uses actual entry/exit distance for Beer attenuation, and treats
 * partial alpha as expected thin-interface coverage. Environment reads now
 * share one whole-RGB finite-binary32 fail-closed scaling stage. Unlit
 * receivers are terminal emission closures at both the first hit and later
 * dielectric-suffix hits. Direct layered lighting keeps absolute thin-film R
 * separate from base-closure film T, including the R>0/T=0 endpoint.
 * Explicit material-sample validity, checked atlas/metadata addressing, and
 * scale-safe normal/bump evaluation are included in the 2026-08-03 source
 * freeze; focused material and optical-transport tests accompany this digest.
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
      'reflectionResponse * mat.reflectionLayerTransmission;',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'out.localSurface = rcApplyHomogeneousVolumeSingleScatter(',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'if (!suppressOpaqueSubstrate && !explicitBulkSegment)',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      '(mat.flags & MATERIAL_FLAG_SKIP_EMITTER) != 0u;',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let segmentTransfer = rcMediumRadianceSegmentTransfer(',
    );
  });

  it('matches the captured length + sha256 golden', () => {
    const length = PROBE_RAY_CAST_WGSL.length;
    const sha256 = createHash('sha256').update(PROBE_RAY_CAST_WGSL, 'utf8').digest('hex');
    expect({ length, sha256 }).toEqual({
      length: 239658,
      sha256: 'e734e4df172a7e535f1faacfa4fdbe286f7011b9b00f6a7850e593344ac66128',
    });
  });
});
