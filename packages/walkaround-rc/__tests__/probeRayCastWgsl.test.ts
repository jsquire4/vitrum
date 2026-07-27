import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { PROBE_RAY_CAST_WGSL } from '../src/wgsl/probeRayCast.wgsl.js';
import { buildMaterialAtlasOffsetConstsWGSL } from '@vitrum/shared-bvh';

function readRepoText(path: string): string {
  return readFileSync(new URL(`../../${path}`, import.meta.url), 'utf8');
}

function numberFrom(source: string, re: RegExp, label: string): number {
  const match = source.match(re);
  if (!match) throw new Error(`Missing ${label}`);
  return Number(match[1]);
}

describe('PROBE_RAY_CAST_WGSL material UV decode', () => {
  it('matches walkaround f16 UV packing in vec4.w lanes', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcPackedUvFromVec4(v: vec4f) -> vec2f');
    expect(PROBE_RAY_CAST_WGSL).toContain('unpack2x16float(bitcast<u32>(v.w))');
    expect(PROBE_RAY_CAST_WGSL).not.toContain('unpack2x16unorm(bitcast<u32>(v.w))');
  });

  it('keeps RC material metadata stride aligned with the atlas producer and main material shader', () => {
    // The MATERIAL_MAP_META_TEXELS_PER_TRI stride authority moved to the CPU
    // pack module (T6-2 / I3-2); the pipeline file now re-exports it. The WGSL
    // offset-const block is single-sourced from @vitrum/shared-bvh (T4-2,
    // 2026-07-20). The shade shader emits its stride from that generator, so
    // the "main shader stride" is read from the shared generator's own output
    // (the authority both the shade + RC copies now consume).
    const atlasSource = readRepoText('walkaround-hybrid/src/bvh/materialTextureAtlasPack.ts');
    const sharedOffsetConsts = buildMaterialAtlasOffsetConstsWGSL({
      prefix: '',
      include: ['META_TEXELS_PER_TRI'],
    });
    const hostStride = numberFrom(
      atlasSource,
      /MATERIAL_MAP_META_TEXELS_PER_TRI\s*=\s*(\d+)/,
      'host material atlas stride',
    );
    const mainShaderStride = numberFrom(
      sharedOffsetConsts,
      /const MATERIAL_MAP_META_TEXELS_PER_TRI:\s*u32\s*=\s*(\d+)u/,
      'main material atlas shader stride',
    );
    const rcShaderStride = numberFrom(
      PROBE_RAY_CAST_WGSL,
      /const RC_MATERIAL_MAP_META_TEXELS_PER_TRI:\s*u32\s*=\s*(\d+)u/,
      'RC material atlas shader stride',
    );

    expect(rcShaderStride).toBe(hostStride);
    expect(rcShaderStride).toBe(mainShaderStride);
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset',
    );
  });

  it('multiplies scalar roughness/metallic maps by the authored scalar fallback', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'return clamp(fallback * rcMaterialMapChannel(texel, channel), 0.0, 1.0);',
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain(
      'return clamp(rcMaterialMapChannel(texel, channel), 0.0, 1.0);',
    );
  });
});

describe('PROBE_RAY_CAST_WGSL environment transform', () => {
  it('uses one H6 RY(-rotationY) directional lookup helper at both RC environment reads', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'return vec3f(c * dir.x - s * dir.z, dir.y, s * dir.x + c * dir.z);',
    );
    expect(
      PROBE_RAY_CAST_WGSL.match(/dirToEquirectUV\(rcEnvRotateYNeg\(/g),
    ).toHaveLength(1);
    expect(
      PROBE_RAY_CAST_WGSL.match(/rcEnvironmentRadiance\((?:ray\.direction|rayDir)\)/g),
    ).toHaveLength(2);
    expect(PROBE_RAY_CAST_WGSL).toContain(').rgb * rc_u.envIntensity;');

    const rotationY = Math.PI / 2;
    const direction: readonly [number, number, number] = [1, 0, 0];
    const c = Math.cos(rotationY);
    const s = Math.sin(rotationY);
    const lookup = [
      c * direction[0] - s * direction[2],
      direction[1],
      s * direction[0] + c * direction[2],
    ];
    expect(lookup[0]).toBeCloseTo(0, 12);
    expect(lookup[1]).toBe(0);
    expect(lookup[2]).toBeCloseTo(1, 12);
  });

  it('selects scalar sky radiance only when no directional payload is active', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('if (rc_u.hasDirectionalEnv == 0u)');
    expect(PROBE_RAY_CAST_WGSL).toContain('return rc_u.scalarSkyRadiance;');
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'radiance = rcEnvironmentRadiance(rayDir);',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let env = rcEnvironmentRadiance(ray.direction);',
    );
  });
});
