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

function functionBody(source: string, name: string): string {
  const marker = `fn ${name}(`;
  const start = source.indexOf(marker);
  if (start < 0) throw new Error(`Missing ${name}`);
  const brace = source.indexOf('{', start);
  if (brace < 0) throw new Error(`Missing ${name} body`);
  let depth = 0;
  for (let i = brace; i < source.length; i += 1) {
    const ch = source[i];
    if (ch === '{') depth += 1;
    if (ch === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(brace + 1, i);
    }
  }
  throw new Error(`Unterminated ${name} body`);
}

describe('PROBE_RAY_CAST_WGSL material UV decode', () => {
  it('converts only the top 24 RNG bits into a reachable f32 value below one', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'return f32(seed >> 8u) / 16777216.0;',
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain('/ 4294967296.0');
  });

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
      'let idTableTexel = triangleMaterialBase + idTableOffset;',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let materialOffset = materialId * RC_MATERIAL_MAP_META_TEXELS_PER_TRI;',
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain(
      'triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset',
    );
  });

  it('multiplies scalar roughness/metallic maps by the authored scalar fallback', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'return clamp(fallback * rcMaterialMapChannel(texel.value, channel), 0.0, 1.0);',
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain(
      'return clamp(rcMaterialMapChannel(texel, channel), 0.0, 1.0);',
    );
  });

  it('uses the no-layer identity when optional material metadata is absent', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'if (!rcMaterialMetaAvailable(triIndex, offset)) {',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'return vec4f(1.0, 1.0, 1.0, -1.0);',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'out.dielectricLayerTransmission = clamp(',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'out.reflectionLayerTransmission = out.dielectricLayerTransmission;',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'out.layerTransmission = out.reflectionLayerTransmission;',
    );
  });

  it('rejects non-finite selected UVs before every wrap, filter, LOD, and texel cast path', () => {
    const body = functionBody(
      PROBE_RAY_CAST_WGSL,
      'rcSampleMaterialAtlasRawAtOffsetDelta',
    );
    const guard = body.indexOf('if (!rcMaterialAtlasFiniteVec2(uv))');
    const affine = body.indexOf('let scaled = uv * meta1.xy;');
    const wrap = body.indexOf('let wrapped = rcWrapMaterialUv(');
    const lod = body.indexOf('let lodCandidate = log2(');

    expect(guard).toBeGreaterThan(0);
    expect(body.slice(guard, affine)).toContain(
      'return rcMaterialAtlasInvalidSample();',
    );
    expect(guard).toBeLessThan(affine);
    expect(guard).toBeLessThan(wrap);
    expect(guard).toBeLessThan(lod);
    expect(PROBE_RAY_CAST_WGSL).toContain('if (mode == 2u) {');
    expect(PROBE_RAY_CAST_WGSL).toContain('if (mipFilter == 1u) {');
    expect(PROBE_RAY_CAST_WGSL).toContain('let finiteLod = select(0.0, lod, rcMaterialAtlasFiniteF32(lod));');

    for (const uvLane of [Number.NaN, Number.POSITIVE_INFINITY]) {
      for (const wrapMode of ['repeat', 'mirrored-repeat'] as const) {
        for (const filter of ['nearest', 'linear'] as const) {
          const sampled = Number.isFinite(uvLane)
            ? `${wrapMode}:${filter}`
            : -1;
          expect(sampled).toBe(-1);
        }
      }
    }
  });

  it('propagates explicit finite sample validity through taps and mips', () => {
    const validSample = (value: readonly number[]): boolean =>
      value.every((lane) => Number.isFinite(lane) && Math.abs(lane) <= 3.402823466e38);

    expect(validSample([-1, -0.5, 0, 1])).toBe(true);
    expect(validSample([Number.NEGATIVE_INFINITY, 0, 0, 1])).toBe(false);
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'struct RCMaterialAtlasSampleResult {',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain('encoding: u32,');
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'c00.valid == 0u || c10.valid == 0u || c01.valid == 0u || c11.valid == 0u',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'if (c0.valid == 0u || c1.valid == 0u)',
    );
    const decodedFinite = PROBE_RAY_CAST_WGSL.indexOf(
      'let decoded = rcMaterialAtlasValidSample(value, address.encoding);',
    );
    const srgbDecode = PROBE_RAY_CAST_WGSL.indexOf(
      'if (address.decodeSrgb != 0u)',
      decodedFinite,
    );
    expect(decodedFinite).toBeGreaterThan(0);
    expect(decodedFinite).toBeLessThan(srgbDecode);
  });

  it('uses source encoding for SNORM normal decode and degenerate fallback', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'texelColor.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'texelColor.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'if (!rcCanNormalize(tangentSampleRaw))',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let perturbed = rcSafeNormalizeOr(perturbedRaw, fallbackNormal);',
    );
  });

  it('uses source encoding for anisotropy direction decode', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'anisoMap.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'anisoMap.encoding == RC_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'clamp(anisoMap.value.rg, vec2f(0.0), vec2f(1.0)) * 2.0 - vec2f(1.0)',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'clamp(anisoMap.value.rg, vec2f(-1.0), vec2f(1.0))',
    );
  });

  it('validates emitter float metadata before every integer conversion', () => {
    const exactF32IntegerInRange = (
      value: number,
      minimum: number,
      maximum: number,
    ): boolean => {
      const represented = Math.fround(value);
      return Number.isFinite(represented) &&
        Number.isInteger(represented) &&
        represented >= minimum &&
        represented <= maximum;
    };

    expect(exactF32IntegerInRange(3, 0, 3)).toBe(true);
    expect(exactF32IntegerInRange(4, 0, 3)).toBe(false);
    expect(exactF32IntegerInRange(Number.NaN, 0, 3)).toBe(false);
    expect(exactF32IntegerInRange(16, 1, 16)).toBe(true);
    expect(exactF32IntegerInRange(16.5, 1, 16)).toBe(false);
    expect(exactF32IntegerInRange(Number.MAX_VALUE, -16_777_216, 16_777_216)).toBe(false);

    expect(PROBE_RAY_CAST_WGSL).not.toContain(
      'u32(max(emitter.emitterFlags, 0.0))',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      '!rcMaterialAtlasFiniteF32(flags)',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'flags > 3.0 ||\n    floor(flags) != flags',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      '!rcMaterialAtlasFiniteF32(e._padA)',
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain(
      'i32(round(e._padA))',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'e._padA > 16777216.0 ||\n    floor(e._padA) != e._padA',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let subdivOrdinalCount = subdivLevel * subdivLevel;',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'if (e._padC >= f32(subdivOrdinalCount))',
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain(
      'u32(round(max(levelF, 1.0)))',
    );
  });

  it('does not turn a traversable glass back interface into two-sided emission', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'hit.side >= 0.0 ||',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      '(mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u',
    );
    expect(
      PROBE_RAY_CAST_WGSL.match(/rcSampleSurfaceEmissiveMap\(hit,/g),
    ).toHaveLength(1);
  });

  it('keeps partial transmission, visible volume, suffix transport, and unlit ownership distinct', () => {
    const directResponse = functionBody(
      PROBE_RAY_CAST_WGSL,
      'rcEvaluateProbeDirectResponse',
    );
    expect(directResponse).toContain(
      '(1.0 - clamp(mat.transmission, 0.0, 1.0)) * RC_INV_PI',
    );
    const sources = functionBody(
      PROBE_RAY_CAST_WGSL,
      'rcShadeTransmissionInterfaceSources',
    );
    const suffix = functionBody(
      PROBE_RAY_CAST_WGSL,
      'rcTraceDielectricSuffixChannel',
    );
    const kernel = functionBody(PROBE_RAY_CAST_WGSL, 'probeRayCastKernel');
    expect(sources).toContain(
      'rcSampleLightMapIrradiance(hit) * probeMat.layerTransmission;',
    );
    expect(sources).toContain('out.localSurface = rcApplyHomogeneousVolumeSingleScatter(');
    expect(sources).toContain('if (!suppressOpaqueSubstrate && !explicitBulkSegment)');
    expect(sources).toContain('out.emission = select(');
    expect(suffix).toContain(
      'rcSuffixTransferredChannel(throughput, localSurfaceSource, channel);',
    );
    expect(suffix).not.toContain('opaquePhysicalWeight');
    expect(kernel).toContain('let firstSources = rcShadeTransmissionInterfaceSources(');
    expect(kernel).toContain('if (firstSources.unlit != 0u)');
    expect(suffix).toContain(
      'if (interfaceUnlit != 0u) {',
    );
    expect(suffix).toContain('(mat.flags & MATERIAL_FLAG_SKIP_EMITTER) != 0u;');
    expect(kernel).toContain('rcTraceDielectricSuffixChannel(');
    expect(kernel).not.toContain('localSurfaceRadiance + transContrib');
  });
});

describe('PROBE_RAY_CAST_WGSL environment transform', () => {
  it('uses the repository-wide top-to-bottom equirectangular V convention', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'return vec2f(phi / (2.0 * 3.14159265) + 0.5, theta / 3.14159265);',
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain('1.0 - theta / 3.14159265');
  });

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
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'return rcScaleEnvironmentRadiance(texel.rgb, rc_u.envIntensity);',
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain(
      ').rgb * rc_u.envIntensity;',
    );

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
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'return rcScaleEnvironmentRadiance(rc_u.scalarSkyRadiance, 1.0);',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'radiance = rcEnvironmentRadiance(rayDir);',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let env = rcEnvironmentRadiance(ray.direction);',
    );
  });

  it('composes one whole-RGB fail-closed environment scaler', () => {
    expect(PROBE_RAY_CAST_WGSL.match(
      /fn rcScaleEnvironmentRadiance\(/g,
    )).toHaveLength(1);
    expect(PROBE_RAY_CAST_WGSL.match(
      /return rcScaleEnvironmentRadiance\(/g,
    )).toHaveLength(2);
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'if (!all(scaled == scaled) || any(abs(scaled) > vec3f(maxFinite)))',
    );
  });
});
