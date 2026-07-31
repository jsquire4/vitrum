import { describe, expect, it } from 'vitest';
import { RENDER_MAIN } from './renderMain.glsl.js';
import { ATTENUATE_HIT_BASIC_GLSL } from './render/attenuate_hit_basic.glsl.js';
import * as FullAttenuationModule from './render/attenuate_hit_function.glsl.js';
import { ATTENUATE_HIT_MAPPED_PBR_GLSL } from './render/attenuate_hit_mapped_pbr.glsl.js';
import { ATTENUATE_HIT_SCALAR_RICH_GLSL } from './render/attenuate_hit_scalar_rich.glsl.js';
import * as DirectLightModule from './render/direct_light_contribution_function.glsl.js';
import * as BdptConnectionModule from './render/bdpt_connection.glsl.js';
import * as BdptLightSubpathModule from './render/bdpt_light_subpath.glsl.js';
import * as EquirectSamplingModule from './shader/sampling/equirect_sampling_functions.glsl.js';
import * as LightSamplingModule from './shader/sampling/light_sampling_functions.glsl.js';
import * as UtilModule from './shader/common/util_functions.glsl.js';

function glslChunk(module: Record<string, unknown>, name: string): string {
  const value = module[name];
  if (typeof value !== 'string') {
    throw new TypeError(`Expected GLSL module "${name}" to export a string.`);
  }
  return value;
}

const attenuate_hit_function = glslChunk(
  FullAttenuationModule,
  'attenuate_hit_function',
);
const direct_light_contribution_function = glslChunk(
  DirectLightModule,
  'direct_light_contribution_function',
);
const bdpt_connection = glslChunk(
  BdptConnectionModule,
  'bdpt_connection',
);
const bdpt_light_subpath = glslChunk(
  BdptLightSubpathModule,
  'bdpt_light_subpath',
);
const equirect_functions = glslChunk(
  EquirectSamplingModule,
  'equirect_functions',
);
const light_sampling_functions = glslChunk(
  LightSamplingModule,
  'light_sampling_functions',
);
const util_functions = glslChunk(UtilModule, 'util_functions');
const GLSL_INFINITY = Math.fround(3.402823466e38);
const GLSL_MIN_SUBNORMAL = 2 ** -149;

function isInfiniteVisibilityDistance(distance: number): boolean {
  return !Number.isFinite(distance) || distance >= GLSL_INFINITY;
}

function stablePowerHeuristic(a: number, b: number): number {
  if (!(a >= 0) || !(b >= 0) || !Number.isFinite(a) || !Number.isFinite(b)) {
    return 0;
  }
  const scale = Math.max(a, b);
  if (!(scale > 0)) return 0;
  const scaledA = a / scale;
  const scaledB = b / scale;
  return (scaledA * scaledA) / (scaledA * scaledA + scaledB * scaledB);
}

function remainingVisibilityDistance(rayDistance: number, traveledDistance: number): number {
  return isInfiniteVisibilityDistance(rayDistance)
    ? Number.POSITIVE_INFINITY
    : Math.max(rayDistance - traveledDistance, 0);
}

function endpointReached(
  hitFaceIndex: number,
  hitDistance: number,
  rayDistance: number,
  hasTargetFace: boolean,
  targetFaceIndex: number,
): boolean {
  if (hasTargetFace && hitFaceIndex === targetFaceIndex) return true;
  return !isInfiniteVisibilityDistance(rayDistance) && hitDistance >= rayDistance;
}

function nextUpFloat32(value: number): number {
  const float = new Float32Array([value]);
  const bits = new Uint32Array(float.buffer);
  bits[0] = bits[0]! + 1;
  return float[0]!;
}

function finiteEquirectScaledColor(
  value: readonly [number, number, number],
  scale: number,
): [number, number, number] {
  const f32Value = value.map(Math.fround) as [number, number, number];
  const f32Scale = Math.fround(scale);
  if (
    f32Value.some((component) => component < 0 || !Number.isFinite(component)) ||
    f32Scale < 0 ||
    !Number.isFinite(f32Scale)
  ) {
    return [0, 0, 0];
  }
  const scaled = f32Value.map((component) =>
    Math.fround(component * f32Scale)
  ) as [number, number, number];
  if (scaled.some((component) => !Number.isFinite(component))) {
    return [0, 0, 0];
  }
  return scaled;
}

function finiteEquirectRadiance(
  texel: readonly [number, number, number],
  environmentIntensity: number,
  materialIntensity: number,
): [number, number, number] {
  return finiteEquirectScaledColor(
    finiteEquirectScaledColor(texel, environmentIntensity),
    materialIntensity,
  );
}

describe('pt-webgl2 transport scale safety', () => {
  it('normalizes MIS PDFs before squaring at both f32 dynamic-range ends', () => {
    expect(stablePowerHeuristic(3e38, 1.5e38)).toBeCloseTo(0.8, 14);
    expect(stablePowerHeuristic(1e-38, 2e-38)).toBeCloseTo(0.2, 14);
    expect(stablePowerHeuristic(0, 0)).toBe(0);
    expect(stablePowerHeuristic(Number.POSITIVE_INFINITY, 1)).toBe(0);
    expect(util_functions).toContain('float pdfScale = max( a, b );');
    expect(util_functions).toContain('float scaledA = a / pdfScale;');
    expect(util_functions).not.toContain('float aa = a * a;');
  });

  it('fails closed before the combined environment radiance can overflow', () => {
    const formerGuardBoundary = Math.fround(
      Math.fround(3.402822e38) / Math.fround(1.5),
    );
    const nextRepresentableValue = Math.fround(2.2685481662938083e38);
    expect(nextRepresentableValue).toBe(nextUpFloat32(formerGuardBoundary));
    const nextRepresentableProduct = Math.fround(
      nextRepresentableValue * Math.fround(1.5),
    );
    expect(Number.isFinite(nextRepresentableProduct)).toBe(true);

    expect(
      finiteEquirectRadiance([1, 1, 1], 2, GLSL_INFINITY),
    ).toEqual([0, 0, 0]);
    expect(
      finiteEquirectRadiance([0.25, 0.5, 1], 2, 3),
    ).toEqual([1.5, 3, 6]);
    expect(
      finiteEquirectRadiance([1, 1, 1], 1, GLSL_INFINITY),
    ).toEqual([GLSL_INFINITY, GLSL_INFINITY, GLSL_INFINITY]);
    expect(
      finiteEquirectRadiance([Number.NaN, 1, 1], 1, 1),
    ).toEqual([0, 0, 0]);
    expect(
      finiteEquirectScaledColor(
        [nextRepresentableValue, 0, 0],
        1.5,
      ),
    ).toEqual([nextRepresentableProduct, 0, 0]);
    expect(
      finiteEquirectRadiance(
        [nextRepresentableValue, 0, 0],
        1,
        1.5,
      ),
    ).toEqual([nextRepresentableProduct, 0, 0]);
    expect(
      finiteEquirectScaledColor([GLSL_INFINITY, 0, 0], 2),
    ).toEqual([0, 0, 0]);
    expect(
      finiteEquirectScaledColor([0.25, 0.5, 1], 2),
    ).toEqual([0.5, 1, 2]);
    expect(
      finiteEquirectScaledColor([GLSL_INFINITY, 1, GLSL_MIN_SUBNORMAL], 0),
    ).toEqual([0, 0, 0]);
    expect(
      finiteEquirectScaledColor([1, 1, 1], -1),
    ).toEqual([0, 0, 0]);
    expect(
      finiteEquirectScaledColor([-GLSL_MIN_SUBNORMAL, 1, 1], 1),
    ).toEqual([0, 0, 0]);
    expect(
      finiteEquirectScaledColor([GLSL_MIN_SUBNORMAL, 0, 0], 1),
    ).toEqual([GLSL_MIN_SUBNORMAL, 0, 0]);
    expect(
      finiteEquirectScaledColor([GLSL_MIN_SUBNORMAL, 0, 0], 0.5),
    ).toEqual([0, 0, 0]);
    expect(
      finiteEquirectScaledColor([1, 1, 1], GLSL_INFINITY),
    ).toEqual([GLSL_INFINITY, GLSL_INFINITY, GLSL_INFINITY]);

    const representabilityGuard = equirect_functions.indexOf(
      'vec3 scaled = value * scale;',
    );
    const guardedMultiply = equirect_functions.indexOf('? value * scale');
    expect(representabilityGuard).toBeGreaterThan(-1);
    expect(guardedMultiply).toBeGreaterThan(representabilityGuard);
    expect(equirect_functions).toContain(
      '! any( isnan( scaled ) ) && ! any( isinf( scaled ) )',
    );
    expect(equirect_functions).toContain(
      'finiteEquirectScaledColor( color, environmentIntensity )',
    );
    expect(equirect_functions).toContain(
      'materialIntensity',
    );

    expect(
      direct_light_contribution_function.match(
        /finiteEquirectRadiance\(\s*envColor, surf\.envMapIntensity\s*\)/g,
      ) ?? [],
    ).toHaveLength(2);
    expect(RENDER_MAIN.match(
      /finiteEquirectRadiance\(\s*envColor, state\.envMapIntensity\s*\)/g,
    ) ?? []).toHaveLength(2);
    expect(bdpt_light_subpath).toContain(
      'finiteEquirectRadiance( envColor, 1.0 )',
    );
    expect(bdpt_light_subpath).toContain(
      'incomingPathThroughput = finiteEquirectScaledColor(',
    );
    const bdptMaterialScale = bdpt_light_subpath.indexOf(
      'incomingPathThroughput = finiteEquirectScaledColor(',
    );
    const bdptExtensionProduct = bdpt_light_subpath.indexOf(
      'incomingPathThroughput * segmentRatioWeight *',
    );
    expect(bdptMaterialScale).toBeGreaterThan(-1);
    expect(bdptExtensionProduct).toBeGreaterThan(bdptMaterialScale);

    const environmentTransportSource = [
      direct_light_contribution_function,
      RENDER_MAIN,
      bdpt_light_subpath,
    ].join('\n');
    expect(environmentTransportSource).not.toMatch(
      /(?:surf|state)\.envMapIntensity\s*\*\s*environmentIntensity/,
    );
    expect(environmentTransportSource).not.toContain(
      'envColor * environmentIntensity',
    );
    expect(environmentTransportSource).not.toContain(
      'scatterThroughput *= newSurf.envMapIntensity',
    );
  });

  it('keeps infinite visibility independent of traveled distance in every tier', () => {
    expect(
      remainingVisibilityDistance(
        Number.POSITIVE_INFINITY,
        Number.POSITIVE_INFINITY,
      ),
    ).toBe(Number.POSITIVE_INFINITY);
    expect(
      remainingVisibilityDistance(GLSL_INFINITY, Number.POSITIVE_INFINITY),
    ).toBe(Number.POSITIVE_INFINITY);
    expect(remainingVisibilityDistance(100, 37)).toBe(63);
    expect(util_functions).toContain(
      'return isinf( distance ) || distance >= INFINITY;',
    );

    for (const source of [
      attenuate_hit_function,
      ATTENUATE_HIT_SCALAR_RICH_GLSL,
      ATTENUATE_HIT_BASIC_GLSL,
      ATTENUATE_HIT_MAPPED_PBR_GLSL,
    ]) {
      const compact = source.replace(/\s+/g, ' ');
      expect(compact).toContain(
        'bool finiteRayDistance = ! vitrumIsInfiniteDistance( rayDist );',
      );
      expect(compact).toContain(
        'float remainingDistance = finiteRayDistance ? max( rayDist - traveledDistance, 0.0 ) : INFINITY;',
      );
      expect(compact).toContain(
        'if ( finiteRayDistance ) { traveledDistance += max( surfaceHit.dist, 0.0 ); }',
      );
      expect(compact).not.toContain(
        'float remainingDistance = max( rayDist - traveledDistance, 0.0 );',
      );
      expect(compact).not.toContain('distance( startPoint');
    }
  });

  it('terminates finite mesh visibility by exact target face before distance tolerance', () => {
    // A target hit may round one ray-parameter ULP before the reconstructed
    // endpoint distance; identity still owns termination, while another face
    // at the same parameter remains an occluder.
    const endpointDistance = 1_269_827.625;
    const roundedTargetHit = 1_261_635.125;
    expect(
      endpointReached(0x1_0000, roundedTargetHit, endpointDistance, true, 0x1_0000),
    ).toBe(true);
    expect(
      endpointReached(7, roundedTargetHit, endpointDistance, true, 0x1_0000),
    ).toBe(false);

    for (const source of [
      attenuate_hit_function,
      ATTENUATE_HIT_SCALAR_RICH_GLSL,
      ATTENUATE_HIT_BASIC_GLSL,
      ATTENUATE_HIT_MAPPED_PBR_GLSL,
    ]) {
      const compact = source.replace(/\s+/g, ' ');
      expect(compact).toContain('bool hasTargetFace, uint targetFaceIndex,');
      const targetClassification = compact.indexOf(
        'hasTargetFace && surfaceHit.faceIndices.w == targetFaceIndex',
      );
      const distanceClassification = compact.indexOf(
        'surfaceHit.dist >= remainingDistance',
      );
      const traveledUpdate = compact.indexOf(
        'traveledDistance += max( surfaceHit.dist, 0.0 );',
      );
      expect(targetClassification).toBeGreaterThan(-1);
      expect(distanceClassification).toBeGreaterThan(targetClassification);
      expect(traveledUpdate).toBeGreaterThan(distanceClassification);
    }

    const samplingCompact = light_sampling_functions.replace(/\s+/g, ' ');
    expect(samplingCompact).toContain(
      't.sourceFaceWords = s5.ra; t.sourceFaceIndex = meshLightSourceFaceIndex( t.sourceFaceWords );',
    );
    expect(samplingCompact).toContain(
      'rec.hasTargetFace = true; rec.targetFaceIndex = tri.sourceFaceIndex;',
    );

    const directCompact = direct_light_contribution_function.replace(/\s+/g, ' ');
    expect(directCompact).toContain(
      'lightSample.hasTargetFace = lightRec.hasTargetFace; lightSample.targetFaceIndex = lightRec.targetFaceIndex;',
    );
    expect(directCompact).toContain(
      'lightSample.distance, lightSample.hasTargetFace, lightSample.targetFaceIndex, attenuatedColor',
    );

    const subpathCompact = bdpt_light_subpath.replace(/\s+/g, ' ');
    expect(subpathCompact).toContain(
      'tri.sourceFaceWords.x, tri.sourceFaceWords.y',
    );
    expect(subpathCompact).toContain(
      'vec4( light.castShadowDisabled, 0.0, -1.0, 0.0 )',
    );
    const connectionCompact = bdpt_connection.replace(/\s+/g, ' ');
    expect(connectionCompact).toContain(
      'bool surfaceVertexHasTarget = ! lightIsEndpoint && ! lightIsMedium;',
    );
    expect(connectionCompact).toContain(
      'targetFaceIndex = meshLightSourceFaceIndex( lv4.zw );',
    );
    expect(connectionCompact).toContain(
      'targetFaceIndex = meshLightSourceFaceIndex( lv7.zw );',
    );
    expect(connectionCompact).toContain(
      'len, hasTargetFace, targetFaceIndex, attenColor',
    );
  });

  it('rebuilds finite visibility rays from the offset origin to the sampled endpoint', () => {
    expect(direct_light_contribution_function).toContain('vec3 point;');
    expect(direct_light_contribution_function).toContain(
      'if ( ! vitrumIsInfiniteDistance( lightSample.distance ) )',
    );
    expect(direct_light_contribution_function).toContain(
      'vec3 toLightEndpoint = lightSample.point - lightRay.origin;',
    );
    expect(direct_light_contribution_function).toContain(
      'lightRay.direction = toLightEndpoint / endpointDistance;',
    );
    expect(direct_light_contribution_function).not.toContain(
      'lightRec.dist - 2.0 * RAY_OFFSET',
    );
  });

  it('measures forward-hit PDFs from the preceding accepted scattering vertex', () => {
    const compact = RENDER_MAIN.replace(/\s+/g, ' ');
    expect(compact).toContain('vec3 incomingAcceptedVertexPoint = ray.origin;');
    expect(compact).toContain(
      'forwardAreaLightRec.point - incomingAcceptedVertexPoint',
    );
    expect(compact).toContain('geometricHitPoint - incomingAcceptedVertexPoint');
    expect(compact).toContain('incomingAcceptedVertexPoint = geometricHitPoint;');
    expect(compact).toContain('gbufLinearDepth = vitrumSaturatedLengthVec3(');
  });
});
