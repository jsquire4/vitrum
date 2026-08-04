import { describe, expect, it } from 'vitest';
import { RENDER_MAIN } from './renderMain.glsl.js';
import * as BdptConnectionModule from './render/bdpt_connection.glsl.js';
import * as BdptLightSubpathModule from './render/bdpt_light_subpath.glsl.js';
import * as UtilModule from './shader/common/util_functions.glsl.js';

function glslChunk(module: Record<string, unknown>, name: string): string {
  const value = module[name];
  if (typeof value !== 'string') {
    throw new TypeError(`Expected GLSL module "${name}" to export a string.`);
  }
  return value;
}

const connection = glslChunk(BdptConnectionModule, 'bdpt_connection');
const lightSubpath = glslChunk(
  BdptLightSubpathModule,
  'bdpt_light_subpath',
);
const util = glslChunk(UtilModule, 'util_functions');

type Vec3 = readonly [number, number, number];

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function offsetSide(direction: Vec3, geometricNormal: Vec3): number {
  return dot(direction, geometricNormal) < 0 ? -1 : 1;
}

function packFaceIndexWords(faceIndex: number): readonly [number, number] {
  return [faceIndex & 0xffff, faceIndex >>> 16];
}

function unpackFaceIndexWords(words: readonly [number, number]): number {
  return (words[0] | (words[1] << 16)) >>> 0;
}

describe('pt-webgl2 final BDPT numerical and identity closure', () => {
  it('chooses visibility offset from the connection lobe while keeping MIS positions geometric', () => {
    const normal: Vec3 = [0, 0, 1];
    const sampledContinuation: Vec3 = [0, 0, -1];
    const explicitConnection: Vec3 = [0, 0, 1];

    // A mixed reflection/transmission BSDF can select the continuation below
    // the surface while an independently evaluated connection remains above it.
    expect(offsetSide(sampledContinuation, normal)).toBe(-1);
    expect(offsetSide(explicitConnection, normal)).toBe(1);

    const compactConnection = connection.replace(/\s+/g, ' ');
    const compactMain = RENDER_MAIN.replace(/\s+/g, ' ');
    expect(compactConnection).toContain(
      'float visibilitySide = dot( connDir, geometricNormal ) < 0.0 ? -1.0 : 1.0;',
    );
    expect(compactConnection).toContain(
      'vec3 visibilityOrigin = stepRayOrigin( eyePos, vec3( 0.0 ), visibilityOffset, 0.0 );',
    );
    expect(compactConnection).toContain(
      'bdptVisibilityAttenuation( visibilityOrigin, lightPos,',
    );
    expect(compactMain).toContain(
      'bdptEyePos[ bdptEyeDepth ] = geometricHitPoint;',
    );
    expect(compactMain).toContain(
      'pc_fragColor.rgb += evaluateBdptConnection( geometricHitPoint,',
    );
    expect(compactMain).toContain('bdptPrevPos = geometricHitPoint;');
    expect(compactMain).not.toContain(
      'bdptEyePos[ bdptEyeDepth ] = hitPoint;',
    );
  });

  it('keeps a representable anisotropic area-to-solid-angle quotient out of the 0/0 domain', () => {
    const f32 = Math.fround;
    const edgeScale = f32(1e20);
    const area = f32(1e10);
    const distance = f32(1e-3);
    const cosine = f32(1e-20);
    const normalizedArea = f32(f32(area / edgeScale) / edgeScale);
    const distanceOverEdge = f32(distance / edgeScale);
    const oldNumerator = f32(distanceOverEdge * distanceOverEdge);
    const oldDenominator = f32(normalizedArea * cosine);

    expect(oldNumerator).toBe(0);
    expect(oldDenominator).toBe(0);
    expect(oldNumerator / oldDenominator).toBeNaN();

    const logDomain = 2 ** (
      2 * Math.log2(distance) -
      Math.log2(area) -
      Math.log2(cosine)
    );
    expect(logDomain).toBeCloseTo(1e4, 2);

    const compact = util.replace(/\s+/g, ' ');
    expect(compact).toContain(
      'float logResult = 2.0 * log2( distance ) - log2( measure.area ) - log2( cosine );',
    );
    expect(compact).toContain('float result = exp2( logResult );');
    expect(compact).not.toContain(
      'distanceOverEdge * distanceOverEdge',
    );
  });

  it('uses the exact sampled reverse density and fails only that alternate strategy closed', () => {
    const f32 = Math.fround;
    const edge: Vec3 = [f32(2e20), f32(1e20), 0];
    const rawLengthSquared = f32(
      f32(edge[0] * edge[0]) +
      f32(edge[1] * edge[1]),
    );
    expect(rawLengthSquared).toBe(Number.POSITIVE_INFINITY);

    const scale = Math.max(...edge.map(Math.abs));
    const scaled = edge.map((component) => component / scale);
    const stableLength = Math.hypot(...scaled);
    const stableDirection = scaled.map((component) => component / stableLength);
    expect(stableDirection.every(Number.isFinite)).toBe(true);
    expect(Math.hypot(...stableDirection)).toBeCloseTo(1, 14);

    const compactMain = RENDER_MAIN.replace(/\s+/g, ' ');
    expect(compactMain).toContain(
      'float candidateReverseDensity = scatterRec.pdfRev * bdptEyeSegmentReverseDensity;',
    );
    expect(compactMain).toContain(
      'scatterRec.pdfRev >= 0.0 && ! isnan( scatterRec.pdfRev ) && ! isinf( scatterRec.pdfRev )',
    );
    expect(compactMain).toContain(
      'candidateReverseDensity >= 0.0 && ! isnan( candidateReverseDensity ) && ! isinf( candidateReverseDensity )',
    );
    expect(compactMain).toContain(
      'float bdptPatchedReverseDensity = 0.0;',
    );
    expect(compactMain).not.toContain(
      'float bdptSwappedRev = bsdfPdfResult(',
    );
  });

  it('round-trips every uint32 surface face identity above the f32 exact-integer limit', () => {
    const faceIndex = 0x0100_0001;
    expect(Math.fround(faceIndex)).toBe(0x0100_0000);
    const words = packFaceIndexWords(faceIndex);
    expect(words).toEqual([1, 256]);
    expect(unpackFaceIndexWords(words)).toBe(faceIndex);

    const compactSubpath = lightSubpath.replace(/\s+/g, ' ');
    const compactConnection = connection.replace(/\s+/g, ' ');
    expect(compactSubpath).toContain(
      'vec2 bdptPackFaceIndexWords( uint faceIndex )',
    );
    expect(compactSubpath).toContain(
      'v7.zw = bdptPackFaceIndexWords( hit.faceIndices.w );',
    );
    expect(compactSubpath).toContain(
      'uint triIndex = meshLightSourceFaceIndex( faceIndexWords );',
    );
    expect(compactConnection).toContain(
      'targetFaceIndex = meshLightSourceFaceIndex( lv7.zw );',
    );
    expect(compactConnection).not.toContain(
      'targetFaceIndex = uint( round( lv4.x ) );',
    );
  });
});
