/**
 * A9 — BDPT production-quality structure tests: the REAL glossy/specular light
 * subpath, the 7-row light-path vertex carrying the light-vertex BSDF, hit-local
 * material payload, and interface eta metadata, the raised bounce cap,
 * and the isotropic point emitter.
 *
 * These pin the WGSL structure + the host-side caps + the CPU oracle parity. The
 * GPU radiometric A/Bs (equal-spp variance vs the megakernel on a glass Cornell;
 * the BDPT caustic scene) are V28 queue entries — see road-to-100 A9.
 */
import { describe, expect, it } from 'vitest';

import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import {
  bdptEmitterThroughputOracle,
  spectralCombinedReflectanceAtHeroOracle,
  spectralRgbFactorAtHeroOracle,
} from './spectralScalarOracle.js';
const FRONT_FACE_BIT = 0x80000000 >>> 0;
const TRI_INDEX_MASK = 0x7fffffff >>> 0;

type Vec3 = readonly [number, number, number];

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

function mul3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
}

function scale3(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function clamp3(a: Vec3, lo = 0, hi = 1): Vec3 {
  return [clamp(a[0], lo, hi), clamp(a[1], lo, hi), clamp(a[2], lo, hi)];
}

function mix3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [
    a[0] * (1 - t) + b[0] * t,
    a[1] * (1 - t) + b[1] * t,
    a[2] * (1 - t) + b[2] * t,
  ];
}

function expectVecClose(actual: Vec3, expected: Vec3, precision = 6): void {
  expect(actual[0]).toBeCloseTo(expected[0], precision);
  expect(actual[1]).toBeCloseTo(expected[1], precision);
  expect(actual[2]).toBeCloseTo(expected[2], precision);
}

function packBdptPayloadTriWord(triIndex: number, isFrontFace: boolean): number {
  return (((triIndex >>> 0) & TRI_INDEX_MASK) | (isFrontFace ? FRONT_FACE_BIT : 0)) >>> 0;
}

function unpackBdptPayloadTriWord(word: number): { triIndex: number; isFrontFace: boolean } {
  const raw = word >>> 0;
  return {
    triIndex: (raw & TRI_INDEX_MASK) >>> 0,
    isFrontFace: (raw & FRONT_FACE_BIT) !== 0,
  };
}

function cauchyIorAtLambdaOracle(lambdaNm: number, baseIor: number, abbeV: number): number {
  if (abbeV < 1) {
    return baseIor;
  }
  const lambdaUm = lambdaNm * 0.001;
  const lam2 = lambdaUm * lambdaUm;
  const lamF = 0.4861;
  const lamC = 0.6563;
  const denom = 1 / (lamF * lamF) - 1 / (lamC * lamC);
  const B = (baseIor - 1) / Math.max(abbeV, 1) / Math.max(denom, 1e-6);
  return baseIor + B / lam2;
}

function jakobHanikaReflectanceOracle(coeffs: Vec3, lambdaNm: number): number {
  const x = coeffs[0] + coeffs[1] * lambdaNm + coeffs[2] * lambdaNm * lambdaNm;
  return 0.5 + x / (2 * Math.sqrt(1 + x * x));
}

interface BdptPayloadMaterialFixture {
  baseColor: Vec3;
  vertexColor: Vec3;
  baseColorTex: Vec3;
  ao: number;
  roughness: number;
  metallic: number;
  orm: Vec3;
  transmission: number;
  transmissionTex: number;
  ior: number;
  dispersionAbbe: number;
  clearcoat: number;
  clearcoatTex: number;
  clearcoatRoughness: number;
  clearcoatRoughnessTex: number;
  sheen: number;
  sheenRoughness: number;
  sheenRoughnessTex: number;
  sheenColor: Vec3;
  sheenColorTex: Vec3;
  iridescence: number;
  iridescenceTex: number;
  iridescenceThicknessMin: number;
  iridescenceThicknessMax: number;
  iridescenceThicknessTex: number;
  specularColor: Vec3;
  specularColorTex: Vec3;
  specularIntensity: number;
  specularIntensityTex: number;
  anisotropy: number;
  anisotropyRotation: number;
  frontLayerTx: Vec3;
  frontLayerRoughness: number;
  backLayerTx: Vec3;
  backLayerRoughness: number;
  thinFilmLayerCount: number;
  thinFilmReflectTint: Vec3;
  spectralReflCoeffs: Vec3;
  hasSpectralReflectance: boolean;
}

function sampleBdptPayloadMaterialOracle(
  mat: BdptPayloadMaterialFixture,
  opts: { isFrontFace: boolean; spectralEnabled: boolean; heroLambdaNm: number; thinFilmEnabled: boolean },
) {
  const authoredBaseColor = mat.baseColor;
  let baseColor = scale3(mul3(mul3(mat.baseColor, mat.vertexColor), mat.baseColorTex), mat.ao);
  const roughnessFromOrm = clamp(mat.roughness * mat.orm[1], 0, 1);
  const metallic = clamp(mat.metallic * mat.orm[2], 0, 1);
  const transmission = clamp(mat.transmission * mat.transmissionTex, 0, 1);
  const ior =
    opts.spectralEnabled && mat.dispersionAbbe >= 1
      ? cauchyIorAtLambdaOracle(opts.heroLambdaNm, mat.ior, mat.dispersionAbbe)
      : mat.ior;
  const clearcoat = clamp(mat.clearcoat * mat.clearcoatTex, 0, 1);
  const clearcoatRoughness = clamp(mat.clearcoatRoughness * mat.clearcoatRoughnessTex, 0, 1);
  const sheenRoughness = clamp(mat.sheenRoughness * mat.sheenRoughnessTex, 0, 1);
  const sheenColor = clamp3(mul3(mat.sheenColor, mat.sheenColorTex));
  const iridescence = clamp(mat.iridescence * mat.iridescenceTex, 0, 1);
  const iridescenceThickness = mat.iridescenceThicknessTex >= 0
    ? mat.iridescenceThicknessMin +
      (mat.iridescenceThicknessMax - mat.iridescenceThicknessMin) * mat.iridescenceThicknessTex
    : mat.iridescenceThicknessMin;
  const specularColor = clamp3(mul3(mat.specularColor, mat.specularColorTex));
  const specularIntensity = clamp(mat.specularIntensity * mat.specularIntensityTex, 0, 1);
  const layerTx = clamp3(opts.isFrontFace ? mat.frontLayerTx : mat.backLayerTx);
  const layerRoughness = opts.isFrontFace ? mat.frontLayerRoughness : mat.backLayerRoughness;
  const roughness = layerRoughness >= 0 ? clamp(layerRoughness, 0, 1) : roughnessFromOrm;
  baseColor = mul3(baseColor, layerTx);
  if (opts.thinFilmEnabled) {
    const layerStrength = clamp(0.12 + 0.06 * mat.thinFilmLayerCount, 0, 0.55);
    const filmStrength = clamp(layerStrength * (1 - roughness), 0, 0.6);
    baseColor = mix3(baseColor, mul3(baseColor, clamp3(mat.thinFilmReflectTint)), filmStrength);
  }
  if (opts.spectralEnabled) {
    const authoredReflectance = mat.hasSpectralReflectance
      ? jakobHanikaReflectanceOracle(mat.spectralReflCoeffs, opts.heroLambdaNm)
      : spectralRgbFactorAtHeroOracle(authoredBaseColor, opts.heroLambdaNm);
    const refl = spectralCombinedReflectanceAtHeroOracle(
      baseColor, authoredBaseColor, authoredReflectance, opts.heroLambdaNm,
    );
    baseColor = [refl, refl, refl];
  }

  return {
    baseColor,
    roughness,
    metallic,
    transmission,
    ior,
    clearcoat,
    clearcoatRoughness,
    sheen: mat.sheen,
    sheenRoughness,
    sheenColor,
    iridescence,
    iridescenceThicknessMin: iridescenceThickness,
    iridescenceThicknessMax: iridescenceThickness,
    specularColor,
    specularIntensity,
    anisotropy: mat.anisotropy,
    anisotropyRotation: mat.anisotropyRotation,
  };
}

describe('A9 — glossy/specular BDPT light subpath', () => {
  it('samples the REAL BSDF (glossy partition + cosine diffuse), not Lambertian-only', () => {
    // BDPT light-subpath estimator coherence (2026-06-10): the scatter direction
    // is sampled at the PREVIOUS vertex (prevPos) using its stored outgoing direction
    // (woAtPrev), and that SAME direction is used to extend the path (trace) AND
    // to compute the stored throughput / pdfFwd. The old two-step
    // (cosine-hemisphere trace + discard + real-BSDF sample at newPos) is gone.
    // The BSDF is sampled at prevPos through the shared main-path sampler so the
    // scalar clearcoat/sheen source-lobe mixture and sampled PDF stay coherent.
    //
    // PTWG-BDPT-01 (2026-06-15): finite area emitters need the cos/pdfΩ = π
    // factor after the first traced hit; legacy pseudo emitters keep the old
    // INV_PI branch because their bounce-0 normalization already includes the
    // direction-density term.
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let bsPrev = sampleNextBounceDirectionWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevMat.clearcoat,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevMat.sheenRoughness,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let f0BasePrev = materialSpecularF0(');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let f0Prev = iridescenceModifiedF0(');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevMat.iridescenceThicknessMax,');
    // f and throughput computed at prevPos (prevMat/prevNormal/woAtPrev/scatterDir).
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('surfaceThroughputMul = bsPrev.throughputMul;');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('surfaceRayOrigin = bsPrev.newRayOrigin;');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('var newThroughput = prevThroughput * surfaceThroughputMul * segmentWeight;');
    // pdfFwd = scatter pdf at prevPos (SA, no baked-in geometry term).
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('pdfScatter = bsPrev.sampledEventPdf;');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let pdfFwd = pdfScatter * segmentForwardDensity;');
    // pdfRev(prevCol) is patched to the TRUE reverse density (Item-3 fix 2026-06-10):
    // for surface vertices, brdfDirectionalPdf(prevNormal, scatterDir, woAtPrev) —
    // NOT pdfFwd, which was the forward pdf and only equal for symmetric BSDFs.
    // For emitter vertices (prevMatId < 0), Lambertian cosine hemisphere IS symmetric
    // so pdfFwd == pdfRev; the emitter branch correctly falls back to pdfFwd.
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let predecessorCol = prevCol - 1;');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('swappedDirectionalPdf = bdptMarginalSurfacePdf(');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('swappedDirectionalPdf * reverseEdgeDensity,');
  });

  it('finite-area emitter extension keeps the needed π factor without a bounce-0 direction PDF', () => {
    const rgbThroughput = bdptEmitterThroughputOracle([4, 2, 1], 630, 0.2, false);
    expectVecClose(rgbThroughput, [20, 10, 5], 12);

    const spectralThroughput = bdptEmitterThroughputOracle([4, 2, 1], 630, 0.2, true);
    expect(spectralThroughput[0]).toBeGreaterThan(0);
    expect(spectralThroughput[1]).toBe(spectralThroughput[0]);
    expect(spectralThroughput[2]).toBe(spectralThroughput[0]);
    const twicePdf = bdptEmitterThroughputOracle([4, 2, 1], 630, 0.4, true);
    expect(spectralThroughput[0] / twicePdf[0]).toBeCloseTo(2, 12);

    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('if (prevMatId == BDPT_LV_AREA_EMITTER_MATID) {');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let hemi = cosineHemisphereSample(&rng, prevNormal);');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('fPrev = vec3f(1.0);');
    // Surface vertices still use the real extension-aware BRDF.
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('surfaceThroughputMul = bsPrev.throughputMul;');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('newThroughput = prevThroughput * fPrev * cosPrev / pdfScatter * segmentWeight;');
  });

  it('launches directional and environment roots from the scene-bounding disk', () => {
    const code = PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL
      .split("\n")
      .filter((line) => !line.trim().startsWith("//"))
      .join("\n");
    expect(code).toContain('params.directionalLightCount');
    expect(code).toContain('fn bdptInfiniteLaunchDisk(');
    expect(code).toContain('BDPT_LV_DIRECTIONAL_EMITTER_MATID');
    expect(code).toContain('BDPT_LV_ENVIRONMENT_EMITTER_MATID');
    expect(code).toContain('1.0 / (PI * radius * radius),');
  });

  it('mirrors emitter castShadow:false into BDPT bounce-0 connection visibility', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('fn bdptWriteLvEmitterPayload(');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'select(0.0, 1.0, castShadowDisabled), cutoffDistance, decay, spotCosOuter,',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('ptExtra.z > 0.5,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('spExtra.z > 0.5,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('rbase.w > 0.5,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('meshAreaLights[mb + 3u].w > 0.5,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'let lightEmitterCastShadowDisabled = lightVtxIdx == 0 && lvMatId < 0.0 && lv4.x > 0.5;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'if (!lightEmitterCastShadowDisabled && traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {',
    );
  });

  it('records the light-vertex matId + wo-toward-prev so the connection can evaluate the real BSDF', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('bdptWriteLvBsdf(col, f32(matIdx), woLp);');
    // The emitter vertex is marked with the sentinel matId < 0 (Lambertian/emission).
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('const BDPT_LV_EMITTER_MATID: f32 = -1.0;');
  });

  it('stores hit-local material payload for texture-mapped light-path vertices', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let sideBit = select(0u, 0x80000000u, isFrontFace);',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'bdptLightPath[bdptLightPathIndex(col, 4u)] = vec4f(bitcast<f32>((triIndex & 0x7fffffffu) | sideBit), baryVW.x, baryVW.y, bitcast<f32>(instanceIndex));',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'bdptWriteLvMaterialPayload(col, hit.triIndex, hit.baryVW, hit.instanceIndex, isFrontFaceHit);',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let prevPayload = bdptLightPath[bdptLightPathIndex(prevCol, 4u)];',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let prevMat = bdptSampleMaterialAtPayload(',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let triWord = bitcast<u32>(payload.x);');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let triIndex = triWord & 0x7fffffffu;');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let isFrontFace = (triWord & 0x80000000u) != 0u;');
  });

  it('numeric oracle pins row-4 tri/front-face payload packing', () => {
    const front = packBdptPayloadTriWord(123_456, true);
    expect(front).toBe(0x8001e240);
    expect(unpackBdptPayloadTriWord(front)).toEqual({
      triIndex: 123_456,
      isFrontFace: true,
    });

    const back = packBdptPayloadTriWord(123_456, false);
    expect(back).toBe(0x0001e240);
    expect(unpackBdptPayloadTriWord(back)).toEqual({
      triIndex: 123_456,
      isFrontFace: false,
    });

    const masked = packBdptPayloadTriWord(0xffffffff, false);
    expect(masked).toBe(TRI_INDEX_MASK);
    expect(unpackBdptPayloadTriWord(masked)).toEqual({
      triIndex: TRI_INDEX_MASK,
      isFrontFace: false,
    });
  });

  it('samples texture-map material payloads before BDPT light-subpath scatter', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'out.baseColor = mat.baseColor * sampleVertexColor(triIndex, baryVW).rgb * sampleBaseColorTexture(matId, triIndex, baryVW, instanceIndex).rgb;',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let orm = sampleOrmTexture(matId, triIndex, baryVW, instanceIndex);');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'out.clearcoat = clamp(mat.clearcoat * sampleClearcoatTexture(matId, triIndex, baryVW, instanceIndex), 0.0, 1.0);',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'out.clearcoatNormal = applyClearcoatNormalMap(matId, triIndex, baryVW, shadingNormal, instanceIndex);',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevNormal,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevMat.clearcoatNormal,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'out.specularColor = clamp(mat.specularColor * sampleSpecularColorTexture(matId, triIndex, baryVW, instanceIndex), vec3f(0.0), vec3f(1.0));',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'nsFront = applyNormalMap(matIdx, hit.triIndex, hit.baryVW, nsFront, hit.instanceIndex, isFrontFaceHit);',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevMat.anisotropy,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevMat.anisotropyRotation,');
  });

  it('numeric oracle pins mapped material payload transforms used by BDPT light vertices', () => {
    const fixture: BdptPayloadMaterialFixture = {
      baseColor: [0.9, 0.6, 0.3],
      vertexColor: [0.5, 0.25, 0.75],
      baseColorTex: [0.8, 0.5, 0.25],
      ao: 0.4,
      roughness: 0.7,
      metallic: 0.8,
      orm: [0.2, 0.5, 1.25],
      transmission: 0.6,
      transmissionTex: 0.5,
      ior: 1.5,
      dispersionAbbe: 50,
      clearcoat: 0.75,
      clearcoatTex: 1.4,
      clearcoatRoughness: 0.9,
      clearcoatRoughnessTex: 0.2,
      sheen: 0.65,
      sheenRoughness: 0.5,
      sheenRoughnessTex: 3,
      sheenColor: [0.9, 0.7, 0.5],
      sheenColorTex: [0.5, 2, 0.25],
      iridescence: 0.9,
      iridescenceTex: 0.5,
      iridescenceThicknessMin: 100,
      iridescenceThicknessMax: 500,
      iridescenceThicknessTex: 0.25,
      specularColor: [0.9, 0.6, 0.2],
      specularColorTex: [0.5, 0.25, 2],
      specularIntensity: 0.8,
      specularIntensityTex: 1.5,
      anisotropy: 0.35,
      anisotropyRotation: 0.7,
      frontLayerTx: [0.5, 0.75, 1.25],
      frontLayerRoughness: 0.01,
      backLayerTx: [0.2, 0.3, 0.4],
      backLayerRoughness: 0.6,
      thinFilmLayerCount: 4,
      thinFilmReflectTint: [0.25, 0.5, 1],
      spectralReflCoeffs: [-3, 0.01, -0.000005],
      hasSpectralReflectance: true,
    };

    const sampled = sampleBdptPayloadMaterialOracle(fixture, {
      isFrontFace: true,
      spectralEnabled: false,
      heroLambdaNm: 510,
      thinFilmEnabled: true,
    });

    expectVecClose(sampled.baseColor, [0.0527544, 0.0184905, 0.0225]);
    expect(sampled.roughness).toBeCloseTo(0.01, 8);
    expect(sampled.metallic).toBeCloseTo(1, 8);
    expect(sampled.transmission).toBeCloseTo(0.3, 8);
    expect(sampled.ior).toBeCloseTo(1.5, 8);
    expect(sampled.clearcoat).toBeCloseTo(1, 8);
    expect(sampled.clearcoatRoughness).toBeCloseTo(0.18, 8);
    expect(sampled.sheen).toBeCloseTo(0.65, 8);
    expect(sampled.sheenRoughness).toBeCloseTo(1, 8);
    expectVecClose(sampled.sheenColor, [0.45, 1, 0.125]);
    expect(sampled.iridescence).toBeCloseTo(0.45, 8);
    expect(sampled.iridescenceThicknessMin).toBeCloseTo(200, 8);
    expect(sampled.iridescenceThicknessMax).toBeCloseTo(200, 8);
    expectVecClose(sampled.specularColor, [0.45, 0.15, 0.4]);
    expect(sampled.specularIntensity).toBeCloseTo(1, 8);
    expect(sampled.anisotropy).toBeCloseTo(0.35, 8);
    expect(sampled.anisotropyRotation).toBeCloseTo(0.7, 8);

    const backSide = sampleBdptPayloadMaterialOracle(fixture, {
      isFrontFace: false,
      spectralEnabled: false,
      heroLambdaNm: 510,
      thinFilmEnabled: false,
    });
    expectVecClose(backSide.baseColor, [0.0288, 0.009, 0.009]);
    expect(backSide.roughness).toBeCloseTo(0.6, 8);
  });

  it('applies layer, spectral albedo, and Cauchy IOR to BDPT light-side material payloads', () => {
    for (const line of [
      'out.ior = cauchyIorAtLambda(heroLambda, mat.ior, mat.dispersionAbbe);',
      'let layerTx = clamp(select(mat.backLayerTx, mat.frontLayerTx, isFrontFace), vec3f(0.0), vec3f(1.0));',
      'out.roughness = clamp(layerRoughness, 0.0, 1.0);',
      'activeLayerWeightRgb(layerTx, heroLambda, true)',
      'spectralCombinedReflectanceAtHero(',
      'out.baseColor = vec3f(reflScalar);',
    ]) {
      expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(line);
    }
  });

  it('numeric oracle preserves Jakob-Hanika while applying runtime base factors at hero lambda', () => {
    const fixture: BdptPayloadMaterialFixture = {
      baseColor: [0.9, 0.6, 0.3],
      vertexColor: [0.5, 0.25, 0.75],
      baseColorTex: [0.8, 0.5, 0.25],
      ao: 0.4,
      roughness: 0.7,
      metallic: 0.8,
      orm: [0.2, 0.5, 1.25],
      transmission: 0.6,
      transmissionTex: 0.5,
      ior: 1.5,
      dispersionAbbe: 50,
      clearcoat: 0.75,
      clearcoatTex: 1.4,
      clearcoatRoughness: 0.9,
      clearcoatRoughnessTex: 0.2,
      sheen: 0.65,
      sheenRoughness: 0.5,
      sheenRoughnessTex: 3,
      sheenColor: [0.9, 0.7, 0.5],
      sheenColorTex: [0.5, 2, 0.25],
      iridescence: 0.9,
      iridescenceTex: 0.5,
      iridescenceThicknessMin: 100,
      iridescenceThicknessMax: 500,
      iridescenceThicknessTex: 0.25,
      specularColor: [0.9, 0.6, 0.2],
      specularColorTex: [0.5, 0.25, 2],
      specularIntensity: 0.8,
      specularIntensityTex: 1.5,
      anisotropy: 0.35,
      anisotropyRotation: 0.7,
      frontLayerTx: [0.5, 0.75, 1.25],
      frontLayerRoughness: 0.01,
      backLayerTx: [0.2, 0.3, 0.4],
      backLayerRoughness: 0.6,
      thinFilmLayerCount: 4,
      thinFilmReflectTint: [0.25, 0.5, 1],
      spectralReflCoeffs: [-3, 0.01, -0.000005],
      hasSpectralReflectance: true,
    };

    const sampled = sampleBdptPayloadMaterialOracle(fixture, {
      isFrontFace: true,
      spectralEnabled: true,
      heroLambdaNm: 510,
      thinFilmEnabled: true,
    });
    const authoredOnly = sampleBdptPayloadMaterialOracle({
      ...fixture,
      vertexColor: [1, 1, 1],
      baseColorTex: [1, 1, 1],
      ao: 1,
      frontLayerTx: [1, 1, 1],
      frontLayerRoughness: -1,
    }, {
      isFrontFace: true, spectralEnabled: true, heroLambdaNm: 510, thinFilmEnabled: false,
    });


    expect(sampled.ior).toBeCloseTo(1.52012509542473, 10);
    expectVecClose(authoredOnly.baseColor, [0.8122284453397484, 0.8122284453397484, 0.8122284453397484], 10);
    expectVecClose(sampled.baseColor, [0.03699294454299884, 0.03699294454299884, 0.03699294454299884], 10);
    expect(sampled.baseColor[0]).toBeLessThan(authoredOnly.baseColor[0]);
  });

  it('the §10.3 connection evaluates the REAL light-vertex BSDF + pdfs for a surface vertex', () => {
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('eyeClearcoatNormal: vec3f,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('specularColor: vec3f,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('specularIntensity: f32,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('eyeBrdf = evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'baseColor, roughness, metallic, transmission, etaTOverI,',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('specularColor, specularIntensity,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('var revLc = 0.0;');
    // lightBsdfCosTheta uses the real BSDF when matId >= 0 (was always cosθ/π),
    // without re-multiplying cosLight (the geometry term owns edge cosines).
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let lv4 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 4u)];');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('if (lvMatId >= 0.0) {');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let lvMat = bdptSampleMaterialAtPayload(u32(lvMatId), lv4, lightNormal, lvWoPrev, bdptInvocationHeroLambdaNm);');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let lvBrdf = evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('lightNormal, lvMat.clearcoatNormal, lvWoPrev, -connDir,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('lightBsdfCosTheta = lvBrdf;');
    // The MIS pdf bookkeeping (fwdEe + revLcMinus) also uses the real BSDF pdf.
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('fwdEe = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('revLc = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('lightNormal, lvMatF.clearcoatNormal, lvWoPrev, lcToE,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'lvMatF.clearcoat, lvMatF.clearcoatRoughness, lvMatF.sheen, lvMatF.sheenRoughness,',
    );
    // Legacy pseudo emitters keep the Lambertian emission profile; finite area
    // emitters use a distinct sentinel and contribute no extra endpoint factor.
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('if (lvMatId == -1.0) {');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('lightBsdfCosTheta = vec3f(cosLight / PI);');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('const BDPT_LV_AREA_EMITTER_MATID: f32 = -2.0;');
  });

  it('the light-path vertex is 7 rows (row 6 = both medium-side budgets)', () => {
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain('const BDPT_LIGHT_PATH_ROWS = 7u;');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'bdptWriteLvInterfaceEta(col, hitEtaTOverI, incidentIor, transmittedIor);',
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain(
      'hit-local tri/bary/instance payload for texture-map material sampling. The high',
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain(
      'bit of row-4.x stores the front-face flag; real triangle indices are required to',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'var<private> bdptEyeStackPrivate: array<BdptEyeVtx, 8>;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('bdptEyeStackPrivate[d] = v;');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'incidentMediumMatId: u32,',
    );
  });

  it('keeps the eye stack invocation-local with the production depth bound', () => {
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('const BDPT_MAX_EYE_DEPTH: u32 = 8u;');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).not.toContain('var<storage');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).not.toContain('bdptCurrentPixel');
  });
});

describe('A9 — raised bounce cap + isotropic point emitter', () => {
  it('bounds the invocation-local light path to eight 7-row vertices', () => {
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain('const BDPT_MAX_LIGHT_DEPTH = 8u;');
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain(
      'var<private> bdptLightPath: array<vec4f, 56>;',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let maxB = i32(min(params.bdptMaxLightBounces, BDPT_MAX_LIGHT_DEPTH));',
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).not.toContain('@group(2) @binding(5)');
  });

  it('the point emitter is ISOTROPIC (uniform sphere, 1/4π), not cosine-up', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('scatterDir = uniformSphere(vec2f(rand_f32(&rng), rand_f32(&rng)));');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('pdfScatter = 0.25 * INV_PI;');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('BDPT_LV_POINT_EMITTER_MATID,');
  });
});
