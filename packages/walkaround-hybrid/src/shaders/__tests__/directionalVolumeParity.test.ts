import { describe, expect, it } from 'vitest';

import { RESTIR_GI_MATERIAL_WGSL } from '../restirGiMaterial.wgsl.js';
import { NATIVE_GLASS_GI_WGSL } from '../risGiGlassWalk.wgsl.js';
import { NRC_INDEPENDENT_SUFFIX_WGSL } from '../nrcIndependentSuffix.wgsl.js';
import { RESTIR_PHAT_WGSL } from '../restirPHat.wgsl.js';
import { SHADE_WGSL } from '../shade.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';
import { TRANSPARENT_OIT_WGSL } from '../transparentOit.wgsl.js';

type Vec3 = readonly [number, number, number];

const REC709: Vec3 = [0.2126, 0.7152, 0.0722];

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scale(a: Vec3, factor: number): Vec3 {
  return [a[0] * factor, a[1] * factor, a[2] * factor];
}

function multiply(a: Vec3, b: Vec3): Vec3 {
  return [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
}

function luminance(value: Vec3): number {
  return value[0] * REC709[0] + value[1] * REC709[1] +
    value[2] * REC709[2];
}

function henyeyGreenstein(cosTheta: number, g: number): number {
  const anisotropy = Math.max(-0.99, Math.min(0.99, g));
  const denominator = 1 + anisotropy * anisotropy -
    2 * anisotropy * Math.max(-1, Math.min(1, cosTheta));
  return (1 - anisotropy * anisotropy) /
    (4 * Math.PI * denominator * Math.sqrt(denominator));
}

function directionalVolume(
  radiance: Vec3,
  albedo: Vec3,
  sigmaS: Vec3,
  pathLength: number,
  projectedCosine: number,
  propagationViewCosine: number,
  g: number,
): Vec3 {
  if (sigmaS.every((channel) => channel <= 0) || pathLength <= 0) {
    return radiance;
  }
  const phase = henyeyGreenstein(propagationViewCosine, g);
  const source = scale(albedo, luminance(radiance) * phase);
  if (projectedCosine <= 0) return source;
  const distance = pathLength / projectedCosine;
  return [0, 1, 2].map((channel) => {
    const transmittance = Math.exp(-sigmaS[channel]! * distance);
    return radiance[channel]! * transmittance +
      source[channel]! * (1 - transmittance);
  }) as unknown as Vec3;
}

function logAdd2(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const hi = Math.max(a, b);
  const lo = Math.min(a, b);
  return hi + Math.log2(1 + 2 ** (lo - hi));
}

function directionalVolumeLog2(
  logRadiance: Vec3,
  albedo: Vec3,
  sigmaS: Vec3,
  pathLength: number,
  projectedCosine: number,
  propagationViewCosine: number,
  g: number,
): Vec3 {
  if (sigmaS.every((channel) => channel <= 0) || pathLength <= 0) {
    return logRadiance;
  }
  let logLuminance = -Infinity;
  for (let channel = 0; channel < 3; channel += 1) {
    logLuminance = logAdd2(
      logLuminance,
      logRadiance[channel]! + Math.log2(REC709[channel]!),
    );
  }
  const logPhase = Math.log2(
    henyeyGreenstein(propagationViewCosine, g),
  );
  const logSource = [0, 1, 2].map((channel) =>
    albedo[channel]! > 0
      ? logLuminance + Math.log2(albedo[channel]!) + logPhase
      : -Infinity,
  ) as unknown as Vec3;
  if (projectedCosine <= 0) return logSource;
  const distance = pathLength / projectedCosine;
  return [0, 1, 2].map((channel) => {
    const transmittance = Math.exp(-sigmaS[channel]! * distance);
    const logAttenuated = transmittance > 0
      ? logRadiance[channel]! + Math.log2(transmittance)
      : -Infinity;
    const sourceShare = 1 - transmittance;
    const logScattered = sourceShare > 0
      ? logSource[channel]! + Math.log2(sourceShare)
      : -Infinity;
    return logAdd2(logAttenuated, logScattered);
  }) as unknown as Vec3;
}

function safeDemodulate(physical: Vec3, albedo: Vec3): Vec3 {
  return [0, 1, 2].map((channel) =>
    albedo[channel]! > 0 && Number.isFinite(albedo[channel]!) &&
      physical[channel]! >= 0 && Number.isFinite(physical[channel]!)
      ? physical[channel]! / albedo[channel]!
      : 0,
  ) as unknown as Vec3;
}

function functionBody(source: string, name: string): string {
  const signature = source.indexOf(`fn ${name}(`);
  expect(signature, `${name} is declared`).toBeGreaterThanOrEqual(0);
  const open = source.indexOf('{', signature);
  let depth = 0;
  for (let i = open; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1;
    if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(open + 1, i);
    }
  }
  throw new Error(`${name} has an unterminated body`);
}

describe('directional volume ownership and parity', () => {
  it('keeps the physical volume operator linear across diffuse and specular lanes', () => {
    const diffuse: Vec3 = [0.7, 0.12, 0.03];
    const specular: Vec3 = [0.08, 0.3, 0.65];
    const albedo: Vec3 = [0.9, 0.45, 0.2];
    const sigmaS: Vec3 = [0.2, 0.7, 1.1];
    const args = [albedo, sigmaS, 1.8, 0.63, 0.76, 0.68] as const;
    const combined = directionalVolume(add(diffuse, specular), ...args);
    const split = add(
      directionalVolume(diffuse, ...args),
      directionalVolume(specular, ...args),
    );
    for (let channel = 0; channel < 3; channel += 1) {
      expect(split[channel]).toBeCloseTo(combined[channel]!, 13);
    }
  });

  it('filters the complete colored incident contribution before cross-channel scatter', () => {
    const incident: Vec3 = [0.03, 4.2, 0.17];
    const brdf: Vec3 = [0.82, 0.04, 0.37];
    const albedo: Vec3 = [0.91, 0.38, 0.12];
    const sigmaS: Vec3 = [0.25, 0.8, 1.4];
    const args = [albedo, sigmaS, 1.7, 0.58, -0.31, 0.71] as const;
    const correct = directionalVolume(multiply(incident, brdf), ...args);
    const retiredPostMultiply = multiply(
      incident,
      directionalVolume(brdf, ...args),
    );
    expect(Math.max(...correct.map((value, channel) =>
      Math.abs(value - retiredPostMultiply[channel]!),
    ))).toBeGreaterThan(1e-3);

    const pHatResponse = functionBody(
      RESTIR_PHAT_WGSL,
      'restir_di_eval_surface_response',
    );
    const oitResponse = functionBody(
      TRANSPARENT_OIT_WGSL,
      'oitLayerDirectionalResponse',
    );
    const nrcResponse = functionBody(
      NRC_INDEPENDENT_SUFFIX_WGSL,
      'nrcTeacherDirectionalIncidentResponse',
    );
    expect(pHatResponse).toContain('incidentRadiance * brdf,');
    expect(oitResponse).toContain('incidentRadiance * layeredClosure,');
    expect(nrcResponse).toContain(
      'let rawResponse = incidentRadiance * nrcTeacherMaterialResponse(',
    );
    expect(nrcResponse).toContain(
      'applyHomogeneousVolumeSingleScatterDirectional(\n    rawResponse,',
    );
  });

  it('matches the log-domain operator to the linear formula over a wide range', () => {
    const radiance: Vec3 = [2 ** -120, 2 ** 80, 2 ** -35];
    const logRadiance = radiance.map(Math.log2) as unknown as Vec3;
    const albedo: Vec3 = [1e-9, 0.35, 0.95];
    const sigmaS: Vec3 = [0.01, 0.8, 2.4];
    const args = [albedo, sigmaS, 3.25, 0.42, -0.55, 0.73] as const;
    const linear = directionalVolume(radiance, ...args);
    const logarithmic = directionalVolumeLog2(logRadiance, ...args);
    for (let channel = 0; channel < 3; channel += 1) {
      expect(logarithmic[channel]).toBeCloseTo(
        Math.log2(linear[channel]!),
        11,
      );
    }
  });

  it('preserves zero albedo and round-trips tiny positive albedo without epsilon', () => {
    const albedo: Vec3 = [0, 1e-30, 0.7];
    const lighting: Vec3 = [5, 3, 2];
    const rawPhysical: Vec3 = [
      lighting[0] * albedo[0],
      lighting[1] * albedo[1],
      lighting[2] * albedo[2],
    ];
    const physical = directionalVolume(
      rawPhysical,
      albedo,
      [0.2, 0.4, 0.8],
      1.2,
      0.8,
      0.4,
      0.5,
    );
    const demodulated = safeDemodulate(physical, albedo);
    expect(demodulated[0]).toBe(0);
    expect(demodulated[1]).toBeGreaterThan(0);
    for (let channel = 0; channel < 3; channel += 1) {
      expect(demodulated[channel]! * albedo[channel]!).toBeCloseTo(
        physical[channel]!,
        13,
      );
    }
  });

  it('processes four distinct GI directions before bilinear accumulation', () => {
    const radiances: readonly Vec3[] = [
      [0.9, 0.1, 0.2],
      [0.2, 0.8, 0.1],
      [0.1, 0.3, 1.1],
      [0.7, 0.6, 0.2],
    ];
    const directionCosines = [-0.85, -0.2, 0.45, 0.92] as const;
    const albedo: Vec3 = [0.8, 0.55, 0.3];
    const sigmaS: Vec3 = [0.4, 0.6, 0.9];
    let perDirection: Vec3 = [0, 0, 0];
    let aggregateInput: Vec3 = [0, 0, 0];
    for (let index = 0; index < 4; index += 1) {
      perDirection = add(perDirection, scale(directionalVolume(
        radiances[index]!, albedo, sigmaS, 1.5, 0.7,
        directionCosines[index]!, 0.8,
      ), 0.25));
      aggregateInput = add(aggregateInput, scale(radiances[index]!, 0.25));
    }
    const retiredIsotropic = directionalVolume(
      aggregateInput, albedo, sigmaS, 1.5, 0.7, 0, 0,
    );
    expect(Math.abs(perDirection[0] - retiredIsotropic[0])).toBeGreaterThan(1e-3);
    expect(Math.abs(perDirection[1] - retiredIsotropic[1])).toBeGreaterThan(1e-3);
  });

  it('keeps the RC-disabled blend exactly on the processed ReSTIR or DDGI lane', () => {
    const restirOrDdgi: Vec3 = [0.125, 1e-30, 7.5];
    const rcProcessed: Vec3 = [9, 8, 7];
    const wRc = 0;
    const wRestirGi = 1 - wRc;
    const blended = add(
      scale(restirOrDdgi, wRestirGi),
      scale(rcProcessed, wRc),
    );
    expect(blended).toEqual(restirOrDdgi);

    const indirect = functionBody(SHADING_TERMS_WGSL, 'lo_indirect');
    const fallbackIndex = indirect.indexOf('sampleDDGIAtPoint(pos, normal)');
    const rcSampleIndex = indirect.indexOf('sampleCascadeC0(pos, normal)');
    const rcProcessIndex = indirect.indexOf(
      'let Lo_rcDemodulated = restirShadeAggregateDiffuseDemodulated(',
    );
    const blendIndex = indirect.indexOf(
      'return wRestirGi * Lo_indirect + wRc * Lo_rcDemodulated;',
    );
    expect(fallbackIndex).toBeGreaterThanOrEqual(0);
    expect(rcSampleIndex).toBeGreaterThan(fallbackIndex);
    expect(rcProcessIndex).toBeGreaterThan(rcSampleIndex);
    expect(blendIndex).toBeGreaterThan(rcProcessIndex);
    expect(indirect).not.toMatch(
      /return\s+\([^;]*Lo_indirect[^;]*\)\s*\*\s*(?:albedo|diffuseWeight|layerTransmission)/,
    );
  });

  it('wires producer and final consumers to exact directional ownership', () => {
    const giCore = functionBody(
      RESTIR_GI_MATERIAL_WGSL,
      'restir_gi_receiver_contribution_core',
    );
    const direct = functionBody(SHADING_TERMS_WGSL, 'lo_direct');
    const indirect = functionBody(SHADING_TERMS_WGSL, 'lo_indirect');
    const specular = functionBody(
      SHADING_TERMS_WGSL,
      'lo_indirectSpecular',
    );
    expect(giCore).toContain(
      '(1.0 - clamp(payload.metal, 0.0, 1.0)) *',
    );
    expect(giCore).toContain(
      'return applyHomogeneousVolumeSingleScatterDirectional(',
    );
    expect(direct.match(/restirShadeDirectionalVolumeLog\(/g))
      .toHaveLength(3);
    expect(indirect).toContain('restirShadeDirectionalVolumeLog(');
    expect(indirect).toContain('restirShadeDemodulateLog(');
    expect(indirect).toContain('restirShadeAggregateDiffuseDemodulated(');
    expect(indirect).toContain(
      'return wRestirGi * Lo_indirect + wRc * Lo_rcDemodulated;',
    );
    expect(specular).toContain('reflectionLayerTransmission,');
    expect(specular).toContain('restirShadeDirectionalVolumeLog(');
  });

  it('assigns every shade, native, and OIT term to exactly one volume lane', () => {
    const shade = functionBody(SHADE_WGSL, 'shadeMain');
    const nativeArea = functionBody(
      NATIVE_GLASS_GI_WGSL,
      'nativeGlassGiAreaEmitterNee',
    );
    const nativeReceiver = functionBody(
      NATIVE_GLASS_GI_WGSL,
      'evaluateNativeGlassGiReceiver',
    );
    const oitBrdf = functionBody(TRANSPARENT_OIT_WGSL, 'oitLayerSurfaceBrdf');
    const oitResponse = functionBody(
      TRANSPARENT_OIT_WGSL,
      'oitLayerDirectionalResponse',
    );
    const oitLayer = functionBody(TRANSPARENT_OIT_WGSL, 'oitLayerRadiance');
    expect(shade).toContain('let aggregateSurfaceDirect = applyHomogeneousVolumeSingleScatter(');
    expect(shade).toContain(
      'let visiblePrimaryDirect = aggregateSurfaceDirect + Lo_indirectSpec +',
    );
    expect(shade).toContain('let indirectRadiance = Lo_indirect * ao;');
    expect(shade).toContain(
      'let directRadiance = visiblePrimaryDirect + Lo_transmittedGI;',
    );
    expect(nativeArea).toContain(
      'applyHomogeneousVolumeSingleScatterDirectional(',
    );
    expect(nativeReceiver).toContain(
      'let receiverLocalDirect = applyHomogeneousVolumeSingleScatter(',
    );
    expect(nativeReceiver).toContain(
      'let receiverDirect = receiverLocalDirect + receiverAnalytic + receiverSun +',
    );
    expect(oitBrdf).not.toContain(
      'return applyHomogeneousVolumeSingleScatterDirectional(',
    );
    expect(oitResponse).toContain(
      'return applyHomogeneousVolumeSingleScatterDirectional(',
    );
    expect(oitResponse).toContain('incidentRadiance * layeredClosure,');
    expect(oitLayer).toContain(
      'let localSurfaceRadiance = applyHomogeneousVolumeSingleScatter(',
    );
    expect(oitLayer).toContain(
      'return skyAmbient + sunDirect + analyticDirect + areaDirect +',
    );
  });
});
