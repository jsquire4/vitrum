import { describe, expect, it } from 'vitest';

import { makeProbeUpdateRaysWGSL } from '../wgsl/probeUpdateRays.wgsl.js';

function functionBody(source: string, name: string): string {
  const start = source.indexOf(`fn ${name}(`);
  expect(start, `${name} should exist`).toBeGreaterThanOrEqual(0);
  const brace = source.indexOf('{', start);
  let depth = 0;
  for (let index = brace; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1;
    if (source[index] === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(brace + 1, index);
    }
  }
  throw new Error(`Could not find the end of ${name}`);
}

describe('DDGI probe radiometry regressions', () => {
  const source = makeProbeUpdateRaysWGSL(4);

  it('keeps area-emitter NEE in incident-irradiance form until one Lambertian conversion', () => {
    const emitterNee = functionBody(source, 'ddgiEmitterNEE');
    expect(source).toContain('fn ddgiEmitterNEE(hitPos: vec3f, n: vec3f, seed0: u32)');
    expect(emitterNee).not.toContain('0.31831');
    expect(emitterNee).toContain('irradiance = irradiance + Le * G * area * shadowT');
    expect(source).toContain('let direct = evalDirectLighting(');
    expect(source).toContain(') + ddgiEmitterNEE(');
    expect(source).toContain(
      'let directRadiance = direct * probeMat.albedo * (1.0 / PI);',
    );

    const incidentIrradiance = 3;
    const outgoing = (albedo: number) =>
      incidentIrradiance * albedo / Math.PI;
    expect(outgoing(0.8) / outgoing(0.4)).toBeCloseTo(2, 12);
  });

  it('addresses the reciprocal glass face layer by triangle material id', () => {
    expect(source).toContain(
      'let slabLayer = ddgiSampleFaceLayerControls(',
    );
    expect(source).toContain(
      'currentHit.indices.w, slabFrontFacing,',
    );
    expect(source).not.toContain('ddgiSampleFaceLayerControls(materialId,');
  });
});
