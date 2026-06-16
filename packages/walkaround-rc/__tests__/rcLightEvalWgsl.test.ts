import { describe, expect, it } from 'vitest';
import { PROBE_RAY_CAST_WGSL } from '../src/index.js';

function functionBody(source: string, name: string): string {
  const marker = `fn ${name}(`;
  const start = source.indexOf(marker);
  expect(start, `${name} should be present`).toBeGreaterThanOrEqual(0);
  const brace = source.indexOf('{', start);
  expect(brace, `${name} should have a body`).toBeGreaterThanOrEqual(0);

  let depth = 0;
  for (let i = brace; i < source.length; i += 1) {
    const ch = source[i];
    if (ch === '{') depth += 1;
    if (ch === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(brace + 1, i);
    }
  }
  throw new Error(`Could not find end of ${name}`);
}

describe('RC light-eval WGSL contract', () => {
  it('uses glass-skip visibility for rect emitter and point/spot direct-light shadows', () => {
    const emitterNee = functionBody(PROBE_RAY_CAST_WGSL, 'rcEmitterNEE');
    expect(emitterNee).toContain(
      'e.castShadowDisabled < 0.5 && shadowTMax > 0.0 && rcTraceAnyCastShadow(hitPos + n * normalBias, wi, shadowTMax, triEps, true)',
    );
    expect(emitterNee).toContain('Emitter castShadow:false rides the shared EmitterTri fifth-vec4 .w lane.');
    expect(emitterNee).not.toContain('let sHit = rcTraceFirstHit');
    expect(emitterNee).not.toContain('sHit.didHit && sHit.dist < dist - normalBias');

    const pointSpot = functionBody(PROBE_RAY_CAST_WGSL, 'evalRCPointSpotLights');
    expect(PROBE_RAY_CAST_WGSL).toContain('sunCastShadowDisabled: u32');
    expect(PROBE_RAY_CAST_WGSL).toContain('RC_LIGHT_CAST_SHADOW_DISABLED');
    expect(pointSpot).toContain('let kind = light.kind & RC_LIGHT_KIND_MASK;');
    expect(pointSpot).toContain('let castShadowDisabled = (light.kind & RC_LIGHT_CAST_SHADOW_DISABLED) != 0u;');
    expect(pointSpot).toContain('if (!castShadowDisabled && shadowTMax > 0.0 && rcTraceAnyCastShadow');
    expect(pointSpot).toContain(
      'rcTraceAnyCastShadow(hitPos + n * normalBias, lightDir, shadowTMax, triEps, true)',
    );
    expect(pointSpot).not.toContain('let shadow = rcTraceFirstHit');
    expect(pointSpot).not.toContain('shadow.didHit && shadow.dist < dist - normalBias');
  });

  it('gates both RC direct-sun visibility calls with sunCastShadowDisabled', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('if (u.sunCastShadowDisabled == 0u)');
    expect(PROBE_RAY_CAST_WGSL.match(/traceSunVisibility/g)?.length).toBeGreaterThanOrEqual(3);
  });

  it('skips primitive castShadow:false geometry in RC GI shadow traversal', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('fn bvhCastShadowDisabledForTri(triIdx: u32) -> bool');
    expect(PROBE_RAY_CAST_WGSL).toContain('MATERIAL_FLAG_CAST_SHADOW_DISABLED');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcTraceAnyCastShadow(');
    expect(functionBody(PROBE_RAY_CAST_WGSL, 'traceSunVisibility')).toContain(
      'if ((sMat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u)',
    );
  });

  it('samples mapped material-backed emitter radiance for RC emitter NEE', () => {
    const emitterNee = functionBody(PROBE_RAY_CAST_WGSL, 'rcEmitterNEE');
    expect(PROBE_RAY_CAST_WGSL).toContain('@group(0) @binding(16) var                      rc_materialTextureAtlas: texture_2d_array<f32>;');
    expect(PROBE_RAY_CAST_WGSL).toContain('@group(0) @binding(17) var                      rc_materialMapMeta:      texture_2d<f32>;');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcSampleEmitterLeAtBary(e: EmitterTri, localBary: vec3f, scalarEmission: vec3f) -> vec3f');
    expect(PROBE_RAY_CAST_WGSL).toContain('let encodedSourceTri = i32(round(e._padA));');
    expect(PROBE_RAY_CAST_WGSL).toContain('return scalarEmission * texel.rgb;');
    expect(emitterNee).toContain('let localBary = vec3f(1.0 - su, su * (1.0 - s1), su * s1);');
    expect(emitterNee).toContain('let Le = rcSampleEmitterLeAtBary(e, localBary, e.Le);');
    expect(emitterNee).toContain('Lo = Lo + albedo * 0.31831 * Le * G * e.area;');
  });
});
