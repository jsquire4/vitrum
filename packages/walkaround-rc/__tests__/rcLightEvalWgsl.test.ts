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
  it('uses alpha transmittance for rect emitter and point/spot direct-light shadows', () => {
    const emitterNee = functionBody(PROBE_RAY_CAST_WGSL, 'rcEmitterNEE');
    expect(emitterNee).toContain('shadowT = rcTraceShadowTransmittance(');
    expect(emitterNee).toContain('max(max(shadowT.x, shadowT.y), shadowT.z) <= 0.0');
    expect(emitterNee).toContain('Emitter castShadow:false rides the shared EmitterTri fifth-vec4 .w lane.');
    expect(emitterNee).toContain('let response = rcEvaluateProbeDirectResponse(material, n, wo, wi);');
    expect(emitterNee).toContain('return response * Le * G * e.area * shadowT / draw.pmf;');
    expect(emitterNee).not.toContain('rcTraceAnyCastShadow(hitPos + n * normalBias, wi, shadowTMax, triEps, true)');
    expect(emitterNee).not.toContain('let sHit = rcTraceFirstHit');
    expect(emitterNee).not.toContain('sHit.didHit && sHit.dist < dist - normalBias');

    const pointSpot = functionBody(PROBE_RAY_CAST_WGSL, 'evalRCPointSpotLights');
    expect(pointSpot).toContain('let count = min(rc_lights[0u], rc_u.lightCount);');
    expect(PROBE_RAY_CAST_WGSL).toContain('sunCastShadowDisabled: u32');
    expect(PROBE_RAY_CAST_WGSL).toContain('RC_LIGHT_CAST_SHADOW_DISABLED');
    expect(pointSpot).toContain('let kind = light.kind & RC_LIGHT_KIND_MASK;');
    expect(pointSpot).toContain('let castShadowDisabled = (light.kind & RC_LIGHT_CAST_SHADOW_DISABLED) != 0u;');
    expect(pointSpot).toContain('shadowT = rcTraceShadowTransmittance(');
    expect(pointSpot).toContain('let response = rcEvaluateProbeDirectResponse(material, n, wo, lightDir);');
    expect(pointSpot).toContain('return response * light.color * atten * coneFalloff * shadowT / draw.pmf;');
    expect(pointSpot).not.toContain('rcTraceAnyCastShadow(hitPos + n * normalBias, lightDir, shadowTMax, triEps, true)');
    expect(pointSpot).not.toContain('let shadow = rcTraceFirstHit');
    expect(pointSpot).not.toContain('shadow.didHit && shadow.dist < dist - normalBias');
  });

  it('gates both RC direct-sun visibility calls with sunCastShadowDisabled', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('if (u.sunCastShadowDisabled == 0u)');
    expect(PROBE_RAY_CAST_WGSL.match(/traceSunVisibility/g)?.length).toBeGreaterThanOrEqual(3);
  });

  it('evaluates spot lights with the forward beam axis and guards hard-edge cones', () => {
    const pointSpot = functionBody(PROBE_RAY_CAST_WGSL, 'evalRCPointSpotLights');
    const distanceAttenuation = functionBody(
      PROBE_RAY_CAST_WGSL,
      'rcPointSpotDistanceAttenuation',
    );

    expect(pointSpot).toContain('let cosToP = dot(-light.direction * inverseSqrt(axisLen2), lightDir);');
    expect(pointSpot).toContain('light.innerCone == light.outerCone');
    expect(pointSpot).toContain('rcPointSpotDistanceAttenuation(');
    expect(pointSpot).toContain('let dist2Floor = max(normalBias * normalBias, triEps * triEps);');
    expect(distanceAttenuation).toContain('if (cutoffDistance > 0.0)');
    expect(distanceAttenuation).toContain(
      'let x = clamp(1.0 - pow(dist / cutoffDistance, 4.0), 0.0, 1.0);',
    );
    expect(pointSpot).toContain('let atten = light.intensity * distanceAttenuation;');
    expect(pointSpot).not.toContain('dot(lightDir, light.direction * inverseSqrt(axisLen2))');
    expect(pointSpot).not.toContain('coneFalloff = smoothstep(light.outerCone, light.innerCone, cosToP);\n      if');
  });

  it('skips primitive castShadow:false geometry in RC GI shadow traversal', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('MATERIAL_FLAG_CAST_SHADOW_DISABLED');
    expect(functionBody(PROBE_RAY_CAST_WGSL, 'rcTraceShadowTransmittance')).toContain(
      'if ((mat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) == 0u)',
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain('fn rcTraceAnyCastShadow(');
  });

  it('samples material-atlas alpha coverage for RC shadow transmittance', () => {
    const sunVisibility = functionBody(PROBE_RAY_CAST_WGSL, 'traceSunVisibility');
    const alphaCoverage = functionBody(PROBE_RAY_CAST_WGSL, 'rcMaterialAlphaCoverageForHit');
    const alphaT = functionBody(PROBE_RAY_CAST_WGSL, 'rcAlphaShadowTransmittanceForHit');
    const traceT = functionBody(PROBE_RAY_CAST_WGSL, 'rcTraceShadowTransmittance');

    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_SLOT_BASE_COLOR: u32 = 0u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_SLOT_ALPHA: u32 = 4u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET: u32 = 10u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcHitMaterialUvs(hit: IntersectionResult) -> RCHitMaterialUvs');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcBvhVertexColorTexel(vertexIndex: u32) -> vec4f');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcSampleVertexColorForHit(hit: IntersectionResult) -> vec4f');
    expect(alphaCoverage).toContain('let baseColorTexel = rcSampleMaterialAtlasRaw(hit.indices.w, RC_MATERIAL_MAP_SLOT_BASE_COLOR, uvs.uv0, uvs.uv1);');
    expect(alphaCoverage).toContain('let alphaTexel = rcSampleMaterialAtlasRaw(hit.indices.w, RC_MATERIAL_MAP_SLOT_ALPHA, uvs.uv0, uvs.uv1);');
    expect(alphaCoverage).toContain('let vertexColorAlpha = rcSampleVertexColorForHit(hit).a;');
    expect(alphaCoverage).toContain('out.coverage = clamp(opacity * vertexColorAlpha * baseColorAlpha * alphaMapCoverage, 0.0, 1.0);');
    expect(alphaT).toContain('return clamp(1.0 - alpha.coverage, 0.0, 1.0);');
    expect(traceT).toContain('let alphaT = rcAlphaShadowTransmittanceForHit(hit);');
    expect(traceT).toContain('let rgbBeer = beerLambertTransmittanceRgb(');
    expect(traceT).toContain('mediumMaterial[mediumDepth - 1u] == matId');
    expect(traceT).toContain('mediumInstance[mediumDepth - 1u] == hit.instanceIndex');
    expect(traceT).toContain('let coverage = clamp(1.0 - alphaT, 0.0, 1.0);');
    expect(traceT).toContain('let surfaceBudget = rcWorldSurfaceBudget();');
    expect(sunVisibility).toContain(
      'return rcTraceShadowTransmittance(origin, sunDir, 1e15, triEps);',
    );
  });

  it('samples mapped material controls for RC probe-hit direct material response', () => {
    const probeMaterial = functionBody(PROBE_RAY_CAST_WGSL, 'rcSampleProbeHitMaterial');
    const probeKernel = functionBody(PROBE_RAY_CAST_WGSL, 'probeRayCastKernel');
    const opaqueReceiver = functionBody(
      PROBE_RAY_CAST_WGSL,
      'rcShadeOpaqueTransmissionReceiver',
    );
    const response = functionBody(PROBE_RAY_CAST_WGSL, 'rcEvaluateProbeDirectResponse');

    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_SLOT_ROUGHNESS: u32 = 1u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_SLOT_METALLIC: u32 = 2u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET: u32 = 24u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET: u32 = 26u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET: u32 = 22u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET: u32 = 28u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET: u32 = 30u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_SHEEN_COLOR_TEXEL_OFFSET: u32 = 23u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET: u32 = 32u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET: u32 = 34u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET: u32 = 39u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_ANISOTROPY_SCALAR_TEXEL_OFFSET: u32 = 41u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET: u32 = 42u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET: u32 = 44u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_IRIDESCENCE_SCALAR_TEXEL_OFFSET: u32 = 46u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcSampleMaterialScalarMap(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcSampleSpecularControls(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcSampleClearcoatControls(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcSampleSheenControls(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcSampleAnisotropyControls(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcSampleIridescenceControls(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcApplyClearcoatNormalMapForHit(');
    expect(PROBE_RAY_CAST_WGSL).toContain('struct RCProbeHitMaterial');
    expect(PROBE_RAY_CAST_WGSL).toContain('roughness: f32,');
    expect(PROBE_RAY_CAST_WGSL).toContain('metalness: f32,');
    expect(PROBE_RAY_CAST_WGSL).toContain('specular: vec4f,');
    expect(PROBE_RAY_CAST_WGSL).toContain('clearcoat: vec2f,');
    expect(PROBE_RAY_CAST_WGSL).toContain('clearcoatNormal: vec3f,');
    expect(PROBE_RAY_CAST_WGSL).toContain('sheen: vec4f,');
    expect(PROBE_RAY_CAST_WGSL).toContain('anisotropy: vec2f,');
    expect(PROBE_RAY_CAST_WGSL).toContain('iridescence: vec4f,');
    expect(probeMaterial).toContain('out.albedo = scalarBaseColor * baseColorTexel.rgb;');
    expect(probeMaterial).toContain('RC_MATERIAL_MAP_SLOT_BASE_COLOR');
    expect(probeMaterial).toContain('RC_MATERIAL_MAP_SLOT_ROUGHNESS');
    expect(probeMaterial).toContain('RC_MATERIAL_MAP_SLOT_METALLIC');
    expect(probeMaterial).toContain('out.specular = rcSampleSpecularControls(hit.indices.w, uvs.uv0, uvs.uv1);');
    expect(probeMaterial).toContain('out.clearcoat = rcSampleClearcoatControls(hit.indices.w, uvs.uv0, uvs.uv1);');
    expect(probeMaterial).toContain('out.clearcoatNormal = rcApplyClearcoatNormalMapForHit(hit, frameNormal, shadingNormal);');
    expect(probeMaterial).toContain('out.sheen = rcSampleSheenControls(hit.indices.w, uvs.uv0, uvs.uv1);');
    expect(probeMaterial).toContain('out.sheenRoughness = rcSampleSheenRoughness(hit.indices.w, uvs.uv0, uvs.uv1);');
    expect(probeMaterial).toContain('out.anisotropy = rcSampleAnisotropyControls(hit.indices.w, uvs.uv0, uvs.uv1);');
    expect(probeMaterial).toContain('out.iridescence = rcSampleIridescenceControls(hit.indices.w, uvs.uv0, uvs.uv1);');
    expect(response).toContain('let diffuse = mat.albedo * (1.0 - clamp(mat.metalness, 0.0, 1.0)) * RC_INV_PI;');
    expect(response).toContain('let F0 = rcIridescenceModifiedF0(rcBaseMaterialF0(mat), mat.iridescence, vDotH);');
    expect(response).toContain('rcDistributionGGXAnisotropic');
    expect(response).toContain('rcEvalClearcoatLobe(mat.clearcoat, mat.clearcoatNormal, v, l)');
    expect(response).toContain('rcEvalSheenLobe(mat.sheen, mat.sheenRoughness, n, v, l)');
    expect(probeKernel).toContain('let probeMat = rcSampleProbeHitMaterial(');
    expect(probeKernel).toContain('mat.transmission, mat.ior, wo,');
    expect(probeKernel).toContain('let matColor    = probeMat.albedo;');
    expect(probeKernel).toContain('let toSun = rcSoftSunDirection(u.sunDirection, u.sunAngularRadius, hitPos, u.roomSize, u.cascadeIndex);');
    expect(probeKernel).toContain('let directSun = u.sunColor * rcEvaluateProbeDirectResponse(probeMat, n, wo, toSun) * sunVis;');
    expect(opaqueReceiver).toContain('let toSun = rcSoftSunDirection(');
    expect(opaqueReceiver).toContain(
      'rcEvaluateProbeDirectResponse(probeMat, n, wo, toSun) * sunVis;',
    );
    // Emitter sampling deliberately derives from the producer/receiver-shared
    // stratified ray seed. rcEmitterNEE hashes that seed again per emitter, so
    // the area sample is deterministic without reusing either ray-UV variate.
    expect(probeKernel).toContain('let raySeed = rcStratifiedRaySeed(probeIdx, rayIdx, u.frameSeed);');
    expect(probeKernel).toContain('let emitterNEE = rcEmitterNEE(hitPos, n, wo, probeMat, u.emitterCount, raySeed, triEps, normalBias);');
    expect(probeKernel).toContain('let pointSpotLights = evalRCPointSpotLights(hitPos, n, wo, probeMat, normalBias, triEps, raySeed);');
  });

  it('applies mapped normal and bump maps to RC probe-hit lighting normals', () => {
    const probeKernel = functionBody(PROBE_RAY_CAST_WGSL, 'probeRayCastKernel');

    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_NORMAL_TEXEL_OFFSET: u32 = 15u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET: u32 = 49u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcSmoothNormalForHit(hit: IntersectionResult, fallbackNormal: vec3f) -> vec3f');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcMaterialTangentFrameForHit(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcBvhTangentTexel(vertexIndex: u32) -> vec4f');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcPreferAuthoredTangentFrameForHit(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcApplyNormalMapForHit(hit: IntersectionResult, baseNormal: vec3f) -> vec3f');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcApplyBumpMapForHit(hit: IntersectionResult, shadingNormal: vec3f) -> vec3f');
    expect(PROBE_RAY_CAST_WGSL).toContain('return rcPreferAuthoredTangentFrameForHit(hit, frameNormal, tangent, bitangent);');
    expect(probeKernel).toContain('let geoNormal = hit.normal;');
    expect(probeKernel).toContain('let smoothNormal = rcSmoothNormalForHit(hit, geoNormal);');
    expect(probeKernel).toContain('let normalMapped = rcApplyNormalMapForHit(hit, smoothNormal);');
    expect(probeKernel).toContain('let n = rcApplyBumpMapForHit(hit, normalMapped);');
    expect(PROBE_RAY_CAST_WGSL).toContain('let shadingNormal = rcApplyBumpMapForHit(');
    expect(PROBE_RAY_CAST_WGSL).toContain('smoothNormal, shadingNormal, mat.transmission, mat.ior, -ray.direction,');
  });

  it('samples mapped material-backed emitter radiance for RC emitter NEE', () => {
    const emitterNee = functionBody(PROBE_RAY_CAST_WGSL, 'rcEmitterNEE');
    expect(PROBE_RAY_CAST_WGSL).toContain('@group(0) @binding(16) var                      rc_materialTextureAtlas: texture_2d_array<u32>;');
    expect(PROBE_RAY_CAST_WGSL).toContain('@group(0) @binding(17) var                      rc_materialMapMeta:      texture_2d<f32>;');
    expect(PROBE_RAY_CAST_WGSL).toContain('@group(0) @binding(18) var<storage, read>       rc_geom_normal:           array<vec4f>;');
    expect(PROBE_RAY_CAST_WGSL).toContain('@group(0) @binding(19) var                      rc_geom_tangent:          texture_2d<f32>;');
    expect(PROBE_RAY_CAST_WGSL).toContain('@group(0) @binding(20) var                      rc_geom_vertex_color:     texture_2d<f32>;');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcSampleEmitterLeAtBary(e: EmitterTri, localBary: vec3f, scalarEmission: vec3f) -> vec3f');
    expect(PROBE_RAY_CAST_WGSL).toContain('let encodedSourceTri = i32(round(e._padA));');
    expect(PROBE_RAY_CAST_WGSL).toContain('RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET,');
    expect(PROBE_RAY_CAST_WGSL).toContain('uv0,');
    expect(PROBE_RAY_CAST_WGSL).toContain('uv1,');
    expect(PROBE_RAY_CAST_WGSL).toContain('let uv1a = rcPackedUvFromVec4(rc_geom_normal[tri.x]);');
    expect(PROBE_RAY_CAST_WGSL).not.toContain('UV1-authored emissive maps intentionally fall back to UV0');
    expect(PROBE_RAY_CAST_WGSL).toContain('return scalarEmission * texel.rgb;');
    expect(emitterNee).toContain('let localBary = vec3f(1.0 - su, su * (1.0 - s1), su * s1);');
    expect(emitterNee).toContain('let Le = rcSampleEmitterLeAtBary(e, localBary, e.Le);');
    expect(emitterNee).toContain('return response * Le * G * e.area * shadowT / draw.pmf;');
  });

  it('samples emissive maps for RC direct probe-hit surface emission', () => {
    const surfaceEmission = functionBody(PROBE_RAY_CAST_WGSL, 'rcSampleSurfaceEmissiveMap');
    const probeKernel = functionBody(PROBE_RAY_CAST_WGSL, 'probeRayCastKernel');

    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET: u32 = 11u;');
    expect(surfaceEmission).toContain('let uvs = rcHitMaterialUvs(hit);');
    expect(surfaceEmission).toContain('RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET');
    expect(surfaceEmission).toContain('return scalarEmission * texel.rgb;');
    expect(probeKernel).toContain('let matEmissive = mat.emissive;');
    expect(probeKernel).toContain('let emissive = select(');
    expect(probeKernel).toContain('rcSampleSurfaceEmissiveMap(hit, matEmissive),');
    expect(probeKernel).toContain('hit.side >= 0.0 ||');
    expect(probeKernel).toContain('(mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u,');
    expect(probeKernel).not.toContain('let emissive = matEmissive;');
  });
});
