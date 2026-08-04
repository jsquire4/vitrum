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

type Rgb = readonly [number, number, number];

function probeAlbedoOracle(scalar: Rgb, vertex: Rgb, texel?: Rgb): Rgb {
  return [
    scalar[0] * vertex[0] * (texel?.[0] ?? 1),
    scalar[1] * vertex[1] * (texel?.[1] ?? 1),
    scalar[2] * vertex[2] * (texel?.[2] ?? 1),
  ];
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
      'if ((mat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u)',
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
    expect(traceT).toContain('visibility = visibility * rcMediumShadowExtinction(');
    expect(traceT).not.toContain('rcMediumRadianceSegmentTransfer(');
    expect(traceT).toContain('mediumBoundary[top] != event.encodedBoundaryId ||');
    expect(traceT).toContain('mediumRepresented[top] != event.representedPrimitiveInstanceId');
    expect(traceT).toContain('var coverage = clamp(1.0 - alphaT, 0.0, 1.0);');
    expect(traceT).toContain('coverageStatus == RC_CONTAINMENT_COVERAGE_SOLID');
    expect(traceT).toContain('rcTraceExactSurfaceEvent(');
    expect(traceT).toContain('let surfaceBudget = rcWorldSurfaceBudget();');
    expect(sunVisibility).toContain(
      'return rcTraceShadowTransmittance(origin, sunDir, 1e15, triEps);',
    );
  });

  it('samples mapped material controls for RC probe-hit direct material response', () => {
    const probeMaterial = functionBody(PROBE_RAY_CAST_WGSL, 'rcSampleProbeHitMaterial');
    const probeKernel = functionBody(PROBE_RAY_CAST_WGSL, 'probeRayCastKernel');
    const interfaceSources = functionBody(
      PROBE_RAY_CAST_WGSL,
      'rcShadeTransmissionInterfaceSources',
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
    expect(probeMaterial).toContain('out.albedo = out.albedo * baseColorTexel.value.rgb;');
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
    expect(response).toContain('(1.0 - clamp(mat.transmission, 0.0, 1.0)) * RC_INV_PI;');
    expect(response).toContain('let F0 = rcIridescenceModifiedF0(rcBaseMaterialF0(mat), mat.iridescence, vDotH);');
    expect(response).toContain('rcDistributionGGXAnisotropic');
    expect(response).toContain('rcEvalClearcoatLobe(mat.clearcoat, mat.clearcoatNormal, v, l)');
    expect(response).toContain('rcEvalSheenLobe(mat.sheen, mat.sheenRoughness, n, v, l)');
    expect(probeKernel).toContain('let firstSources = rcShadeTransmissionInterfaceSources(');
    expect(probeKernel).toContain('rcTraceDielectricSuffixChannel(');
    expect(interfaceSources).toContain('let probeMat = rcSampleProbeHitMaterial(');
    expect(interfaceSources).toContain('mat.transmission, mat.ior, wo,');
    expect(interfaceSources).toContain('let toSun = rcSoftSunDirection(');
    expect(interfaceSources).toContain(
      'rcEvaluateProbeDirectResponse(directMat, n, wo, toSun) * sunVis;',
    );
    // Emitter sampling deliberately derives from the producer/receiver-shared
    // stratified ray seed. rcEmitterNEE hashes that seed again per emitter, so
    // the area sample is deterministic without reusing either ray-UV variate.
    expect(probeKernel).toContain('let raySeed = rcStratifiedRaySeed(probeIdx, rayIdx, u.frameSeed);');
    expect(interfaceSources).toContain('let emitterNEE = rcEmitterNEE(');
    expect(interfaceSources).toContain('receiverPos, n, wo, directMat, u.emitterCount, raySeed, triEps, normalBias,');
    expect(interfaceSources).toContain('let pointSpotLights = evalRCPointSpotLights(');
  });

  it('modulates textured and untextured probe albedo by interpolated vertex RGB exactly once', () => {
    const probeMaterial = functionBody(PROBE_RAY_CAST_WGSL, 'rcSampleProbeHitMaterial');
    const vertexSample = 'let vertexColor = rcSampleVertexColorForHit(hit);';
    const vertexModulation = 'out.albedo = scalarBaseColor * vertexColor.rgb;';
    const uvs = 'let uvs = rcHitMaterialUvs(hit);';
    const textureModulation = 'out.albedo = out.albedo * baseColorTexel.value.rgb;';

    expect(probeMaterial.match(/rcSampleVertexColorForHit\(hit\)/g)).toHaveLength(1);
    expect(probeMaterial).toContain(vertexSample);
    expect(probeMaterial).toContain(vertexModulation);
    expect(probeMaterial.indexOf(vertexModulation)).toBeLessThan(probeMaterial.indexOf(uvs));
    expect(probeMaterial).toContain(textureModulation);
    expect(probeMaterial).not.toContain('scalarBaseColor * baseColorTexel.rgb');

    const scalar: Rgb = [0.8, 0.5, 0.25];
    const vertex: Rgb = [0.25, 0.4, 0.8];
    expect(probeAlbedoOracle(scalar, vertex)).toEqual([0.2, 0.2, 0.2]);
    expect(probeAlbedoOracle(scalar, vertex, [0.5, 0.25, 1])).toEqual([0.1, 0.05, 0.2]);
  });

  it('applies mapped normal and bump maps to RC probe-hit lighting normals', () => {
    const interfaceSources = functionBody(
      PROBE_RAY_CAST_WGSL,
      'rcShadeTransmissionInterfaceSources',
    );

    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_NORMAL_TEXEL_OFFSET: u32 = 15u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET: u32 = 49u;');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcSmoothNormalForHit(hit: IntersectionResult, fallbackNormal: vec3f) -> vec3f');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcMaterialTangentFrameForHit(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcBvhTangentTexel(vertexIndex: u32) -> vec4f');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcPreferAuthoredTangentFrameForHit(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcApplyNormalMapForHit(hit: IntersectionResult, baseNormal: vec3f) -> vec3f');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcApplyBumpMapForHit(hit: IntersectionResult, shadingNormal: vec3f) -> vec3f');
    expect(PROBE_RAY_CAST_WGSL).toContain('return rcPreferAuthoredTangentFrameForHit(hit, frameNormal, tangent, bitangent);');
    expect(interfaceSources).toContain('let smoothNormal = rcSmoothNormalForHit(hit, hit.normal);');
    expect(interfaceSources).toContain('let normalMapped = rcApplyNormalMapForHit(hit, smoothNormal);');
    expect(interfaceSources).toContain('let n = rcApplyBumpMapForHit(hit, normalMapped);');
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
    expect(PROBE_RAY_CAST_WGSL).toContain('let encodedSourceTri = i32(e._padA);');
    expect(PROBE_RAY_CAST_WGSL).toContain('!rcMaterialAtlasFiniteF32(e._padA)');
    expect(PROBE_RAY_CAST_WGSL).toContain('RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET,');
    expect(PROBE_RAY_CAST_WGSL).toContain('uv0,');
    expect(PROBE_RAY_CAST_WGSL).toContain('uv1,');
    expect(PROBE_RAY_CAST_WGSL).toContain('let uv1a = rcPackedUvFromVec4(rc_geom_normal[tri.x]);');
    expect(PROBE_RAY_CAST_WGSL).not.toContain('UV1-authored emissive maps intentionally fall back to UV0');
    expect(PROBE_RAY_CAST_WGSL).toContain('rcMaterialAtlasFiniteNonNegativeRadianceOrBlack(');
    expect(PROBE_RAY_CAST_WGSL).toContain('scalarEmission * texel.value.rgb,');
    expect(emitterNee).toContain('let localBary = vec3f(1.0 - su, su * (1.0 - s1), su * s1);');
    expect(emitterNee).toContain('let Le = rcSampleEmitterLeAtBary(e, localBary, e.Le);');
    expect(emitterNee).toContain('return response * Le * G * e.area * shadowT / draw.pmf;');
  });

  it('samples emissive maps for RC direct probe-hit surface emission', () => {
    const surfaceEmission = functionBody(PROBE_RAY_CAST_WGSL, 'rcSampleSurfaceEmissiveMap');
    const interfaceSources = functionBody(
      PROBE_RAY_CAST_WGSL,
      'rcShadeTransmissionInterfaceSources',
    );

    expect(PROBE_RAY_CAST_WGSL).toContain('const RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET: u32 = 11u;');
    expect(surfaceEmission).toContain('let uvs = rcHitMaterialUvs(hit);');
    expect(surfaceEmission).toContain('RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET');
    expect(surfaceEmission).toContain('rcMaterialAtlasFiniteNonNegativeRadianceOrBlack(');
    expect(surfaceEmission).toContain('scalarEmission * texel.value.rgb,');
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'max(texel.value.rgb, vec3f(0.0)) * max(intensity, 0.0),',
    );
    expect(interfaceSources).toContain('out.emission = select(');
    expect(interfaceSources).toContain('rcSampleSurfaceEmissiveMap(hit, mat.emissive),');
    expect(interfaceSources).toContain('hit.side >= 0.0 ||');
    expect(interfaceSources).toContain('(mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u,');
  });
});
