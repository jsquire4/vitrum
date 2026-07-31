import type { WgslModule } from '../pipeline/wgslComposer.js';

export const RESTIR_GI_MATERIAL_WGSL = /* wgsl */ `

struct RestirGIHitMaterial {
  normal: vec3f,
  Lo: vec3f,
  albedo: vec3f,
  rough: f32,
};

@group(1) @binding(12) var restir_gi_bvh_emissive: texture_2d<f32>;

fn restir_gi_receiver_key_mix(state: u32, value: u32) -> u32 {
  var mixed = (state ^ value) * 0x85ebca6bu;
  mixed = (mixed ^ (mixed >> 13u)) * 0xc2b2ae35u;
  return mixed ^ (mixed >> 16u);
}

// Exact receiver-domain identity for scaled GI reservoirs. Besides primitive
// identity, hash every mapped parameter that participates in the producer
// target or shade-side specular evaluation. A texture/material discontinuity
// therefore fails closed to receiver-local DDGI instead of borrowing a
// representative reservoir whose proposal may not cover the current lobe.
fn restir_gi_receiver_domain_key(
  matColorPacked: u32,
  materialWord: u32,
  triangleId: u32,
  instanceId: u32,
  payload: RestirDIMaterialPayload,
) -> u32 {
  var h = restir_gi_receiver_key_mix(0x9e3779b9u, matColorPacked);
  h = restir_gi_receiver_key_mix(h, materialWord);
  h = restir_gi_receiver_key_mix(h, triangleId);
  h = restir_gi_receiver_key_mix(h, instanceId);

  let albedo = bitcast<vec3u>(payload.albedo);
  h = restir_gi_receiver_key_mix(h, albedo.x);
  h = restir_gi_receiver_key_mix(h, albedo.y);
  h = restir_gi_receiver_key_mix(h, albedo.z);
  h = restir_gi_receiver_key_mix(h, bitcast<u32>(payload.rough));
  h = restir_gi_receiver_key_mix(h, bitcast<u32>(payload.metal));
  h = restir_gi_receiver_key_mix(h, bitcast<u32>(payload.envMapIntensity));

  let clearcoatNormal = bitcast<vec3u>(payload.clearcoatNormal);
  h = restir_gi_receiver_key_mix(h, clearcoatNormal.x);
  h = restir_gi_receiver_key_mix(h, clearcoatNormal.y);
  h = restir_gi_receiver_key_mix(h, clearcoatNormal.z);
  let specular = bitcast<vec4u>(payload.specular);
  h = restir_gi_receiver_key_mix(h, specular.x);
  h = restir_gi_receiver_key_mix(h, specular.y);
  h = restir_gi_receiver_key_mix(h, specular.z);
  h = restir_gi_receiver_key_mix(h, specular.w);
  let anisotropy = bitcast<vec2u>(payload.anisotropy);
  h = restir_gi_receiver_key_mix(h, anisotropy.x);
  h = restir_gi_receiver_key_mix(h, anisotropy.y);
  let anisotropyTangent = bitcast<vec3u>(payload.anisotropyTangent);
  h = restir_gi_receiver_key_mix(h, anisotropyTangent.x);
  h = restir_gi_receiver_key_mix(h, anisotropyTangent.y);
  h = restir_gi_receiver_key_mix(h, anisotropyTangent.z);
  let anisotropyBitangent = bitcast<vec3u>(payload.anisotropyBitangent);
  h = restir_gi_receiver_key_mix(h, anisotropyBitangent.x);
  h = restir_gi_receiver_key_mix(h, anisotropyBitangent.y);
  h = restir_gi_receiver_key_mix(h, anisotropyBitangent.z);

  let iridescence = bitcast<vec4u>(payload.iridescence);
  h = restir_gi_receiver_key_mix(h, iridescence.x);
  h = restir_gi_receiver_key_mix(h, iridescence.y);
  h = restir_gi_receiver_key_mix(h, iridescence.z);
  h = restir_gi_receiver_key_mix(h, iridescence.w);
  let clearcoat = bitcast<vec2u>(payload.clearcoat);
  h = restir_gi_receiver_key_mix(h, clearcoat.x);
  h = restir_gi_receiver_key_mix(h, clearcoat.y);
  let sheen = bitcast<vec4u>(payload.sheen);
  h = restir_gi_receiver_key_mix(h, sheen.x);
  h = restir_gi_receiver_key_mix(h, sheen.y);
  h = restir_gi_receiver_key_mix(h, sheen.z);
  h = restir_gi_receiver_key_mix(h, sheen.w);
  h = restir_gi_receiver_key_mix(h, bitcast<u32>(payload.sheenRoughness));

  let layerTransmission = bitcast<vec3u>(payload.layerTransmission);
  h = restir_gi_receiver_key_mix(h, layerTransmission.x);
  h = restir_gi_receiver_key_mix(h, layerTransmission.y);
  h = restir_gi_receiver_key_mix(h, layerTransmission.z);
  let volumeScattering = bitcast<vec4u>(payload.volumeScattering);
  h = restir_gi_receiver_key_mix(h, volumeScattering.x);
  h = restir_gi_receiver_key_mix(h, volumeScattering.y);
  h = restir_gi_receiver_key_mix(h, volumeScattering.z);
  h = restir_gi_receiver_key_mix(h, volumeScattering.w);
  h = restir_gi_receiver_key_mix(h, bitcast<u32>(payload.bulkThickness));

  // Reserve zero for migrated snapshots whose old padding word was zero.
  return h | 1u;
}

fn restir_gi_smooth_normal_for_hit(hit: IntersectionResult, geoNormal: vec3f) -> vec3f {
  let isTlas = ubo.bvhMode == 1u;
  let base = hit.instanceIndex * 4u;
  let ok = isTlas && base + 2u < tlasWorldToLocalColumnCount();
  let i = select(0u, base, ok);
  return smoothShadingNormal(
    hit,
    geoNormal,
    sceneLoadBvhNormal(hit.indices.x).xyz,
    sceneLoadBvhNormal(hit.indices.y).xyz,
    sceneLoadBvhNormal(hit.indices.z).xyz,
    ok,
    tlasLoadWorldToLocalColumn(i),
    tlasLoadWorldToLocalColumn(i + 1u),
    tlasLoadWorldToLocalColumn(i + 2u),
  );
}

fn restir_gi_shading_normal_for_hit(hit: IntersectionResult, geoNormal: vec3f) -> vec3f {
  let smoothNormal = restir_gi_smooth_normal_for_hit(hit, geoNormal);
  let normalMapped = applyNormalMapForHit(hit, smoothNormal);
  return applyBumpMapForHit(hit, normalMapped);
}

fn restir_gi_receiver_has_specular_lobes(payload: RestirDIMaterialPayload) -> bool {
  let specularDelta = max(
    max(abs(payload.specular.r - 1.0), abs(payload.specular.g - 1.0)),
    max(abs(payload.specular.b - 1.0), abs(payload.specular.a - 1.0)),
  );
  return payload.metal > 0.0
      || payload.rough < 0.6
      || specularDelta > 0.0
      || abs(payload.anisotropy.x) > 0.0
      || payload.iridescence.x > 0.0
      || payload.clearcoat.x > 0.0
      || payload.sheen.a > 0.0;
}

fn restir_gi_receiver_contribution_from_payload(
  receiverNormal: vec3f,
  receiverClearcoatNormal: vec3f,
  receiverWo: vec3f,
  payload: RestirDIMaterialPayload,
  wi: vec3f,
  Lo: vec3f,
) -> vec3f {
  let cosTheta = max(0.0, dot(receiverNormal, wi));
  if (cosTheta <= 0.0) {
    return vec3f(0.0);
  }

  // Keep the diffuse-indirect channel albedo-demodulated for SVGF/atrous
  // compatibility. Rich receivers add their true glossy/clearcoat/sheen target
  // so reservoir selection/reuse is no longer diffuse-only.
  var contribution = Lo * cosTheta * INV_PI;
  if (restir_gi_receiver_has_specular_lobes(payload)) {
    let specBrdf = evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(
      payload.albedo,
      payload.rough,
      payload.metal,
      payload.specular.rgb,
      payload.specular.a,
      payload.anisotropy.x,
      payload.anisotropy.y,
      payload.iridescence,
      payload.clearcoat.x,
      payload.clearcoat.y,
      payload.sheen.a,
      payload.sheenRoughness,
      payload.sheen.rgb,
      payload.anisotropyTangent,
      payload.anisotropyBitangent,
      receiverNormal,
      receiverClearcoatNormal,
      receiverWo,
      wi,
    );
    contribution = contribution + Lo * specBrdf;
  }
  return applyHomogeneousVolumeSingleScatterDirectional(
    contribution * payload.layerTransmission,
    payload.albedo,
    payload.volumeScattering,
    payload.bulkThickness,
    receiverNormal,
    receiverWo,
    wi,
  );
}

fn restir_gi_receiver_phat_from_payload(
  receiverPos: vec3f,
  receiverNormal: vec3f,
  receiverClearcoatNormal: vec3f,
  receiverWo: vec3f,
  payload: RestirDIMaterialPayload,
  xs: vec3f,
  Lo: vec3f,
) -> f32 {
  let d = xs - receiverPos;
  if (!(safe_length(d) > 0.0)) { return 0.0; }
  let wi = safe_normalize(d);
  return luminance(restir_gi_receiver_contribution_from_payload(
    receiverNormal,
    receiverClearcoatNormal,
    receiverWo,
    payload,
    wi,
    Lo,
  ));
}

fn restir_gi_receiver_phat_from_geometry(
  xv: vec3f,
  nv: vec3f,
  xs: vec3f,
  Lo: vec3f,
) -> f32 {
  let d = xs - xv;
  if (!(safe_length(d) > 0.0)) { return 0.0; }
  let wi = safe_normalize(d);
  let cosTheta = max(0.0, dot(nv, wi));
  return luminance(Lo) * cosTheta * INV_PI;
}

fn restir_gi_receiver_phat_from_surface(
  surf: PrimarySurface,
  xs: vec3f,
  Lo: vec3f,
) -> f32 {
  if (!surf.hit) { return 0.0; }
  var payload: RestirDIMaterialPayload;
  payload.albedo = surf.albedo;
  payload.rough = surf.rough;
  payload.metal = surf.metal;
  payload.envMapIntensity = surf.envMapIntensity;
  payload.clearcoatNormal = surf.clearcoatNormal;
  payload.specular = surf.specular;
  payload.anisotropy = surf.anisotropy;
  payload.anisotropyTangent = surf.anisotropyTangent;
  payload.anisotropyBitangent = surf.anisotropyBitangent;
  payload.iridescence = surf.iridescence;
  payload.clearcoat = surf.clearcoat;
  payload.sheen = surf.sheen;
  payload.sheenRoughness = surf.sheenRoughness;
  payload.layerTransmission = surf.layerTransmission;
  payload.volumeScattering = surf.volumeScattering;
  payload.bulkThickness = surf.bulkThickness;
  return restir_gi_receiver_phat_from_payload(
    surf.pos,
    surf.normal,
    surf.clearcoatNormal,
    surf.wo,
    payload,
    xs,
    Lo,
  );
}

fn restir_gi_receiver_phat_from_surface_or_geometry(
  surf: PrimarySurface,
  xv: vec3f,
  nv: vec3f,
  xs: vec3f,
  Lo: vec3f,
) -> f32 {
  // Glass-primary reservoirs store the post-glass diffuse receiver, not the
  // camera-visible pane. If a recast surface does not match the reservoir's
  // receiver point, keep the geometric diffuse target rather than borrowing the
  // wrong material.
  if (surf.hit && length(surf.pos - xv) <= 5e-2) {
    return restir_gi_receiver_phat_from_surface(surf, xs, Lo);
  }
  return restir_gi_receiver_phat_from_geometry(xv, nv, xs, Lo);
}

fn restir_gi_surface_emission_for_hit(hit: IntersectionResult) -> vec3f {
  if (!materialEmissionSideAdmittedForHit(hit)) {
    return vec3f(0.0);
  }
  let coord = vec2u(
    hit.indices.w % BVH_MATERIAL_TEX_WIDTH,
    hit.indices.w / BVH_MATERIAL_TEX_WIDTH,
  );
  let scalarEmission = textureLoad(restir_gi_bvh_emissive, vec2i(coord), 0).rgb;
  return sampleEmissiveMap(
    hit.indices.w,
    hit.uv,
    materialAtlasUv1ForHit(hit),
    scalarEmission,
  );
}

fn restir_gi_surface_source_for_hit(
  hit: IntersectionResult,
  albedo: vec3f,
) -> vec3f {
  let uv1 = materialAtlasUv1ForHit(hit);
  // MaterialSpec.lightMap is receiver-local baked irradiance. Convert it to
  // diffuse outgoing radiance exactly once at the hit surface; this term then
  // participates in ordinary ReSTIR-GI transport like self-emission.
  let bakedLo = albedo * INV_PI * sampleLightMap(hit);
  return restir_gi_surface_emission_for_hit(hit) + bakedLo;
}

fn sampleRestirGIHitMaterialForHit(
  hit: IntersectionResult,
  smoothNormal: vec3f,
  shadingNormal: vec3f,
  incomingIrradiance: vec3f,
  wiVisibleToHit: vec3f,
  materialWord: u32,
) -> RestirGIHitMaterial {
  let scalarMat = decodeMaterialColor(hit.matColorPacked);
  let payload = sampleRestirDIMaterialPayloadForHit(
    hit,
    smoothNormal,
    shadingNormal,
    scalarMat.rgb,
    materialWord,
    safe_normalize(-wiVisibleToHit),
  );

  var out: RestirGIHitMaterial;
  out.normal = shadingNormal;
  out.albedo = payload.albedo;
  out.rough = payload.rough;

  let surfaceSource = restir_gi_surface_source_for_hit(hit, payload.albedo);
  // DDGI supplies hemispherically integrated irradiance E, not directional
  // radiance Li(wi).  Therefore the only measure-correct local reflection is
  // Lambertian E * albedo / pi.  Rich lobes remain fully evaluated at the
  // visible ReSTIR receiver, where the actual reconnection direction exists;
  // inventing a reflected proxy direction here would change units and energy.
  let diffuseLo = incomingIrradiance * payload.albedo * INV_PI;
  out.Lo = applyHomogeneousVolumeSingleScatter(
    (surfaceSource + diffuseLo) * payload.layerTransmission,
    payload.albedo,
    payload.volumeScattering,
    payload.bulkThickness,
    shadingNormal,
    safe_normalize(-wiVisibleToHit),
  );

  return out;
}
`;

export const RESTIR_GI_MATERIAL_MODULE: WgslModule = {
  name: 'restirGiMaterial',
  source: RESTIR_GI_MATERIAL_WGSL,
  requires: ['walkaroundUbo', 'reservoirGi', 'sceneTraversal', 'sharedPrimitives', 'ggxBrdf', 'materialDecode', 'materialAtlas'],
};
