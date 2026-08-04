import type { WgslModule } from '../pipeline/wgslComposer.js';

export const RESTIR_GI_MATERIAL_WGSL = /* wgsl */ `

struct RestirGIHitMaterial {
  normal: vec3f,
  Lo: vec3f,
  // Split terms retain the exact linear decomposition needed by a stochastic
  // opaque-vs-dielectric suffix branch. Emission remains deterministic;
  // opaqueLo is the unweighted light-map + DDGI diffuse response, while the
  // dielectric branch owns its persistent Fresnel reflection/refraction walk.
  emissionLo: vec3f,
  opaqueLo: vec3f,
  albedo: vec3f,
  rough: f32,
  transmission: f32,
  // The 28-word GI reservoir stores final Lo, not the material/irradiance state
  // needed to re-evaluate an angular suffix at another receiver. Such a sample
  // remains a valid native estimate but must not supply a shifted candidate.
  localSelectedSuffix: u32,
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
  transmission: f32,
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
  h = restir_gi_receiver_key_mix(h, bitcast<u32>(transmission));

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
  let reflectionLayerTransmission = bitcast<vec3u>(
    payload.reflectionLayerTransmission,
  );
  h = restir_gi_receiver_key_mix(h, reflectionLayerTransmission.x);
  h = restir_gi_receiver_key_mix(h, reflectionLayerTransmission.y);
  h = restir_gi_receiver_key_mix(h, reflectionLayerTransmission.z);
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

fn restir_gi_receiver_contribution_core(
  receiverNormal: vec3f,
  receiverClearcoatNormal: vec3f,
  receiverWo: vec3f,
  payload: RestirDIMaterialPayload,
  wi: vec3f,
  Lo: vec3f,
  diffuseReflectance: vec3f,
  transmission: f32,
) -> vec3f {
  let cosTheta = max(0.0, dot(receiverNormal, wi));
  if (cosTheta <= 0.0) {
    return vec3f(0.0);
  }

  var contribution = Lo * diffuseReflectance *
    (1.0 - clamp(payload.metal, 0.0, 1.0)) *
    (1.0 - clamp(transmission, 0.0, 1.0)) * cosTheta * INV_PI *
    payload.layerTransmission;
  if (transmission > 0.0 || restir_gi_receiver_has_specular_lobes(payload)) {
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
    contribution = contribution + Lo * specBrdf *
      payload.reflectionLayerTransmission;
  }
  return applyHomogeneousVolumeSingleScatterDirectional(
    contribution,
    payload.albedo,
    payload.volumeScattering,
    payload.bulkThickness,
    receiverNormal,
    receiverWo,
    wi,
  );
}

fn restir_gi_receiver_contribution_from_payload(
  receiverNormal: vec3f,
  receiverClearcoatNormal: vec3f,
  receiverWo: vec3f,
  payload: RestirDIMaterialPayload,
  transmission: f32,
  wi: vec3f,
  Lo: vec3f,
) -> vec3f {
  // p-hat may be any support-preserving importance target; using the physical
  // diffuse reflectance here is a target-fidelity and variance choice, not an
  // unbiasedness requirement. Shade evaluates the same physical response per
  // selected direction, then safely demodulates only the diffuse buffer before
  // indirectCombine restores albedo after filtering.
  return restir_gi_receiver_contribution_core(
    receiverNormal,
    receiverClearcoatNormal,
    receiverWo,
    payload,
    wi,
    Lo,
    payload.albedo,
    transmission,
  );
}

fn restir_gi_receiver_phat_from_payload(
  receiverPos: vec3f,
  receiverNormal: vec3f,
  receiverClearcoatNormal: vec3f,
  receiverWo: vec3f,
  payload: RestirDIMaterialPayload,
  transmission: f32,
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
    transmission,
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
  return luminance(restir_gi_receiver_contribution_from_geometry(
    xv,
    nv,
    xs,
    Lo,
  ));
}

fn restir_gi_receiver_contribution_from_geometry(
  xv: vec3f,
  nv: vec3f,
  xs: vec3f,
  Lo: vec3f,
) -> vec3f {
  let d = xs - xv;
  if (!(safe_length(d) > 0.0)) { return vec3f(0.0); }
  let wi = safe_normalize(d);
  let cosTheta = max(0.0, dot(nv, wi));
  return Lo * cosTheta * INV_PI;
}

fn restir_gi_receiver_phat_from_surface(
  surf: PrimarySurface,
  xs: vec3f,
  Lo: vec3f,
) -> f32 {
  return luminance(restir_gi_receiver_contribution_from_surface(
    surf,
    xs,
    Lo,
  ));
}

fn restir_gi_receiver_contribution_from_surface(
  surf: PrimarySurface,
  xs: vec3f,
  Lo: vec3f,
) -> vec3f {
  if (!surf.hit) { return vec3f(0.0); }
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
  payload.reflectionLayerTransmission = surf.reflectionLayerTransmission;
  payload.layerTransmission = surf.layerTransmission;
  payload.volumeScattering = surf.volumeScattering;
  payload.bulkThickness = surf.bulkThickness;
  let d = xs - surf.pos;
  if (!(safe_length(d) > 0.0)) { return vec3f(0.0); }
  return restir_gi_receiver_contribution_from_payload(
    surf.normal,
    surf.clearcoatNormal,
    surf.wo,
    payload,
    surf.transmission,
    safe_normalize(d),
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
  return luminance(restir_gi_receiver_contribution_from_surface_or_geometry(
    surf,
    xv,
    nv,
    xs,
    Lo,
  ));
}

fn restir_gi_receiver_contribution_from_surface_or_geometry(
  surf: PrimarySurface,
  xv: vec3f,
  nv: vec3f,
  xs: vec3f,
  Lo: vec3f,
) -> vec3f {
  // Glass-primary reservoirs store the post-glass diffuse receiver, not the
  // camera-visible pane. If a recast surface does not match the reservoir's
  // receiver point, keep the geometric diffuse target rather than borrowing the
  // wrong material.
  if (surf.hit && length(surf.pos - xv) <= 5e-2) {
    return restir_gi_receiver_contribution_from_surface(surf, xs, Lo);
  }
  return restir_gi_receiver_contribution_from_geometry(xv, nv, xs, Lo);
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
  let transmission = sampleTransmissionMapForHit(hit, scalarMat.a);
  out.transmission = transmission;

  let outgoingDirection = safe_normalize(-wiVisibleToHit);
  if (decodeIsUnlitMaterial(materialWord)) {
    // Match the camera-visible unlit contract exactly. Unlit is an authored
    // outgoing-radiance source, independent of DDGI and the connection angle.
    out.Lo = payload.albedo * payload.layerTransmission;
    out.emissionLo = out.Lo;
    out.opaqueLo = vec3f(0.0);
    out.localSelectedSuffix = 0u;
    return out;
  }

  // Thin-film transmission changes continuously with the outgoing angle. The
  // homogeneous-volume proxy changes its Beer path length by 1/|n.wo|. Neither
  // angular input is retained by the reservoir ABI, so classify these selected
  // suffixes as local instead of silently reusing source-evaluated Lo.
  let film = materialThinFilmResponse(
    hit.indices.w,
    hit.side >= 0.0,
    abs(dot(shadingNormal, outgoingDirection)),
  );
  let volumeIsAngular =
    payload.bulkThickness > 0.0 &&
    any(payload.volumeScattering.rgb > vec3f(0.0));
  out.localSelectedSuffix = select(
    0u,
    1u,
    transmission > 0.0 || film.present != 0u || volumeIsAngular,
  );

  let surfaceEmission = restir_gi_surface_emission_for_hit(hit);
  let bakedDiffuse = payload.albedo * INV_PI * sampleLightMap(hit);
  // DDGI supplies hemispherically integrated irradiance E, not directional
  // radiance Li(wi).  Therefore the only measure-correct local reflection is
  // Lambertian E * albedo / pi.  Rich lobes remain fully evaluated at the
  // visible ReSTIR receiver, where the actual reconnection direction exists;
  // inventing a reflected proxy direction here would change units and energy.
  let diffuseLo = (bakedDiffuse +
    incomingIrradiance * payload.albedo * INV_PI) *
    (1.0 - clamp(transmission, 0.0, 1.0));
  let rawEmissionLo = surfaceEmission * payload.layerTransmission;
  let rawOpaqueLo =
    (bakedDiffuse + incomingIrradiance * payload.albedo * INV_PI) *
    payload.layerTransmission;
  let scatteredEmissionLo = applyHomogeneousVolumeSingleScatter(
    rawEmissionLo,
    payload.albedo,
    payload.volumeScattering,
    payload.bulkThickness,
    shadingNormal,
    outgoingDirection,
  );
  let scatteredOpaqueLo = applyHomogeneousVolumeSingleScatter(
    rawOpaqueLo,
    payload.albedo,
    payload.volumeScattering,
    payload.bulkThickness,
    shadingNormal,
    outgoingDirection,
  );
  // Explicit dielectric suffixes own the actual in-medium segment and its
  // 3x3 absorption/scattering transfer. Feeding them the straight-view volume
  // proxy here would apply scattering a second time before the path is known.
  out.emissionLo = select(
    scatteredEmissionLo, rawEmissionLo, transmission > 0.0,
  );
  out.opaqueLo = select(
    scatteredOpaqueLo, rawOpaqueLo, transmission > 0.0,
  );
  out.Lo = applyHomogeneousVolumeSingleScatter(
    (surfaceEmission + diffuseLo) * payload.layerTransmission,
    payload.albedo,
    payload.volumeScattering,
    payload.bulkThickness,
    shadingNormal,
    outgoingDirection,
  );

  return out;
}
`;

export const RESTIR_GI_MATERIAL_MODULE: WgslModule = {
  name: 'restirGiMaterial',
  source: RESTIR_GI_MATERIAL_WGSL,
  requires: ['walkaroundUbo', 'reservoirGi', 'sceneTraversal', 'sharedPrimitives', 'ggxBrdf', 'materialDecode', 'materialAtlas'],
};
