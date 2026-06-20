import type { WgslModule } from '../pipeline/wgslComposer.js';

export const RESTIR_GI_MATERIAL_WGSL = /* wgsl */ `

struct RestirGIHitMaterial {
  normal: vec3f,
  Lo: vec3f,
  albedo: vec3f,
  rough: f32,
};

@group(1) @binding(12) var restir_gi_bvh_emissive: texture_2d<f32>;

fn restir_gi_smooth_normal_for_hit(hit: IntersectionResult, geoNormal: vec3f) -> vec3f {
  let isTlas = ubo.bvhMode == 1u;
  let base = hit.instanceIndex * 4u;
  let ok = isTlas && base + 2u < arrayLength(&tlasInstanceWorldToLocal);
  let i = select(0u, base, ok);
  return smoothShadingNormal(
    hit,
    geoNormal,
    bvh_normal[hit.indices.x].xyz,
    bvh_normal[hit.indices.y].xyz,
    bvh_normal[hit.indices.z].xyz,
    ok,
    tlasInstanceWorldToLocal[i],
    tlasInstanceWorldToLocal[i + 1u],
    tlasInstanceWorldToLocal[i + 2u],
  );
}

fn restir_gi_shading_normal_for_hit(hit: IntersectionResult, geoNormal: vec3f) -> vec3f {
  let smoothNormal = restir_gi_smooth_normal_for_hit(hit, geoNormal);
  let normalMapped = applyNormalMapForHit(hit, smoothNormal);
  return applyBumpMapForHit(hit, normalMapped);
}

fn restir_gi_has_rich_suffix_payload(payload: RestirDIMaterialPayload) -> bool {
  let specularDelta = max(
    max(abs(payload.specular.r - 1.0), abs(payload.specular.g - 1.0)),
    max(abs(payload.specular.b - 1.0), abs(payload.specular.a - 1.0)),
  );
  return payload.metal > 1e-4
      || payload.rough < 0.84
      || specularDelta > 1e-4
      || abs(payload.anisotropy.x) > 1e-4
      || payload.iridescence.x > 1e-4
      || payload.clearcoat.x > 1e-4
      || payload.sheen.a > 1e-4;
}

fn restir_gi_proxy_incoming_dir(normal: vec3f, woToVisible: vec3f) -> vec3f {
  var wi = reflect(-woToVisible, normal);
  if (dot(wi, normal) <= 1e-4) {
    wi = normal;
  }
  return safe_normalize(wi);
}

fn restir_gi_receiver_has_specular_lobes(payload: RestirDIMaterialPayload) -> bool {
  let specularDelta = max(
    max(abs(payload.specular.r - 1.0), abs(payload.specular.g - 1.0)),
    max(abs(payload.specular.b - 1.0), abs(payload.specular.a - 1.0)),
  );
  return payload.metal > 1e-4
      || payload.rough < 0.6
      || specularDelta > 1e-4
      || abs(payload.anisotropy.x) > 1e-4
      || payload.iridescence.x > 1e-4
      || payload.clearcoat.x > 1e-4
      || payload.sheen.a > 1e-4;
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
  if (cosTheta <= 1e-6) {
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
  return contribution;
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
  let dist2 = dot(d, d);
  if (dist2 < 1e-8) { return 0.0; }
  let wi = d * inverseSqrt(dist2);
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
  let dist2 = dot(d, d);
  if (dist2 < 1e-8) { return 0.0; }
  let wi = d * inverseSqrt(dist2);
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
  );

  var out: RestirGIHitMaterial;
  out.normal = shadingNormal;
  out.albedo = payload.albedo;
  out.rough = payload.rough;

  let surfaceEmission = restir_gi_surface_emission_for_hit(hit);
  let diffuseLo = incomingIrradiance * payload.albedo * INV_PI;
  if (restir_gi_has_rich_suffix_payload(payload)) {
    let woToVisible = safe_normalize(-wiVisibleToHit);
    let proxyWi = restir_gi_proxy_incoming_dir(shadingNormal, woToVisible);
    let brdf = evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(
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
      shadingNormal,
      payload.clearcoatNormal,
      woToVisible,
      proxyWi,
    );
    out.Lo = surfaceEmission + incomingIrradiance * brdf;
  } else {
    out.Lo = surfaceEmission + diffuseLo;
  }

  return out;
}
`;

export const RESTIR_GI_MATERIAL_MODULE: WgslModule = {
  name: 'restirGiMaterial',
  source: RESTIR_GI_MATERIAL_WGSL,
  requires: ['walkaroundUbo', 'reservoirGi', 'sceneTraversal', 'sharedPrimitives', 'ggxBrdf', 'materialDecode', 'materialAtlas'],
};
