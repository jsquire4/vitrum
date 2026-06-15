import type { WgslModule } from '../pipeline/wgslComposer.js';

export const RESTIR_GI_MATERIAL_WGSL = /* wgsl */ `

struct RestirGIHitMaterial {
  normal: vec3f,
  Lo: vec3f,
  albedo: vec3f,
  rough: f32,
};

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

  let diffuseLo = incomingIrradiance * payload.albedo * INV_PI;
  if (restir_gi_has_rich_suffix_payload(payload)) {
    let woToVisible = safe_normalize(-wiVisibleToHit);
    let proxyWi = restir_gi_proxy_incoming_dir(shadingNormal, woToVisible);
    let brdf = evalGGXWithSpecularClearcoatSheen(
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
      shadingNormal,
      payload.clearcoatNormal,
      woToVisible,
      proxyWi,
    );
    out.Lo = incomingIrradiance * brdf;
  } else {
    out.Lo = diffuseLo;
  }

  return out;
}
`;

export const RESTIR_GI_MATERIAL_MODULE: WgslModule = {
  name: 'restirGiMaterial',
  source: RESTIR_GI_MATERIAL_WGSL,
  requires: ['walkaroundUbo', 'sceneTraversal', 'sharedPrimitives', 'ggxBrdf', 'materialDecode', 'materialAtlas'],
};
