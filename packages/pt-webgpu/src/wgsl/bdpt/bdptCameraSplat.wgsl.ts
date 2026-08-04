/**
 * BDPT t=1 light-subpath-to-camera splats.
 *
 * One full-tier invocation builds one invocation-private light subpath. Every
 * connectible L_c (c>=1) is projected onto the pinhole sensor and contributes
 * the Veach s=c+1,t=1 strategy to that projected pixel. Because the target pixel
 * is unrelated to the invocation that generated the light path, RGB is summed
 * through a CAS-based atomic-f32 buffer. A second compute entry point resolves
 * that per-frame buffer into the ordinary persistent accumulator; the main
 * kernel stages its own eye sample into the same buffer first, so each pixel's
 * variance sample is the complete base+splat estimate for that frame.
 *
 * Camera measurement follows PBRT PerspectiveCamera::{We,Pdf_We,Sample_Wi}:
 *
 *   pdfDir(camera -> L_c) = 1 / (A cos^3(theta))
 *   We / pdfPosition      = 1 / (A cos^3(theta) distance^2)
 *
 * where A is reconstructed from invViewProj at the unit-forward image plane.
 *
 * References:
 * - Veach 1997 §10.3 (s=n-1,t=1 strategy; power-heuristic MIS).
 * - Pharr, Jakob & Humphreys, PBRT 3e, Perspective Camera `We`, `Pdf_We`,
 *   `Sample_Wi`, and BDPT `ConnectBDPT`'s t=1 branch.
 */
export const PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL = /* wgsl */ `
// Four atomic words per pixel: RGB float bit patterns plus one padding word.
// WebGPU has no native atomic<f32>; bdptAtomicAddFiniteF32 performs a bounded
// compare-exchange loop and rejects NaN/Inf/negative physical contributions.
@group(0) @binding(14) var<storage, read_write>
  bdptCameraSplatBuffer: array<atomic<u32>>;

struct BdptCameraGeometry {
  forward: vec3f,
  imagePlaneArea: f32,
  valid: bool,
}

struct BdptCameraProjection {
  pixelIndex: u32,
  cameraToVertex: vec3f,
  cameraDirectionalPdf: f32,
  sampleWiOverPdf: f32,
  valid: bool,
}

fn bdptFiniteNonNegativeRgb(value: vec3f) -> bool {
  let finiteProbe = value - value;
  return all(finiteProbe == vec3f(0.0)) && all(value >= vec3f(0.0));
}

fn bdptAtomicAddFiniteF32(
  wordIndex: u32,
  value: f32,
) {
  if (!(value > 0.0) || value - value != 0.0) {
    return;
  }
  var expected = atomicLoad(&bdptCameraSplatBuffer[wordIndex]);
  loop {
    let oldValue = bitcast<f32>(expected);
    if (oldValue - oldValue != 0.0 || oldValue < 0.0) {
      return;
    }
    let nextValue = oldValue + value;
    if (nextValue - nextValue != 0.0 || nextValue < oldValue) {
      // A finite f32 overflow must not publish Inf into the beauty accumulator.
      return;
    }
    let exchanged = atomicCompareExchangeWeak(
      &bdptCameraSplatBuffer[wordIndex], expected, bitcast<u32>(nextValue),
    );
    if (exchanged.exchanged) {
      return;
    }
    expected = exchanged.old_value;
  }
}

fn bdptAtomicAddCameraRgb(pixelIndex: u32, value: vec3f) {
  if (!bdptFiniteNonNegativeRgb(value)) {
    return;
  }
  let base = pixelIndex * 4u;
  bdptAtomicAddFiniteF32(base, value.x);
  bdptAtomicAddFiniteF32(base + 1u, value.y);
  bdptAtomicAddFiniteF32(base + 2u, value.z);
}

fn bdptLoadCameraRgb(pixelIndex: u32) -> vec3f {
  let base = pixelIndex * 4u;
  return vec3f(
    bitcast<f32>(atomicLoad(&bdptCameraSplatBuffer[base])),
    bitcast<f32>(atomicLoad(&bdptCameraSplatBuffer[base + 1u])),
    bitcast<f32>(atomicLoad(&bdptCameraSplatBuffer[base + 2u])),
  );
}

fn bdptCameraDirectionForNdc(ndc: vec2f) -> vec3f {
  return unproject_ray_common(params.invViewProj, ndc).direction;
}

fn bdptBuildCameraGeometry() -> BdptCameraGeometry {
  var result: BdptCameraGeometry;
  result.valid = false;
  let rasterCenterDirection =
    bdptCameraDirectionForNdc(vec2f(0.0));
  let near00h = finite_homogeneous_point_common(
    params.invViewProj * vec4f(-1.0, 1.0, -1.0, 1.0),
  );
  let near10h = finite_homogeneous_point_common(
    params.invViewProj * vec4f(1.0, 1.0, -1.0, 1.0),
  );
  let near01h = finite_homogeneous_point_common(
    params.invViewProj * vec4f(-1.0, -1.0, -1.0, 1.0),
  );
  if (near00h.w == 0.0 || near10h.w == 0.0 || near01h.w == 0.0) {
    return result;
  }
  let near00 = near00h.xyz;
  let near10 = near10h.xyz;
  let near01 = near01h.xyz;
  result.forward = safe_normalize(
    cross(near10 - near00, near01 - near00),
  );
  if (dot(result.forward, rasterCenterDirection) < 0.0) {
    result.forward = -result.forward;
  }
  let d00 = bdptCameraDirectionForNdc(vec2f(-1.0, 1.0));
  let d10 = bdptCameraDirectionForNdc(vec2f(1.0, 1.0));
  let d01 = bdptCameraDirectionForNdc(vec2f(-1.0, -1.0));
  let c00 = dot(d00, result.forward);
  let c10 = dot(d10, result.forward);
  let c01 = dot(d01, result.forward);
  if (!(c00 > 0.0) || !(c10 > 0.0) || !(c01 > 0.0)) {
    return result;
  }
  let p00 = d00 / c00;
  let p10 = d10 / c10;
  let p01 = d01 / c01;
  result.imagePlaneArea = length(cross(p10 - p00, p01 - p00));
  if (
    !(result.imagePlaneArea > 0.0) ||
    result.imagePlaneArea - result.imagePlaneArea != 0.0
  ) {
    return result;
  }
  result.valid = true;
  return result;
}

fn bdptCameraDirectionalPdfForDirection(direction: vec3f) -> f32 {
  let camera = bdptBuildCameraGeometry();
  if (!camera.valid) {
    return 0.0;
  }
  let rayDirection = safe_normalize(direction);
  let cosTheta = dot(rayDirection, camera.forward);
  if (!(cosTheta > 0.0)) {
    return 0.0;
  }
  let pdf =
    1.0 / (camera.imagePlaneArea * cosTheta * cosTheta * cosTheta);
  if (!(pdf > 0.0) || pdf - pdf != 0.0) {
    return 0.0;
  }
  return pdf;
}

fn bdptProjectCameraSplat(
  vertexPosition: vec3f,
  camera: BdptCameraGeometry,
) -> BdptCameraProjection {
  var result: BdptCameraProjection;
  result.valid = false;
  let rawClip = params.viewProj * vec4f(vertexPosition, 1.0);
  let clipScale = max(
    max(abs(rawClip.x), abs(rawClip.y)),
    max(abs(rawClip.z), abs(rawClip.w)),
  );
  if (!(clipScale > 0.0) || clipScale > 3.402823e38) {
    return result;
  }
  let clip = rawClip / clipScale;
  if (!(clip.w > 0.0)) {
    return result;
  }
  let ndc = clip.xy / clip.w;
  if (
    ndc.x < -1.0 || ndc.x >= 1.0 ||
    ndc.y <= -1.0 || ndc.y > 1.0
  ) {
    return result;
  }
  let cameraToVertexVector = vertexPosition - params.cameraPos.xyz;
  let distanceSquared = dot(cameraToVertexVector, cameraToVertexVector);
  if (!(distanceSquared > 0.0)) {
    return result;
  }
  result.cameraToVertex = cameraToVertexVector * inverseSqrt(distanceSquared);
  let cosTheta = dot(result.cameraToVertex, camera.forward);
  if (!(cosTheta > 0.0)) {
    return result;
  }
  result.cameraDirectionalPdf =
    1.0 / (camera.imagePlaneArea * cosTheta * cosTheta * cosTheta);
  result.sampleWiOverPdf = result.cameraDirectionalPdf / distanceSquared;
  if (
    !(result.cameraDirectionalPdf > 0.0) ||
    !(result.sampleWiOverPdf > 0.0) ||
    result.cameraDirectionalPdf - result.cameraDirectionalPdf != 0.0 ||
    result.sampleWiOverPdf - result.sampleWiOverPdf != 0.0
  ) {
    return result;
  }
  let raster = vec2f(
    (ndc.x * 0.5 + 0.5) * f32(params.width),
    (1.0 - (ndc.y * 0.5 + 0.5)) * f32(params.height),
  );
  let pixel = vec2u(floor(raster));
  if (pixel.x >= params.width || pixel.y >= params.height) {
    return result;
  }
  result.pixelIndex = pixel.y * params.width + pixel.x;
  result.valid = true;
  return result;
}

fn bdptCameraSplatInvalidResult() -> vec4f {
  return vec4f(0.0);
}

// Return RGB contribution + target pixel encoded in .w as a u32 bit pattern.
fn bdptEvaluateCameraSplatVertex(
  c: u32,
  camera: BdptCameraGeometry,
  heroPdf: f32,
  rng: ptr<function, PtRngState>,
) -> vec4f {
  if (c == 0u) {
    return bdptCameraSplatInvalidResult();
  }
  let lv0 = bdptLightPath[bdptLightPathIndex(i32(c), 0u)];
  let lv1 = bdptLightPath[bdptLightPathIndex(i32(c), 1u)];
  let lv2 = bdptLightPath[bdptLightPathIndex(i32(c), 2u)];
  let lv3 = bdptLightPath[bdptLightPathIndex(i32(c), 3u)];
  let lv4 = bdptLightPath[bdptLightPathIndex(i32(c), 4u)];
  let lv5 = bdptLightPath[bdptLightPathIndex(i32(c), 5u)];
  let lv6 = bdptLightPath[bdptLightPathIndex(i32(c), 6u)];
  let lv7 = bdptLightPath[bdptLightPathIndex(i32(c), 7u)];
  if (
    lv0.w == BDPT_KIND_INVALID ||
    lv3.w < 0.0
  ) {
    return bdptCameraSplatInvalidResult();
  }
  let activeCausticMode = causticMode();
  if (
    (activeCausticMode == 2u && bdptLightPrefixContainsInteriorDelta(c)) ||
    (activeCausticMode == 1u && bdptLightPrefixIsMneeOwned(c))
  ) {
    return bdptCameraSplatInvalidResult();
  }

  let lightPosition = lv0.xyz;
  let projection = bdptProjectCameraSplat(lightPosition, camera);
  if (!projection.valid) {
    return bdptCameraSplatInvalidResult();
  }
  let toCamera = -projection.cameraToVertex;
  let cameraDistance =
    safe_length(lightPosition - params.cameraPos.xyz);
  if (!(cameraDistance > 0.0)) {
    return bdptCameraSplatInvalidResult();
  }
  let lightNormal = lv1.xyz;
  let surfaceCosine = abs(dot(lightNormal, toCamera));
  if (!(surfaceCosine > 0.0)) {
    return bdptCameraSplatInvalidResult();
  }

  // The camera endpoint starts in vacuum. A light vertex may connect only
  // through its camera-facing vacuum side; medium-boundary integration along an
  // unmatched edge is deliberately rejected by the same endpoint-medium rule
  // used by ordinary BDPT connections.
  let lightConnectionMedium = bdptSelectEndpointMedium(
    false,
    lightNormal,
    toCamera,
    bitcast<u32>(lv5.w),
    lv7.x,
    lv6.y,
    bitcast<u32>(lv6.x),
    lv7.x,
    lv6.y,
    bitcast<u32>(lv6.z),
    lv7.y,
    lv6.w,
  );
  let cameraConnectionMedium = bdptNoEndpointMedium();
  let connectionMedium = bdptSharedEdgeMedium(
    lightConnectionMedium, cameraConnectionMedium,
  );
  if (connectionMedium.remainingDistance < 0.0) {
    return bdptCameraSplatInvalidResult();
  }
  var connectionTransmittance = vec3f(1.0);
  var connectionIor = 1.0;
  if (connectionMedium.matId != BDPT_NO_MEDIUM) {
    let connectionMaterial = decodeMaterial(connectionMedium.matId);
    connectionIor = max(connectionMaterial.ior, 1e-4);
    if (
      params.spectralEnabled != 0u &&
      connectionMaterial.dispersionAbbe > 0.0
    ) {
      connectionIor = cauchyIorAtLambda(
        bdptInvocationHeroLambdaNm,
        connectionMaterial.ior,
        connectionMaterial.dispersionAbbe,
      );
    }
    let connectionSigmaT = bdptMaterialSigmaT(
      connectionMedium.matId,
      connectionMaterial,
      bdptInvocationHeroLambdaNm,
    );
    connectionTransmittance = materialBeer(
      connectionSigmaT,
      min(cameraDistance, max(connectionMedium.remainingDistance, 0.0)),
    );
  }

  let visibilityRay = Ray(
    lightPosition + toCamera * ptRayOriginBias(),
    toCamera,
  );
  let visibility = traceSurfaceVisibility(
    visibilityRay,
    ptRayTMin(),
    ptFiniteSegmentTMax(cameraDistance),
    bdptInvocationHeroLambdaNm,
    connectionIor,
    rng,
  );
  if (!visibility.visible) {
    return bdptCameraSplatInvalidResult();
  }
  connectionTransmittance = connectionTransmittance *
    visibility.transmittance;

  let lightMaterial = bdptSampleMaterialAtPayload(
    u32(lv3.w), lv4, lightNormal, lv3.xyz,
    bdptInvocationHeroLambdaNm,
  );
  if (!bsdfHasFiniteConnectionSupport(
    lightMaterial.roughness,
    lightMaterial.metallic,
    lightMaterial.transmission,
    lightMaterial.clearcoat,
    lightMaterial.sheen,
  )) {
    return bdptCameraSplatInvalidResult();
  }
  let lightScatter = evaluateFiniteBsdfFullWithClearcoatNormal(
    lightMaterial.baseColor,
    lightMaterial.roughness,
    lightMaterial.metallic,
    lightMaterial.transmission,
    max(lv5.x, 1e-4),
    lightNormal,
    lightMaterial.clearcoatNormal,
    lv3.xyz,
    toCamera,
    lightMaterial.clearcoat,
    lightMaterial.clearcoatRoughness,
    lightMaterial.sheen,
    lightMaterial.sheenRoughness,
    lightMaterial.sheenColor,
    lightMaterial.iridescence,
    lightMaterial.iridescenceIor,
    lightMaterial.iridescenceThicknessMin,
    lightMaterial.iridescenceThicknessMax,
    lightMaterial.specularColor,
    lightMaterial.specularIntensity,
    lightMaterial.anisotropy,
    lightMaterial.anisotropyRotation,
    lightMaterial.thinFilm,
    true,
  );
  if (!bdptFiniteNonNegativeRgb(lightScatter)) {
    return bdptCameraSplatInvalidResult();
  }

  let predecessor0 =
    bdptLightPath[bdptLightPathIndex(i32(c - 1u), 0u)];
  let predecessor1 =
    bdptLightPath[bdptLightPathIndex(i32(c - 1u), 1u)];
  let predecessor3 =
    bdptLightPath[bdptLightPathIndex(i32(c - 1u), 3u)];
  let predecessor5 =
    bdptLightPath[bdptLightPathIndex(i32(c - 1u), 5u)];
  let predecessor6 =
    bdptLightPath[bdptLightPathIndex(i32(c - 1u), 6u)];
  let predecessor7 =
    bdptLightPath[bdptLightPathIndex(i32(c - 1u), 7u)];
  let lcToPredecessor = safe_normalize(predecessor0.xyz - lightPosition);
  var revLcMinus = bdptMarginalSurfacePdf(
    lightMaterial.baseColor,
    lightMaterial.roughness,
    lightMaterial.metallic,
    lightMaterial.transmission,
    max(lv5.x, 1e-4),
    lightNormal,
    lightMaterial.clearcoatNormal,
    toCamera,
    lcToPredecessor,
    lightMaterial.clearcoat,
    lightMaterial.clearcoatRoughness,
    lightMaterial.sheen,
    lightMaterial.sheenRoughness,
    lightMaterial.iridescence,
    lightMaterial.iridescenceIor,
    lightMaterial.iridescenceThicknessMin,
    lightMaterial.iridescenceThicknessMax,
    lightMaterial.specularColor,
    lightMaterial.specularIntensity,
    lightMaterial.anisotropy,
    lightMaterial.anisotropyRotation,
    lightMaterial.thinFilm,
  );
  let predecessorIsMedium =
    predecessor3.w == BDPT_LV_MEDIUM_MATID;
  let currentPrefixMedium = bdptSelectEndpointMedium(
    false,
    lightNormal,
    lcToPredecessor,
    bitcast<u32>(lv5.w),
    lv7.x,
    lv6.y,
    bitcast<u32>(lv6.x),
    lv7.x,
    lv6.y,
    bitcast<u32>(lv6.z),
    lv7.y,
    lv6.w,
  );
  let predecessorPrefixMedium = bdptSelectEndpointMedium(
    predecessorIsMedium,
    predecessor1.xyz,
    -lcToPredecessor,
    bitcast<u32>(predecessor5.w),
    predecessor7.x,
    predecessor6.y,
    bitcast<u32>(predecessor6.x),
    predecessor7.x,
    predecessor6.y,
    bitcast<u32>(predecessor6.z),
    predecessor7.y,
    predecessor6.w,
  );
  revLcMinus = revLcMinus * bdptEndpointEdgeDistanceDensity(
    currentPrefixMedium,
    predecessorPrefixMedium,
    distance(lightPosition, predecessor0.xyz),
    predecessorIsMedium,
    bdptInvocationHeroLambdaNm,
  );

  // Camera Sample_Wi induces pdfRev(L_c). The selected t=1 path contains
  // v[0..c] plus the camera endpoint, so n=c+2 and selectedS=c+1=n-1.
  let n = c + 2u;
  let selectedS = c + 1u;
  let root1 = bdptLightPath[bdptLightPathIndex(0, 1u)];
  let root3 = bdptLightPath[bdptLightPathIndex(0, 3u)];
  let root4 = bdptLightPath[bdptLightPathIndex(0, 4u)];
  let infiniteRoot =
    root3.w == BDPT_LV_DIRECTIONAL_EMITTER_MATID ||
    root3.w == BDPT_LV_ENVIRONMENT_EMITTER_MATID;
  let infiniteEnvironmentRoot =
    root3.w == BDPT_LV_ENVIRONMENT_EMITTER_MATID;
  let infiniteSourceDirection = safe_normalize(root1.xyz);
  let infiniteNeePdf = select(0.0, root4.z, infiniteRoot);
  let infiniteLaunchPdf = select(0.0, root1.w * root4.y, infiniteRoot);
  var infiniteEyeEscapePdf = 0.0;
  var infiniteEyeEscapeDelta = false;
  if (infiniteRoot) {
    let first0 = bdptLightPath[bdptLightPathIndex(1, 0u)];
    let first1 = bdptLightPath[bdptLightPathIndex(1, 1u)];
    let first3 = bdptLightPath[bdptLightPathIndex(1, 3u)];
    let first4 = bdptLightPath[bdptLightPathIndex(1, 4u)];
    let first5 = bdptLightPath[bdptLightPathIndex(1, 5u)];
    let firstIsMedium = first3.w == BDPT_LV_MEDIUM_MATID;
    var firstCamerawardPosition = params.cameraPos.xyz;
    if (c >= 2u) {
      firstCamerawardPosition =
        bdptLightPath[bdptLightPathIndex(2, 0u)].xyz;
    }
    let firstCamerawardDirection = safe_normalize(
      firstCamerawardPosition - first0.xyz,
    );
    infiniteEyeEscapeDelta = first0.w == BDPT_KIND_DELTA;
    if (firstIsMedium) {
      infiniteEyeEscapePdf = hgPhase(
        dot(-firstCamerawardDirection, infiniteSourceDirection),
        first5.y,
      );
    } else if (first3.w >= 0.0 && !infiniteEyeEscapeDelta) {
      let firstMaterial = bdptSampleMaterialAtPayload(
        u32(first3.w),
        first4,
        first1.xyz,
        firstCamerawardDirection,
        bdptInvocationHeroLambdaNm,
      );
      if (bsdfHasFiniteConnectionSupport(
        firstMaterial.roughness,
        firstMaterial.metallic,
        firstMaterial.transmission,
        firstMaterial.clearcoat,
        firstMaterial.sheen,
      )) {
        infiniteEyeEscapePdf = bdptMarginalSurfacePdf(
          firstMaterial.baseColor,
          firstMaterial.roughness,
          firstMaterial.metallic,
          firstMaterial.transmission,
          max(first5.x, 1e-4),
          first1.xyz,
          firstMaterial.clearcoatNormal,
          firstCamerawardDirection,
          infiniteSourceDirection,
          firstMaterial.clearcoat,
          firstMaterial.clearcoatRoughness,
          firstMaterial.sheen,
          firstMaterial.sheenRoughness,
          firstMaterial.iridescence,
          firstMaterial.iridescenceIor,
          firstMaterial.iridescenceThicknessMin,
          firstMaterial.iridescenceThicknessMax,
          firstMaterial.specularColor,
          firstMaterial.specularIntensity,
          firstMaterial.anisotropy,
          firstMaterial.anisotropyRotation,
          firstMaterial.thinFilm,
        );
      }
    }
  }
  let misWeight = bdptMISWeightFull(
    c,
    0u,
    n,
    selectedS,
    infiniteRoot,
    infiniteEnvironmentRoot,
    infiniteSourceDirection,
    infiniteNeePdf,
    infiniteLaunchPdf,
    infiniteEyeEscapePdf,
    infiniteEyeEscapeDelta,
    1.0,
    1.0,
    projection.cameraDirectionalPdf,
    revLcMinus,
    params.cameraPos.xyz,
    camera.forward,
  );
  if (!(misWeight > 0.0)) {
    return bdptCameraSplatInvalidResult();
  }

  var contribution =
    lv2.xyz * lightScatter * surfaceCosine *
    projection.sampleWiOverPdf * misWeight * connectionTransmittance;
  if (!bdptFiniteNonNegativeRgb(contribution)) {
    return bdptCameraSplatInvalidResult();
  }
  if (params.spectralEnabled != 0u) {
    contribution = max(
      heroWavelengthToRgb(
        bdptInvocationHeroLambdaNm,
        luminance(contribution),
        heroPdf,
      ),
      vec3f(0.0),
    );
  }
  if (!bdptFiniteNonNegativeRgb(contribution)) {
    return bdptCameraSplatInvalidResult();
  }
  return vec4f(contribution, bitcast<f32>(projection.pixelIndex));
}

fn bdptAccumulateCameraSplatStrategies(pixel: vec2u, heroPdf: f32) {
  let camera = bdptBuildCameraGeometry();
  if (!camera.valid) {
    return;
  }
  let maxLightVertices = min(
    params.bdptMaxLightBounces,
    BDPT_MAX_LIGHT_DEPTH,
  );
  for (var c = 1u; c < maxLightVertices; c = c + 1u) {
    let vertex =
      bdptLightPath[bdptLightPathIndex(i32(c), 0u)];
    if (vertex.w == BDPT_KIND_INVALID) {
      break;
    }
    var visibilityRng = pcgInit(
      pixel.x ^ (c * 0x6c8e9cf5u),
      pixel.y ^ (c * 0x9e3779b9u),
      ptRngFrameKey(
        params.frameSeed ^ 0xca6e12f5u,
        params.frameIndex,
      ),
    );
    let splat = bdptEvaluateCameraSplatVertex(
      c, camera, heroPdf, &visibilityRng,
    );
    if (any(splat.xyz > vec3f(0.0))) {
      bdptAtomicAddCameraRgb(bitcast<u32>(splat.w), splat.xyz);
    }
  }
}
`;
