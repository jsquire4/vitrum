import { RESTIR_GI_DIELECTRIC_SUFFIX_WGSL } from './risGi.wgsl.js';

/**
 * Full-resolution native camera-prefix glass GI.
 *
 * A refracted camera prefix is tied to one exact primary ray: its rough-BTDF
 * draws, TIR decisions, nested-medium state, and post-glass receiver cannot be
 * shifted to another pixel. The ordinary GI pass is at least half resolution,
 * so storing this estimator in its reservoir made unrelated full-resolution
 * pixels share one refractive path. This raw-string fragment instead runs the
 * bounded prefix and its suffix RIS directly in shadeMain for each glass pixel.
 * No camera-prefix record enters temporal, spatial, or bilinear GI reuse.
 */
export const NATIVE_GLASS_GI_WGSL = /* wgsl */ `
${RESTIR_GI_DIELECTRIC_SUFFIX_WGSL}

fn nativeGlassGiChannel(value: vec3f, channel: u32) -> f32 {
  if (channel == 0u) { return value.r; }
  if (channel == 1u) { return value.g; }
  return value.b;
}

fn nativeGlassGiThicknessMapScaleForHit(hit: IntersectionResult) -> f32 {
  let thicknessMap = sampleMaterialAtlasRawAtOffsetForHit(
    hit,
    MATERIAL_MAP_THICKNESS_TEXEL_OFFSET,
  );
  if (thicknessMap.valid == 0u) { return 1.0; }
  return materialOpticalThicknessMapScale(
    hit.indices.w,
    thicknessMap.value.g,
  );
}

fn nativeGlassGiTransferHasSupport(transfer: mat3x3f) -> bool {
  if (!restirGiSuffixTransferFinite(transfer)) { return false; }
  // Every transfer assembled by this walk is non-negative. Multiplying by a
  // unit radiance vector therefore exposes any surviving diagonal or
  // cross-channel in-scatter support without collapsing the operator to the
  // old diagonal-only throughput proxy.
  let support = transfer * vec3f(1.0);
  return max(support.x, max(support.y, support.z)) > 0.0;
}

struct NativeGlassGiContainingMedia {
  valid: u32,
  depth: u32,
  ior: array<vec3f, 8>,
  tri: array<u32, 8>,
  materialId: array<u32, 8>,
  instance: array<u32, 8>,
  beer: array<vec3f, 8>,
  thickness: array<f32, 8>,
  thicknessMapScale: array<f32, 8>,
  albedo: array<vec3f, 8>,
  scatter: array<vec4f, 8>,
};

// Recover the closed bulk-media stack with the shared exact outward scan. The
// actual exit face supplies all optical payloads; component/range IDs, not a
// material slot, own LIFO pairing.
fn classifyNativeGlassGiContainingMedia(
  cameraOrigin: vec3f,
  forwardDirection: vec3f,
) -> NativeGlassGiContainingMedia {
  var out: NativeGlassGiContainingMedia;
  out.valid = 0u;
  out.depth = 0u;
  let classified = materialShadowClassifyContainingMedia(
    ubo.bvhMode,
    ubo.tlasNodeCount,
    cameraOrigin,
    forwardDirection,
    ubo.triIntersectEpsilon,
    bvh_material,
    BVH_MATERIAL_TEX_WIDTH,
    bvh_beer,
  );
  if (classified.valid == 0u || classified.state.depth > 8u) { return out; }
  for (var slot = 0u; slot < classified.state.depth; slot = slot + 1u) {
    let triIndex = classified.state.tri[slot];
    let coord = vec2u(
      triIndex % BVH_MATERIAL_TEX_WIDTH,
      triIndex / BVH_MATERIAL_TEX_WIDTH,
    );
    let materialWord = textureLoad(bvh_material, vec2i(coord), 0).r;
    out.ior[slot] = materialDispersionIorRgb(
      triIndex,
      decodeIor(materialWord),
    );
    out.tri[slot] = triIndex;
    out.materialId[slot] = classified.state.materialId[slot];
    out.instance[slot] = classified.state.instance[slot];
    out.beer[slot] = classified.state.tint[slot];
    out.thickness[slot] = classified.state.thickness[slot];
    out.thicknessMapScale[slot] =
      classified.state.thicknessMapScale[slot];
    out.albedo[slot] = classified.state.albedo[slot];
    out.scatter[slot] = classified.state.scattering[slot];
  }
  out.depth = classified.state.depth;
  out.valid = 1u;
  return out;
}

struct NativeGlassGiReceiver {
  valid: u32,
  environment: u32,
  hit: IntersectionResult,
  pos: vec3f,
  geoNormal: vec3f,
  smoothNormal: vec3f,
  shadingNormal: vec3f,
  wo: vec3f,
  transmission: f32,
  closureMode: u32,
  opaqueShare: f32,
  environmentDirection: vec3f,
  prefixTransfer: mat3x3f,
};

fn traceNativeGlassGiReceiver(
  primaryHitInput: IntersectionResult,
  primaryRay: Ray,
  primaryPos: vec3f,
  primarySmoothNormal: vec3f,
  primaryNormal: vec3f,
  primaryGeoNormal: vec3f,
  primaryTransmission: f32,
  primaryAlbedo: vec3f,
  transportChannel: u32,
  rng: ptr<function, u32>,
) -> NativeGlassGiReceiver {
  const GLASS_WALK_MAX_INTERFACES: u32 = 8u;
  var result: NativeGlassGiReceiver;
  result.valid = 0u;
  result.environment = 0u;
  result.transmission = 0.0;
  result.closureMode = 0u;
  result.opaqueShare = 1.0;

  // The camera primary is selected by the ordinary traversal, but no
  // transmissive continuation may inherit tolerant Moller ownership. Replay
  // the exact represented world triangle and fail closed if it is not the
  // accepted optical source feature.
  var primaryHit = primaryHitInput;
  let exactPrimary = traceSceneRetraceOpticalHit(
    ubo.bvhMode, ubo.tlasNodeCount, primaryRay, primaryHit, 0.0,
  );
  let primarySourceFeature = sceneOpticalSourceFeatureForExactHit(
    ubo.bvhMode, ubo.tlasNodeCount, primaryHit, exactPrimary,
  );
  if (
    !exactPrimary.hit ||
    primarySourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID
  ) { return result; }
  let primaryUseTlas = ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u;
  let primaryExactTriangle = sceneLoadOpticalWorldTriangle(
    primaryUseTlas, primaryHit.indices.w, primaryHit.instanceIndex,
  );
  if (primaryExactTriangle.valid == 0u) { return result; }
  primaryHit.normal = exactPrimary.normal;
  primaryHit.barycoord = exactPrimary.bary;
  primaryHit.side = exactPrimary.side;
  primaryHit.dist = exactPrimary.t;
  primaryHit.uv = exactPrimary.bary.x * primaryExactTriangle.uvA +
    exactPrimary.bary.y * primaryExactTriangle.uvB +
    exactPrimary.bary.z * primaryExactTriangle.uvC;
  let exactPrimaryPos = primaryRay.origin + primaryRay.direction * exactPrimary.t;

  let glassPrimaryRmCoord = vec2u(
    primaryHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
    primaryHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
  );
  let glassPrimaryPacked = textureLoad(
    bvh_material,
    vec2i(glassPrimaryRmCoord),
    0,
  ).r;
  let glassPrimaryRm = decodeRoughMetal(glassPrimaryPacked);
  let glassPrimaryMappedRough = sampleMaterialScalarMap(
    primaryHit,
    MATERIAL_MAP_SLOT_ROUGHNESS,
    1u,
    glassPrimaryRm.x,
  );
  let primaryLayer = sampleFaceLayerControls(
    primaryHit.indices.w,
    primaryHit.side >= 0.0,
  );
  let glassPrimaryRough = faceLayerRoughness(
    glassPrimaryMappedRough,
    primaryLayer,
  );
  let primaryIorRgb = materialDispersionIorRgb(
    primaryHit.indices.w,
    decodeIor(glassPrimaryPacked),
  );
  // The material header defines closed-volume topology and the reference
  // attenuation length. The mapped G channel is already applied to the stored
  // Beer tint, so using it here as topology would make entry/exit disagree.
  let primaryThickness = materialShadowAuthoredThickness(primaryHit);
  let primaryBoundaryId = sceneOpticalEncodedBoundaryId(
    primaryUseTlas, primaryHit.indices.w, primaryHit.instanceIndex,
  );
  let primaryRepresentedId = sceneOpticalRepresentedPrimitiveInstanceId(
    primaryUseTlas, primaryHit.indices.w, primaryHit.instanceIndex,
  );
  if (primaryRepresentedId == 0u) { return result; }
  let primaryBulkMedium = primaryBoundaryId != 0u;

  let d = primaryRay.direction;
  let primaryEntering = primaryHit.side >= 0.0;
  let primaryAlignedNormal = select(
    -primaryNormal,
    primaryNormal,
    dot(primaryNormal, primaryGeoNormal) >= 0.0,
  );
  let primaryFaceNormal = select(
    -primaryAlignedNormal,
    primaryAlignedNormal,
    dot(d, primaryAlignedNormal) < 0.0,
  );
  if (dot(d, primaryFaceNormal) >= 0.0) { return result; }
  let primaryAnisotropy = sampleAnisotropyControls(primaryHit);
  let primaryAnisotropyFrame = materialTangentFrameForHit(
    primaryHit,
    primaryFaceNormal,
    MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
  );

  var mediumDepth: u32 = 0u;
  var mediumIor: array<vec3f, 8>;
  var mediumTri: array<u32, 8>;
  var mediumMaterialId: array<u32, 8>;
  var mediumInstance: array<u32, 8>;
  var mediumBeer: array<vec3f, 8>;
  var mediumThickness: array<f32, 8>;
  var mediumThicknessMapScale: array<f32, 8>;
  var mediumAlbedo: array<vec3f, 8>;
  var mediumScatter: array<vec4f, 8>;
  // A medium entered by this camera ray already paid its authored scalar
  // transmission. A medium containing the camera has not; its first forward
  // exit must pay that factor exactly once.
  var mediumTransmissionPaid: array<u32, 8>;

  let primaryBeerPacked = textureLoad(
    bvh_beer,
    vec2i(glassPrimaryRmCoord),
    0,
  ).r;
  var primaryBeer = vec3f(
    f32((primaryBeerPacked >> 24u) & 0xffu) / 255.0,
    f32((primaryBeerPacked >> 16u) & 0xffu) / 255.0,
    f32((primaryBeerPacked >> 8u) & 0xffu) / 255.0,
  );
  let primaryThicknessMapScale =
    nativeGlassGiThicknessMapScaleForHit(primaryHit);

  if (primaryBulkMedium && !primaryEntering) {
    let containing = classifyNativeGlassGiContainingMedia(
      primaryRay.origin,
      primaryRay.direction,
    );
    mediumDepth = containing.depth;
    mediumIor = containing.ior;
    mediumTri = containing.tri;
    mediumMaterialId = containing.materialId;
    mediumInstance = containing.instance;
    mediumBeer = containing.beer;
    mediumThickness = containing.thickness;
    mediumThicknessMapScale = containing.thicknessMapScale;
    mediumAlbedo = containing.albedo;
    mediumScatter = containing.scatter;
    if (
      containing.valid == 0u ||
      mediumDepth == 0u ||
      mediumMaterialId[mediumDepth - 1u] != primaryBoundaryId ||
      mediumInstance[mediumDepth - 1u] != primaryRepresentedId
    ) {
      return result;
    }
    let primarySlot = mediumDepth - 1u;
    mediumIor[primarySlot] = primaryIorRgb;
    mediumTri[primarySlot] = primaryHit.indices.w;
    mediumMaterialId[primarySlot] = primaryBoundaryId;
    mediumInstance[primarySlot] = primaryRepresentedId;
    mediumBeer[primarySlot] = primaryBeer;
    mediumThickness[primarySlot] = primaryThickness;
    mediumThicknessMapScale[primarySlot] = primaryThicknessMapScale;
    mediumAlbedo[primarySlot] = primaryAlbedo;
    mediumScatter[primarySlot] = sampleVolumeScatteringControls(
      primaryHit.indices.w,
    );
    for (var ci: u32 = 0u; ci < mediumDepth; ci = ci + 1u) {
      mediumTransmissionPaid[ci] = 0u;
    }
  }

  var primaryIncidentIor = vec3f(1.0);
  var primaryTargetIor = primaryIorRgb;
  if (primaryBulkMedium && !primaryEntering) {
    primaryIncidentIor = mediumIor[mediumDepth - 1u];
    primaryTargetIor = vec3f(1.0);
    if (mediumDepth > 1u) {
      primaryTargetIor = mediumIor[mediumDepth - 2u];
    }
  }
  let primaryBtdf = ggxSampleDielectricTransmissionAnisotropyFrame(
    primaryFaceNormal,
    primaryAnisotropyFrame.tangent,
    primaryAnisotropyFrame.bitangent,
    -d,
    glassPrimaryRough,
    primaryAnisotropy.x,
    primaryAnisotropy.y,
    nativeGlassGiChannel(primaryIncidentIor, transportChannel),
    nativeGlassGiChannel(primaryTargetIor, transportChannel),
    rng,
  );
  if (primaryBtdf.valid == 0u) { return result; }

  let primaryCosI = primaryBtdf.microfacetCos;
  var primaryInterfaceT = dielectricInterfaceTransmissionRgb(
    primaryCosI,
    primaryIncidentIor,
    primaryTargetIor,
  );
  let primaryFilm = materialThinFilmResponse(
    primaryHit.indices.w,
    primaryHit.side >= 0.0,
    primaryCosI,
  );
  if (primaryFilm.present != 0u) {
    primaryInterfaceT = primaryFilm.transmittance;
  }
  let primaryInterfaceFactor = primaryInterfaceT *
    (primaryBtdf.weight / primaryBtdf.transmission) *
    faceLayerTransmission(primaryLayer);
  var prefixTransfer = restirGiSuffixDiagonalTransfer(
    primaryInterfaceFactor,
  );
  if (!nativeGlassGiTransferHasSupport(prefixTransfer)) { return result; }

  var refractDir = primaryBtdf.direction;
  var interfaceCount = 1u;

  if (primaryBulkMedium) {
    if (primaryEntering) {
      prefixTransfer = prefixTransfer * restirGiSuffixDiagonalTransfer(
        vec3f(primaryTransmission),
      );
      if (!nativeGlassGiTransferHasSupport(prefixTransfer)) { return result; }
      mediumIor[mediumDepth] = primaryIorRgb;
      mediumTri[mediumDepth] = primaryHit.indices.w;
      mediumMaterialId[mediumDepth] = primaryBoundaryId;
      mediumInstance[mediumDepth] = primaryRepresentedId;
      mediumBeer[mediumDepth] = primaryBeer;
      mediumThickness[mediumDepth] = primaryThickness;
      mediumThicknessMapScale[mediumDepth] = primaryThicknessMapScale;
      mediumAlbedo[mediumDepth] = primaryAlbedo;
      mediumScatter[mediumDepth] = sampleVolumeScatteringControls(
        primaryHit.indices.w,
      );
      mediumTransmissionPaid[mediumDepth] = 1u;
      mediumDepth = mediumDepth + 1u;
    } else {
      // The camera begins inside the innermost classified medium. Account for
      // its actual camera-to-boundary distance before crossing the first exit;
      // the transfer is left-multiplied because this segment is closest to the
      // camera in the reverse radiance operator chain.
      let top = mediumDepth - 1u;
      let segmentDistance = primaryHit.dist;
      let referenceThickness = select(
        1.0, mediumThickness[top], mediumThickness[top] > 0.0,
      );
      let transportDistance = select(
        segmentDistance,
        min(
          segmentDistance,
          referenceThickness * clamp(mediumThicknessMapScale[top], 0.0, 1.0),
        ),
        mediumThickness[top] > 0.0,
      );
      let distanceScale = transportDistance / referenceThickness;
      let rgbBeer = pow(
        clamp(mediumBeer[top], vec3f(0.0), vec3f(1.0)),
        vec3f(distanceScale),
      );
      let segmentAbsorption = materialSpectralAttenuation(
        mediumTri[top],
        transportDistance,
        rgbBeer,
      );
      prefixTransfer = restirGiSuffixSegmentTransfer(
        segmentAbsorption,
        mediumScatter[top],
        mediumAlbedo[top],
        transportDistance,
      ) * prefixTransfer * restirGiSuffixDiagonalTransfer(
        vec3f(primaryTransmission),
      );
      if (!nativeGlassGiTransferHasSupport(prefixTransfer)) { return result; }
      mediumDepth = mediumDepth - 1u;
    }
  } else {
    // A zero-thickness material is a reciprocal two-boundary slab. The primary
    // lane already selected transmission at the visible boundary; pay scalar t
    // once, then retain every exact internal reflection/TIR path until the slab
    // exits or the shared interface budget is exhausted.
    prefixTransfer = prefixTransfer * restirGiSuffixDiagonalTransfer(
      vec3f(primaryTransmission),
    );
    if (!nativeGlassGiTransferHasSupport(prefixTransfer)) { return result; }
    var slabFrontFacing = primaryHit.side < 0.0;
    var slabExited = false;
    loop {
      if (interfaceCount >= GLASS_WALK_MAX_INTERFACES) { break; }
      let slabLayer = sampleFaceLayerControls(
        primaryHit.indices.w,
        slabFrontFacing,
      );
      let slabRough = faceLayerRoughness(
        glassPrimaryMappedRough,
        slabLayer,
      );
      let slabMappedNormal = applyBumpMapForHit(
        primaryHit,
        applyNormalMapForSideForHit(
          primaryHit, primarySmoothNormal, slabFrontFacing,
        ),
      );
      let slabAlignedNormal = select(
        -slabMappedNormal,
        slabMappedNormal,
        dot(slabMappedNormal, primaryHit.normal) >= 0.0,
      );
      let slabNormal = select(
        -slabAlignedNormal,
        slabAlignedNormal,
        dot(-refractDir, slabAlignedNormal) > 0.0,
      );
      let slabFrame = materialTangentFrameForHit(
        primaryHit,
        slabNormal,
        MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
      );
      let slabBranchPdf = represented_bernoulli_probability_f32(0.5);
      let slabChooseTransmission = rand_f32(rng) < slabBranchPdf;
      let slabLobe = restirGiSampleDielectricLobe(
        primaryHit,
        slabNormal,
        slabFrame.tangent,
        slabFrame.bitangent,
        -refractDir,
        slabRough,
        primaryAnisotropy.x,
        primaryAnisotropy.y,
        primaryTargetIor,
        primaryIncidentIor,
        slabFrontFacing,
        slabLayer,
        transportChannel,
        !slabChooseTransmission,
        rng,
      );
      interfaceCount = interfaceCount + 1u;
      if (slabLobe.valid == 0u) { break; }
      let slabSelectedPdf = select(
        1.0 - slabBranchPdf,
        slabBranchPdf,
        slabChooseTransmission,
      );
      let slabFactor = slabLobe.weightRgb / vec3f(slabSelectedPdf);
      prefixTransfer = prefixTransfer * restirGiSuffixDiagonalTransfer(
        slabFactor,
      );
      if (!restirGiSuffixTransferFinite(prefixTransfer)) { break; }
      refractDir = slabLobe.direction;
      if (slabChooseTransmission) {
        slabExited = true;
        break;
      }
      slabFrontFacing = !slabFrontFacing;
    }
    if (!slabExited) { return result; }
  }

  var walkOrigin = exactPrimaryPos;
  var walkHit: IntersectionResult;
  var walkHitPos = vec3f(0.0);
  var walkSmoothNormal = vec3f(0.0);
  var walkShadingNormal = vec3f(0.0);
  var foundSurface = false;
  var receiverTransmission = 0.0;
  var receiverClosureMode = 0u;
  var receiverOpaqueShare = 1.0;
  var continuationSourceFeature = primarySourceFeature;

  // Inspect one terminal hit after the eighth interface. That terminal may be
  // an opaque receiver but may not consume a ninth dielectric interface.
  for (var gi: u32 = 1u; gi <= GLASS_WALK_MAX_INTERFACES; gi = gi + 1u) {
    let walkRay = Ray(walkOrigin, refractDir);
    let sourceAware = traceSceneFirstHitAlphaMaskTexturedWithOpticalSource(
      ubo.bvhMode,
      ubo.tlasNodeCount,
      walkRay,
      continuationSourceFeature,
      bvh_material,
      BVH_MATERIAL_TEX_WIDTH,
      ubo.frameSeed ^ (gi * 0x9e3779b9u) ^ 0x474c4153u,
    );
    if (sourceAware.valid == 0u) { break; }
    walkHit = sourceAware.hit;
    if (!walkHit.didHit) {
      // A miss after the complete medium stack has been exited is a real
      // transmitted environment path. A miss while still inside an open bulk
      // medium has no finite exit distance, so its Beer state is undefined.
      if (mediumDepth == 0u) {
        result.valid = 1u;
        result.environment = 1u;
        result.environmentDirection = refractDir;
        result.prefixTransfer = prefixTransfer;
        return result;
      }
      break;
    }

    var acceptedSourceFeature = opticalSourceFeatureInvalid();
    if (packedMaterialHasTransmission(walkHit.matColorPacked)) {
      let exactWalk = traceSceneRetraceOpticalHit(
        ubo.bvhMode, ubo.tlasNodeCount, walkRay, walkHit, 0.0,
      );
      let walkSourceFeature = sceneOpticalSourceFeatureForExactHit(
        ubo.bvhMode, ubo.tlasNodeCount, walkHit, exactWalk,
      );
      if (
        !exactWalk.hit ||
        walkSourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID
      ) { break; }
      let walkUseTlas = ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u;
      let walkExactTriangle = sceneLoadOpticalWorldTriangle(
        walkUseTlas, walkHit.indices.w, walkHit.instanceIndex,
      );
      if (walkExactTriangle.valid == 0u) { break; }
      walkHit.normal = exactWalk.normal;
      walkHit.barycoord = exactWalk.bary;
      walkHit.side = exactWalk.side;
      walkHit.dist = exactWalk.t;
      walkHit.uv = exactWalk.bary.x * walkExactTriangle.uvA +
        exactWalk.bary.y * walkExactTriangle.uvB +
        exactWalk.bary.z * walkExactTriangle.uvC;
      acceptedSourceFeature = walkSourceFeature;
    }
    let walkUseTlas = ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u;
    let walkBoundaryId = sceneOpticalEncodedBoundaryId(
      walkUseTlas, walkHit.indices.w, walkHit.instanceIndex,
    );
    let walkRepresentedId = sceneOpticalRepresentedPrimitiveInstanceId(
      walkUseTlas, walkHit.indices.w, walkHit.instanceIndex,
    );

    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      let segmentDistance = walkHit.dist;
      var segmentTri = mediumTri[top];
      var segmentBeer = mediumBeer[top];
      var segmentThickness = mediumThickness[top];
      var segmentThicknessMapScale = mediumThicknessMapScale[top];
      var segmentScatter = mediumScatter[top];
      var segmentAlbedo = mediumAlbedo[top];
      if (
        packedMaterialHasTransmission(walkHit.matColorPacked) &&
        walkHit.side < 0.0 &&
        walkBoundaryId == mediumMaterialId[top] &&
        walkRepresentedId == mediumInstance[top]
      ) {
        segmentTri = walkHit.indices.w;
        segmentBeer = materialShadowAuthoredBeerTint(walkHit, bvh_beer);
        segmentThickness = materialShadowAuthoredThickness(walkHit);
        segmentThicknessMapScale = materialShadowThicknessMapScale(walkHit);
        segmentScatter = sampleVolumeScatteringControls(walkHit.indices.w);
        let exitScalar = decodeMaterialColor(walkHit.matColorPacked);
        let exitVertexColor = sampleVertexColorForHit(walkHit);
        segmentAlbedo = sampleBaseColorMap(
          walkHit, exitScalar.rgb * exitVertexColor.rgb,
        );
      }
      let referenceThickness = select(
        1.0, segmentThickness, segmentThickness > 0.0,
      );
      let transportDistance = select(
        segmentDistance,
        min(
          segmentDistance,
          referenceThickness * clamp(segmentThicknessMapScale, 0.0, 1.0),
        ),
        segmentThickness > 0.0,
      );
      let distanceScale = transportDistance / referenceThickness;
      let rgbBeer = pow(
        clamp(segmentBeer, vec3f(0.0), vec3f(1.0)),
        vec3f(distanceScale),
      );
      let segmentAbsorption = materialSpectralAttenuation(
        segmentTri,
        transportDistance,
        rgbBeer,
      );
      prefixTransfer = prefixTransfer * restirGiSuffixSegmentTransfer(
        segmentAbsorption,
        segmentScatter,
        segmentAlbedo,
        transportDistance,
      );
      if (!nativeGlassGiTransferHasSupport(prefixTransfer)) { break; }
    }

    walkHitPos = walkOrigin + refractDir * walkHit.dist;
    let scalarWalkMat = decodeMaterialColor(walkHit.matColorPacked);
    let walkTransmission = sampleTransmissionMapForHit(
      walkHit,
      scalarWalkMat.a,
    );
    // A dark transmission-map texel changes the lobe weight, not whether an
    // authored dielectric boundary exists. This keeps both reflection and bulk
    // stack topology stable across the two faces of a mapped material.
    let walkIsGlass = materialHasTransmission(scalarWalkMat.a);

    let wnIsTlas = ubo.bvhMode == 1u;
    let wnBase = walkHit.instanceIndex * 4u;
    let wnOk = wnIsTlas && wnBase + 2u < tlasWorldToLocalColumnCount();
    let wnI = select(0u, wnBase, wnOk);
    walkSmoothNormal = smoothShadingNormal(
      walkHit,
      walkHit.normal,
      sceneLoadBvhNormal(walkHit.indices.x).xyz,
      sceneLoadBvhNormal(walkHit.indices.y).xyz,
      sceneLoadBvhNormal(walkHit.indices.z).xyz,
      wnOk,
      tlasLoadWorldToLocalColumn(wnI),
      tlasLoadWorldToLocalColumn(wnI + 1u),
      tlasLoadWorldToLocalColumn(wnI + 2u),
    );
    let walkMappedNormal = applyNormalMapForHit(
      walkHit,
      walkSmoothNormal,
    );
    walkShadingNormal = applyBumpMapForHit(
      walkHit,
      walkMappedNormal,
    );

    if (!walkIsGlass) {
      foundSurface = true;
      break;
    }

    let walkCoord = vec2u(
      walkHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
      walkHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
    );
    let walkWord = textureLoad(
      bvh_material,
      vec2i(walkCoord),
      0,
    ).r;
    let walkRm = decodeRoughMetal(walkWord);
    if (decodeIsUnlitMaterial(walkWord)) {
      receiverTransmission = walkTransmission;
      foundSurface = true;
      break;
    }
    let walkMappedRough = sampleMaterialScalarMap(
      walkHit,
      MATERIAL_MAP_SLOT_ROUGHNESS,
      1u,
      walkRm.x,
    );
    let walkLayer = sampleFaceLayerControls(
      walkHit.indices.w,
      walkHit.side >= 0.0,
    );
    let walkRough = faceLayerRoughness(walkMappedRough, walkLayer);
    let walkIor = materialDispersionIorRgb(
      walkHit.indices.w,
      decodeIor(walkWord),
    );
    let walkThickness = materialShadowAuthoredThickness(walkHit);
    if (walkRepresentedId == 0u) { break; }
    let walkBulkMedium = walkBoundaryId != 0u;
    let entering = walkHit.side >= 0.0;
    if (walkBulkMedium && !entering && mediumDepth == 0u) {
      break;
    }
    if (interfaceCount >= GLASS_WALK_MAX_INTERFACES) { break; }

    let alignedNormal = select(
      -walkShadingNormal,
      walkShadingNormal,
      dot(walkShadingNormal, walkHit.normal) >= 0.0,
    );
    let faceNormal = select(
      -alignedNormal,
      alignedNormal,
      dot(refractDir, alignedNormal) < 0.0,
    );
    if (dot(refractDir, faceNormal) >= 0.0) { break; }
    let walkAnisotropy = sampleAnisotropyControls(walkHit);
    let walkAnisotropyFrame = materialTangentFrameForHit(
      walkHit,
      faceNormal,
      MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
    );

    var incidentIor = vec3f(1.0);
    if (mediumDepth > 0u) {
      incidentIor = mediumIor[mediumDepth - 1u];
    } else if (walkBulkMedium && !entering) {
      incidentIor = walkIor;
    }
    var targetIor = walkIor;
    if (walkBulkMedium && !entering) {
      let top = mediumDepth - 1u;
      if (
        mediumMaterialId[top] != walkBoundaryId ||
        mediumInstance[top] != walkRepresentedId
      ) {
        break;
      }
      // Exiting the outermost tracked volume targets air. A nested exit targets
      // its immediately enclosing medium. Leaving targetIor at walkIor here
      // would make the outer exit eta_i == eta_t and erase its refraction and
      // Fresnel response.
      targetIor = vec3f(1.0);
      if (mediumDepth > 1u) {
        targetIor = mediumIor[mediumDepth - 2u];
      }
    }

    var pairedPaidExit = false;
    if (walkBulkMedium && !entering && mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      pairedPaidExit =
        mediumMaterialId[top] == walkBoundaryId &&
        mediumInstance[top] == walkRepresentedId &&
        mediumTransmissionPaid[top] != 0u;
    }

    // Three disjoint estimators own the complete hidden closure: additive
    // emission plus the complementary opaque/base source, exact dielectric
    // reflection continuation, and scalar-weighted transmission continuation.
    // Equal technique probabilities keep every family supported even at t=0,
    // t=1, or exact TIR; every selected result is divided by its explicit PDF.
    let localBranchThreshold = represented_bernoulli_probability_f32(
      1.0 / 3.0,
    );
    let reflectionBranchThreshold = represented_bernoulli_probability_f32(
      2.0 / 3.0,
    );
    let localBranchPdf = localBranchThreshold;
    let reflectionBranchPdf =
      reflectionBranchThreshold - localBranchThreshold;
    let transmissionBranchPdf = 1.0 - reflectionBranchThreshold;
    let eventXi = rand_f32(rng);
    if (eventXi < localBranchThreshold) {
      let localBranchWeight = 1.0 / localBranchPdf;
      prefixTransfer = prefixTransfer * restirGiSuffixDiagonalTransfer(
        vec3f(localBranchWeight),
      );
      if (!restirGiSuffixTransferFinite(prefixTransfer)) { break; }
      receiverTransmission = walkTransmission;
      receiverClosureMode = 1u;
      receiverOpaqueShare = select(
        1.0 - clamp(walkTransmission, 0.0, 1.0),
        0.0,
        pairedPaidExit,
      );
      foundSurface = true;
      break;
    }
    let chooseTransmission = eventXi >= reflectionBranchThreshold;
    let selectedOpticalBranchPdf = select(
      reflectionBranchPdf,
      transmissionBranchPdf,
      chooseTransmission,
    );

    let interfaceLobe = restirGiSampleDielectricLobe(
      walkHit,
      faceNormal,
      walkAnisotropyFrame.tangent,
      walkAnisotropyFrame.bitangent,
      -refractDir,
      walkRough,
      walkAnisotropy.x,
      walkAnisotropy.y,
      incidentIor,
      targetIor,
      walkHit.side >= 0.0,
      walkLayer,
      transportChannel,
      !chooseTransmission,
      rng,
    );
    interfaceCount = interfaceCount + 1u;
    if (interfaceLobe.valid == 0u) { break; }
    let transmissionPhysicalWeight = select(
      clamp(walkTransmission, 0.0, 1.0),
      1.0,
      pairedPaidExit,
    );
    let interfaceFactor = interfaceLobe.weightRgb /
      vec3f(selectedOpticalBranchPdf) *
      select(1.0, transmissionPhysicalWeight, chooseTransmission);
    prefixTransfer = prefixTransfer * restirGiSuffixDiagonalTransfer(
      interfaceFactor,
    );
    if (!restirGiSuffixTransferFinite(prefixTransfer)) { break; }

    refractDir = interfaceLobe.direction;
    if (!chooseTransmission) {
      // Reflection never crosses the material boundary, never mutates the
      // medium stack, and never pays scalar transmission.
      walkOrigin = walkHitPos;
      continuationSourceFeature = acceptedSourceFeature;
      continue;
    }

    if (!walkBulkMedium) {
      var slabFrontFacing = walkHit.side < 0.0;
      var slabExited = false;
      loop {
        if (interfaceCount >= GLASS_WALK_MAX_INTERFACES) { break; }
        let slabLayer = sampleFaceLayerControls(
          walkHit.indices.w,
          slabFrontFacing,
        );
        let slabRough = faceLayerRoughness(walkMappedRough, slabLayer);
        let slabMappedNormal = applyBumpMapForHit(
          walkHit,
          applyNormalMapForSideForHit(
            walkHit, walkSmoothNormal, slabFrontFacing,
          ),
        );
        let slabAlignedNormal = select(
          -slabMappedNormal,
          slabMappedNormal,
          dot(slabMappedNormal, walkHit.normal) >= 0.0,
        );
        let slabNormal = select(
          -slabAlignedNormal,
          slabAlignedNormal,
          dot(-refractDir, slabAlignedNormal) > 0.0,
        );
        let slabFrame = materialTangentFrameForHit(
          walkHit,
          slabNormal,
          MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
        );
        let slabBranchPdf = represented_bernoulli_probability_f32(0.5);
        let slabChooseTransmission = rand_f32(rng) < slabBranchPdf;
        let slabLobe = restirGiSampleDielectricLobe(
          walkHit,
          slabNormal,
          slabFrame.tangent,
          slabFrame.bitangent,
          -refractDir,
          slabRough,
          walkAnisotropy.x,
          walkAnisotropy.y,
          targetIor,
          incidentIor,
          slabFrontFacing,
          slabLayer,
          transportChannel,
          !slabChooseTransmission,
          rng,
        );
        interfaceCount = interfaceCount + 1u;
        if (slabLobe.valid == 0u) { break; }
        let slabSelectedPdf = select(
          1.0 - slabBranchPdf,
          slabBranchPdf,
          slabChooseTransmission,
        );
        let slabFactor = slabLobe.weightRgb / vec3f(slabSelectedPdf);
        prefixTransfer = prefixTransfer * restirGiSuffixDiagonalTransfer(
          slabFactor,
        );
        if (!restirGiSuffixTransferFinite(prefixTransfer)) { break; }
        refractDir = slabLobe.direction;
        if (slabChooseTransmission) {
          slabExited = true;
          break;
        }
        slabFrontFacing = !slabFrontFacing;
      }
      if (!slabExited) { break; }
      walkOrigin = walkHitPos;
      continuationSourceFeature = acceptedSourceFeature;
      continue;
    }

    if (!entering &&
      mediumDepth > 0u &&
      mediumTransmissionPaid[mediumDepth - 1u] == 0u
    ) {
      // This stack entry came from camera containment rather than a forward
      // entry boundary, so its authored material-transmission factor has not
      // yet been represented on the path.
      mediumTransmissionPaid[mediumDepth - 1u] = 1u;
    }
    if (entering) {
      if (mediumDepth >= GLASS_WALK_MAX_INTERFACES) { break; }
      let walkBeerPacked = textureLoad(
        bvh_beer,
        vec2i(walkCoord),
        0,
      ).r;
      var walkBeer = vec3f(
        f32((walkBeerPacked >> 24u) & 0xffu) / 255.0,
        f32((walkBeerPacked >> 16u) & 0xffu) / 255.0,
        f32((walkBeerPacked >> 8u) & 0xffu) / 255.0,
      );
      mediumIor[mediumDepth] = walkIor;
      mediumTri[mediumDepth] = walkHit.indices.w;
      mediumMaterialId[mediumDepth] = walkBoundaryId;
      mediumInstance[mediumDepth] = walkRepresentedId;
      mediumBeer[mediumDepth] = walkBeer;
      mediumThickness[mediumDepth] = walkThickness;
      mediumThicknessMapScale[mediumDepth] =
        nativeGlassGiThicknessMapScaleForHit(walkHit);
      let walkVertexColor = sampleVertexColorForHit(walkHit);
      mediumAlbedo[mediumDepth] = sampleBaseColorMap(
        walkHit,
        scalarWalkMat.rgb * walkVertexColor.rgb,
      );
      mediumScatter[mediumDepth] = sampleVolumeScatteringControls(
        walkHit.indices.w,
      );
      mediumTransmissionPaid[mediumDepth] = 1u;
      mediumDepth = mediumDepth + 1u;
    } else if (mediumDepth > 0u) {
      mediumDepth = mediumDepth - 1u;
    }
    walkOrigin = walkHitPos;
    continuationSourceFeature = acceptedSourceFeature;
  }

  if (!foundSurface) { return result; }
  result.valid = 1u;
  result.environment = 0u;
  result.hit = walkHit;
  result.pos = walkHitPos;
  result.geoNormal = walkHit.normal;
  result.smoothNormal = walkSmoothNormal;
  result.shadingNormal = walkShadingNormal;
  result.wo = safe_normalize(-refractDir);
  result.transmission = receiverTransmission;
  result.closureMode = receiverClosureMode;
  result.opaqueShare = receiverOpaqueShare;
  result.prefixTransfer = prefixTransfer;
  return result;
}

const NATIVE_GLASS_GI_AREA_SAMPLE_COUNT: u32 = 4u;

fn nativeGlassGiReceiverBrdf(
  receiver: NativeGlassGiReceiver,
  payload: RestirDIMaterialPayload,
  wi: vec3f,
) -> vec3f {
  if (receiver.closureMode == 0u) {
    return evalDirectSurfaceBrdf(
      payload.albedo,
      payload.rough,
      payload.metal,
      payload.specular,
      payload.anisotropy,
      payload.anisotropyTangent,
      payload.anisotropyBitangent,
      payload.iridescence,
      payload.clearcoat,
      payload.sheen,
      payload.sheenRoughness,
      receiver.shadingNormal,
      payload.clearcoatNormal,
      receiver.wo,
      wi,
      receiver.transmission,
      payload.layerTransmission,
      payload.reflectionLayerTransmission,
    );
  }
  // The event walk owns exact dielectric reflection separately. Subtract the
  // persistent reflection closure from the opaque material closure, leaving one
  // unit of base response that can be scaled by the authored (1-t) share.
  let opaqueClosure = evalDirectSurfaceBrdf(
    payload.albedo,
    payload.rough,
    payload.metal,
    payload.specular,
    payload.anisotropy,
    payload.anisotropyTangent,
    payload.anisotropyBitangent,
    payload.iridescence,
    payload.clearcoat,
    payload.sheen,
    payload.sheenRoughness,
    receiver.shadingNormal,
    payload.clearcoatNormal,
    receiver.wo,
    wi,
    0.0,
    payload.layerTransmission,
    payload.reflectionLayerTransmission,
  );
  let reflectionClosure = evalDirectSurfaceBrdf(
    payload.albedo,
    payload.rough,
    payload.metal,
    payload.specular,
    payload.anisotropy,
    payload.anisotropyTangent,
    payload.anisotropyBitangent,
    payload.iridescence,
    payload.clearcoat,
    payload.sheen,
    payload.sheenRoughness,
    receiver.shadingNormal,
    payload.clearcoatNormal,
    receiver.wo,
    wi,
    1.0,
    payload.layerTransmission,
    payload.reflectionLayerTransmission,
  );
  return max(opaqueClosure - reflectionClosure, vec3f(0.0)) *
    receiver.opaqueShare;
}

fn nativeGlassGiAreaEmitterNee(
  receiver: NativeGlassGiReceiver,
  payload: RestirDIMaterialPayload,
  shadowContainingMedia: MaterialShadowContainingMedia,
  rng: ptr<function, u32>,
) -> vec3f {
  let count = min(ubo.emitterCount, sceneEmitterCount());
  if (count == 0u) { return vec3f(0.0); }

  var Lo = vec3f(0.0);
  for (
    var si: u32 = 0u;
    si < NATIVE_GLASS_GI_AREA_SAMPLE_COUNT;
    si = si + 1u
  ) {
    // Stratify both the exact emitter CDF and one triangle coordinate. Every
    // draw still has the represented CDF PMF and uniform-area triangle PDF.
    let emitterXi = (f32(si) + rand_f32(rng)) /
      f32(NATIVE_GLASS_GI_AREA_SAMPLE_COUNT);
    let lid = sampleEmitterIdx(count, emitterXi);
    let emitterPmf = emitterCdfPmf(count, lid);
    if (!(emitterPmf > 0.0)) { continue; }
    let e = sceneLoadEmitter(lid);
    let xi = vec2f(
      (f32(si) + rand_f32(rng)) /
        f32(NATIVE_GLASS_GI_AREA_SAMPLE_COUNT),
      rand_f32(rng),
    );
    let ls = sampleEmitterPoint(e, xi);
    let toL = ls.pos - receiver.pos;
    let dist2 = dot(toL, toL);
    if (!(dist2 > 0.0) || !(ls.area > 0.0)) { continue; }
    let dist = sqrt(dist2);
    let wi = toL / dist;
    let nDotL = dot(receiver.shadingNormal, wi);
    let nlDotL = emitterTriCosineTowardReceiver(e, -wi);
    if (!(nDotL > 0.0) || !(nlDotL > 0.0)) { continue; }

    var shadowTint = vec3f(1.0);
    if (!emitterTriCastShadowDisabled(e)) {
      shadowTint = traceSceneAlphaTintTransmittanceTexturedWithContainingMedia(
        ubo.bvhMode,
        ubo.tlasNodeCount,
        receiver.pos + receiver.geoNormal * walkaroundRayOriginBias(),
        wi,
        max(0.0, dist - walkaroundRayEndMargin()),
        ubo.triIntersectEpsilon,
        bvh_material,
        BVH_MATERIAL_TEX_WIDTH,
        bvh_beer,
        shadowContainingMedia,
        manifoldNeeOwnsMaterialTransmission(),
      );
      if (max(shadowTint.r, max(shadowTint.g, shadowTint.b)) <= 0.0) {
        continue;
      }
    }

    let layeredBrdf = nativeGlassGiReceiverBrdf(receiver, payload, wi);
    let geometry = emitterGeometry(
      nlDotL,
      dist2,
      ubo.emitterDist2Floor,
    );
    let estimatorWeight = ls.area /
      (f32(NATIVE_GLASS_GI_AREA_SAMPLE_COUNT) * emitterPmf);
    let unvolumedContribution = sampleEmitterLeAtXi(e, xi) * shadowTint *
      layeredBrdf * geometry * estimatorWeight;
    let contribution = applyHomogeneousVolumeSingleScatterDirectional(
      unvolumedContribution,
      payload.albedo,
      payload.volumeScattering,
      payload.bulkThickness,
      receiver.shadingNormal,
      receiver.wo,
      wi,
    );
    Lo = Lo + contribution;
  }
  return Lo;
}

fn nativeGlassGiAnalyticNee(
  receiver: NativeGlassGiReceiver,
  payload: RestirDIMaterialPayload,
  transmission: f32,
  shadowContainingMedia: MaterialShadowContainingMedia,
) -> vec3f {
  return lo_analyticNEE(
    receiver.pos,
    receiver.shadingNormal,
    payload.clearcoatNormal,
    receiver.geoNormal,
    shadowContainingMedia,
    payload.albedo,
    payload.rough,
    payload.metal,
    payload.specular,
    payload.anisotropy,
    payload.anisotropyTangent,
    payload.anisotropyBitangent,
    payload.iridescence,
    payload.clearcoat,
    payload.sheen,
    payload.sheenRoughness,
    receiver.wo,
    transmission,
    payload.layerTransmission,
    payload.reflectionLayerTransmission,
    payload.volumeScattering,
    payload.bulkThickness,
    payload.metal >= 1.0,
  );
}

fn nativeGlassGiSunNee(
  fullPx: vec2u,
  receiver: NativeGlassGiReceiver,
  payload: RestirDIMaterialPayload,
  transmission: f32,
  shadowContainingMedia: MaterialShadowContainingMedia,
) -> vec3f {
  return lo_sunNEE(
    fullPx,
    receiver.pos,
    receiver.shadingNormal,
    payload.clearcoatNormal,
    receiver.geoNormal,
    shadowContainingMedia,
    payload.albedo,
    payload.rough,
    payload.metal,
    payload.specular,
    payload.anisotropy,
    payload.anisotropyTangent,
    payload.anisotropyBitangent,
    payload.iridescence,
    payload.clearcoat,
    payload.sheen,
    payload.sheenRoughness,
    receiver.wo,
    transmission,
    payload.layerTransmission,
    payload.reflectionLayerTransmission,
    payload.volumeScattering,
    payload.bulkThickness,
  );
}

fn evaluateNativeGlassGiReceiver(
  fullPx: vec2u,
  receiver: NativeGlassGiReceiver,
  rng: ptr<function, u32>,
) -> vec3f {
  if (receiver.valid == 0u) { return vec3f(0.0); }
  if (receiver.environment != 0u) {
    // This is a camera-background transmission path, not material IBL. The
    // primary material's envMapIntensity scales reflected illumination only.
    let transmittedEnvironment = receiver.prefixTransfer * envRadiance(
      receiver.environmentDirection,
    );
    return transmittedEnvironment;
  }

  let receiverCoord = vec2u(
    receiver.hit.indices.w % BVH_MATERIAL_TEX_WIDTH,
    receiver.hit.indices.w / BVH_MATERIAL_TEX_WIDTH,
  );
  let receiverWord = textureLoad(
    bvh_material,
    vec2i(receiverCoord),
    0,
  ).r;
  let receiverPayload = sampleRestirDIMaterialPayloadForHit(
    receiver.hit,
    receiver.smoothNormal,
    receiver.shadingNormal,
    decodeMaterialColor(receiver.hit.matColorPacked).rgb,
    receiverWord,
    receiver.wo,
  );
  if (decodeIsUnlitMaterial(receiverWord)) {
    return receiver.prefixTransfer *
      (receiverPayload.albedo * receiverPayload.layerTransmission);
  }

  let receiverShadowOrigin = receiver.pos +
    receiver.geoNormal * walkaroundRayOriginBias();
  let receiverShadowContainingMedia = materialShadowClassifyContainingMedia(
    ubo.bvhMode,
    ubo.tlasNodeCount,
    receiverShadowOrigin,
    receiver.wo,
    ubo.triIntersectEpsilon,
    bvh_material,
    BVH_MATERIAL_TEX_WIDTH,
    bvh_beer,
  );

  var receiverAnalytic = nativeGlassGiAnalyticNee(
    receiver,
    receiverPayload,
    receiver.transmission,
    receiverShadowContainingMedia,
  );
  var receiverSun = nativeGlassGiSunNee(
    fullPx,
    receiver,
    receiverPayload,
    receiver.transmission,
    receiverShadowContainingMedia,
  );
  if (receiver.closureMode != 0u) {
    receiverAnalytic = max(
      nativeGlassGiAnalyticNee(
        receiver, receiverPayload, 0.0, receiverShadowContainingMedia,
      ) -
        nativeGlassGiAnalyticNee(
          receiver, receiverPayload, 1.0, receiverShadowContainingMedia,
        ),
      vec3f(0.0),
    ) * receiver.opaqueShare;
    receiverSun = max(
      nativeGlassGiSunNee(
        fullPx, receiver, receiverPayload, 0.0,
        receiverShadowContainingMedia,
      ) -
        nativeGlassGiSunNee(
          fullPx, receiver, receiverPayload, 1.0,
          receiverShadowContainingMedia,
        ),
      vec3f(0.0),
    ) * receiver.opaqueShare;
  }
  let receiverArea = nativeGlassGiAreaEmitterNee(
    receiver,
    receiverPayload,
    receiverShadowContainingMedia,
    rng,
  );
  let receiverLightMapShare = select(
    1.0 - clamp(receiver.transmission, 0.0, 1.0),
    receiver.opaqueShare,
    receiver.closureMode != 0u,
  );
  let receiverLocalSource = restir_gi_surface_emission_for_hit(receiver.hit) +
    receiverPayload.albedo * INV_PI * sampleLightMap(receiver.hit) *
      receiverLightMapShare;
  let receiverLocalDirect = applyHomogeneousVolumeSingleScatter(
    receiverLocalSource * receiverPayload.layerTransmission,
    receiverPayload.albedo,
    receiverPayload.volumeScattering,
    receiverPayload.bulkThickness,
    receiver.shadingNormal,
    receiver.wo,
  );
  let receiverDirect = receiverLocalDirect + receiverAnalytic + receiverSun +
    receiverArea;
  let transmittedReceiverDirect = receiver.prefixTransfer * receiverDirect;

  var reservoir = emptyReservoirGI();
  var wrs = representedWrsInit();
  reservoir.xv = receiver.pos;
  reservoir.nv = receiver.shadingNormal;
  let historyEpoch = bitcast<u32>(ubo.sunAngular.y);

  // Pure cosine sampling is deliberately used here. The PPG storage bindings
  // are not part of shade's portable bind contract; cosine sampling has full
  // support over the receiver hemisphere and remains an unbiased estimator.
  let tierRaw = textureLoad(gi_tier, vec2i(fullPx), 0).r;
  let tier = clamp(tierRaw, 1u, 4u);
  let candidateCount = 8u * tier / 2u;

  for (var i: u32 = 0u; i < candidateCount; i = i + 1u) {
    let wi = sampleCosineHemisphere(receiver.shadingNormal, rng);
    let cosTheta = max(0.0, dot(receiver.shadingNormal, wi));
    if (!reservoirGiFinite(cosTheta) || !(cosTheta > 0.0)) {
      recordInvalidReservoirGICandidate(
        &reservoir,
        GI_SAMPLE_SURFACE,
        historyEpoch,
      );
      continue;
    }

    let bounceRay = Ray(
      receiver.pos + receiver.geoNormal * walkaroundRayOriginBias(),
      wi,
    );
    let bounceHit = traceSceneFirstHitAlphaMaskTextured(
      ubo.bvhMode,
      ubo.tlasNodeCount,
      bounceRay,
      ubo.triIntersectEpsilon,
      bvh_material,
      BVH_MATERIAL_TEX_WIDTH,
      ubo.frameSeed ^
        (fullPx.x * 0x9e3779b9u) ^
        (fullPx.y * 0x85ebca6bu) ^
        (i * 0xc2b2ae35u) ^
        0x474c5358u,
    );

    var xs: vec3f;
    var ns: vec3f;
    var suffixLo: vec3f;
    var sampleKind = GI_SAMPLE_ENVIRONMENT;
    if (bounceHit.didHit) {
      sampleKind = GI_SAMPLE_SURFACE;
      xs = bounceRay.origin + wi * bounceHit.dist;
      ns = bounceHit.normal;
      let smoothNs = restir_gi_smooth_normal_for_hit(
        bounceHit,
        bounceHit.normal,
      );
      let shadingNs = applyBumpMapForHit(
        bounceHit,
        applyNormalMapForHit(bounceHit, smoothNs),
      );
      let irrAtXs = min(
        sampleDDGIAtPoint(xs, shadingNs),
        vec3f(ubo.restirGiIrrClamp),
      );
      let xsCoord = vec2u(
        bounceHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
        bounceHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
      );
      let xsWord = textureLoad(
        bvh_material,
        vec2i(xsCoord),
        0,
      ).r;
      let xsPayload = sampleRestirGIHitMaterialForHit(
        bounceHit,
        smoothNs,
        shadingNs,
        irrAtXs,
        wi,
        xsWord,
      );
      suffixLo = xsPayload.Lo;
      let xsTransmission = clamp(xsPayload.transmission, 0.0, 1.0);
      if (
        !decodeIsUnlitMaterial(xsWord) &&
        xsTransmission > 0.0
      ) {
        // The first suffix interface may itself be another dielectric. Continue
        // with the same bounded nested-medium/TIR state machine as ordinary RIS
        // instead of treating that second glass surface as a terminal DDGI proxy.
        suffixLo = traceRestirGiDielectricSuffix(
          bounceHit,
          xs,
          bounceHit.dist,
          shadingNs,
          wi,
          xsPayload.emissionLo,
          xsPayload.opaqueLo,
          xsTransmission,
          rng,
        );
      }
    } else {
      xs = receiver.pos + wi * walkaroundReconnectMaxDistance();
      ns = -wi;
      suffixLo = walkaroundScaleEnvironmentRadiance(
        envRadiance(wi),
        receiverPayload.envMapIntensity,
      );
    }

    // The suffix first-hit walk already realized partial-alpha occupancy. The
    // receiver response is evaluated in full RGB, then the camera-prefix
    // Fresnel/transmission/Beer throughput is applied exactly once after the
    // receiver's own face layer and homogeneous-volume response.
    let receiverBrdf = nativeGlassGiReceiverBrdf(
      receiver,
      receiverPayload,
      wi,
    );
    let receiverResponse = applyHomogeneousVolumeSingleScatterDirectional(
      suffixLo * receiverBrdf,
      receiverPayload.albedo,
      receiverPayload.volumeScattering,
      receiverPayload.bulkThickness,
      receiver.shadingNormal,
      receiver.wo,
      wi,
    );
    let receiverContribution = receiver.prefixTransfer * receiverResponse;
    let logPHat = reservoirGiLogPositive(luminance(receiverContribution));
    let logPSrc = reservoirGiLogPositive(cosTheta * INV_PI);
    if (!reservoirGiValidLog(logPHat) || !reservoirGiValidLog(logPSrc)) {
      recordInvalidReservoirGICandidate(
        &reservoir,
        sampleKind,
        historyEpoch,
      );
      continue;
    }
    let logWeight = logPHat - logPSrc;
    if (!reservoirGiValidLog(logWeight)) {
      recordInvalidReservoirGICandidate(
        &reservoir,
        sampleKind,
        historyEpoch,
      );
      continue;
    }
    updateReservoirGIWithMetadata(
      &reservoir,
      &wrs,
      xs,
      ns,
      receiverContribution,
      sampleKind,
      wi,
      0u,
      logPHat,
      1.0,
      historyEpoch,
      logWeight,
      rng,
    );
  }

  finaliseGIReservoirFromNativeWrs(
    &reservoir,
    wrs,
    ubo.restirGiWCap,
  );
  var indirect = vec3f(0.0);
  if (restirShadeValidLog(reservoir.logW)) {
    indirect = restirShadeExp2Clamped3(
      restirShadeLogProduct3(
        reservoir.logW,
        reservoir.Lo,
        vec3f(1.0),
        vec3f(1.0),
      ),
    );
  }
  let scaledIndirect = indirect * ubo.glassMixScale;
  let clampedIndirect = min(
    scaledIndirect,
    ubo.indirectFireflyClamp * ubo.glassMixScale,
  );
  return transmittedReceiverDirect + clampedIndirect;
}

fn lo_transmittedGI(
  fullPx: vec2u,
  isGlass: bool,
  primaryHit: IntersectionResult,
  primaryRay: Ray,
  primaryPos: vec3f,
  primarySmoothNormal: vec3f,
  primaryNormal: vec3f,
  primaryGeoNormal: vec3f,
  primaryTransmission: f32,
  primaryAlbedo: vec3f,
  rng: ptr<function, u32>,
) -> vec3f {
  if (!isGlass) { return vec3f(0.0); }

  // Hero-channel geometry is evaluated for all three RGB basis wavelengths.
  // Each copy starts from the same RNG state, correlating its VNDF draws while
  // allowing Snell direction, BTDF PDF/Jacobian, intersection sequence, and
  // terminal radiance to diverge with the authored dispersion IOR.
  let sharedRng = (*rng);
  var rngR = sharedRng;
  var rngG = sharedRng;
  var rngB = sharedRng;
  let receiverR = traceNativeGlassGiReceiver(
    primaryHit,
    primaryRay,
    primaryPos,
    primarySmoothNormal,
    primaryNormal,
    primaryGeoNormal,
    primaryTransmission,
    primaryAlbedo,
    0u,
    &rngR,
  );
  let receiverG = traceNativeGlassGiReceiver(
    primaryHit,
    primaryRay,
    primaryPos,
    primarySmoothNormal,
    primaryNormal,
    primaryGeoNormal,
    primaryTransmission,
    primaryAlbedo,
    1u,
    &rngG,
  );
  let receiverB = traceNativeGlassGiReceiver(
    primaryHit,
    primaryRay,
    primaryPos,
    primarySmoothNormal,
    primaryNormal,
    primaryGeoNormal,
    primaryTransmission,
    primaryAlbedo,
    2u,
    &rngB,
  );
  let radianceR = evaluateNativeGlassGiReceiver(fullPx, receiverR, &rngR);
  let radianceG = evaluateNativeGlassGiReceiver(fullPx, receiverG, &rngG);
  let radianceB = evaluateNativeGlassGiReceiver(fullPx, receiverB, &rngB);
  (*rng) = rngR ^ (rngG * 0x9e3779b9u) ^ (rngB * 0x85ebca6bu);
  return vec3f(radianceR.r, radianceG.g, radianceB.b);
}
`;
