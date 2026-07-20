/**
 * Per-frame bind groups shared by all compute passes (W4b).
 *
 * Centralises the frame / scene / ubo / hybrid-layers construction that
 * {@link WalkaroundGPUPipeline.renderFrame} previously inlined.
 */

import type { BGLCache } from './bindGroupLayouts.js';
import {
  buildFrameBindGroup,
  buildRisGiFrameBindGroup,
  buildSceneBindGroup,
  buildUboBindGroup,
  buildCompositeBindGroup,
} from './bindGroupBuilders.js';
import type { SceneBindGroupResources } from './BvhBufferHost.js';
import type { OptionalSubsystemBindingState } from './OptionalSubsystemBindingState.js';
import type { FrameResources } from './resourceManager.js';
import { cachedBindGroup, type PipelineResourceCache } from './PipelineResourceCache.js';

export interface PerFrameBindGroups {
  readonly frame: GPUBindGroup;
  readonly risGiFrame: GPUBindGroup;
  readonly scene: GPUBindGroup;
  readonly ubo: GPUBindGroup;
  readonly hybridLayers: GPUBindGroup;
  readonly shadeHybridLayers: GPUBindGroup;
}

export function buildPerFrameBindGroups(
  device: GPUDevice,
  cache: BGLCache,
  resources: FrameResources,
  scene: SceneBindGroupResources,
  ddgi: OptionalSubsystemBindingState,
  placeholderView: GPUTextureView,
  resourceCache?: PipelineResourceCache,
): PerFrameBindGroups {
  const { common, restirDI, restirGI, gtao, svgf } = resources;
  const buildFrame = (): GPUBindGroup => buildFrameBindGroup(device, cache, {
    placeholderView,
    reservoirCurrentBuffer: restirDI.reservoirCurrentBuffer,
    reservoirPreviousBuffer: restirDI.reservoirPreviousBuffer,
    reservoirSpatialBuffer: restirDI.reservoirSpatialBuffer,
    hdrColorTexture: common.hdrColorTexture,
    nearestSampler: common.nearestSampler,
    gNormalDepthTexture: common.gNormalDepthTexture,
    reservoirGiCurrentBuffer: restirGI.reservoirGiCurrentBuffer,
    hdrIndirectTexture: common.hdrIndirectTexture,
    hdrTotalTexture: common.hdrTotalTexture,
    albedoTexture: common.albedoTexture,
    svgfCurrentObjectIdTexture: svgf.svgfCurrentObjectIdTexture,
  }, resourceCache);
  const buildScene = (): GPUBindGroup => buildSceneBindGroup(device, cache, scene);
  const buildRisGiFrame = (): GPUBindGroup => buildRisGiFrameBindGroup(
    device,
    cache,
    common.gNormalDepthTexture,
    restirGI.reservoirGiCurrentBuffer,
    resourceCache,
  );
  const aoFullView = resourceCache?.textureView(gtao.aoFullTexture) ?? gtao.aoFullTexture.createView();
  const tierView = resourceCache?.textureView(common.tierTexture) ?? common.tierTexture.createView();
  const buildUbo = (): GPUBindGroup => buildUboBindGroup(
    device,
    cache,
    common.uboBuffer,
    aoFullView,
    tierView,
  );
  // The cache key MUST list EVERY resource the scene bind group binds, in any
  // order — a missing entry is a stale-binding bug: the memoized group keeps
  // referencing a DESTROYED buffer/texture after an update path swaps it.
  // The Wave A/B additions (analytic lights @13, rough-metal @14, env @15-19,
  // material texture atlas @20-21, tangent texture @22, beer @5, emissive @12,
  // normals @11) recreate their resources in
  // `updateAnalyticLights` / `updateEmitters` / `updateEnvironment`
  // (BvhBufferHost) — destroy() + fresh upload changes identity, so listing
  // them here makes those refresh paths auto-invalidate the scene group on the
  // next frame without an explicit cache.clear(). `envSampler`/`envParamsBuffer`
  // are reused across env swaps (stable identity) but are listed for
  // completeness so a future lifecycle change can't silently desync.
  const sceneKey = [
    scene.bvhNodesBuffer,
    scene.bvhIndexBuffer,
    scene.bvhPositionBuffer,
    scene.emitterBuffer,
    scene.emitterCdfBuffer,
    scene.bvhBeerTextureView,        // 5
    scene.tlasNodesBuffer,
    scene.tlasInstanceIndicesBuffer,
    scene.tlasBlasRootsBuffer,
    scene.tlasInstanceWorldToLocalBuffer,
    scene.tlasInstanceLocalToWorldBuffer,
    scene.bvhNormalBuffer,           // 11
    scene.bvhEmissiveTextureView,    // 12
    scene.analyticLightsTextureView, // 13 — recreated by updateAnalyticLights
    scene.bvhRoughMetalTextureView,  // 14 — B1
    scene.envMapTextureView,         // 15 — B3, recreated by updateEnvironment
    scene.envMarginalTextureView,    // 16
    scene.envConditionalTextureView, // 17
    scene.envSampler,                // 18
    scene.envParamsBuffer,           // 19
    scene.materialTextureAtlasView,  // 20 — Phase-3D baseColorMap atlas
    scene.baseColorMapMetaTextureView, // 21
    scene.bvhTangentTextureView,     // 22 — authored/generated tangent.xyzw
    scene.bvhVertexColorTextureView, // 23 — COLOR_0 vertex colors
  ] as const;
  return {
    frame: cachedBindGroup(resourceCache, 'per-frame:frame', [
      placeholderView,
      restirDI.reservoirCurrentBuffer,
      restirDI.reservoirPreviousBuffer,
      restirDI.reservoirSpatialBuffer,
      common.hdrColorTexture,
      common.nearestSampler,
      common.gNormalDepthTexture,
      restirGI.reservoirGiCurrentBuffer,
      common.hdrIndirectTexture,
      common.hdrTotalTexture,
      common.albedoTexture,
      svgf.svgfCurrentObjectIdTexture,
    ], buildFrame),
    risGiFrame: cachedBindGroup(resourceCache, 'per-frame:ris-gi-frame', [
      common.gNormalDepthTexture,
      restirGI.reservoirGiCurrentBuffer,
    ], buildRisGiFrame),
    scene: cachedBindGroup(resourceCache, 'per-frame:scene', sceneKey, buildScene),
    ubo: cachedBindGroup(resourceCache, 'per-frame:ubo', [
      common.uboBuffer,
      gtao.aoFullTexture,
      common.tierTexture,
    ], buildUbo),
    hybridLayers: ddgi.buildBindGroup(device, cache, resources, resourceCache),
    shadeHybridLayers: ddgi.buildShadeBindGroup(device, cache, resources, resourceCache),
  };
}

export function buildCompositePresentBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  resolvedTexture: GPUTexture,
  compositeSampler: GPUSampler,
  compositeUbo: GPUBuffer,
  resourceCache?: PipelineResourceCache,
): GPUBindGroup {
  const build = (): GPUBindGroup => buildCompositeBindGroup(
    device,
    cache,
    resourceCache?.textureView(resolvedTexture) ?? resolvedTexture.createView(),
    compositeSampler,
    compositeUbo,
  );
  return cachedBindGroup(
    resourceCache,
    'present:composite',
    [resolvedTexture, compositeSampler, compositeUbo],
    build,
  );
}
