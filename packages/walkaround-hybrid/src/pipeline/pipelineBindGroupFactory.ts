/**
 * Per-frame bind groups shared by all compute passes (W4b).
 *
 * Centralises the frame / scene / ubo / hybrid-layers construction that
 * {@link WalkaroundGPUPipeline.renderFrame} previously inlined.
 */

import type { BGLCache } from './bindGroupLayouts.js';
import {
  buildFrameBindGroup,
  buildSceneBindGroup,
  buildUboBindGroup,
  buildCompositeBindGroup,
} from './bindGroupBuilders.js';
import type { SceneBindGroupResources } from './BvhBufferHost.js';
import type { DDGIBindingState } from './DDGIBindingState.js';
import type { FrameResources } from './resourceManager.js';
import type { PipelineResourceCache } from './PipelineResourceCache.js';

export interface PerFrameBindGroups {
  readonly frame: GPUBindGroup;
  readonly scene: GPUBindGroup;
  readonly ubo: GPUBindGroup;
  readonly hybridLayers: GPUBindGroup;
}

export function buildPerFrameBindGroups(
  device: GPUDevice,
  cache: BGLCache,
  resources: FrameResources,
  scene: SceneBindGroupResources,
  ddgi: DDGIBindingState,
  placeholderView: GPUTextureView,
  resourceCache?: PipelineResourceCache,
): PerFrameBindGroups {
  const { common, restirDI, restirGI, gtao } = resources;
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
  }, resourceCache);
  const buildScene = (): GPUBindGroup => buildSceneBindGroup(device, cache, scene);
  const aoFullView = resourceCache?.textureView(gtao.aoFullTexture) ?? gtao.aoFullTexture.createView();
  const tierView = resourceCache?.textureView(common.tierTexture) ?? common.tierTexture.createView();
  const buildUbo = (): GPUBindGroup => buildUboBindGroup(
    device,
    cache,
    common.uboBuffer,
    aoFullView,
    tierView,
  );
  const sceneKey = [
    scene.bvhNodesBuffer,
    scene.bvhIndexBuffer,
    scene.bvhPositionBuffer,
    scene.emitterBuffer,
    scene.emitterCdfBuffer,
    scene.bvhBeerTextureView,
    scene.bvhNormalBuffer,
    scene.bvhEmissiveTextureView,
    scene.tlasNodesBuffer,
    scene.tlasInstanceIndicesBuffer,
    scene.tlasBlasRootsBuffer,
    scene.tlasInstanceWorldToLocalBuffer,
    scene.tlasInstanceLocalToWorldBuffer,
  ] as const;
  return {
    frame: resourceCache?.bindGroup('per-frame:frame', [
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
    ], buildFrame) ?? buildFrame(),
    scene: resourceCache?.bindGroup('per-frame:scene', sceneKey, buildScene) ?? buildScene(),
    ubo: resourceCache?.bindGroup('per-frame:ubo', [
      common.uboBuffer,
      gtao.aoFullTexture,
      common.tierTexture,
    ], buildUbo) ?? buildUbo(),
    hybridLayers: ddgi.buildBindGroup(device, cache, resources, resourceCache),
  };
}

export function buildCompositePresentBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  resolvedTexture: GPUTexture,
  compositeSampler: GPUSampler,
  resourceCache?: PipelineResourceCache,
): GPUBindGroup {
  const build = (): GPUBindGroup => buildCompositeBindGroup(
    device,
    cache,
    resourceCache?.textureView(resolvedTexture) ?? resolvedTexture.createView(),
    compositeSampler,
  );
  return resourceCache?.bindGroup(
    'present:composite',
    [resolvedTexture, compositeSampler],
    build,
  ) ?? build();
}
