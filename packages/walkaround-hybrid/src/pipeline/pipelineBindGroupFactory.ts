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
): PerFrameBindGroups {
  const { common, restirDI, restirGI, gtao } = resources;
  return {
    frame: buildFrameBindGroup(device, cache, {
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
    }),
    scene: buildSceneBindGroup(device, cache, scene),
    ubo: buildUboBindGroup(
      device,
      cache,
      common.uboBuffer,
      gtao.aoFullTexture.createView(),
      common.tierTexture.createView(),
    ),
    hybridLayers: ddgi.buildBindGroup(device, cache, resources),
  };
}

export function buildCompositePresentBindGroup(
  device: GPUDevice,
  cache: BGLCache,
  resolvedTexture: GPUTexture,
  compositeSampler: GPUSampler,
): GPUBindGroup {
  return buildCompositeBindGroup(
    device,
    cache,
    resolvedTexture.createView(),
    compositeSampler,
  );
}
