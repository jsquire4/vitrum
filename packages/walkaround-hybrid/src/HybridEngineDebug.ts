/**
 * Debug-introspection surface for {@link HybridEngine} (W4e).
 */
import type { EngineDebugSurface, GpuMemoryBreakdown } from '@vitrum/core';
import { packBvhNodesForDebug } from './debug/packBvhNodesForDebug.js';
import { estimateFrameResourcesMemory } from './pipeline/gpuMemoryEstimate.js';
import type { FrameResources } from './pipeline/resourceManager.js';
import type { PipelineDebugTextures } from './pipeline/PipelineDebugTextures.js';

export interface HybridEngineDebugDeps {
  device: () => GPUDevice | null;
  readAtlas: () => { irradiance: GPUTexture; visibility: GPUTexture } | null;
  bvhNodesCpu: () => ArrayBuffer | null | undefined;
  /**
   * Narrow debug-texture handles for the GI-signal overlay
   * (`giSignalTextures`). Backed by `WalkaroundGPUPipeline.getDebugTextures()`.
   */
  debugTextures: () => PipelineDebugTextures | null;
  /**
   * Full frame-resources struct for the GPU memory estimator
   * (`estimatedGpuMemoryBytes`). Backed by the `@internal`
   * `WalkaroundGPUPipeline.frameResources` getter.
   */
  pipelineResources: () => FrameResources | null;
  denoiserPassEnabled: () => boolean;
  setDenoiserPassEnabled: (enabled: boolean) => void;
  setPipelineDenoiserPassEnabled: (enabled: boolean) => void;
}

export function createHybridEngineDebugSurface(deps: HybridEngineDebugDeps): EngineDebugSurface {
  return {
    device: deps.device,
    atlasTexture: (): GPUTexture | null => deps.readAtlas()?.irradiance ?? null,
    visibilityAtlasTexture: (): GPUTexture | null => deps.readAtlas()?.visibility ?? null,
    bvhNodes: (): Float32Array | null => {
      const buf = deps.bvhNodesCpu();
      if (buf == null) return null;
      return packBvhNodesForDebug(buf);
    },
    giSignalTextures: () => {
      const textures = deps.debugTextures();
      if (textures == null) return null;
      return {
        direct:   textures.hdrColorTexture,
        indirect: textures.hdrIndirectTexture,
        ao:       textures.aoFullTexture,
        total:    null,
      };
    },
    isDenoiserEnabled: deps.denoiserPassEnabled,
    setDenoiserEnabled: (enabled: boolean): void => {
      deps.setDenoiserPassEnabled(enabled);
      deps.setPipelineDenoiserPassEnabled(enabled);
    },
    estimatedGpuMemoryBytes: (): GpuMemoryBreakdown | null => {
      const res = deps.pipelineResources();
      if (res == null) return null;
      return estimateFrameResourcesMemory(res);
    },
  };
}
