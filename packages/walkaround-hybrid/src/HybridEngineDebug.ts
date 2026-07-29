/**
 * Debug-introspection surface for {@link HybridEngine} (W4e).
 */
import type { EngineDebugSurface, GpuMemoryBreakdown, Scene } from '@vitrum/core';
import { packBvhNodesForDebug } from './debug/packBvhNodesForDebug.js';
import { pickPrimitiveCpu, type PickCamera } from '@vitrum/shared-bvh';
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
   * GPU memory breakdown delegate — backed by
   * `WalkaroundGPUPipeline.getMemoryBreakdown()` (I3.2). Returns null before
   * the pipeline initialises.
   */
  getMemoryBreakdown: () => GpuMemoryBreakdown | null;
  denoiserPassEnabled: () => boolean;
  setDenoiserPassEnabled: (enabled: boolean) => void;
  setPipelineDenoiserPassEnabled: (enabled: boolean) => void;
  /** Retained core scene for CPU click-to-pick (`pickPrimitive`, T3.G). */
  pickScene: () => Scene | null;
  /** Last-frame camera for click-to-pick ray unprojection. */
  pickCamera: () => PickCamera | null;
  /** Canvas pixel size for the screen→NDC mapping. */
  pickSize: () => { width: number; height: number };
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
        total:    textures.hdrTotalTexture,
      };
    },
    isDenoiserEnabled: deps.denoiserPassEnabled,
    setDenoiserEnabled: (enabled: boolean): void => {
      deps.setDenoiserPassEnabled(enabled);
      deps.setPipelineDenoiserPassEnabled(enabled);
    },
    estimatedGpuMemoryBytes: deps.getMemoryBreakdown,
    // T3.G click-to-pick: CPU ray-cast of pixel (x,y) against the retained core
    // scene using the last-frame camera. Returns null before the first frame
    // (no camera), before a scene is set, or on a miss.
    pickPrimitive: (x: number, y: number): string | null => {
      const scene = deps.pickScene();
      const camera = deps.pickCamera();
      if (scene == null || camera == null) return null;
      const { width, height } = deps.pickSize();
      return pickPrimitiveCpu(scene, camera, x, y, width, height);
    },
  };
}
