/**
 * DDGIBindingState — owns the DDGI atlas binding state that
 * {@link WalkaroundGPUPipeline} hands to the shade / GI-RIS passes via the
 * `hybridLayers` bind group.
 *
 * Extracted from {@link WalkaroundGPUPipeline} in the 2026-05-18 refactor
 * sweep. Three pieces of state move here:
 *
 *  - `_irrTex` / `_visTex` — the host-supplied irradiance + visibility
 *    atlases (or `null` when DDGI is disabled, in which case the placeholder
 *    1×1 textures from {@link FrameResources.ddgi} are bound instead).
 *  - `_placeholderUBO` — a cached `Float32Array(16)` so that
 *    {@link setInputs}(`null`) doesn't allocate a new buffer every time a
 *    host toggles DDGI off (HOT-1 fix from the 2026-05-16 sweep).
 *
 * Public surface mirrors the pre-refactor pipeline methods: `setInputs` is
 * called from `HybridLayeredStage`, and `buildBindGroup` is called once per
 * frame from `renderFrame`.
 */

import { buildHybridLayersBindGroup } from './bindGroupBuilders.js';
import {
  buildDDGIPlaceholderUBO,
  type FrameResources,
} from './resourceManager.js';
import type { BGLCache } from './bindGroupLayouts.js';

export interface DDGISetInputs {
  irradianceTex: GPUTexture;
  visibilityTex: GPUTexture;
  gridParams: ArrayBuffer;
}

export class DDGIBindingState {
  private readonly _device: GPUDevice;
  /** DDGI inputs (layered hybrid). Null → placeholder textures. */
  private _irrTex: GPUTexture | null = null;
  private _visTex: GPUTexture | null = null;
  /** Cached DDGI placeholder UBO — reused by setInputs(null) so we don't
   *  allocate a fresh Float32Array(16) every frame when DDGI is disabled.
   *  Populated lazily on first setInputs(null) call. */
  private _placeholderUBO: Float32Array | null = null;

  constructor(device: GPUDevice) {
    this._device = device;
  }

  /**
   * Bind a DDGI atlas. Pass `null` to revert to placeholder (no-DDGI
   * fallback). The shade pass continues to render with hardcoded sky color
   * when isDDGIWired() returns false (dimsX ≤ 1 in the placeholder UBO).
   *
   * Caller (HybridLayeredStage) provides:
   *  - irradianceTex / visibilityTex: GPUTexture with TEXTURE_BINDING usage.
   *  - gridParams: 64-byte ArrayBuffer matching the WGSL DDGIGridUniform
   *    layout (origin vec3 + spacing f32 + dims vec3u + pad u32 +
   *    irradianceAtlasW/H + visibilityAtlasW/H).
   */
  setInputs(inputs: DDGISetInputs | null, frameResources: FrameResources): void {
    if (inputs === null) {
      this._irrTex = null;
      this._visTex = null;
      // Restore placeholder UBO (dims=1×1×1) so shade.wgsl's
      // isDDGIWired() check returns false and Lo_ddgi drops to zero.
      // Cache the placeholder to avoid allocating Float32Array(16) every
      // frame when DDGI is disabled (HOT-1 fix).
      if (this._placeholderUBO === null) {
        this._placeholderUBO = buildDDGIPlaceholderUBO();
      }
      this._device.queue.writeBuffer(frameResources.ddgi.ddgiUboBuffer, 0, this._placeholderUBO.buffer);
    } else {
      this._irrTex = inputs.irradianceTex;
      this._visTex = inputs.visibilityTex;
      if (inputs.gridParams.byteLength > 0) {
        this._device.queue.writeBuffer(frameResources.ddgi.ddgiUboBuffer, 0, inputs.gridParams);
      }
    }
  }

  /**
   * Build the hybrid-layers bind group for the current frame, falling back
   * to the placeholder textures when no host-supplied DDGI atlases have
   * been bound. Delegates to {@link buildHybridLayersBindGroup} so the
   * caller layout (binding slots 0..3) stays in one place.
   */
  buildBindGroup(
    device: GPUDevice,
    bglCache: BGLCache,
    frameResources: FrameResources,
  ): GPUBindGroup {
    return buildHybridLayersBindGroup(device, bglCache, {
      ddgiIrrTex:             this._irrTex,
      ddgiVisTex:             this._visTex,
      ddgiPlaceholderRgba16f: frameResources.ddgi.ddgiPlaceholderRgba16f,
      ddgiPlaceholderRg16f:   frameResources.ddgi.ddgiPlaceholderRg16f,
      nearestSampler:         frameResources.common.nearestSampler,
      ddgiUboBuffer:          frameResources.ddgi.ddgiUboBuffer,
    });
  }

  /** Release held atlas references (the host owns those GPUTextures; we only
   *  drop our references). The cached placeholder UBO is a plain
   *  Float32Array — no GPU resource to release. */
  dispose(): void {
    this._irrTex = null;
    this._visTex = null;
    this._placeholderUBO = null;
  }
}
