/**
 * PipelineDebugTextures — narrow interface exposing only the GPU texture
 * handles that the debug overlays actually read.
 *
 * Extracted from the full {@link FrameResources} god-struct (complexity sweep
 * 2026-06-02) so that debug consumers (`HybridEngineDebug`, `@vitrum/dev`
 * overlays) are decoupled from the 8-sub-struct FrameResources shape.
 *
 * Fields chosen by reading `HybridEngineDebug.giSignalTextures`: it accesses
 * exactly the direct, indirect, total, and AO split targets.
 *
 * Returned handles are owned by the pipeline (caller MUST NOT destroy them);
 * they are invalidated on the next `setScene()` / `dispose()` / `resize()`.
 */
export interface PipelineDebugTextures {
  /**
   * Direct-channel HDR render target (`rgba16float`, full-res).
   * Written by the shade pass; overwritten by the denoiser chain.
   * Corresponds to `FrameResources.common.hdrColorTexture`.
   */
  hdrColorTexture: GPUTexture | null;

  /**
   * Indirect-channel HDR render target (`rgba16float`, full-res).
   * Written by the shade pass as `Lo_indirect * ao`; processed by
   * the indirect-combine pass.
   * Corresponds to `FrameResources.common.hdrIndirectTexture`.
   */
  hdrIndirectTexture: GPUTexture | null;

  /** Combined direct + indirect HDR signal before final denoised composition. */
  hdrTotalTexture: GPUTexture | null;

  /**
   * Full-resolution GTAO ambient-occlusion factor (`rgba16float`).
   * Written by `gtaoUpsampleMain`; sampled by shade to modulate diffuse terms.
   * Corresponds to `FrameResources.gtao.aoFullTexture`.
   */
  aoFullTexture: GPUTexture | null;
}
