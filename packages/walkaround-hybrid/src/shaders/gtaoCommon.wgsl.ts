/**
 * GTAOUniforms struct — shared between the GTAO compute pass (gtao.wgsl.ts)
 * and the bilateral upsample pass (gtaoUpsample.wgsl.ts). Both shaders bind
 * the same UBO buffer; the shared struct declaration is extracted here so it
 * has exactly one source of truth.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const GTAO_COMMON_WGSL = /* wgsl */ `
struct GTAOUniforms {
  // tan(fov/2) along screen height; used to convert pixel offsets to view-space
  // angles for the horizon integration. Packed from the host's camera.
  tanFovHalf: f32,
  // Sampling radius in screen pixels (full-res space). 32 px is a typical
  // contact-AO radius; larger broadens the AO to medium-range occlusion.
  radiusPx:   f32,
  // AO intensity exponent. ao = pow(ao_raw, intensity). 1.0 = linear,
  // 2.0 = stronger contact darkening.
  intensity:  f32,
  // Maximum depth difference (world units) to consider a sample for the
  // horizon test. Larger gap = treat as background → no occlusion. Prevents
  // halos around foreground silhouettes.
  depthThresh: f32,
  // Audit B3: bilateral upsample depth-weight sigma (world units). Used by
  // gtaoUpsample.wgsl for the joint-bilateral half→full upsample. Previously
  // hardcoded to 4.0 in the shader (σ ≈ 0.25 m), which was Cornell-scale-only.
  // Hosts should set ~(sceneDiagonal * 0.01) so the half-life of the depth
  // weight is ~1% of the scene's longest axis.
  bilateralDepthSigma: f32,
  // AO compute downscale factor (integer, stored as f32). 2 = half-res
  // (gtaoMode 'on'); 4 = quarter-res (gtaoMode 'quarter'). Both gtao.wgsl
  // and gtaoUpsample.wgsl read this to map between AO-grid and full-res coords,
  // replacing the prior hardcoded div-by-2 / mul-by-2. Was the inert _pad0 slot.
  gtaoDownscale: f32,
  // Pad to 32 bytes (8-element struct) for WebGPU 16-byte UBO alignment.
  _pad1: f32,
  _pad2: f32,
  // World-to-view transform. gNormalDepth stores world-space normals while
  // GTAO constructs its horizon slices in view space; keeping both operands in
  // one space is required for camera rotations to leave AO invariant.
  viewMatrix: mat4x4f,
};
`;

/** GTAO shared uniforms struct — required by both gtao and gtaoUpsample. */
export const GTAO_COMMON_MODULE: WgslModule = {
  name: 'gtaoCommon',
  source: GTAO_COMMON_WGSL,
  requires: [],
};
