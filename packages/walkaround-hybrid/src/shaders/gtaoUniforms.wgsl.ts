/**
 * Canonical GTAOUniforms struct WGSL — single source for both the GTAO
 * compute pass (gtao.wgsl.ts) and the bilateral upsample pass
 * (gtaoUpsample.wgsl.ts). Both shaders bind the same uboBuffer, so the
 * struct layout MUST match byte-for-byte (host packer in uboUpdater.ts
 * relies on this). Pre-C12 this struct was declared twice (verbatim) in
 * gtao.wgsl.ts:58-82 and gtaoUpsample.wgsl.ts:20-29 with an
 * anti-drift-by-comment explaining why duplication was tolerated.
 *
 * Distribution: registered in the W1-R6 include-graph as the
 * `gtaoUniforms` module. Both consumers declare
 * `requires: ['gtaoUniforms']` and the composer prepends this string
 * exactly once before each shader's own source.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const GTAO_UNIFORMS_WGSL = /* wgsl */ `
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
  // Pad to 32 bytes (8-element struct) for WebGPU 16-byte UBO alignment.
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};
`;

/** W1-R6 — declarative include-graph entry. Pure struct fragment; no
 *  bindings, no entry points. Both gtao + gtaoUpsample declare
 *  `requires: ['gtaoUniforms']` so the composer emits this struct once. */
export const GTAO_UNIFORMS_MODULE: WgslModule = {
  name: 'gtaoUniforms',
  source: GTAO_UNIFORMS_WGSL,
  requires: [],
};
