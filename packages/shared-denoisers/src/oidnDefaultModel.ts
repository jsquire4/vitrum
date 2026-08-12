/**
 * Default Intel Open Image Denoise RT HDR albedo+normal ONNX for stills.
 *
 * Vitrum does not vendor the multi-megabyte weight file. `auto` / omitted
 * `modelUrl` resolve to this version-pinned, CORS-enabled jsDelivr URL.
 * The weights are Intel's Apache-2.0 OIDN RT filter, converted to ONNX by
 * pmndrs/denoiser-weights (tag `models-v1`). Hosts may override with a
 * self-hosted copy.
 *
 * Graph contract: one NCHW `input` (9 channels: color, albedo, normal) and
 * one NCHW `output` (3 channels). The OIDN bridge concatenates aux planes
 * when the session declares that single input.
 */
export const DEFAULT_OIDN_RT_HDR_ALB_NRM_MODEL_URL =
  'https://cdn.jsdelivr.net/gh/pmndrs/denoiser-weights@models-v1/models/rt_hdr_alb_nrm.onnx';

export function resolveOidnModelUrl(modelUrl: string | undefined): string {
  if (typeof modelUrl === 'string' && modelUrl.trim().length > 0) {
    return modelUrl.trim();
  }
  return DEFAULT_OIDN_RT_HDR_ALB_NRM_MODEL_URL;
}

export function oidnModelUrlIsHostProvided(modelUrl: string | undefined): boolean {
  return typeof modelUrl === 'string' && modelUrl.trim().length > 0;
}
