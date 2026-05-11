/**
 * HDR luminance bilateral (5×5) — WebGPU compute, color edge-stop only.
 *
 * No G-buffer: preserves sharp silhouettes vs depth/normal-guided SVGF, but runs
 * entirely on linear HDR radiance read back from WebGL accumulation — suitable
 * for Cornell preview/export without a raster prepass.
 *
 * References: bilateral filtering (Tomasi & Manduchi); luminance edge-stop common in HDR denoise probes.
 */

/** Must match `@workgroup_size` below (dispatch uses this value). */
export const HDR_LUMINANCE_BILATERAL_WORKGROUP_SIZE = 8 as const;

export const HDR_LUMINANCE_BILATERAL_WGSL = /* wgsl */ `
struct BilateralParams {
  sigmaLuminance: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var texIn: texture_2d<f32>;
@group(0) @binding(1) var texOut: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: BilateralParams;

fn luminance(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8, 1)
fn hdrLuminanceBilateralMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(texIn);
  if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) {
    return;
  }
  let p0 = vec2<i32>(i32(gid.x), i32(gid.y));
  let c0 = textureLoad(texIn, p0, 0).rgb;
  let L0 = luminance(c0);
  var acc = vec3<f32>(0.0);
  var wsum = 0.0;
  let sig = max(params.sigmaLuminance, 1e-6);
  let invTwoSig2 = 1.0 / (2.0 * sig * sig);

  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let p = p0 + vec2<i32>(dx, dy);
      if (p.x < 0 || p.y < 0 || p.x >= dims.x || p.y >= dims.y) {
        continue;
      }
      let c = textureLoad(texIn, p, 0).rgb;
      let L = luminance(c);
      let spatial = f32(dx * dx + dy * dy);
      let ws = exp(-spatial / 18.0); // 2 * sigma_spatial^2, sigma_spatial=3 (5x5 kernel radius)
      let wr = exp(-(L - L0) * (L - L0) * invTwoSig2);
      let w = ws * wr;
      acc += c * w;
      wsum += w;
    }
  }

  let outRgb = acc / max(wsum, 1e-6);
  textureStore(texOut, p0, vec4<f32>(outRgb, 1.0));
}
`;

export const HDR_LUMINANCE_BILATERAL_ENTRY = 'hdrLuminanceBilateralMain' as const;
