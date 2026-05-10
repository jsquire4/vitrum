/**
 * Seed compute kernel for the early pt-webgpu backend.
 *
 * This is intentionally minimal: it validates device/pipeline/bindgroup
 * plumbing and writes a deterministic debug image to the accumulation target.
 * Real path integration (BVH traversal + BSDF + NEE + accumulation policy) is
 * layered on top of this scaffold.
 */
export const PT_WEBGPU_SEED_WGSL = /* wgsl */ `
struct FrameParams {
  width: u32,
  height: u32,
  frameIndex: u32,
  frameSeed: u32,
};

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: FrameParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let uv = vec2f(
    f32(gid.x) / max(1.0, f32(params.width - 1u)),
    f32(gid.y) / max(1.0, f32(params.height - 1u)),
  );
  let t = f32(params.frameIndex & 255u) / 255.0;
  let r = uv.x;
  let g = uv.y;
  let b = 0.35 + 0.65 * fract(t + f32(params.frameSeed & 255u) * 0.0039215686);
  textureStore(outputTexture, vec2i(gid.xy), vec4f(r, g, b, 1.0));
}
`;
