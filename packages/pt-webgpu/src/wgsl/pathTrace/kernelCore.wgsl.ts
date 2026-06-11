/**
 * Kernel CORE — the byte-identical shared prefix of both the full-tier
 * (`kernel.wgsl.ts`) and lite-tier (`kernelLite.wgsl.ts`) kernel modules.
 *
 * Bundled here (verbatim, shared by both tiers):
 *  - `generatePrimaryRay` — inverse-VP camera ray + sub-pixel jitter
 *  - `projectToNdc` — VP-clip projection used for motion vectors
 *  - `causticMode` — UBO accessor for the caustic-strategy selector
 *  - `RRResult` struct + `russianRoulette` — bounce-termination helper
 *
 * The full tier prepends HG phase helpers (when SSS is enabled) and appends
 * `accumulateFrame` + the `@compute` entry point; the lite tier appends its
 * reduced `accumulateFrame` + `@compute` entry point. Both compositions remain
 * byte-identical to the pre-extraction monolithic strings.
 *
 * `accumulateFrame` is NOT extracted here: it differs between tiers (full tier
 * writes variance-moments + motion vectors; lite tier does not).
 *
 * No leading/trailing newline is added here: each tier interpolates this const
 * directly where the shared body used to be inlined.
 */
export const PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL = /* wgsl */ `fn generatePrimaryRay(px: u32, py: u32, jitter: vec2f) -> Ray {
  let uv = (vec2f(f32(px), f32(py)) + jitter) / vec2f(f32(params.width), f32(params.height));
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let far4 = params.invViewProj * vec4f(ndc, 1.0, 1.0);
  let near4 = params.invViewProj * vec4f(ndc, -1.0, 1.0);
  let farW = far4.xyz / far4.w;
  let nearW = near4.xyz / near4.w;
  var ray: Ray;
  ray.origin = params.cameraPos.xyz;
  ray.direction = safe_normalize(farW - nearW);
  return ray;
}

fn projectToNdc(pos: vec3f, vp: mat4x4f) -> vec2f {
  let clip = vp * vec4f(pos, 1.0);
  let invW = 1.0 / max(abs(clip.w), 1e-8);
  return clip.xy * invW;
}


fn causticMode() -> u32 {
  return params.causticStrategy;
}

struct RRResult {
  survives: bool,
  throughputMul: f32,
}

fn russianRoulette(rng: ptr<function, u32>, throughput: vec3f) -> RRResult {
  let survival = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.1, 0.95);
  var result: RRResult;
  if (rand_f32(rng) > survival) {
    result.survives = false;
    result.throughputMul = 1.0;
    return result;
  }
  result.survives = true;
  result.throughputMul = 1.0 / survival;
  return result;
}

// D9.11 — Shirley & Chiu 1997 concentric-disc mapping: maps the unit square [−1,1]²
// uniformly to the unit disc. xi = vec2f(a, b) in [−1,1]² (pre-remapped from [0,1]²
// by the caller). Returns the 2-D disc point (x, y) with |(x,y)| ≤ 1.
// Ref: Shirley & Chiu, "A Low Distortion Map Between Disk and Square," JGT 1997.
fn concentricDiscSample(xi: vec2f) -> vec2f {
  let a = xi.x; let b = xi.y;
  var cr: f32; var cphi: f32;
  if (abs(a) >= abs(b)) {
    cr = a; cphi = (PI / 4.0) * (b / max(abs(a), 1e-9));
  } else {
    cr = b; cphi = (PI / 2.0) - (PI / 4.0) * (a / max(abs(b), 1e-9));
  }
  return vec2f(cr * cos(cphi), cr * sin(cphi));
}`;
