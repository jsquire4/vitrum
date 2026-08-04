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
  return unproject_ray_common(params.invViewProj, ndc);
}

fn projectToNdc(pos: vec3f, vp: mat4x4f) -> vec2f {
  let rawClip = vp * vec4f(pos, 1.0);
  let scale = max(
    max(abs(rawClip.x), abs(rawClip.y)),
    max(abs(rawClip.z), abs(rawClip.w)),
  );
  if (!(scale > 0.0) || scale > 3.402823e38) {
    return vec2f(2.0);
  }
  let clip = rawClip / scale;
  if (!(clip.w > 0.0)) {
    return vec2f(2.0);
  }
  return clip.xy / clip.w;
}


fn causticMode() -> u32 {
  return params.causticStrategy;
}

struct RRResult {
  survives: bool,
  throughputMul: f32,
}

fn russianRoulette(rng: ptr<function, PtRngState>, throughput: vec3f) -> RRResult {
  let survival = represented_bernoulli_probability_f32(
    clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.1, 0.95),
  );
  var result: RRResult;
  if (!(rand_f32(rng) < survival)) {
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
  if (a == 0.0 && b == 0.0) {
    return vec2f(0.0);
  }
  var cr: f32; var cphi: f32;
  if (abs(a) >= abs(b)) {
    cr = a; cphi = (PI / 4.0) * (b / a);
  } else {
    cr = b; cphi = (PI / 2.0) - (PI / 4.0) * (a / b);
  }
  return vec2f(cr * cos(cphi), cr * sin(cphi));
}`;
