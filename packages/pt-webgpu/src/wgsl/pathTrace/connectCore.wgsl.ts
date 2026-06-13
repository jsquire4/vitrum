/**
 * Connect CORE — Y-rotation helpers and the legacy `sampleSky` helper shared by
 * both the full-tier (`connect.wgsl.ts`) and lite-tier (`connectLite.wgsl.ts`)
 * connect modules.
 *
 * Bundled here (verbatim, shared by both tiers):
 *  - `sampleSky` — analytic sky helper kept for shader compatibility. Authored
 *    procedural sky is now baked into the HDRI path; no-environment lookups must
 *    return black rather than calling this helper.
 *
 * The full tier appends the HDRI bookkeeping helpers (`hasEnvironmentMap`,
 * `environmentDimensions`, `sampleEnvironmentColor`, `environmentPdf`,
 * `sampleEnvironmentImportance`) and the area-light MIS connection functions;
 * the lite tier appends its stub / procedural-only implementations.
 * Both compositions remain byte-identical to the pre-extraction monolithic
 * strings.
 *
 * No leading/trailing newline is added here: each tier interpolates this const
 * directly where the shared body used to be inlined.
 */
export const PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL = /* wgsl */ `// D9.13 — Y-rotation helpers shared by both connect tiers (full + lite). Used for
// HDRI env-map rotation: params.environmentTint.w stores the Y-rotation angle (radians).
//
// rotateYNeg(dir, rotY) = RY(−rotY) * dir — used when sampling the env map:
// the map is stored unrotated, so the lookup dir is counter-rotated.
//   x' =  cos(rotY)·x − sin(rotY)·z
//   y' =  y
//   z' =  sin(rotY)·x + cos(rotY)·z
fn rotateYNeg(dir: vec3f, rotY: f32) -> vec3f {
  let c = cos(rotY); let s = sin(rotY);
  return vec3f(c * dir.x - s * dir.z, dir.y, s * dir.x + c * dir.z);
}

// rotateYPos(dir, rotY) = RY(+rotY) * dir — used when converting the sampled map
// direction back to world space (inverse of rotateYNeg).
//   x' =  cos(rotY)·x + sin(rotY)·z
//   y' =  y
//   z' = −sin(rotY)·x + cos(rotY)·z
fn rotateYPos(dir: vec3f, rotY: f32) -> vec3f {
  let c = cos(rotY); let s = sin(rotY);
  return vec3f(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
}

fn sampleSky(dir: vec3f) -> vec3f {
  let t = 0.5 * (dir.y + 1.0);
  var sky = mix(vec3f(0.06, 0.08, 0.12), vec3f(0.45, 0.62, 0.95), clamp(t, 0.0, 1.0));
  let sunDir = safe_normalize(params.environmentSun.xyz);
  let sunGlow = pow(max(0.0, dot(dir, sunDir)), 512.0) * params.environmentSun.w;
  sky = sky + vec3f(1.0, 0.95, 0.85) * sunGlow;
  return sky * params.environmentTint.rgb;
}`;
