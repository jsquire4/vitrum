/**
 * B3 (road-to-100) — directional IBL sampling WGSL.
 *
 * Provides the scene-group environment bindings + the directional lookup /
 * importance-sample helpers consumed by the sky-miss paths (ris, shade) and the
 * GI-escape path (risGi). When no pixel-backed HDRI is present the host writes
 * `envParams.hasEnv = 0u` and every helper returns the scalar-tint fallback
 * (`ubo.skyTint * ubo.skyIrradiance`) — so no-HDRI scenes are byte-identical.
 *
 * Bindings (scene group(1), appended after the B1 bvh_material at binding 14):
 *   @binding(15) env_map         rgba32float  : unit-intensity radiance (.rgb)
 *   @binding(16) env_marginal    r32float     : H×1 forward row CDF
 *   @binding(17) env_conditional r32float     : W×H forward column CDF
 *   @binding(18) env_pdf         r32float     : per-texel solid-angle density
 *   @binding(19) envParams       uniform      : { hasEnv, width, height, rotationY,
 *                                               intensity } (own small uniform —
 *                                               the 416B WalkaroundUBO is frozen)
 *
 * Direction convention (H6 — matches HdriEnvironment.rotationY + pt-webgpu
 * connect.wgsl): a CCW dome rotationY means a world dir d looks up the UNROTATED
 * map at rotateYNeg(d, rotationY); a CDF-sampled (unrotated) dir is rotated by
 * +rotationY to world. rotationY = 0 ⇒ both are identity.
 *
 * The equirect UV convention matches buildDirectionalEnv (PBRT):
 *   u = phi/(2π) + 0.5 with phi = atan2(z, x);  v = theta/π with theta = acos(y).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { WALKAROUND_ENVIRONMENT_RADIANCE_SCALE_WGSL } from '../environment/environmentRadianceScale.js';

export const ENVIRONMENT_SAMPLE_WGSL = /* wgsl */ `
${WALKAROUND_ENVIRONMENT_RADIANCE_SCALE_WGSL}

// ── B3 directional IBL — scene-group bindings ────────────────────────────────
@group(1) @binding(15) var env_map: texture_2d<f32>;
@group(1) @binding(16) var env_marginal: texture_2d<f32>;
@group(1) @binding(17) var env_conditional: texture_2d<f32>;
@group(1) @binding(18) var env_pdf: texture_2d<f32>;
struct EnvParams {
  hasEnv:    u32,
  width:     u32,
  height:    u32,
  rotationY: f32,
  intensity: f32,
};
@group(1) @binding(19) var<uniform> envParams: EnvParams;

fn envHasMap() -> bool {
  return envParams.hasEnv == 1u && envParams.width > 0u && envParams.height > 0u;
}

// H6 — RY(-rotY)·dir (world → unrotated-map lookup direction).
fn envRotateYNeg(dir: vec3f, rotY: f32) -> vec3f {
  let c = cos(rotY);
  let s = sin(rotY);
  return vec3f(c * dir.x - s * dir.z, dir.y, s * dir.x + c * dir.z);
}
// H6 — RY(+rotY)·dir (unrotated-map sampled dir → world).
fn envRotateYPos(dir: vec3f, rotY: f32) -> vec3f {
  let c = cos(rotY);
  let s = sin(rotY);
  return vec3f(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
}

struct EnvSample {
  color: vec3f,   // radiance × intensity along .dir
  dir:   vec3f,   // world-space direction
  pdf:   f32,     // solid-angle pdf of selecting .dir (0 ⇒ no map)
};

fn envCdfXi(xi: f32) -> f32 {
  // rand_f32 is [0,1), but make the search total for synthetic/debug callers
  // that provide 1 exactly. 0x3f7fffff is the greatest f32 below one.
  return clamp(xi, 0.0, bitcast<f32>(0x3f7fffffu));
}

fn envMarginalRowFromCdf(xiRaw: f32, h: i32) -> i32 {
  let xi = envCdfXi(xiRaw);
  var lo = 0;
  var hi = h - 1;
  while (lo < hi) {
    let mid = (lo + hi) / 2;
    let cdf = textureLoad(env_marginal, vec2i(mid, 0), 0).r;
    if (cdf <= xi) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

fn envConditionalColumnFromCdf(xiRaw: f32, row: i32, w: i32) -> i32 {
  let xi = envCdfXi(xiRaw);
  var lo = 0;
  var hi = w - 1;
  while (lo < hi) {
    let mid = (lo + hi) / 2;
    let cdf = textureLoad(env_conditional, vec2i(mid, row), 0).r;
    if (cdf <= xi) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Directional radiance along a WORLD direction. Falls back to the scalar sky
// (skyTint·skyIrradiance) when no map is bound — preserving the no-HDRI contract.
fn envRadiance(dir: vec3f) -> vec3f {
  if (!envHasMap()) {
    return walkaroundScaleEnvironmentRadiance(
      ubo.skyTint,
      ubo.skyIrradiance,
    );
  }
  let w = i32(envParams.width);
  let h = i32(envParams.height);
  let lookupDir = envRotateYNeg(safe_normalize(dir), envParams.rotationY);
  let phi = atan2(lookupDir.z, lookupDir.x);
  let theta = acos(clamp(lookupDir.y, -1.0, 1.0));
  let u = fract(phi * INV_PI * 0.5 + 0.5);
  let v = clamp(theta * INV_PI, 0.0, 0.999999);
  let x = clamp(i32(floor(u * f32(w))), 0, w - 1);
  let y = clamp(i32(floor(v * f32(h))), 0, h - 1);
  let texel = textureLoad(env_map, vec2i(x, y), 0);
  return walkaroundScaleEnvironmentRadiance(
    texel.rgb,
    envParams.intensity,
  );
}

// Importance-sample the environment via the PBRT 2D distribution (marginal row
// then conditional column), followed by a continuous solid-angle sample inside
// the selected cell. Four RNG draws. Returns pdf=0 when no map. The sampled
// direction is in WORLD space (rotateYPos applied).
fn envImportanceSample(rng: ptr<function, u32>) -> EnvSample {
  var s: EnvSample;
  s.color = vec3f(0.0);
  s.dir = vec3f(0.0, 1.0, 0.0);
  s.pdf = 0.0;
  if (!envHasMap()) { return s; }
  let w = i32(envParams.width);
  let h = i32(envParams.height);

  // Exact discrete inversion: select the first forward-CDF entry > ξ. The
  // realized selection interval is therefore the source texel's exact PMF,
  // matching the solid-angle density stored in env_pdf.
  let yTexel = envMarginalRowFromCdf(rand_f32(rng), h);
  let xTexel = envConditionalColumnFromCdf(rand_f32(rng), yTexel, w);

  // Preserve exact discrete CDF inversion, then sample continuously and
  // uniformly in solid angle inside the selected equirect cell. Using the
  // texel centre here would collapse every cell's probability mass to a delta
  // direction and bias visibility/MIS even though the discrete PMF is exact.
  let uc = (f32(xTexel) + rand_f32(rng)) / f32(w);
  let theta0 = f32(yTexel) * PI / f32(h);
  let theta1 = f32(yTexel + 1) * PI / f32(h);
  let cosTheta = mix(cos(theta0), cos(theta1), rand_f32(rng));
  let phi = (uc - 0.5) * (2.0 * PI);
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  let mapDir = vec3f(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
  let texel = textureLoad(env_map, vec2i(xTexel, yTexel), 0);

  s.dir = safe_normalize(envRotateYPos(mapDir, envParams.rotationY));
  s.color = walkaroundScaleEnvironmentRadiance(
    texel.rgb,
    envParams.intensity,
  );
  s.pdf = max(textureLoad(env_pdf, vec2i(xTexel, yTexel), 0).r, 0.0);
  return s;
}

`;

export const ENVIRONMENT_SAMPLE_MODULE: WgslModule = {
  name: 'environmentSample',
  source: ENVIRONMENT_SAMPLE_WGSL,
  // Depends on walkaroundUbo (WalkaroundUBO/PI/INV_PI), sharedPrimitives
  // (safe_normalize/rand_f32). `common` aggregates these; consumers that pull
  // environmentSample in transitively have them in scope.
  requires: [],
};
