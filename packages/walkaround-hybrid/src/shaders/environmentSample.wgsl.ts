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
 *   @binding(15) env_map         rgba16float  : unit-intensity radiance (.rgb) +
 *                                               per-texel solid-angle pdf (.a)
 *   @binding(16) env_marginal    r32float     : H×1 inverse-CDF (random→row v;
 *                                               width=H height=1; textureLoad x=row y=0)
 *   @binding(17) env_conditional r32float     : W×H inverse-CDF (random→col u)
 *   @binding(18) env_sampler     sampler      : (declared for completeness; the
 *                                               lookups use textureLoad, not sample)
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

export const ENVIRONMENT_SAMPLE_WGSL = /* wgsl */ `

// ── B3 directional IBL — scene-group bindings ────────────────────────────────
@group(1) @binding(15) var env_map: texture_2d<f32>;
@group(1) @binding(16) var env_marginal: texture_2d<f32>;
@group(1) @binding(17) var env_conditional: texture_2d<f32>;
@group(1) @binding(18) var env_sampler: sampler;
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

// Directional radiance along a WORLD direction. Falls back to the scalar sky
// (skyTint·skyIrradiance) when no map is bound — preserving the no-HDRI contract.
fn envRadiance(dir: vec3f) -> vec3f {
  if (!envHasMap()) {
    return ubo.skyTint * ubo.skyIrradiance;
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
  return texel.rgb * max(envParams.intensity, 0.0);
}

// Importance-sample the environment via the PBRT 2D distribution (marginal row
// then conditional column). Two RNG draws. Returns pdf=0 when no map. The
// sampled direction is in WORLD space (rotateYPos applied).
fn envImportanceSample(rng: ptr<function, u32>) -> EnvSample {
  var s: EnvSample;
  s.color = vec3f(0.0);
  s.dir = vec3f(0.0, 1.0, 0.0);
  s.pdf = 0.0;
  if (!envHasMap()) { return s; }
  let w = i32(envParams.width);
  let h = i32(envParams.height);

  // Marginal: random → row centre v (stored in .r of the H×1 marginal texture;
  // width=H height=1, so textureLoad x=row y=0).
  let xiV = rand_f32(rng);
  let row = clamp(i32(floor(xiV * f32(h))), 0, h - 1);
  let vCenter = textureLoad(env_marginal, vec2i(row, 0), 0).r;
  let yTexel = clamp(i32(floor(vCenter * f32(h))), 0, h - 1);

  // Conditional: random → column centre u for the chosen row.
  let xiU = rand_f32(rng);
  let col = clamp(i32(floor(xiU * f32(w))), 0, w - 1);
  let uCenter = textureLoad(env_conditional, vec2i(col, yTexel), 0).r;
  let xTexel = clamp(i32(floor(uCenter * f32(w))), 0, w - 1);

  // Texel centre → direction (unrotated-map space), then rotate to world.
  let uc = (f32(xTexel) + 0.5) / f32(w);
  let vc = (f32(yTexel) + 0.5) / f32(h);
  let phi = (uc - 0.5) * (2.0 * PI);
  let theta = vc * PI;
  let sinTheta = sin(theta);
  let mapDir = vec3f(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta);
  let texel = textureLoad(env_map, vec2i(xTexel, yTexel), 0);

  s.dir = safe_normalize(envRotateYPos(mapDir, envParams.rotationY));
  s.color = texel.rgb * max(envParams.intensity, 0.0);
  s.pdf = max(texel.w, 0.0);
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
