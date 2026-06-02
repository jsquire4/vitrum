/**
 * Connect CORE — the byte-identical `sampleSky` helper shared by both the
 * full-tier (`connect.wgsl.ts`) and lite-tier (`connectLite.wgsl.ts`)
 * connect modules.
 *
 * Bundled here (verbatim, shared by both tiers):
 *  - `sampleSky` — analytic sky fallback (sun glow + zenith tint)
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
export const PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL = /* wgsl */ `fn sampleSky(dir: vec3f) -> vec3f {
  let t = 0.5 * (dir.y + 1.0);
  var sky = mix(vec3f(0.06, 0.08, 0.12), vec3f(0.45, 0.62, 0.95), clamp(t, 0.0, 1.0));
  let sunDir = safe_normalize(params.environmentSun.xyz);
  let sunGlow = pow(max(0.0, dot(dir, sunDir)), 512.0) * params.environmentSun.w;
  sky = sky + vec3f(1.0, 0.95, 0.85) * sunGlow;
  return sky * params.environmentTint.rgb;
}`;
