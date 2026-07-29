/**
 * Connect CORE — Y-rotation helpers shared by the full-tier
 * (`connect.wgsl.ts`) and lite-tier (`connectLite.wgsl.ts`) connect modules.
 *
 * The full tier appends the HDRI bookkeeping helpers (`hasEnvironmentMap`,
 * `environmentDimensions`, `sampleEnvironmentColor`, `environmentPdf`,
 * `sampleEnvironmentImportance`) and the area-light MIS connection functions;
 * the lite tier appends its disabled-feature / procedural-only implementations.
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
`;
