/**
 * Camera helpers shared by RIS / temporal / spatial / shade.
 *
 * Split out of common.wgsl.ts (T9-stepA): `invertMat4_common` (cofactor
 * 4×4 inverse) and `generatePrimaryRay_common` (screen → world primary ray).
 * `generatePrimaryRay_common` returns a `Ray` (from the sceneTraversal /
 * shared-bvh injection) and uses `safe_normalize` (shared primitives).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const CAMERA_RAYS_WGSL = /* wgsl */ `// ============================================================
// Camera helpers (shared by RIS / temporal / spatial / shade)
// ============================================================
// Invert a 4x4 matrix (standard cofactor method).  Used to unproject screen
// coords → world rays for primary-ray-cast mode.
fn invertMat4_common(m: mat4x4f) -> mat4x4f {
  let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
  let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
  let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
  let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];
  let b00 = a00*a11-a01*a10; let b01 = a00*a12-a02*a10; let b02 = a00*a13-a03*a10;
  let b03 = a01*a12-a02*a11; let b04 = a01*a13-a03*a11; let b05 = a02*a13-a03*a12;
  let b06 = a20*a31-a21*a30; let b07 = a20*a32-a22*a30; let b08 = a20*a33-a23*a30;
  let b09 = a21*a32-a22*a31; let b10 = a21*a33-a23*a31; let b11 = a22*a33-a23*a32;
  let det = b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if (abs(det) < 1e-10) { return mat4x4f(); }
  let inv = 1.0/det;
  return mat4x4f(
    vec4f((a11*b11-a12*b10+a13*b09)*inv, (-a01*b11+a02*b10-a03*b09)*inv,
           (a31*b05-a32*b04+a33*b03)*inv,  (-a21*b05+a22*b04-a23*b03)*inv),
    vec4f((-a10*b11+a12*b08-a13*b07)*inv, (a00*b11-a02*b08+a03*b07)*inv,
           (-a30*b05+a32*b02-a33*b01)*inv,  (a20*b05-a22*b02+a23*b01)*inv),
    vec4f((a10*b10-a11*b08+a13*b06)*inv, (-a00*b10+a01*b08-a03*b06)*inv,
           (a30*b04-a31*b02+a33*b00)*inv,  (-a20*b04+a21*b02-a23*b00)*inv),
    vec4f((-a10*b09+a11*b07-a12*b06)*inv, (a00*b09-a01*b07+a02*b06)*inv,
           (-a30*b03+a31*b01-a32*b00)*inv,  (a20*b03-a21*b01+a22*b00)*inv)
  );
}

// Generate a world-space primary ray for pixel (px, py) given the inverse
// view-projection matrix.  Ray origin = camera position; direction unprojects
// the pixel center through near→far in NDC.  Used by ALL passes that need
// to cast primary rays (RIS, shade, temporal, spatial).
fn generatePrimaryRay_common(
  px: u32, py: u32, w: u32, h: u32,
  camPos: vec3f, invVP: mat4x4f,
) -> Ray {
  let uv  = (vec2f(f32(px), f32(py)) + 0.5) / vec2f(f32(w), f32(h));
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let far4  = invVP * vec4f(ndc,  1.0, 1.0);
  let near4 = invVP * vec4f(ndc, -1.0, 1.0);
  // Guard against degenerate-camera invVP. invertMat4_common returns the zero
  // matrix when |det| < 1e-10; that would set far4/near4 = (0,0,0,0), and the
  // raw /w divides would yield NaN, which downstream safe_normalize does not
  // catch (it only handles zero-length). On real perspective cameras far4.w
  // and near4.w are well above 1e-30, so the guard is inert.
  let farW  = far4.xyz  / select(1.0, far4.w,  abs(far4.w)  > 1e-30);
  let nearW = near4.xyz / select(1.0, near4.w, abs(near4.w) > 1e-30);
  var ray: Ray;
  ray.origin    = camPos;
  ray.direction = safe_normalize(farW - nearW);
  return ray;
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const CAMERA_RAYS_MODULE: WgslModule = {
  name: "cameraRays",
  source: CAMERA_RAYS_WGSL,
  requires: [],
};
