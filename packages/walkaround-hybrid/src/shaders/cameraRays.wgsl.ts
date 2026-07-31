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
fn cameraFiniteF32(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823e38;
}

fn cameraMaxAbs4(value: vec4f) -> f32 {
  return max(max(abs(value.x), abs(value.y)), max(abs(value.z), abs(value.w)));
}

fn cameraFiniteMat4(value: mat4x4f) -> bool {
  for (var column = 0u; column < 4u; column += 1u) {
    for (var row = 0u; row < 4u; row += 1u) {
      if (!cameraFiniteF32(value[column][row])) {
        return false;
      }
    }
  }
  return true;
}

// A non-zero cofactor determinant alone is insufficient in f32: reject an
// overflowed or numerically non-reciprocal candidate in either product order.
fn cameraMat4ProductIsIdentity(left: mat4x4f, right: mat4x4f) -> bool {
  for (var column = 0u; column < 4u; column += 1u) {
    for (var row = 0u; row < 4u; row += 1u) {
      var value = 0.0;
      var absoluteTermSum = 0.0;
      for (var inner = 0u; inner < 4u; inner += 1u) {
        let term = left[inner][row] * right[column][inner];
        if (!cameraFiniteF32(term)) {
          return false;
        }
        value += term;
        absoluteTermSum += abs(term);
      }
      if (!cameraFiniteF32(absoluteTermSum)) {
        return false;
      }
      let expected = select(0.0, 1.0, column == row);
      let tolerance = 1e-4 * max(1.0, absoluteTermSum);
      if (!cameraFiniteF32(value) || abs(value - expected) > tolerance) {
        return false;
      }
    }
  }
  return true;
}

fn invertMat4_common(m: mat4x4f) -> mat4x4f {
  if (!cameraFiniteMat4(m)) {
    return mat4x4f();
  }
  let scales = vec4f(
    cameraMaxAbs4(m[0]),
    cameraMaxAbs4(m[1]),
    cameraMaxAbs4(m[2]),
    cameraMaxAbs4(m[3]),
  );
  if (any(scales <= vec4f(0.0)) || !all(scales == scales)) {
    return mat4x4f();
  }
  let normalized = mat4x4f(
    m[0] / scales.x,
    m[1] / scales.y,
    m[2] / scales.z,
    m[3] / scales.w,
  );
  let a00 = normalized[0][0]; let a01 = normalized[0][1]; let a02 = normalized[0][2]; let a03 = normalized[0][3];
  let a10 = normalized[1][0]; let a11 = normalized[1][1]; let a12 = normalized[1][2]; let a13 = normalized[1][3];
  let a20 = normalized[2][0]; let a21 = normalized[2][1]; let a22 = normalized[2][2]; let a23 = normalized[2][3];
  let a30 = normalized[3][0]; let a31 = normalized[3][1]; let a32 = normalized[3][2]; let a33 = normalized[3][3];
  let b00 = a00*a11-a01*a10; let b01 = a00*a12-a02*a10; let b02 = a00*a13-a03*a10;
  let b03 = a01*a12-a02*a11; let b04 = a01*a13-a03*a11; let b05 = a02*a13-a03*a12;
  let b06 = a20*a31-a21*a30; let b07 = a20*a32-a22*a30; let b08 = a20*a33-a23*a30;
  let b09 = a21*a32-a22*a31; let b10 = a21*a33-a23*a31; let b11 = a22*a33-a23*a32;
  let det = b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if (det == 0.0 || !cameraFiniteF32(det)) { return mat4x4f(); }
  let inv = 1.0/det;
  let normalizedInverse = mat4x4f(
    vec4f((a11*b11-a12*b10+a13*b09)*inv, (-a01*b11+a02*b10-a03*b09)*inv,
           (a31*b05-a32*b04+a33*b03)*inv,  (-a21*b05+a22*b04-a23*b03)*inv),
    vec4f((-a10*b11+a12*b08-a13*b07)*inv, (a00*b11-a02*b08+a03*b07)*inv,
           (-a30*b05+a32*b02-a33*b01)*inv,  (a20*b05-a22*b02+a23*b01)*inv),
    vec4f((a10*b10-a11*b08+a13*b06)*inv, (-a00*b10+a01*b08-a03*b06)*inv,
           (a30*b04-a31*b02+a33*b00)*inv,  (-a20*b04+a21*b02-a23*b00)*inv),
    vec4f((-a10*b09+a11*b07-a12*b06)*inv, (a00*b09-a01*b07+a02*b06)*inv,
           (-a30*b03+a31*b01-a32*b00)*inv,  (a20*b03-a21*b01+a22*b00)*inv)
  );
  let candidate = mat4x4f(
    normalizedInverse[0] / scales,
    normalizedInverse[1] / scales,
    normalizedInverse[2] / scales,
    normalizedInverse[3] / scales,
  );
  if (
    !cameraFiniteMat4(candidate) ||
    !cameraMat4ProductIsIdentity(m, candidate) ||
    !cameraMat4ProductIsIdentity(candidate, m)
  ) {
    return mat4x4f();
  }
  return candidate;
}

fn cameraRayFromInvVP_common(
  px: u32, py: u32, w: u32, h: u32,
  invVP: mat4x4f,
) -> Ray {
  let uv = (vec2f(f32(px), f32(py)) + 0.5) / vec2f(f32(w), f32(h));
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  var farH = invVP * vec4f(ndc, 1.0, 1.0);
  var nearH = invVP * vec4f(ndc, -1.0, 1.0);
  let farScale = cameraMaxAbs4(farH);
  let nearScale = cameraMaxAbs4(nearH);
  var ray: Ray;
  ray.origin = vec3f(0.0);
  ray.direction = vec3f(0.0);
  if (
    !cameraFiniteMat4(invVP) ||
    !cameraFiniteF32(farScale) || !(farScale > 0.0) ||
    !cameraFiniteF32(nearScale) || !(nearScale > 0.0)
  ) {
    return ray;
  }
  // Positive equilibration preserves each homogeneous point while avoiding
  // overflow/underflow in the division-free direction numerator.
  farH /= farScale;
  nearH /= nearScale;
  if (nearH.w == 0.0) {
    return ray;
  }
  let nearPoint = nearH.xyz / nearH.w;
  // When farH.w is zero the numerator already contains nearH.w, which anchors
  // the direction of the projective point at infinity. Multiplying by its sign
  // again would make a globally negated (but equivalent) matrix flip the ray.
  var orientation = 1.0;
  if (farH.w != 0.0) {
    orientation = sign(farH.w * nearH.w);
  }
  let directionNumerator =
    (farH.xyz * nearH.w - nearH.xyz * farH.w) * orientation;
  let directionScale = max(
    abs(directionNumerator.x),
    max(abs(directionNumerator.y), abs(directionNumerator.z)),
  );
  if (
    !all(nearPoint == nearPoint) ||
    any(abs(nearPoint) > vec3f(3.402823e38)) ||
    !all(directionNumerator == directionNumerator) ||
    any(abs(directionNumerator) > vec3f(3.402823e38)) ||
    !(directionScale > 0.0)
  ) {
    return ray;
  }
  // The unprojected near point is the correct origin for orthographic cameras
  // and is intersection-equivalent for perspective cameras. It also enforces
  // the projection matrix's near clipping plane.
  ray.origin = nearPoint;
  ray.direction = safe_normalize(directionNumerator);
  return ray;
}

// Generate a world-space primary ray for pixel (px, py) given the inverse
// view-projection matrix. The ray starts on the unprojected near plane and its
// direction remains valid when the far plane is at infinity. Used by every
// pass that casts primary rays (RIS, shade, temporal, spatial).
fn generatePrimaryRay_common(
  px: u32, py: u32, w: u32, h: u32,
  _camPos: vec3f, invVP: mat4x4f,
) -> Ray {
  return cameraRayFromInvVP_common(px, py, w, h, invVP);
}

// Reconstruct a camera ray using only an inverse view-projection matrix. This
// is used for temporal surface correspondence, where the UBO intentionally
// stores the previous VP but not a fabricated previous object-motion field.
// Starting at the unprojected near-plane point is intersection-equivalent to
// the perspective camera origin and is also correct for orthographic cameras.
fn generatePrimaryRayFromInvVP_common(
  px: u32, py: u32, w: u32, h: u32,
  invVP: mat4x4f,
) -> Ray {
  return cameraRayFromInvVP_common(px, py, w, h, invVP);
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const CAMERA_RAYS_MODULE: WgslModule = {
  name: "cameraRays",
  source: CAMERA_RAYS_WGSL,
  requires: [],
};
