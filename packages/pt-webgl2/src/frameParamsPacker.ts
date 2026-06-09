// frameParamsPacker — the per-frame std140 UBO assembly for the THREE-free
// WebGL2 path tracer (WS5 §2). Mirrors pt-webgpu's `packFrameParams` /
// `frameParamsLayout.generated.ts`, but lays out for a GLSL `std140` uniform
// block rather than a WGSL storage struct.
//
// std140 rules used here:
//   - scalars (float/int/uint/bool) occupy one slot (4 bytes), tightly packed
//     into the leading region until the first vec4-aligned aggregate;
//   - a `vec4` must start on a 16-byte (4-slot) boundary;
//   - a `mat4` (= 4 × vec4 columns) must start on a 16-byte boundary.
// FRAME_PARAMS_SLOTS therefore lists the scalar region first (one slot each),
// inserts explicit pad slots to reach the next 4-slot boundary, then the two
// mat4s at 16-byte-aligned offsets.
//
// The big CMF arrays (uCmfX/Y/Z[81], CDFs[82]) and the samplers are NOT packed
// here — std140 array padding is wasteful for them, so the GLSL kernel binds
// them as direct uniforms / textures (WS5 §2 + §5).
//
// Mat4 helpers: @vitrum/core exports only asMat4/isMat4 (no invert/multiply),
// and pt-webgpu's invert/multiply live in its own package. To keep this file
// self-contained (per the WS5 brief), the 4×4 inverse + multiply are inlined
// below. Matrices are column-major (three.js / WebGL convention), matching
// `FrameInput.viewMatrix` / `projMatrix`.

import type { FrameInput } from '@vitrum/core';
import { X_CMF_INTEGRAL, Y_CMF_INTEGRAL, Z_CMF_INTEGRAL } from '@vitrum/shared-samplers';

/** Per-frame engine configuration the packer reads (mirror of pt-webgpu's
 *  FrameParamsEngineConfig, narrowed to the WebGL2 surface). */
export interface FrameParamsConfig {
  readonly maxBounces: number;
  readonly spectral: boolean;
  readonly causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly mneeMaxIterations: number;
  readonly mneeMaxChainLength: number;
  readonly bdpt: boolean;
}

/** The scene-derived counts/flags the packer reads. */
export interface FrameParamsScene {
  readonly triangleCount: number;
  readonly bvhNodeCount: number;
  readonly lightCount: number;
  readonly hasEnvironmentMap: boolean;
  readonly environmentMapWidth: number;
  readonly environmentMapHeight: number;
}

/**
 * std140 float-slot offsets for the GLSL `FrameParams` uniform block. Each value
 * is a float index (× 4 = byte offset). Scalars occupy one slot; `vec4` and
 * `mat4` members start on 4-slot (16-byte) boundaries.
 *
 * Scalar region (slots 0..21):
 *   resolution = (width,height) packed as two scalars; the rest are the
 *   per-frame integers/floats. Slots 22 and 23 are explicit pad to reach the
 *   next 16-byte boundary before the first mat4.
 *
 * The GLSL block this maps to:
 *
 *   layout(std140) uniform FrameParams {
 *     uint  resolutionX;            // slot 0
 *     uint  resolutionY;            // slot 1
 *     uint  frameSeed;              // slot 2
 *     uint  bounces;                // slot 3
 *     uint  triangleCount;          // slot 4
 *     uint  bvhNodeCount;           // slot 5
 *     uint  lightCount;             // slot 6
 *     uint  hasEnvironmentMap;      // slot 7
 *     uint  environmentMapWidth;    // slot 8
 *     uint  environmentMapHeight;   // slot 9
 *     uint  uSpectralRendering;     // slot 10
 *     float heroLambdaNm;           // slot 11
 *     float heroPdf;                // slot 12
 *     float cmfIntegralX;           // slot 13
 *     float cmfIntegralY;           // slot 14
 *     float cmfIntegralZ;           // slot 15
 *     uint  uCausticStrategy;       // slot 16
 *     uint  uMneeMaxIterations;     // slot 17
 *     uint  uMneeMaxChainLength;    // slot 18
 *     uint  uBdptEnabled;           // slot 19
 *     uint  uBdptMaxEyeDepth;       // slot 20
 *     uint  _pad0;                  // slot 21
 *     uint  _pad1;                  // slot 22
 *     uint  _pad2;                  // slot 23
 *     mat4  cameraWorldMatrix;      // slots 24..39
 *     mat4  invProjectionMatrix;    // slots 40..55
 *   };
 */
export const FRAME_PARAMS_SLOTS = {
  resolutionX: 0,
  resolutionY: 1,
  frameSeed: 2,
  bounces: 3,
  triangleCount: 4,
  bvhNodeCount: 5,
  lightCount: 6,
  hasEnvironmentMap: 7,
  environmentMapWidth: 8,
  environmentMapHeight: 9,
  uSpectralRendering: 10,
  heroLambdaNm: 11,
  heroPdf: 12,
  cmfIntegralX: 13,
  cmfIntegralY: 14,
  cmfIntegralZ: 15,
  uCausticStrategy: 16,
  uMneeMaxIterations: 17,
  uMneeMaxChainLength: 18,
  uBdptEnabled: 19,
  uBdptMaxEyeDepth: 20,
  _pad0: 21,
  _pad1: 22,
  _pad2: 23,
  cameraWorldMatrix: 24,
  invProjectionMatrix: 40,
} as const;

/** Total float slots = 40 (start of last mat4) + 16 (its 4 columns) = 56. */
export const FRAME_PARAMS_F32_SLOTS = 56;
/** std140 byte size of the packed block. */
export const FRAME_PARAMS_BYTE_SIZE = FRAME_PARAMS_F32_SLOTS * 4;

/** Hero wavelength used when spectral rendering is enabled (550 nm = green peak). */
const HERO_LAMBDA_NM = 550.0;

/**
 * Pack the per-frame uniform block into a std140-laid-out ArrayBuffer.
 *
 * Computes `cameraWorldMatrix = inverse(input.viewMatrix)` and
 * `invProjectionMatrix = inverse(input.projMatrix)` (throws on a singular
 * matrix). Scalars are written at their FRAME_PARAMS_SLOTS offsets; the two
 * mat4s are `.set()` at their 16-byte-aligned slots (column-major, so they map
 * 1:1 onto a GLSL `mat4`).
 */
export function packFrameParams(
  cfg: FrameParamsConfig,
  sceneInfo: FrameParamsScene,
  input: FrameInput,
  width: number,
  height: number,
): ArrayBuffer {
  const cameraWorldMatrix = invertMat4(input.viewMatrix);
  if (cameraWorldMatrix == null) {
    throw new Error('packFrameParams: non-invertible viewMatrix (singular)');
  }
  const invProjectionMatrix = invertMat4(input.projMatrix);
  if (invProjectionMatrix == null) {
    throw new Error('packFrameParams: non-invertible projMatrix (singular)');
  }

  const buf = new ArrayBuffer(FRAME_PARAMS_BYTE_SIZE);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);

  const S = FRAME_PARAMS_SLOTS;
  u32[S.resolutionX] = width >>> 0;
  u32[S.resolutionY] = height >>> 0;
  u32[S.frameSeed] = input.frameSeed >>> 0;
  u32[S.bounces] = cfg.maxBounces >>> 0;
  u32[S.triangleCount] = sceneInfo.triangleCount >>> 0;
  u32[S.bvhNodeCount] = sceneInfo.bvhNodeCount >>> 0;
  u32[S.lightCount] = sceneInfo.lightCount >>> 0;
  u32[S.hasEnvironmentMap] = sceneInfo.hasEnvironmentMap ? 1 : 0;
  u32[S.environmentMapWidth] = sceneInfo.environmentMapWidth >>> 0;
  u32[S.environmentMapHeight] = sceneInfo.environmentMapHeight >>> 0;

  u32[S.uSpectralRendering] = cfg.spectral ? 1 : 0;
  f32[S.heroLambdaNm] = HERO_LAMBDA_NM;
  f32[S.heroPdf] = 1.0;
  f32[S.cmfIntegralX] = X_CMF_INTEGRAL;
  f32[S.cmfIntegralY] = Y_CMF_INTEGRAL;
  f32[S.cmfIntegralZ] = Z_CMF_INTEGRAL;

  u32[S.uCausticStrategy] =
    cfg.causticStrategy === 'manifold-nee'
      ? 1
      : cfg.causticStrategy === 'photon-map'
        ? 2
        : 0;
  u32[S.uMneeMaxIterations] = cfg.mneeMaxIterations >>> 0;
  u32[S.uMneeMaxChainLength] = cfg.mneeMaxChainLength >>> 0;

  u32[S.uBdptEnabled] = cfg.bdpt ? 1 : 0;
  // Eye-subpath scratch depth = the active per-frame bounce limit.
  u32[S.uBdptMaxEyeDepth] = cfg.maxBounces >>> 0;

  f32.set(cameraWorldMatrix, S.cameraWorldMatrix);
  f32.set(invProjectionMatrix, S.invProjectionMatrix);

  return buf;
}

// ────────────────────────────────────────────────────────────────────────────
// Inline 4×4 column-major linear algebra (self-contained — see header note).
// ────────────────────────────────────────────────────────────────────────────

/**
 * Multiply two column-major 4×4 matrices: `out = a · b`. Exported for callers
 * that need a view-projection product (e.g. an `invViewProj` follow-up); the
 * packer itself only needs the two single-matrix inverses.
 */
export function multiplyMat4(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[col * 4 + row] =
        (a[0 * 4 + row] ?? 0) * (b[col * 4 + 0] ?? 0) +
        (a[1 * 4 + row] ?? 0) * (b[col * 4 + 1] ?? 0) +
        (a[2 * 4 + row] ?? 0) * (b[col * 4 + 2] ?? 0) +
        (a[3 * 4 + row] ?? 0) * (b[col * 4 + 3] ?? 0);
    }
  }
  return out;
}

/**
 * Invert a column-major 4×4 matrix via cofactor expansion. Returns `null` when
 * the determinant is ~0 (singular). Column-major in, column-major out, so the
 * result feeds straight into a std140 `mat4` slot.
 *
 * Reference: the standard adjugate/determinant inverse (e.g. gl-matrix
 * `mat4.invert`, MESA `__gluInvertMatrixd`).
 */
export function invertMat4(m: ArrayLike<number>): Float32Array | null {
  const a00 = m[0] ?? 0,
    a01 = m[1] ?? 0,
    a02 = m[2] ?? 0,
    a03 = m[3] ?? 0;
  const a10 = m[4] ?? 0,
    a11 = m[5] ?? 0,
    a12 = m[6] ?? 0,
    a13 = m[7] ?? 0;
  const a20 = m[8] ?? 0,
    a21 = m[9] ?? 0,
    a22 = m[10] ?? 0,
    a23 = m[11] ?? 0;
  const a30 = m[12] ?? 0,
    a31 = m[13] ?? 0,
    a32 = m[14] ?? 0,
    a33 = m[15] ?? 0;

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  const det =
    b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) {
    return null;
  }
  const invDet = 1.0 / det;

  const out = new Float32Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * invDet;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * invDet;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * invDet;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * invDet;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * invDet;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * invDet;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * invDet;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * invDet;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;
  return out;
}
