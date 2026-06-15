// transforms.ts — Node TRS / matrix → flat world transforms.
//
// glTF node hierarchy is a tree. We walk it depth-first, accumulating
// column-major 4×4 world transforms by multiplying parent × local.
//
// TRS → local matrix follows glTF §3.5 convention:
//   M = T · R · S
// where T, R, S are standard 4×4 column-major matrices.
//
// Column-major storage: M[col*4 + row] (same as WebGL / three.js convention).

import type { GltfJson, GltfNode } from './gltfTypes.js';
import { asMat4 } from '@vitrum/core';
import type { Mat4 } from '@vitrum/core';

/** Identity 4×4 column-major matrix. */
export const IDENTITY_MAT4 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

/** Multiply two column-major 4×4 matrices: result = a * b. */
export function mat4Mul(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += (a[k * 4 + row] ?? 0) * (b[col * 4 + k] ?? 0);
      }
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

/** Invert a column-major 4x4 matrix. Returns null for singular matrices. */
export function mat4Invert(m: ArrayLike<number>): Float32Array | null {
  const a00 = m[0]!, a01 = m[1]!, a02 = m[2]!, a03 = m[3]!;
  const a10 = m[4]!, a11 = m[5]!, a12 = m[6]!, a13 = m[7]!;
  const a20 = m[8]!, a21 = m[9]!, a22 = m[10]!, a23 = m[11]!;
  const a30 = m[12]!, a31 = m[13]!, a32 = m[14]!, a33 = m[15]!;

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
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null;
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

/**
 * Normalize a quaternion [x, y, z, w] in-place. Returns the (possibly
 * un-normalized) input if it is the zero quaternion (degenerate).
 */
function normalizeQuat(q: [number, number, number, number]): [number, number, number, number] {
  const len = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (len < 1e-10) return [0, 0, 0, 1];
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

/** Quaternion xyzw → column-major 4×4 rotation matrix. */
function quatToMat4(qIn: [number, number, number, number]): Float32Array {
  const [x, y, z, w] = normalizeQuat(qIn);
  const out = new Float32Array(16);
  // Column-major:
  out[0] = 1 - 2 * (y * y + z * z);
  out[1] = 2 * (x * y + z * w);
  out[2] = 2 * (x * z - y * w);
  out[3] = 0;

  out[4] = 2 * (x * y - z * w);
  out[5] = 1 - 2 * (x * x + z * z);
  out[6] = 2 * (y * z + x * w);
  out[7] = 0;

  out[8] = 2 * (x * z + y * w);
  out[9] = 2 * (y * z - x * w);
  out[10] = 1 - 2 * (x * x + y * y);
  out[11] = 0;

  out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
  return out;
}

/** Build a column-major 4×4 matrix from TRS. */
export function composeTrsMat4(
  t: [number, number, number],
  r: [number, number, number, number],
  s: [number, number, number],
): Float32Array {
  // Scale matrix
  const S = new Float32Array(IDENTITY_MAT4);
  S[0] = s[0]; S[5] = s[1]; S[10] = s[2];

  // Rotation matrix
  const R = quatToMat4(r);

  // Translation matrix
  const T = new Float32Array(IDENTITY_MAT4);
  T[12] = t[0]; T[13] = t[1]; T[14] = t[2];

  // M = T * R * S
  return mat4Mul(T, mat4Mul(R, S));
}

/** Extract the local matrix from a glTF node (matrix || TRS || identity). */
export function nodeLocalMatrix(node: GltfNode): Float32Array {
  if (node.matrix) {
    if (node.matrix.length !== 16) {
      throw new Error('[vitrum/gltf-adapter] Node matrix must have 16 elements');
    }
    return new Float32Array(node.matrix);
  }

  const t: [number, number, number] = node.translation ?? [0, 0, 0];
  const r: [number, number, number, number] = node.rotation ?? [0, 0, 0, 1];
  const s: [number, number, number] = node.scale ?? [1, 1, 1];
  return composeTrsMat4(t, r, s);
}

/**
 * Walk the node hierarchy of a glTF scene and return a map from node index →
 * world-space column-major 4×4 Mat4.
 *
 * Only nodes reachable from the given root nodes are traversed.
 */
export function buildWorldTransforms(
  gltf: GltfJson,
  rootNodeIndices: number[],
): Map<number, Mat4> {
  const result = new Map<number, Mat4>();
  const nodes = gltf.nodes ?? [];

  // Iterative DFS to avoid stack overflow on deep hierarchies.
  const stack: Array<{ nodeIdx: number; parentWorld: Float32Array }> = rootNodeIndices.map(
    (idx) => ({ nodeIdx: idx, parentWorld: new Float32Array(IDENTITY_MAT4) }),
  );

  while (stack.length > 0) {
    const entry = stack.pop()!;
    const node = nodes[entry.nodeIdx];
    if (!node) continue;
    if (result.has(entry.nodeIdx)) continue; // cycle guard

    const local = nodeLocalMatrix(node);
    const world = mat4Mul(entry.parentWorld, local);
    result.set(entry.nodeIdx, asMat4(world));

    for (const childIdx of node.children ?? []) {
      stack.push({ nodeIdx: childIdx, parentWorld: world });
    }
  }

  return result;
}
