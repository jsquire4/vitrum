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
const IDENTITY_16 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

/** Multiply two column-major 4×4 matrices: result = a * b. */
function mat4Mul(a: Float32Array, b: Float32Array): Float32Array {
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
function trsToMat4(
  t: [number, number, number],
  r: [number, number, number, number],
  s: [number, number, number],
): Float32Array {
  // Scale matrix
  const S = new Float32Array(IDENTITY_16);
  S[0] = s[0]; S[5] = s[1]; S[10] = s[2];

  // Rotation matrix
  const R = quatToMat4(r);

  // Translation matrix
  const T = new Float32Array(IDENTITY_16);
  T[12] = t[0]; T[13] = t[1]; T[14] = t[2];

  // M = T * R * S
  return mat4Mul(T, mat4Mul(R, S));
}

/** Extract the local matrix from a glTF node (matrix || TRS || identity). */
function nodeLocalMatrix(node: GltfNode): Float32Array {
  if (node.matrix) {
    if (node.matrix.length !== 16) {
      throw new Error('[vitrum/gltf-adapter] Node matrix must have 16 elements');
    }
    return new Float32Array(node.matrix);
  }

  const t: [number, number, number] = node.translation ?? [0, 0, 0];
  const r: [number, number, number, number] = node.rotation ?? [0, 0, 0, 1];
  const s: [number, number, number] = node.scale ?? [1, 1, 1];
  return trsToMat4(t, r, s);
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
    (idx) => ({ nodeIdx: idx, parentWorld: new Float32Array(IDENTITY_16) }),
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
