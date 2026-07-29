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
import type { ImportResourceLedger } from './importResourceBudget.js';

/** Identity 4×4 column-major matrix. */
export const IDENTITY_MAT4 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

function chargeMatrixAllocation(
  resourceLedger: ImportResourceLedger | undefined,
  allocationPath: string,
): void {
  resourceLedger?.chargeDecodedGeometryBytes(
    16 * Float32Array.BYTES_PER_ELEMENT,
    allocationPath,
  );
}

/** Multiply two column-major 4×4 matrices: result = a * b. */
export function mat4Mul(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  resourceLedger?: ImportResourceLedger,
  allocationPath = 'matrix multiplication result',
): Float32Array {
  chargeMatrixAllocation(resourceLedger, allocationPath);
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
export function mat4Invert(
  m: ArrayLike<number>,
  resourceLedger?: ImportResourceLedger,
  allocationPath = 'matrix inverse',
): Float32Array | null {
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
  if (!Number.isFinite(det) || det === 0) return null;
  const invDet = 1.0 / det;

  chargeMatrixAllocation(resourceLedger, allocationPath);
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

  // A fixed absolute determinant threshold is not scale invariant: a valid
  // uniform scale(1e-5) has determinant 1e-15 but a perfectly representable
  // Float32 reciprocal. Validate the reciprocal we will actually publish
  // instead. The relative term-sum tolerance matches the import preflight's
  // Float32 reciprocity proof and still rejects ill-conditioned/non-finite
  // inverses.
  for (const component of out) {
    if (!Number.isFinite(component)) return null;
  }
  for (const [left, right] of [[m, out], [out, m]] as const) {
    for (let column = 0; column < 4; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        let product = 0;
        let absoluteTermSum = 0;
        for (let inner = 0; inner < 4; inner += 1) {
          const term = (left[inner * 4 + row] ?? 0) *
            (right[column * 4 + inner] ?? 0);
          product += term;
          absoluteTermSum += Math.abs(term);
        }
        const expected = row === column ? 1 : 0;
        const tolerance = 1e-5 * Math.max(1, absoluteTermSum);
        if (!Number.isFinite(product) || Math.abs(product - expected) > tolerance) {
          return null;
        }
      }
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

/** Build a column-major 4×4 matrix from TRS. */
export function composeTrsMat4(
  t: [number, number, number],
  r: [number, number, number, number],
  s: [number, number, number],
  resourceLedger?: ImportResourceLedger,
  allocationPath = 'composed TRS matrix',
): Float32Array {
  const [x, y, z, w] = normalizeQuat(r);
  const sx = s[0];
  const sy = s[1];
  const sz = s[2];
  chargeMatrixAllocation(resourceLedger, allocationPath);
  const out = new Float32Array(16);

  // M = T · R · S, written directly in column-major form. The previous
  // implementation materialised S, R, T and two multiplication results; this
  // single allocation is both cheaper and exactly budgetable.
  out[0] = (1 - 2 * (y * y + z * z)) * sx;
  out[1] = (2 * (x * y + z * w)) * sx;
  out[2] = (2 * (x * z - y * w)) * sx;
  out[3] = 0;

  out[4] = (2 * (x * y - z * w)) * sy;
  out[5] = (1 - 2 * (x * x + z * z)) * sy;
  out[6] = (2 * (y * z + x * w)) * sy;
  out[7] = 0;

  out[8] = (2 * (x * z + y * w)) * sz;
  out[9] = (2 * (y * z - x * w)) * sz;
  out[10] = (1 - 2 * (x * x + y * y)) * sz;
  out[11] = 0;

  out[12] = t[0];
  out[13] = t[1];
  out[14] = t[2];
  out[15] = 1;
  return out;
}

/** Extract the local matrix from a glTF node (matrix || TRS || identity). */
export function nodeLocalMatrix(
  node: GltfNode,
  resourceLedger?: ImportResourceLedger,
  allocationPath = 'node local matrix',
): Float32Array {
  if (node.matrix) {
    if (node.matrix.length !== 16) {
      throw new Error('[vitrum/gltf-adapter] Node matrix must have 16 elements');
    }
    chargeMatrixAllocation(resourceLedger, allocationPath);
    return new Float32Array(node.matrix);
  }

  const t: [number, number, number] = node.translation ?? [0, 0, 0];
  const r: [number, number, number, number] = node.rotation ?? [0, 0, 0, 1];
  const s: [number, number, number] = node.scale ?? [1, 1, 1];
  return composeTrsMat4(t, r, s, resourceLedger, allocationPath);
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
  resourceLedger?: ImportResourceLedger,
  allocationPath = 'scene node transforms',
): Map<number, Mat4> {
  const result = new Map<number, Mat4>();
  const nodes = gltf.nodes ?? [];

  // Iterative DFS to avoid stack overflow on deep hierarchies.
  const stack: Array<{ nodeIdx: number; parentWorld: Float32Array }> = rootNodeIndices.map(
    (idx) => ({ nodeIdx: idx, parentWorld: IDENTITY_MAT4 }),
  );

  while (stack.length > 0) {
    const entry = stack.pop()!;
    const node = nodes[entry.nodeIdx];
    if (!node) continue;
    if (result.has(entry.nodeIdx)) continue; // cycle guard

    const nodePath = `${allocationPath}.nodes[${entry.nodeIdx}]`;
    const local = nodeLocalMatrix(
      node,
      resourceLedger,
      `${nodePath}.localMatrix`,
    );
    const world = mat4Mul(
      entry.parentWorld,
      local,
      resourceLedger,
      `${nodePath}.worldMatrix`,
    );
    result.set(entry.nodeIdx, asMat4(world));

    for (const childIdx of node.children ?? []) {
      stack.push({ nodeIdx: childIdx, parentWorld: world });
    }
  }

  return result;
}
