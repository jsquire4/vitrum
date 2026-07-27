// tangents.ts — fallback tangent generation for glTF meshes.
//
// glTF recommends authored TANGENT for normal mapped assets, but many assets
// omit it. This fallback accumulates per-triangle tangent frames and
// Gram-Schmidt orthonormalizes against vertex normals; it is intentionally not a
// full MikkTSpace implementation. Backends can derive tangents privately, but
// the adapter should hand downstream engines a predictable core Scene when
// POSITION/NORMAL/TEXCOORD_0 are available.

import type { ImportResourceLedger } from './importResourceBudget.js';

/**
 * Generate per-vertex xyzw tangents from positions, normals, UV0 and triangles.
 * The xyz vector is Gram-Schmidt orthonormalized against the normal; w is the
 * bitangent handedness sign.
 */
export function generateTangents(
  positions: Float32Array,
  normals: Float32Array,
  uvs: Float32Array,
  indices: Uint32Array | Uint16Array | undefined,
  resourceLedger?: ImportResourceLedger,
  allocationPath = 'generated vertex tangents',
): Float32Array | undefined {
  const vertexCount = positions.length / 3;
  if (!Number.isInteger(vertexCount) || vertexCount <= 0) return undefined;
  if (normals.length < vertexCount * 3 || uvs.length < vertexCount * 2) return undefined;

  const tangentElementCount = checkedTangentProduct(
    vertexCount,
    4,
    `${allocationPath} output element count`,
  );
  const accumulatorElementCount = checkedTangentProduct(
    vertexCount,
    3,
    `${allocationPath} accumulator element count`,
  );
  const totalElementCount = checkedTangentSum(
    tangentElementCount,
    checkedTangentProduct(
      accumulatorElementCount,
      2,
      `${allocationPath} accumulator total`,
    ),
    `${allocationPath} total element count`,
  );
  resourceLedger?.chargeDecodedGeometryBytes(
    checkedTangentProduct(
      totalElementCount,
      Float32Array.BYTES_PER_ELEMENT,
      `${allocationPath} byte length`,
    ),
    allocationPath,
  );
  const tangents = new Float32Array(tangentElementCount);
  const tanAccum = new Float32Array(accumulatorElementCount);
  const bitanAccum = new Float32Array(accumulatorElementCount);
  const triCount = indices ? Math.floor(indices.length / 3) : Math.floor(vertexCount / 3);
  let validTriangleCount = 0;

  for (let t = 0; t < triCount; t += 1) {
    const i0 = indices ? (indices[t * 3] ?? 0) : t * 3;
    const i1 = indices ? (indices[t * 3 + 1] ?? 0) : t * 3 + 1;
    const i2 = indices ? (indices[t * 3 + 2] ?? 0) : t * 3 + 2;
    if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) continue;

    const p0x = positions[i0 * 3] ?? 0;
    const p0y = positions[i0 * 3 + 1] ?? 0;
    const p0z = positions[i0 * 3 + 2] ?? 0;
    const e1x = (positions[i1 * 3] ?? 0) - p0x;
    const e1y = (positions[i1 * 3 + 1] ?? 0) - p0y;
    const e1z = (positions[i1 * 3 + 2] ?? 0) - p0z;
    const e2x = (positions[i2 * 3] ?? 0) - p0x;
    const e2y = (positions[i2 * 3 + 1] ?? 0) - p0y;
    const e2z = (positions[i2 * 3 + 2] ?? 0) - p0z;

    const u0 = uvs[i0 * 2] ?? 0;
    const v0 = uvs[i0 * 2 + 1] ?? 0;
    const du1 = (uvs[i1 * 2] ?? 0) - u0;
    const dv1 = (uvs[i1 * 2 + 1] ?? 0) - v0;
    const du2 = (uvs[i2 * 2] ?? 0) - u0;
    const dv2 = (uvs[i2 * 2 + 1] ?? 0) - v0;

    const denom = du1 * dv2 - du2 * dv1;
    if (Math.abs(denom) < 1e-12) continue;
    validTriangleCount += 1;
    const r = 1 / denom;
    const tx = (dv2 * e1x - dv1 * e2x) * r;
    const ty = (dv2 * e1y - dv1 * e2y) * r;
    const tz = (dv2 * e1z - dv1 * e2z) * r;
    const bx = (du1 * e2x - du2 * e1x) * r;
    const by = (du1 * e2y - du2 * e1y) * r;
    const bz = (du1 * e2z - du2 * e1z) * r;

    accumulateTangentFrame(tanAccum, bitanAccum, i0, tx, ty, tz, bx, by, bz);
    accumulateTangentFrame(tanAccum, bitanAccum, i1, tx, ty, tz, bx, by, bz);
    accumulateTangentFrame(tanAccum, bitanAccum, i2, tx, ty, tz, bx, by, bz);
  }

  if (validTriangleCount === 0) return undefined;

  for (let v = 0; v < vertexCount; v += 1) {
    const nx = normals[v * 3] ?? 0;
    const ny = normals[v * 3 + 1] ?? 1;
    const nz = normals[v * 3 + 2] ?? 0;
    let tx = tanAccum[v * 3] ?? 0;
    let ty = tanAccum[v * 3 + 1] ?? 0;
    let tz = tanAccum[v * 3 + 2] ?? 0;

    const bx = bitanAccum[v * 3] ?? 0;
    const by = bitanAccum[v * 3 + 1] ?? 0;
    const bz = bitanAccum[v * 3 + 2] ?? 0;
    const cx = ny * tz - nz * ty;
    const cy = nz * tx - nx * tz;
    const cz = nx * ty - ny * tx;
    const handedness = (cx * bx + cy * by + cz * bz) < 0 ? -1 : 1;

    const ndt = nx * tx + ny * ty + nz * tz;
    tx -= nx * ndt;
    ty -= ny * ndt;
    tz -= nz * ndt;
    let len = Math.sqrt(tx * tx + ty * ty + tz * tz);

    if (len < 1e-8) {
      const sign = nz >= 0 ? 1 : -1;
      const a = -1 / (sign + nz);
      const b = nx * ny * a;
      tx = 1 + sign * nx * nx * a;
      ty = sign * b;
      tz = -sign * nx;
      const fallbackLength = Math.sqrt(tx * tx + ty * ty + tz * tz) || 1;
      tx /= fallbackLength;
      ty /= fallbackLength;
      tz /= fallbackLength;
      len = 1;
    }

    tangents[v * 4] = tx / len;
    tangents[v * 4 + 1] = ty / len;
    tangents[v * 4 + 2] = tz / len;
    tangents[v * 4 + 3] = handedness;
  }

  return tangents;
}

function checkedTangentProduct(left: number, right: number, label: string): number {
  if (
    !Number.isSafeInteger(left) ||
    left < 0 ||
    !Number.isSafeInteger(right) ||
    right < 0 ||
    (left !== 0 && right > Math.floor(Number.MAX_SAFE_INTEGER / left))
  ) {
    throw new RangeError(`[vitrum/gltf-adapter] ${label} exceeds the safe integer range.`);
  }
  return left * right;
}

function checkedTangentSum(left: number, right: number, label: string): number {
  if (
    !Number.isSafeInteger(left) ||
    left < 0 ||
    !Number.isSafeInteger(right) ||
    right < 0 ||
    right > Number.MAX_SAFE_INTEGER - left
  ) {
    throw new RangeError(`[vitrum/gltf-adapter] ${label} exceeds the safe integer range.`);
  }
  return left + right;
}

function accumulateTangentFrame(
  tangents: Float32Array,
  bitangents: Float32Array,
  vertex: number,
  tx: number,
  ty: number,
  tz: number,
  bx: number,
  by: number,
  bz: number,
): void {
  const offset = vertex * 3;
  tangents[offset] = (tangents[offset] ?? 0) + tx;
  tangents[offset + 1] = (tangents[offset + 1] ?? 0) + ty;
  tangents[offset + 2] = (tangents[offset + 2] ?? 0) + tz;
  bitangents[offset] = (bitangents[offset] ?? 0) + bx;
  bitangents[offset + 1] = (bitangents[offset + 1] ?? 0) + by;
  bitangents[offset + 2] = (bitangents[offset + 2] ?? 0) + bz;
}
