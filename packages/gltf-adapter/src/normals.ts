// normals.ts — Vertex-normal generation for meshes that omit NORMAL.
//
// Non-indexed triangle lists receive per-face normals. Indexed meshes cannot
// represent true flat normals without duplicating every shared vertex (and all
// of its attributes), so they receive area-weighted smooth normals instead.
//
// Degenerate triangles (zero area) receive normal [0, 1, 0] as a safe fallback.

import type { ImportResourceLedger } from './importResourceBudget.js';

/**
 * Generate normals for a triangle mesh.
 *
 * @param positions - interleaved xyz triples (length = vertexCount * 3).
 * @param indices   - optional index buffer. Indexed meshes are smoothed by
 *                    accumulating unnormalized face normals at shared vertices.
 * @returns Float32Array of length positions.length with unit vertex normals.
 */
export function generateVertexNormals(
  positions: Float32Array,
  indices: Uint32Array | Uint16Array | undefined,
  resourceLedger?: ImportResourceLedger,
  allocationPath = 'generated vertex normals',
): Float32Array {
  if (positions.length % 3 !== 0) {
    throw new RangeError(
      'generateVertexNormals: positions length must be divisible by 3',
    );
  }
  const vertexCount = Math.floor(positions.length / 3);
  const accumulatorBytes = positions.length * Float64Array.BYTES_PER_ELEMENT;
  resourceLedger?.chargeDecodedGeometryBytes(
    positions.byteLength + accumulatorBytes,
    allocationPath,
  );
  const normals = new Float32Array(positions.length);
  const accumulated = new Float64Array(positions.length);

  const triCount = indices ? Math.floor(indices.length / 3) : Math.floor(vertexCount / 3);
  for (let t = 0; t < triCount; t++) {
    const i0 = indices ? (indices[t * 3] ?? 0) : t * 3;
    const i1 = indices ? (indices[t * 3 + 1] ?? 0) : t * 3 + 1;
    const i2 = indices ? (indices[t * 3 + 2] ?? 0) : t * 3 + 2;

    if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) {
      throw new RangeError(
        `generateVertexNormals: triangle ${t} references a vertex outside [0, ${vertexCount})`,
      );
    }

    const ax = positions[i0 * 3] ?? 0;
    const ay = positions[i0 * 3 + 1] ?? 0;
    const az = positions[i0 * 3 + 2] ?? 0;

    const e1x = (positions[i1 * 3] ?? 0) - ax;
    const e1y = (positions[i1 * 3 + 1] ?? 0) - ay;
    const e1z = (positions[i1 * 3 + 2] ?? 0) - az;

    const e2x = (positions[i2 * 3] ?? 0) - ax;
    const e2y = (positions[i2 * 3 + 1] ?? 0) - ay;
    const e2z = (positions[i2 * 3 + 2] ?? 0) - az;

    // Cross product e1 × e2
    const cx = e1y * e2z - e1z * e2y;
    const cy = e1z * e2x - e1x * e2z;
    const cz = e1x * e2y - e1y * e2x;

    accumulateNormal(accumulated, i0, cx, cy, cz);
    accumulateNormal(accumulated, i1, cx, cy, cz);
    accumulateNormal(accumulated, i2, cx, cy, cz);
  }

  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const base = vertex * 3;
    const x = accumulated[base] ?? 0;
    const y = accumulated[base + 1] ?? 0;
    const z = accumulated[base + 2] ?? 0;
    const scale = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
    if (scale > 0 && Number.isFinite(scale)) {
      const sx = x / scale;
      const sy = y / scale;
      const sz = z / scale;
      const length = Math.hypot(sx, sy, sz);
      normals[base] = sx / length;
      normals[base + 1] = sy / length;
      normals[base + 2] = sz / length;
    } else {
      normals[base] = 0;
      normals[base + 1] = 1;
      normals[base + 2] = 0;
    }
  }

  return normals;
}

function accumulateNormal(
  normals: Float64Array,
  vertex: number,
  x: number,
  y: number,
  z: number,
): void {
  const offset = vertex * 3;
  normals[offset] = (normals[offset] ?? 0) + x;
  normals[offset + 1] = (normals[offset + 1] ?? 0) + y;
  normals[offset + 2] = (normals[offset + 2] ?? 0) + z;
}
