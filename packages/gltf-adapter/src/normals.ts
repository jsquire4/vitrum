// normals.ts — Flat normal generation for meshes that omit NORMAL.
//
// When a glTF primitive has no NORMAL accessor, we generate per-face (flat)
// normals: all three vertices of each triangle receive the same normal equal to
// cross(e1, e2) / |cross(e1, e2)| where e1 = v1 - v0, e2 = v2 - v0.
//
// Degenerate triangles (zero area) receive normal [0, 1, 0] as a safe fallback.

/**
 * Generate flat normals for a triangle mesh.
 *
 * @param positions - interleaved xyz triples (length = vertexCount * 3).
 * @param indices   - optional index buffer. If absent, vertices are a triangle list.
 * @returns Float32Array of length positions.length with flat per-vertex normals.
 */
export function generateFlatNormals(
  positions: Float32Array,
  indices: Uint32Array | Uint16Array | undefined,
): Float32Array {
  const vertexCount = positions.length / 3;
  const normals = new Float32Array(positions.length);

  const triCount = indices ? indices.length / 3 : Math.floor(vertexCount / 3);

  for (let t = 0; t < triCount; t++) {
    const i0 = indices ? (indices[t * 3] ?? 0) : t * 3;
    const i1 = indices ? (indices[t * 3 + 1] ?? 0) : t * 3 + 1;
    const i2 = indices ? (indices[t * 3 + 2] ?? 0) : t * 3 + 2;

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

    const len = Math.sqrt(cx * cx + cy * cy + cz * cz);
    const nx = len > 1e-10 ? cx / len : 0;
    const ny = len > 1e-10 ? cy / len : 1;
    const nz = len > 1e-10 ? cz / len : 0;

    // Assign the same normal to all three vertices of this triangle.
    for (const vi of [i0, i1, i2]) {
      normals[vi * 3] = nx;
      normals[vi * 3 + 1] = ny;
      normals[vi * 3 + 2] = nz;
    }
  }

  return normals;
}
