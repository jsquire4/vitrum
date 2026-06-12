// tangents.ts — MikkTSpace-style fallback tangent generation for glTF meshes.
//
// glTF recommends authored TANGENT for normal mapped assets, but many assets
// omit it. Backends can derive tangents privately, but the adapter should hand
// downstream engines a predictable core Scene when POSITION/NORMAL/TEXCOORD_0
// are available.

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
): Float32Array | undefined {
  const vertexCount = positions.length / 3;
  if (!Number.isInteger(vertexCount) || vertexCount <= 0) return undefined;
  if (normals.length < vertexCount * 3 || uvs.length < vertexCount * 2) return undefined;

  const tangents = new Float32Array(vertexCount * 4);
  const tanAccum = new Float32Array(vertexCount * 3);
  const bitanAccum = new Float32Array(vertexCount * 3);
  const triCount = indices ? Math.floor(indices.length / 3) : Math.floor(vertexCount / 3);

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
    const r = 1 / denom;
    const tx = (dv2 * e1x - dv1 * e2x) * r;
    const ty = (dv2 * e1y - dv1 * e2y) * r;
    const tz = (dv2 * e1z - dv1 * e2z) * r;
    const bx = (du1 * e2x - du2 * e1x) * r;
    const by = (du1 * e2y - du2 * e1y) * r;
    const bz = (du1 * e2z - du2 * e1z) * r;

    for (const vi of [i0, i1, i2]) {
      tanAccum[vi * 3] = (tanAccum[vi * 3] ?? 0) + tx;
      tanAccum[vi * 3 + 1] = (tanAccum[vi * 3 + 1] ?? 0) + ty;
      tanAccum[vi * 3 + 2] = (tanAccum[vi * 3 + 2] ?? 0) + tz;
      bitanAccum[vi * 3] = (bitanAccum[vi * 3] ?? 0) + bx;
      bitanAccum[vi * 3 + 1] = (bitanAccum[vi * 3 + 1] ?? 0) + by;
      bitanAccum[vi * 3 + 2] = (bitanAccum[vi * 3 + 2] ?? 0) + bz;
    }
  }

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
      const fallback = tangentFromNormal(nx, ny, nz);
      tx = fallback[0];
      ty = fallback[1];
      tz = fallback[2];
      len = 1;
    }

    tangents[v * 4] = tx / len;
    tangents[v * 4 + 1] = ty / len;
    tangents[v * 4 + 2] = tz / len;
    tangents[v * 4 + 3] = handedness;
  }

  return tangents;
}

function tangentFromNormal(nx: number, ny: number, nz: number): readonly [number, number, number] {
  const sign = nz >= 0 ? 1 : -1;
  const a = -1 / (sign + nz);
  const b = nx * ny * a;
  const tx = 1 + sign * nx * nx * a;
  const ty = sign * b;
  const tz = -sign * nx;
  const len = Math.sqrt(tx * tx + ty * ty + tz * tz) || 1;
  return [tx / len, ty / len, tz / len];
}
