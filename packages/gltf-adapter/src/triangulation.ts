// triangulation.ts — TRIANGLE_STRIP / TRIANGLE_FAN → indexed triangle list.
//
// glTF 2.0 §3.7.2.1 defines the implicit triangle topology for the non-list
// triangle modes. The adapter converts both to an explicit indexed triangle
// list so the rest of the pipeline (and every @vitrum backend) only ever sees
// triangle lists:
//
//   TRIANGLE_STRIP (mode 5): triangle i =
//       { v[i],   v[i+1], v[i+2] }  when i is even
//       { v[i+1], v[i],   v[i+2] }  when i is odd
//     (the odd-triangle swap preserves a consistent front-face winding).
//
//   TRIANGLE_FAN (mode 6): triangle i = { v[i+1], v[i+2], v[0] }
//     (all triangles pivot on vertex 0; rotation of the tuple preserves
//      winding).
//
// Degenerate triangles (any two corner indices equal) are skipped — strips
// commonly carry repeated indices as restart/stitching degenerates and they
// must not reach BVH builders as zero-area triangles.
//
// Works for both indexed and non-indexed primitives: for non-indexed input the
// caller passes a sequential 0..vertexCount-1 index array (see
// `sequentialIndices`).
//
// Reference: glTF 2.0 specification (Khronos Group), §3.7.2.1
// https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html#geometry-overview

export const GLTF_MODE_TRIANGLE_STRIP = 5;
export const GLTF_MODE_TRIANGLE_FAN = 6;

/** Sequential indices [0, 1, …, count-1] for non-indexed primitives. */
export function sequentialIndices(count: number): Uint32Array {
  const out = new Uint32Array(count);
  for (let i = 0; i < count; i++) out[i] = i;
  return out;
}

/**
 * Convert TRIANGLE_STRIP / TRIANGLE_FAN source indices to an indexed triangle
 * list, preserving glTF winding and skipping degenerate triangles.
 *
 * @param src  - Source vertex indices in strip/fan order.
 * @param mode - 5 (TRIANGLE_STRIP) or 6 (TRIANGLE_FAN).
 * @returns A fresh Uint32Array triangle list (length is a multiple of 3;
 *   empty when fewer than 3 source indices or all triangles are degenerate).
 */
export function triangulateTopology(src: Uint32Array, mode: 5 | 6): Uint32Array {
  const n = src.length;
  if (n < 3) return new Uint32Array(0);
  const out = new Uint32Array((n - 2) * 3);
  let w = 0;
  if (mode === GLTF_MODE_TRIANGLE_STRIP) {
    for (let i = 0; i + 2 < n; i++) {
      const odd = (i & 1) === 1;
      const a = odd ? src[i + 1]! : src[i]!;
      const b = odd ? src[i]! : src[i + 1]!;
      const c = src[i + 2]!;
      if (a === b || b === c || a === c) continue; // degenerate
      out[w++] = a; out[w++] = b; out[w++] = c;
    }
  } else {
    const pivot = src[0]!;
    for (let i = 1; i + 1 < n; i++) {
      const a = src[i]!;
      const b = src[i + 1]!;
      if (a === b || a === pivot || b === pivot) continue; // degenerate
      out[w++] = a; out[w++] = b; out[w++] = pivot;
    }
  }
  return w === out.length ? out : out.slice(0, w);
}
