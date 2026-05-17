/**
 * bdptVertex.ts — BDPT path vertex type definitions for CPU↔GPU exchange.
 *
 * Defines the canonical BDPTVertex record that the CPU library packs and the
 * fork's WebGL2 GLSL kernels read/write via texture ping-pong. Every field is
 * placed at a fixed float offset so both sides agree on the layout without any
 * additional negotiation.
 *
 * GPU texture layout — 3 RGBA32F texels per vertex (12 floats, 48 bytes):
 *
 *   Texel 0 (row 0):  position.x | position.y | position.z | kind (0–3)
 *   Texel 1 (row 1):  normal.x   | normal.y   | normal.z   | pdfFwd
 *   Texel 2 (row 2):  throughput.x | throughput.y | throughput.z | pdfRev
 *
 * Float-by-float index map (base = vertexIndex * BDPT_VERTEX_FLOATS):
 *   base +  0 → position[0]    (world-space x)
 *   base +  1 → position[1]    (world-space y)
 *   base +  2 → position[2]    (world-space z)
 *   base +  3 → kind           (0=light, 1=eye, 2=connection, 3=invalid)
 *   base +  4 → normal[0]      (shading normal x, unit-length)
 *   base +  5 → normal[1]      (shading normal y, unit-length)
 *   base +  6 → normal[2]      (shading normal z, unit-length)
 *   base +  7 → pdfFwd         (forward PDF of this vertex in path-probability units)
 *   base +  8 → throughput[0]  (path weight accumulated to this vertex, red)
 *   base +  9 → throughput[1]  (path weight accumulated to this vertex, green)
 *   base + 10 → throughput[2]  (path weight accumulated to this vertex, blue)
 *   base + 11 → pdfRev         (reverse PDF — used by MIS weight computation)
 *
 * In the fork's GLSL the texture is written from the light-subpath pass as:
 *
 *   layout(location = 0) out vec4 gVertex0;  // position.xyz, kind
 *   layout(location = 1) out vec4 gVertex1;  // normal.xyz, pdfFwd
 *   layout(location = 2) out vec4 gVertex2;  // throughput.rgb, pdfRev
 *
 * and fetched in the connection pass via texelFetch(uLightPathTex, ivec2(vtxIdx, row), 0).
 *
 * References:
 *   - Veach 1997, "Robust Monte Carlo Methods for Light Transport Simulation",
 *     PhD thesis, Stanford. §10 (bidirectional path tracing), Algorithm 10.4.
 *   - Pharr, Jakob, Humphreys 2023, "Physically Based Rendering" (4th ed.),
 *     §16.3 (BDPT vertex formulation).
 */

// ────────────────────────────────────────────────────────────────────────────
// Kind constants
// ────────────────────────────────────────────────────────────────────────────

/** Vertex originated on the light subpath (from emitter toward scene). */
export const BDPT_KIND_LIGHT = 0 as const;

/** Vertex originated on the eye subpath (from camera toward scene). */
export const BDPT_KIND_EYE = 1 as const;

/** Placeholder kind used for explicit connection vertices (rare / debug use). */
export const BDPT_KIND_CONNECTION = 2 as const;

/** Invalid / uninitialized vertex — GLSL uses this to skip a texture slot. */
export const BDPT_KIND_INVALID = 3 as const;

// ────────────────────────────────────────────────────────────────────────────
// Size constants
// ────────────────────────────────────────────────────────────────────────────

/** Number of floats per BDPTVertex in the packed texture representation. */
export const BDPT_VERTEX_FLOATS = 12; // 3 RGBA32F texels

/** Byte size of one BDPTVertex in the packed representation (12 × 4 bytes). */
export const BDPT_VERTEX_BYTES = 48;

// ────────────────────────────────────────────────────────────────────────────
// Bounce limits — moved to @vitrum/pt-webgl in W7-H7
// ────────────────────────────────────────────────────────────────────────────
//
// BDPT_MAX_LIGHT_BOUNCES (=3) and BDPT_MAX_EYE_BOUNCES (=12) used to live
// here. They were always fork-specific budget choices (light-path texture
// size, eye-subpath loop count) — generic BDPT vertex layout doesn't care.
// They now live alongside the rest of the forkUniformBridge wiring in
// `@vitrum/pt-webgl/src/forkUniformBridge.ts`.

// ────────────────────────────────────────────────────────────────────────────
// Type
// ────────────────────────────────────────────────────────────────────────────

/**
 * A single vertex along a BDPT path (light or eye subpath).
 *
 * Encoded into a 3-texel RGBA32F row for texture ping-pong storage in WebGL2.
 * See the module-level comment for the exact float-offset mapping.
 */
export interface BDPTVertex {
  /** World-space position of the vertex. */
  readonly position: readonly [number, number, number];
  /**
   * Vertex kind.
   *   0 = BDPT_KIND_LIGHT       (light subpath vertex)
   *   1 = BDPT_KIND_EYE         (eye subpath vertex)
   *   2 = BDPT_KIND_CONNECTION  (connection vertex, debug)
   *   3 = BDPT_KIND_INVALID     (empty slot — skip in GLSL)
   */
  readonly kind: 0 | 1 | 2 | 3;
  /** Shading normal at the vertex, unit-length. */
  readonly normal: readonly [number, number, number];
  /**
   * Forward PDF of this vertex (probability density of sampling this vertex
   * along the path's traversal direction, in solid-angle measure).
   */
  readonly pdfFwd: number;
  /**
   * RGB path-throughput accumulated from the path origin to this vertex.
   * For light subpath vertices this is the emitter contribution × BSDF product.
   * For eye subpath vertices this is the camera response × BSDF product.
   */
  readonly throughput: readonly [number, number, number];
  /**
   * Reverse PDF of this vertex (solid-angle PDF of arriving at this vertex
   * from the opposite direction — used in Veach MIS weight computation).
   */
  readonly pdfRev: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Pack / unpack helpers
// ────────────────────────────────────────────────────────────────────────────

/**
 * Pack a BDPTVertex into 12 floats starting at `target[offset]`.
 *
 * The target Float32Array must have capacity for at least `offset + 12` floats.
 * No bounds checking is performed for performance (mirrors the GPU convention).
 *
 * Layout (see module header for GLSL correspondence):
 *   [offset +  0] = position.x
 *   [offset +  1] = position.y
 *   [offset +  2] = position.z
 *   [offset +  3] = kind
 *   [offset +  4] = normal.x
 *   [offset +  5] = normal.y
 *   [offset +  6] = normal.z
 *   [offset +  7] = pdfFwd
 *   [offset +  8] = throughput.x (red)
 *   [offset +  9] = throughput.y (green)
 *   [offset + 10] = throughput.z (blue)
 *   [offset + 11] = pdfRev
 */
export function packBDPTVertex(v: BDPTVertex, target: Float32Array, offset: number): void {
  target[offset + 0] = v.position[0];
  target[offset + 1] = v.position[1];
  target[offset + 2] = v.position[2];
  target[offset + 3] = v.kind;
  target[offset + 4] = v.normal[0];
  target[offset + 5] = v.normal[1];
  target[offset + 6] = v.normal[2];
  target[offset + 7] = v.pdfFwd;
  target[offset + 8] = v.throughput[0];
  target[offset + 9] = v.throughput[1];
  target[offset + 10] = v.throughput[2];
  target[offset + 11] = v.pdfRev;
}

/**
 * Unpack a BDPTVertex from 12 floats starting at `source[offset]`.
 *
 * The `kind` field is cast to the discriminated union. If the stored float
 * is not one of 0–3 (e.g. NaN, out-of-range), it is coerced to
 * BDPT_KIND_INVALID (3).
 *
 * @param source - Float32Array containing packed vertex data
 * @param offset - starting float index within `source`
 * @returns a new BDPTVertex value object
 */
export function unpackBDPTVertex(source: Float32Array, offset: number): BDPTVertex {
  const rawKind = source[offset + 3] ?? 3;
  const kind = (rawKind === 0 || rawKind === 1 || rawKind === 2)
    ? (rawKind as 0 | 1 | 2)
    : (3 as 3);

  return {
    position: [
      source[offset + 0] ?? 0,
      source[offset + 1] ?? 0,
      source[offset + 2] ?? 0,
    ],
    kind,
    normal: [
      source[offset + 4] ?? 0,
      source[offset + 5] ?? 0,
      source[offset + 6] ?? 0,
    ],
    pdfFwd: source[offset + 7] ?? 0,
    throughput: [
      source[offset + 8] ?? 0,
      source[offset + 9] ?? 0,
      source[offset + 10] ?? 0,
    ],
    pdfRev: source[offset + 11] ?? 0,
  };
}
