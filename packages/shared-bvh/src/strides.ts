/**
 * strides.ts — Single source of truth for BVH/vertex/matrix layout constants.
 *
 * ## Why this file exists
 *
 * The repo's #1 historical bug class is stride/layout drift: a literal `8`,
 * `4`, or `16` copied across packages falls out of sync with the actual data
 * layout, producing silent GPU corruption that only manifests at runtime.
 * Named instances:
 *
 *   • F-TLAS1  — bvhIndex.xyz stride-3-of-stride-4, all prior pt-webgpu
 *                baselines embedded it (fixed `1dd9e41`)
 *   • H41      — point/spot direct NEE inert because arrayLength-vec4-vs-float
 *                mismatch used wrong stride (fixed `25f4105`)
 *   • Merged-mode 16B-vs-32B black GI — TLAS used 16-byte BGL dummies vs the
 *                real 32-byte BVHNode min; hidden by the >1-mesh TLAS auto-rule
 *                (fixed `0bedd92`)
 *
 * All magic literals equal to 8, 4, or 16 that correspond to these three
 * layout facts MUST reference the constants below, not inline literals.
 *
 * ## Layout reference
 *
 * ### BVH node (BLAS + TLAS shared)
 * ```
 *   f32[0..2]  boundsMin xyz
 *   f32[3..5]  boundsMax xyz
 *   u32[6]     rightChildOrTriOffset  (relative for interior, absolute for leaf)
 *   u32[7]     splitAxisOrLeafCount   (0xFFFF0000 | count for leaves)
 * ```
 * = 8 × 4-byte words = 32 bytes per node.
 * `BVH_NODE_FLOATS` counts float32/uint32 words (not bytes).
 *
 * ### Vertex position / normal / UV layout
 * Vitrum packs all per-vertex attributes as vec4f (16 bytes/vertex):
 * ```
 *   positions[v * 4 + 0..2] = x, y, z   (.w = 0 or packed UV)
 *   normals  [v * 4 + 0..2] = nx, ny, nz (.w = 0)
 *   uvs      [v * 4 + 0..3] = u0, v0, u1, v1
 * ```
 * `VERTEX_STRIDE_F32` = floats per vertex in any of these arrays.
 *
 * ### Mat4 (world-to-local / local-to-world transforms)
 * Column-major 4×4 float matrix stored as a flat Float32Array of 16 floats.
 * `MAT4_STRIDE_F32` = floats per matrix.
 */

/**
 * Number of float32 (= uint32) words per BVH node.
 *
 * Layout: `[minX, minY, minZ, maxX, maxY, maxZ, rightChildOrOffset, splitOrCount]`
 * = 8 words × 4 bytes = **32 bytes per node**.
 *
 * Applies identically to BLAS nodes (triangle leaves) and TLAS nodes (instance
 * leaves); both tree builders use the same 32-byte node layout so the WGSL
 * traversal can share `bvhIntersect.wgsl`.
 */
export const BVH_NODE_FLOATS = 8 as const;

/**
 * Number of float32 words per vertex in the packed positions, normals, and UVs
 * arrays emitted by `packSceneFromCore`.
 *
 * Each vertex occupies one `vec4f` (16 bytes):
 * ```
 *   positions[v * 4 + 0..2] = x, y, z   (.w = 0, or packed UV in bvhCore)
 *   normals  [v * 4 + 0..2] = nx, ny, nz (.w = 0)
 *   uvs      [v * 4 + 0..3] = u0, v0, u1, v1
 * ```
 */
export const VERTEX_STRIDE_F32 = 4 as const;

/**
 * Number of float32 words per 4×4 column-major matrix.
 *
 * Used for `tlasInstanceWorldToLocal` and `tlasInstanceLocalToWorld` arrays
 * (16 floats × N instances) and for `analyticLocalToWorld` / `analyticWorldToLocal`
 * (16 floats × N analytics).
 */
export const MAT4_STRIDE_F32 = 16 as const;

/** Number of child slots in the compressed wide-BVH node. */
export const CWBVH_CHILDREN = 8 as const;

/**
 * Quantized bounds words per CWBVH child:
 * `[minX, minY, minZ, maxX, maxY, maxZ]`, each u16 relative to the parent node
 * bounds. This is the CPU/oracle-side form mirrored by the production WGSL
 * traversal after explicit u16-pair packing.
 */
export const CWBVH_CHILD_BOUNDS_U16 = 6 as const;

/**
 * WGSL storage buffers do not expose a `u16` scalar type for portable storage
 * reads, so CWBVH child bounds are uploaded as three packed u32 words per child
 * (`lo16 | hi16 << 16`). This is byte-equivalent to the six-u16 CPU oracle
 * form, but explicit packing avoids relying on host endianness.
 */
export const CWBVH_CHILD_BOUNDS_PACKED_U32 = 3 as const;

/**
 * Metadata words per CWBVH child:
 * `[kind, nodeIndexOrTriOffset, triCount]`.
 *
 * kind = 0 empty, 1 child wide-node, 2 leaf triangle range.
 */
export const CWBVH_CHILD_META_WORDS = 3 as const;

/** Fixed private-stack budgets compiled into the live WGSL traversals. */
export const BVH_TRAVERSAL_STACK_DEPTH = 60 as const;
export const CWBVH_TRAVERSAL_STACK_DEPTH = 64 as const;
export const TLAS_TRAVERSAL_STACK_DEPTH = 64 as const;

/**
 * Deepest interior level emitted by the canonical binary builder. Derive this
 * from the smallest live binary/wide/TLAS stack, reserving two slots because
 * the nested TLAS/BLAS paths push two children before the next pop.
 */
export const BINARY_BVH_MAX_BUILD_DEPTH = (
  Math.min(
    BVH_TRAVERSAL_STACK_DEPTH,
    CWBVH_TRAVERSAL_STACK_DEPTH,
    TLAS_TRAVERSAL_STACK_DEPTH,
  ) - 2
);
