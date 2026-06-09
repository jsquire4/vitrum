# @vitrum/shared-bvh

Shared BVH builder + canonical layout used by DDGI, RC, and ReSTIR GI engines (plus pt-webgpu's brute-force path tracer).

## Public surface

- Root exports are core-native and do not import host-renderer packages.
- `buildArrayBvh(positions, indices, triMaterialIds, opts)` — THREE-independent binned-SAH builder (Wald 2007). Takes raw `Float32Array`/`Uint32Array` inputs.
- `buildTlas(instances, opts?)` / `refitTlas(data, newAabbs)` / `tlasIntersect(data, origin, dir, tMax?)` — Top-Level Acceleration Structure. Binned SAH over instance world AABBs; same 32-byte node layout as BLAS so `bvhIntersect.wgsl` primitives can be reused with only the leaf-payload decode varying. `tlasIntersect` is a CPU reference oracle (leaf-conservative candidate set; production traversal happens in WGSL via the BLAS). Foundation for multi-mesh moving-object scenes; pipeline wiring downstream is a separate multi-week effort.
- `MaterialEntry` — canonical 16-float (64-byte) material struct + `packMaterials(mats)` packer. Shared by DDGI and RC; ReSTIR uses a different per-triangle u32 packing scheme.
- `BVH_INTERSECT_WGSL` — canonical WGSL string (Möller–Trumbore + AABB slab + 60-deep stack traversal) consumed by every WGSL-side BVH reader.
- `MATERIAL_ENTRY_WGSL` — canonical WGSL struct declaration for the material entry, kept in lockstep with the TS packer.
- `invertMat4` — 4×4 matrix inversion utility shared by TLAS packing and pt-webgpu's scene setup.
- `expandIndicesToStride4(indices, payloadFn?)` — explicit stride-3→stride-4 helper for consumers that need a `vec4u` index buffer payload lane. It is a public convenience, not a production pipeline dependency.

## Node layout

Each node is 32 bytes (8 × u32). Shared between BLAS + TLAS — the leaf
payload semantics differ but the byte layout + leaf-flag encoding match.

- f32[0..2] boundsMin xyz
- f32[3..5] boundsMax xyz
- u32[6] rightChildOrPayloadOffset:
  - interior: relative offset to right child (`rightChildIndex - thisNodeIndex`). Left child is always `thisNodeIndex + 1`. Invariant: `1 ≤ offset < totalNodes`.
  - leaf (BLAS): absolute triangle offset into the reordered-indices array.
  - leaf (TLAS): absolute instance offset into the `instanceIndices` permutation.
- u32[7] splitAxisOrPayloadCount:
  - interior: split axis (0=X, 1=Y, 2=Z)
  - leaf (BLAS): `0xFFFF0000 | triangleCount`
  - leaf (TLAS): `0xFFFF0000 | instanceCount`

Leaf-flag check: `(node[7] >>> 16) === 0xFFFF` (unsigned upper-16; avoids JS's int32-sign trap of `& 0xFFFF0000`).

## Status

Pre-1.0. Stable contract; new consumers should rely on the canonical exports rather than re-implementing the SAH heuristic or node layout.
