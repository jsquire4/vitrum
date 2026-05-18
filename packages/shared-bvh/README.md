# @vitrum/shared-bvh

Shared BVH builder + canonical layout used by DDGI, RC, and ReSTIR GI engines (plus pt-webgpu's brute-force path tracer).

## Public surface

- `buildSceneBVH(roots, opts)` — wraps `three-mesh-bvh` and returns the canonical 32-byte-node layout. THREE-coupled.
- `buildArrayBvh(positions, indices, triMaterialIds, opts)` — THREE-independent binned-SAH builder (Wald 2007). Takes raw `Float32Array`/`Uint32Array` inputs.
- `MaterialEntry` — canonical 16-float (64-byte) material struct + `packMaterials(mats)` packer. Shared by DDGI and RC; ReSTIR uses a different per-triangle u32 packing scheme.
- `BVH_TRAVERSE_WGSL` — canonical WGSL string (Möller–Trumbore + AABB slab + 60-deep stack traversal) consumed by every WGSL-side BVH reader.
- `MATERIAL_ENTRY_WGSL` — canonical WGSL struct declaration for the material entry, kept in lockstep with the TS packer.

## Node layout

Each node is 32 bytes (8 × u32):
- f32[0..2] boundsMin xyz
- f32[3..5] boundsMax xyz
- u32[6] rightChildOrTriOffset:
  - interior: relative offset to right child (`rightChildIndex - thisNodeIndex`). Left child is always `thisNodeIndex + 1`. Invariant: `1 ≤ offset < totalNodes`.
  - leaf: absolute triangle offset into reordered-indices array.
- u32[7] splitAxisOrTriCount:
  - interior: split axis (0=X, 1=Y, 2=Z)
  - leaf: `0xFFFF0000 | triangleCount`

## Status

Pre-1.0. Stable contract; new consumers should rely on the canonical exports rather than re-implementing the SAH heuristic or node layout.
