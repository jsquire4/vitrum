# TSL → Raw WebGPU Mapping

**Context**: Phase 4 Step 4, RC subsystem extraction from `_staging/legacy-source/`.
This is a historical conversion note for the raw WebGPU dispatcher. The package
root no longer exposes a `src/three` or `/three` bridge; current engine-level
composition lives in `@vitrum/walkaround-hybrid`.

---

## TSL primitives encountered

### `storage(buffer, type, count[, access])`

**Files**: `cascadeDispatch.ts`, `cascadePyramid.ts`

**TSL semantics**: Wraps a `StorageBufferAttribute` as a TSL storage-buffer node.
The node declares a `var<storage, ...>` binding in the generated WGSL. The optional
`.toReadOnly()` call sets access mode to `read`; without it defaults to `read_write`.

**In `cascadeDispatch.ts`** (compute kernel):
- `storage(sceneBVH.bvhNodes, 'BVHNode', count).toReadOnly()` → read-only storage
- `storage(sceneBVH.indices,  'vec3u', count).toReadOnly()` → read-only storage
- `storage(sceneBVH.positions, 'vec3f', count).toReadOnly()` → read-only storage
- `storage(sceneBVH.materials, 'MaterialEntry', count).toReadOnly()` → read-only storage
- `storage(sceneBVH.triMaterialId, 'u32', count).toReadOnly()` → read-only storage
- `storage(cascadeAttr, 'vec4f', count)` (no toReadOnly) → read_write storage (cascade output)
- `storage(uniformAttr, 'CascadeUniforms', 1).toReadOnly()` → read-only storage (UBO-as-SSBO)
- `storage(upperAttr, 'vec4f', count).toReadOnly()` → read-only storage (upper cascade)
- `storage(lowerAttr, 'vec4f', count)` → read_write storage (lower cascade merge target)
- `storage(mergeAttr, 'MergeUniforms', 1).toReadOnly()` → read-only storage

**Raw WebGPU equivalent**:

```typescript
// Each storage() call maps to one GPUBindGroupLayoutEntry:
// .toReadOnly() → buffer: { type: 'read-only-storage' }
// no toReadOnly → buffer: { type: 'storage' }

const layoutEntry: GPUBindGroupLayoutEntry = {
  binding: N,
  visibility: GPUShaderStage.COMPUTE,
  buffer: { type: 'read-only-storage' },   // or 'storage' for writable
};
```

**WGSL declaration**:
```wgsl
// read-only:  @group(0) @binding(N) var<storage, read> myBuffer: array<MyType>;
// read-write: @group(0) @binding(N) var<storage, read_write> myBuffer: array<MyType>;
```

**Binding index assignment**: TSL infers binding indices automatically. The raw conversion
uses explicit indices. The convention adopted here (matching parameter order in the
`wgslFn` kernel signature):

**Cast pass bindings** (one bind group per cascade) — current production ABI:
| binding | name                       | access     | WGSL type                                  |
|---------|----------------------------|------------|--------------------------------------------|
| 0       | `rc_bvh`                   | read-only  | `array<BVHNode>`                           |
| 1       | `rc_geom_index`            | read-only  | `array<vec4u>`                             |
| 2       | `rc_geom_position`         | read-only  | `array<vec4f>`                             |
| 3       | `rc_scene_arena`           | read-only  | `array<u32>`                               |
| 5       | `rc_cascadeOut`            | read_write | `array<vec4f>`                             |
| 6       | `rc_envMap`                | texture    | `texture_2d<f32>`                          |
| 7       | `rc_envSampler`            | sampler    | `sampler`                                  |
| 8       | `rc_u`                     | uniform    | `CascadeUniforms`                          |
| 14      | `rc_emitters`              | read-only  | packed emitter words (`array<u32>`)        |
| 15      | `rc_lights`                | read-only  | packed light/alias words (`array<u32>`)    |
| 16      | `rc_materialTextureAtlas`  | texture    | `texture_2d_array<u32>`                    |
| 17      | `rc_materialMapMeta`       | texture    | `texture_2d<f32>`                          |
| 18      | `rc_geom_normal`           | read-only  | `array<vec4f>`                             |
| 19      | `rc_geom_tangent`          | texture    | `texture_2d<f32>`                          |
| 20      | `rc_geom_vertex_color`     | texture    | `texture_2d<f32>`                          |

The scene arena at binding 3 packs materials, triangle-material IDs, and all
five TLAS streams behind a 16-word offset/count header. This keeps the cast
shader at WebGPU's guaranteed eight-storage-buffer floor. Missing environment,
material-map, tangent, vertex-color, emitter, and light inputs bind explicit
well-typed placeholders; scalar/count uniforms select the authored fallback.

**Merge pass bindings** (one bind group per merge step):
| binding | name         | access         | WGSL type              |
|---------|--------------|----------------|------------------------|
| 0       | upperCascade | read-only      | `array<vec4f>`         |
| 1       | lowerCascade | read_write     | `array<vec4f>`         |
| 2       | m_arr        | read-only      | `array<MergeUniforms>` |

**Subtlety — BVH buffers shared across passes**: In TSL, the BVH `storage()` nodes are
created once and reused across all 5 cast passes. In raw WebGPU, the same `GPUBuffer`
is referenced from each pass's bind group; only the cascade output buffer and uniforms
buffer differ per pass. The BVH bind group layout is identical for all cast passes.

---

### `wgslFn(source, includes?)`

**Files**: `probeRayCast.wgsl.ts`, `cascadeMerge.wgsl.ts`

**TSL semantics**: Wraps a WGSL function string as a TSL node. When included in a
`compute()` or another `wgslFn`'s includes list, TSL concatenates all WGSL source
strings in dependency order to produce the final WGSL shader module. The `includes`
array specifies transitive dependencies; TSL deduplicates them.

**Raw WebGPU equivalent**: The `source` string is extracted verbatim and assembled
with all its dependencies into a single WGSL string:

```typescript
const code = `
  ${MATERIAL_ENTRY_WGSL}
  ${BVH_INTERSECT_CORE_WGSL}
  ${TLAS_TRAVERSAL_CORE_WGSL}
  // RC scene-arena loaders and binding declarations
  ${OCTAHEDRAL_CORE_WGSL}
  ${PCG_HASH_TO_F32_WGSL}
  ${RC_MATERIAL_ATLAS_WGSL}
  ${RC_BRDF_WGSL}
  ${RC_SUN_VISIBILITY_WGSL}
  ${RC_NEE_POINTSPOT_WGSL}
  ${PROBE_RAY_CAST_ENTRY}
`;
device.createShaderModule({ code });
```

**`wgsl()` nodes** (structure declarations): These are also WGSL string literals
(not functions), extracted the same way. They appear in the `wgsl` tag call.

**Subtlety — `wgslFn` vs `wgsl`**: TSL `wgsl()` declares structs/constants (no `fn`
prefix required). `wgslFn()` declares functions (string must start with `fn`). In raw
WebGPU both kinds of content go into the same module source string.

**Subtlety — `@compute @workgroup_size`**: TSL's `compute(fn, [workgroupSize])` call
adds the `@compute @workgroup_size(...)` entry point decorator. In raw WebGPU this
decorator is added manually to the kernel entry function in the WGSL string.

**Workgroup size for cast passes**: `.compute(totalRays, [64])` → `@workgroup_size(64)`.
**Dispatch size for cast passes**: `Math.ceil(totalRays / 64)` workgroups.

**Workgroup size for merge passes**: `.compute(totalLower, [64])` → `@workgroup_size(64)`.
**Dispatch size for merge passes**: `Math.ceil(totalLower / 64)` workgroups.

---

### `compute(fn, dispatchSize)`

**Files**: `cascadeDispatch.ts`

**TSL semantics**: Creates a compute node from a `wgslFn` call. The `dispatchSize`
argument is `[totalInvocations, workgroupSize]` where Three.js computes
`ceil(totalInvocations / workgroupSize)` workgroup dispatches.

In practice:
```typescript
// TSL:
wgslFn_node.compute(totalRays, [64])
// means: dispatch ceil(totalRays / 64) workgroups, each with 64 threads
```

**Raw WebGPU equivalent**:
```typescript
const workgroupCount = Math.ceil(totalRays / 64);
passEncoder.dispatchWorkgroups(workgroupCount);
```

**Subtlety**: TSL's `compute()` first argument is the total thread count, not the
workgroup count. Raw WebGPU `dispatchWorkgroups()` takes workgroup count directly.
The conversion must apply `Math.ceil(totalThreads / workgroupSize)`.

---

### `instanceIndex`

**Files**: `cascadeDispatch.ts`

**TSL semantics**: The TSL built-in property node for `@builtin(global_invocation_id).x`.
When passed as a `wgslFn` parameter named `index`, it becomes the thread's flat linear
index within the dispatch.

**Raw WebGPU equivalent**: In the WGSL kernel, `index` is declared as a parameter
with `@builtin(global_invocation_id)`:

```wgsl
@compute @workgroup_size(64)
fn probeRayCastKernel(@builtin(global_invocation_id) globalId: vec3u, ...) -> void {
  let index = globalId.x;
  ...
}
```

**Decision**: In the `probeRayCastKernel` source (from `probeRayCast.wgsl.ts`), the
function signature is `fn probeRayCastKernel(..., index: u32, ...) -> void`. The `index`
parameter receives the `instanceIndex` TSL builtin. In raw WebGPU, the entry function
changes to receive `@builtin(global_invocation_id) globalId: vec3u` and derives
`let index = globalId.x;`. The storage buffer bindings become explicit `@group(0) @binding(N)`
declarations.

---

### `texture(tex)` and `sampler(tex)`

**Files**: `cascadeDispatch.ts`, `applyDDGIShading.ts` (TSL-preserved)

**TSL semantics**: `texture(threeTexture)` wraps a `THREE.Texture` as a TSL
`TextureNode` for use as a shader uniform. `sampler(threeTexture)` creates a TSL
`SamplerNode` for the same texture.

**In `cascadeDispatch.ts` (converted)**: The env texture and sampler are passed to the
compute kernel as `texture_2d<f32>` and `sampler` bindings. In raw WebGPU these require
a `GPUTextureView` and `GPUSampler` created from a `GPUTexture`.

**Raw WebGPU equivalent** (for the env texture in cast passes):
- `GPUBindGroupLayoutEntry` at binding 6: `{ texture: { sampleType: 'float', viewDimension: '2d' } }`
- `GPUBindGroupLayoutEntry` at binding 7: `{ sampler: { type: 'filtering' } }`
- Bind group entry 6: `{ binding: 6, resource: gpuTexture.createView() }`
- Bind group entry 7: `{ binding: 7, resource: gpuSampler }`

**In the current engine integration**: texture and sampler ownership is handled by
`@vitrum/walkaround-hybrid`; `@vitrum/walkaround-rc` remains a raw-runtime package.

---

### `uniform(value, type?)`

**Files**: historical TSL receiver helpers, now represented by engine-level raw
composition in `@vitrum/walkaround-hybrid`.

**TSL semantics**: Creates a TSL uniform node (scalar or vector) that uploads a CPU-side
value to the GPU each frame.

**Historical note**: the old receiver-side TSL wrappers kept `uniform()` nodes in
the host integration layer. The current package does not ship those wrappers.

---

### `Fn(body)` — TSL shader function builder

**Files**: historical receiver helper only.

**TSL semantics**: Creates a TSL node function from a JavaScript closure that uses TSL
node operations. The closure returns a node tree, not a WGSL string.

**Historical note**: the old receiver helper stayed TSL-side during the initial
extraction. Current package exports do not include that helper.

---

### `StorageBufferAttribute` (from `three/webgpu`)

**TSL semantics**: A Three.js `THREE.BufferAttribute` subclass backed by a storage
buffer. The GPU allocates a `GPUBuffer` with `STORAGE | COPY_DST | COPY_SRC` usage.
The `StorageBufferAttribute` is the CPU-side container that TSL's `storage()` node
wraps.

**Historical note**: the removed bridge used `StorageBufferAttribute` and Three.js
renderer-owned backing buffers. The current `RCDispatcher` accepts raw `GPUBuffer`
handles through `RCDispatchOptsRaw` and never reads a Three.js internal
`__gpuBuffer` property.

---

### `wgslFn` includes from `three-mesh-bvh/src/webgpu`

**Files**: `probeRayCast.wgsl.ts`

**TSL semantics**: The BVH traversal functions (`bvhIntersectFirstHit`, `intersectsTriangle`,
`intersectTriangles`, `intersectsBounds`) and their support structs (`rayStruct`,
`bvhNodeStruct`, etc.) are all TSL `wgslFn`/`wgsl` nodes from `three-mesh-bvh`.

**Raw WebGPU equivalent**: Extract the WGSL string bodies from each node. The strings
are concatenated in dependency order:

```
constants → rayStruct → bvhNodeBoundsStruct → bvhNodeStruct → intersectionResultStruct →
intersectsBounds → intersectsTriangle → intersectTriangles → bvhIntersectFirstHit
```

**Subtlety — tab/whitespace**: The three-mesh-bvh WGSL strings use tab indentation;
the probe ray cast kernel uses 2-space indentation. Both are valid WGSL; the validator
is whitespace-agnostic.

---

## Summary: what was TSL-only vs. what was converted

| File | TSL primitives | Converted? |
|------|---------------|-----------|
| `cascadeDispatch.ts` | `storage()`, `compute()`, `instanceIndex`, `texture()`, `sampler()` | YES — converted to raw `GPUComputePipeline` with raw `GPUBuffer` / texture-view inputs |
| `cascadePyramid.ts` | Cascade geometry previously paired with Three.js storage attributes | YES — raw cascade dimensions and runtime validation only |
| `probeRayCast.wgsl.ts` | `wgslFn`, `wgsl` (inline WGSL strings) | YES — package-owned raw WGSL module string |
| `cascadeMerge.wgsl.ts` | `wgslFn`, `wgsl` (inline WGSL strings) | YES — package-owned raw WGSL module string |
| removed bridge files | `StorageBufferAttribute`, `MeshPhysicalNodeMaterial`, receiver-side TSL nodes | Historical only — not exported by `@vitrum/walkaround-rc` |
