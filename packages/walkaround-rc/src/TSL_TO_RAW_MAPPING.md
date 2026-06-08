# TSL → Raw WebGPU Mapping

**Context**: Phase 4 Step 4, RC subsystem extraction from `_staging/legacy-source/`.
**Written before conversion code lands.** All conversion decisions in `cascadeDispatch.ts`
are checked against this document.

---

## TSL primitives encountered

### `storage(buffer, type, count[, access])`

**Files**: `cascadeDispatch.ts`, `src/three/walkaroundDiffuseLighting.ts`, `cascadePyramid.ts`

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

**Cast pass bindings** (one bind group per cascade):
| binding | name          | access         | WGSL type                             |
|---------|---------------|----------------|---------------------------------------|
| 0       | bvh           | read-only      | `array<BVHNode>`                      |
| 1       | geom_index    | read-only      | `array<vec3u>`                        |
| 2       | geom_position | read-only      | `array<vec3f>`                        |
| 3       | materials     | read-only      | `array<MaterialEntry>`                |
| 4       | triMatId      | read-only      | `array<u32>`                          |
| 5       | cascadeOut    | read_write     | `array<vec4f>`                        |
| 6       | envMap        | texture        | `texture_2d<f32>`                     |
| 7       | envSampler    | sampler        | `sampler`                             |
| 8       | u_arr         | read-only      | `array<CascadeUniforms>`              |

**Merge pass bindings** (one bind group per merge step):
| binding | name         | access         | WGSL type          |
|---------|--------------|----------------|--------------------|
| 0       | upperCascade | read-only      | `array<vec4f>`     |
| 1       | lowerCascade | read_write     | `array<vec4f>`     |
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
  ${BVH_CONSTANTS_WGSL}        // from three-mesh-bvh common_functions
  ${BVH_STRUCTS_WGSL}          // Ray, BVHNode, BVHBoundingBox, IntersectionResult
  ${BVH_INTERSECT_FUNCTIONS}   // intersectsTriangle, intersectTriangles, intersectsBounds, bvhIntersectFirstHit
  ${CASCADE_STRUCTS_WGSL}      // CascadeUniforms, MaterialEntry
  ${OCTAHEDRAL_BODY_WGSL}      // octEncode, octDecode (body only, stripped of file header)
  ${PROBE_RAY_HELPERS_WGSL}    // pcgHash, dirToEquirectUV
  ${SUN_VISIBILITY_WGSL}       // traceSunVisibility
  ${PROBE_RAY_CAST_ENTRY}      // @compute @workgroup_size(64) fn probeRayCastKernel(...)
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

**In `applyDDGIShading.ts`, `walkaroundDiffuseLighting.ts` (TSL-preserved)**: `texture()`,
`sampler()`, `uniform()` stay as TSL nodes. These files are NOT converted.

---

### `uniform(value, type?)`

**Files**: `applyDDGIShading.ts`, `walkaroundDiffuseLighting.ts` (both TSL-preserved)

**TSL semantics**: Creates a TSL uniform node (scalar or vector) that uploads a CPU-side
value to the GPU each frame.

**Not converted**: These files are TSL-preserved material wrappers. `uniform()` stays
as-is.

---

### `Fn(body)` — TSL shader function builder

**Files**: `walkaroundDiffuseLighting.ts` (TSL-preserved)

**TSL semantics**: Creates a TSL node function from a JavaScript closure that uses TSL
node operations. The closure returns a node tree, not a WGSL string.

**Not converted**: `walkaroundDiffuseLighting.ts` is TSL-preserved.

---

### `StorageBufferAttribute` (from `three/webgpu`)

**Files**: `src/three/cascadePyramidThree.ts`, `cascadeDispatch.ts`

**TSL semantics**: A Three.js `THREE.BufferAttribute` subclass backed by a storage
buffer. The GPU allocates a `GPUBuffer` with `STORAGE | COPY_DST | COPY_SRC` usage.
The `StorageBufferAttribute` is the CPU-side container that TSL's `storage()` node
wraps.

**In `src/three/cascadePyramidThree.ts`**: `gpuCascades` is an array of `StorageBufferAttribute`.
The attribute's `.count` and `.array` are used by the dispatchers. This file is
**retained as-is** with `StorageBufferAttribute` from `three/webgpu`, because the
cascade output buffers are consumed by TSL material nodes in
`src/three/walkaroundDiffuseLighting.ts` (which is TSL-preserved). Breaking the
`StorageBufferAttribute` out of the bridge would require that receiver node to
also be converted, which the plan explicitly chose against (Option i).

**Historical note**: the removed RC bridge used `StorageBufferAttribute`.
These are referenced in `cascadeDispatch.ts` which now creates raw WebGPU bind groups.
The conversion strategy: `cascadeDispatch.ts` reads the `GPUBuffer` backing from the
`StorageBufferAttribute` via the `__gpuBuffer` internal property (used by Three.js
WebGPU renderer). This is the same pattern the Three.js backend uses.

**Ambiguity**: Accessing `StorageBufferAttribute.__gpuBuffer` is an internal property.
It is set by the Three.js WebGPU renderer when it uploads the buffer; callers must ensure
the renderer has uploaded the buffers before calling `RCDispatcher.initialize()`.
This matches the original code's `gl.computeAsync()` call, which also required the
Three.js renderer to have processed the scene first.

**Resolution**: `src/three/cascadePyramidThree.ts` keeps `StorageBufferAttribute`.
The `RCDispatcher` receives the `WebGPURenderer`'s raw `GPUDevice` and accesses buffer
GPU handles through the renderer's backend, the same as the pre-conversion code did.

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
| `cascadeDispatch.ts` | `storage()`, `compute()`, `instanceIndex`, `texture()`, `sampler()`, `StorageBufferAttribute` refs | YES — converted to raw GPUComputePipeline |
| `src/three/cascadePyramidThree.ts` | `StorageBufferAttribute` (direct instantiation) | TSL-preserved behind `/three` bridge |
| `src/three/cascadeBuffers.ts` | React hooks only (no TSL) | De-React only → `CascadeBufferManager` class |
| `applyDDGIShading.ts` | `wgslFn`, `texture`, `uniform`, `add`, `vec4`, `mul`, `output`, `materialColor`, `renderOutput`, `positionWorld`, `normalWorld` | TSL-preserved per Option (i) |
| `src/three/giReceiver.ts` | `MeshPhysicalNodeMaterial`, `output`, `renderOutput` | TSL-preserved; React hook stripped |
| `src/three/walkaroundDiffuseLighting.ts` | `Fn`, `vec3`, `float`, `uniform`, `storage`, `positionWorld`, `normalWorld`, `dot`, `max`, `clamp` | TSL-preserved per Option (i) |
| `probeRayCast.wgsl.ts` | `wgslFn`, `wgsl` (inline WGSL strings) | WGSL extracted verbatim into `rc/wgsl/` |
| `cascadeMerge.wgsl.ts` | `wgslFn`, `wgsl` (inline WGSL strings) | WGSL extracted verbatim into `rc/wgsl/` |
