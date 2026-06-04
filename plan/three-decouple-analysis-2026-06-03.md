# THREE-decoupling analysis — hybrid geometry ingestion (B1 / campaign P6)

> **Scope.** Decouple the *geometry-ingestion* path of the real-time stack
> (`@vitrum/shared-bvh` + `@vitrum/walkaround-hybrid`) from THREE.js, so the BVH
> is built from the `@vitrum/core` `Scene` contract instead of `THREE.Mesh` /
> `THREE.BufferAttribute`. This is the *ingestion* (CPU scene → GPU BVH/material
> buffers) coupling only; the TSL/`three/webgpu` *render-graph* coupling (DDGI's
> `applyDDGIShading`, `nodeMaterialUpgrade`, the WebGPU renderer backend) is a
> separate, larger decouple and explicitly **out of scope here**.
>
> Every claim below was verified by reading the cited file:function, not by grep.

## 0. TL;DR

The THREE-free *building blocks already exist* in `@vitrum/shared-bvh`:

- `packSceneFromCore(scene, opts)` (`scenePack.ts:812`) builds **per-primitive
  BLAS + TLAS** straight from the core `Scene` — **zero THREE**. Already the
  production path for **pt-webgpu** (`uploadSceneBuffers.ts:349`) and for
  walkaround-hybrid ReSTIR's **TLAS mode** (`sceneBvhFromCore.ts:225`).
- `buildArrayBvh(positions, indices, triMaterialIds, opts)` (`buildArrayBvh.ts:219`)
  — binned-SAH builder over raw typed arrays, **zero THREE**.
- `packMaterials(MaterialEntryInput[], maxCount?)` (`materialEntry.ts:151`) — the
  canonical 64-byte GPU material struct packer, **zero THREE** (operates on a
  pure-data `MaterialEntryInput` bag).

So the decouple is **not** "write a THREE-free builder" (that's done). It is
**"stop reconstructing core data *through* THREE on the way into those
builders."** Every remaining THREE dependency in the ingestion path is one of:

1. **A second, world-space merged BVH build** (`buildSceneBVH`, THREE-only) used
   *only* for the ReSTIR emitter list + `mergedGeometry` handle — because that
   build needs **world-space** geometry and `packSceneFromCore` emits
   **local-space BLAS** + separate TLAS instance matrices.
2. **Material field reads through `THREE.Material`** (`extractThreePbrScalars`,
   `resolveTriColor`, `materialEmissiveLe`, `classifyTriangleEmitter`) — every
   one of these fields exists 1:1 on the core `MaterialSpec`.
3. **`matrixWorld` snapshots** for the transform-only refit fast path
   (`enrichMeshVertexRangesWithMatrix`) — the core primitive's `transform` Mat4
   is the same data.
4. **`THREE.RectAreaLight` → emitter tris** (`collectRectAreaLightEmitterTris`) —
   the core scene already carries these as `'rect-area'` emitters
   (`vitrumSceneToThree.ts:501` round-trips them THREE→core→THREE).

`@vitrum/shared-bvh` already depends on `@vitrum/core` and lists `three` /
`three-mesh-bvh` as **peer** deps (`packages/shared-bvh/package.json:28`), so the
package is *structurally* ready to shed THREE once the two THREE-only files
(`bvhCommon.ts`, `sceneBvh.ts`) lose their last consumers.

## 1. The two ingestion tracks today (verified)

`HybridEngineLifecycle.ts:323-358` picks the source root, then **always** calls
`buildReSTIRSceneBVHForScene(scene, [bvhRoot], opts)`
(`restir/bvhCompute.ts:210`), which branches on
`resolveReSTIRBvhMode` (`sceneBvhFromCore.ts:28`):

| Path | When | Geometry builder | THREE? |
|------|------|------------------|--------|
| **TLAS** | >1 mesh-like prim, or any `instanced-mesh` | `packSceneFromCore` (local BLAS+TLAS) **+** a *second* `buildSceneBVH(sceneRoots)` for the emitter list & `mergedGeometry` | **partial** — geometry THREE-free, but emitter build + material reads go through THREE |
| **merged** | single mesh prim (or empty/escape-hatch THREE scene) | `buildSceneBVH(sceneRoots)` only (one merged world-space SAH) | **yes** — fully THREE |

Both paths converge in `buffersFromScenePack` / `buildReSTIRSceneBVH` on the same
THREE-dependent emitter + material packing. **DDGI** (`SceneBvh.update(scene)` →
`buildSceneBVH`) and **RC** (`buildRCSceneBVH(scene)` → `buildSceneBVH`) are
**merged-only** and fully THREE today.

## 2. Per-function inventory — what consumes THREE, and the core equivalent

### 2a. `@vitrum/shared-bvh` (2 files, the canonical THREE-geometry ingest)

| file:function | THREE API used | Core-`Scene` equivalent |
|---|---|---|
| `bvhCommon.ts:buildSceneBVH` (`:376`) | `THREE.Scene`/`Object3D[]` roots; `traverseVisible`; `mesh.matrixWorld`; `StaticGeometryGenerator` (world-bake + merge); `new MeshBVH` (`_roots[0]` byte read); `BufferAttribute` position/normal/uv/index reads; `Box3`/`Vector3`/`Quaternion`; `MeshStandardMaterial`/`MeshPhysicalMaterial` instanceof | **Geometry:** `packSceneFromCore` already does the equivalent (per-primitive local BLAS via `buildArrayBvh` + TLAS). The only thing `buildSceneBVH` does that `packSceneFromCore` doesn't is **world-space merge into one root**. A THREE-free analogue would: iterate `scene.primitives` (mesh/instanced/skinned), transform each primitive's `positions`/`normals` by its `transform` (or per-instance), concat, then `buildArrayBvh` over the merged world-space arrays. `transformPoint` already exists (`scenePack.ts:132`). **Materials:** the `THREE.Material[]` dedup list maps to a `MaterialSpec[]` in `scene.primitives[*].material` order. |
| `bvhCommon.ts:snapshotPreBuildMaterials` (`:272`) | reads `THREE.Material` PBR fields to build a value-dedup signature; `MeshStandardMaterial`/`MeshPhysicalMaterial` field reads | Core: dedup over `MaterialSpec` (same fields: `baseColor`/`emissive`/`emissiveIntensity`/`roughness`/`metallic`/`transmission`/`ior`/map handles). In the core path **primitives already carry exactly one material each**, so per-vertex matId is just "primitive index" — the whole groups/`StaticGeometryGenerator` dance disappears. |
| `bvhCommon.ts:ensureIndexed`/`emptyBVHResult` (`:256`,`:700`) | `BufferGeometry`/`BufferAttribute` | N/A — core primitives are already typed arrays; `packSceneFromCore` synthesizes a sequential index when `indices` is absent (`scenePack.ts:282`). |
| `bvhCommon.ts:refitBvhBounds`/`validateBvhEncoding`/`normalizeBvhInteriorOffsets`/`packRootBuffer` | **none** (pure typed-array math) | Already THREE-free; **keep as-is**. `refitBvhBounds` is the shared refit kernel used by both tracks. |
| `sceneBvh.ts:SceneBvh.update` (`:89`) | `THREE.Scene`; `traverseVisible`; `BufferAttribute.version` dirty-hash; `DDGI_MESH_FILTER` (instanceof `THREE.Mesh`) | A `SceneBvhFromCore` analogue: dirty-track on a core-scene fingerprint (the repo already has `fingerprintBuffers` in `bufferFingerprint.ts`) and call the THREE-free merged builder from §2a-row-1. DDGI's filter ("any mesh with positions") maps to "any mesh-like primitive". |

### 2b. `@vitrum/walkaround-hybrid` — ingestion-coupled THREE files

| file:function | THREE API used | Core-`Scene` equivalent |
|---|---|---|
| `rc/bvhCompute.ts:buildRCSceneBVH` (`:141`) | `buildSharedBVH(scene)`; wraps results in `three/webgpu` `StorageBufferAttribute`; `threeToMaterialEntryInput` via `extractThreePbrScalars` | Geometry: swap `buildSceneBVH` for the THREE-free merged builder. Materials: `coreMaterialToMaterialEntry(MaterialSpec)` (★ this increment). **Note:** the `StorageBufferAttribute` wrap is a *render-backend* coupling (`three/webgpu`), separate from ingestion — it stays until the RC dispatch is ported off the THREE WebGPU renderer. |
| `restir/sceneBvhFromCore.ts:buffersFromScenePack` (`:88`) | second `buildSharedBVH(sceneRoots)` for **world-space** emitter list + `mergedGeometry`; `buildMaterialResolver` walks `traverseVisible` for the `THREE.Material[]` LUT; `THREE.Vector3` light dir | Emitter list needs world-space tris. Replace the second build with a THREE-free **world-space tri stream** derived from `packSceneFromCore`'s local BLAS + each binding's `localToWorld` (`primitiveTlasBindings` already carry the per-primitive transform via TLAS instances). Materials: `MaterialSpec[]` in `scene.primitives` order (the resolver's `resolveMaterialId(primitiveId)` is already keyed by id). |
| `restir/emitterList.ts:buildEmitterList`/`classifyTriangleEmitter` (`:182`,`:55`) | `materials: THREE.Material[]`; `THREE.Vector3` lightDir; `materialEmissiveLe` (THREE); `MeshPhysicalMaterial` transmission/attenuation/color reads; `THREE.Color` | Geometry math is **already raw typed-array** (reads `positions`/`normals`/`indices`). Only the **material classification** is THREE. A `MaterialSpec`-typed `classifyTriangleEmitterCore` (same priority rules: emissive Le > transmissive-sun) closes it. `lightDir` → plain `Vec3`/tuple. |
| `restir/packingHelpers.ts` (`resolveTriColor`/`materialEmissiveLe`/`applyBeerLambert`/`packBVHIndexW`/`packBVHBeerColors`/`packBVHEmissiveLe`) (`:79`,`:217`,…) | `THREE.Material`/`THREE.Color`; `userData.surfaceTextureId`; `MeshPhysicalMaterial` reads | All read fields that exist on `MaterialSpec` (`surfaceTextureId` → `material.extensions['surfaceTextureId']`). Mirror each as a `MaterialSpec` overload. The **geometry** lanes (UV-into-position.w, vec4u index expansion) are already typed-array. |
| `restir/bvhSceneHelpers.ts:enrichMeshVertexRangesWithMatrix` (`:12`) | `traverseVisible`; `obj.matrixWorld.elements` | The refit fast path needs each primitive's world matrix at build time. Core primitive `transform` (or per-instance Mat4) **is** that matrix — `packSceneFromCore` already retains `primitiveTlasBindings` keyed by `primitiveId`; the world matrix is the binding's resolved `localToWorld`. No traversal needed. |
| `restir/bvhSceneHelpers.ts:collectRectAreaLightEmitterTris` (`:52`) | `THREE.RectAreaLight` (width/height/color/intensity/matrixWorld) | Core scene carries these as `emitters` of `kind:'rect-area'` (round-tripped at `vitrumSceneToThree.ts:501`). Build the two emitter tris from the core emitter's transform + width/height/Le directly. |
| `HybridEnginePrimitiveUpdates.ts:31`, `HybridEngine.ts:38`, `HybridEngineLifecycle.ts:46` | `import * as THREE` — construct `vitrumSceneToThree(scene)` root, `new THREE.Vector3(...primaryLightDir)`, manage the synthesized scene lifetime | Once the builders accept the core `Scene` directly, the **`vitrumSceneToThree` synthesis for the BVH disappears** for the `coreSceneSuppliesMeshes()` path (the dominant path). `THREE.Vector3` light dir → tuple. The escape-hatch `threeScene` path (host hands a raw THREE scene, no core meshes) **keeps** `buildSceneBVH` — that path *is* the reason the THREE builder must survive. |

### 2c. THREE files that are **render-graph**, NOT ingestion (out of scope)

`ddgi/applyDDGIShading.ts` (TSL nodes), `ddgi/probeUpdatePass.ts`
(`renderer.backend.device`), `ddgi/probeGrid.ts`, `lib/nodeMaterialUpgrade.ts`
(`three/webgpu` NodeMaterial), `HybridEngineRC.ts` (`StorageBufferAttribute`
type), `hostScene/types.ts` (`Object3D`/`Scene` type-only). These couple to the
**Three.js WebGPU renderer**, not to geometry ingestion. They are the *next*
decouple wave (port the compute/raster dispatch off the TSL renderer), not this
one.

## 3. The hard part — the world-space emitter/merged build

`packSceneFromCore` emits **local-space** BLAS (vertices in object space) +
**separate** TLAS instance matrices. But three consumers want **world-space**
merged geometry:

1. **ReSTIR emitter list** — `buildEmitterList` derives triangle area, world
   face-normal (for the sun-dot transmissive test), centroid, AABB → all
   world-space. (`sceneBvhFromCore.ts:110-122` documents exactly why the second
   build is load-bearing and not a redundant duplicate.)
2. **`mergedGeometry`** handle (`SceneBVHBuffers.mergedGeometry`) — a
   `THREE.BufferGeometry` kept "for debug / re-upload" (`disposeSceneBVH` only
   calls `.dispose()` on it; grep shows no functional read of its attributes in
   the hybrid pipeline — **candidate for deletion**, verify before removing).
3. **DDGI / RC merged BVH** — these traverse one world-space root.

**The THREE-free answer** is a small **world-space-merge-from-core** helper:
iterate `scene.primitives`, for each (instance) transform the local
`positions`/`normals` by `localToWorld` (`transformPoint` exists), concat into
one world-space `positions`/`normals`/`indices`/`triMaterialId` stream, and
either (a) feed `buildArrayBvh` for a merged root (DDGI/RC) or (b) feed
`buildEmitterList` directly (ReSTIR — it only needs the tri stream, not a BVH).
This is the central new primitive the full decouple needs; it is pure
typed-array math and unit-testable against the existing `buildSceneBVH` output
for golden parity.

## 4. Dependency order (smallest-blast-radius first)

1. **★ `coreMaterialToMaterialEntry(MaterialSpec): MaterialEntryInput`** in
   `@vitrum/shared-bvh/materialEntry.ts` — pure data, zero new deps, no existing
   call site touched. **This increment** (§6). Unblocks the RC + DDGI material
   packers' THREE removal independently of geometry.
2. **`coreEmitterClassify` / `materialSpecEmissiveLe` / Beer-Lambert-from-spec**
   — `MaterialSpec` overloads of the `packingHelpers` + `emitterList`
   classifiers. Pure data. Unblocks the emitter list's material half.
3. **`mergeWorldSpaceFromCore(scene)`** — the §3 world-space tri-stream/merged
   builder. The keystone; everything merged-path depends on it.
4. **`buildEmitterListFromCore`** = (3) + (2): world-space tris + core material
   classify + core RectAreaLight emitter tris. Removes `buildSharedBVH` from
   `buffersFromScenePack`'s **TLAS** path.
5. **`SceneBvhFromCore`** (DDGI) + **`buildRCSceneBVHFromCore`** (RC) on top of
   (3)+(1). Removes `buildSceneBVH` from the DDGI + RC merged paths.
6. **Switch `buildReSTIRSceneBVH` merged path** to (3)+(4)+(2). Last consumer of
   `buildSceneBVH` for the core-scene path; the THREE builder then survives
   **only** for the escape-hatch raw-`threeScene` ingestion.
7. **Drop `import * as THREE`** from the hybrid ingestion files (light dir →
   tuple; delete `vitrumSceneToThree`-for-BVH on the core path). `bvhCommon.ts` /
   `sceneBvh.ts` retained behind the escape hatch (or moved to a
   `three-bindings`-adjacent module if the escape hatch is dropped — a product
   decision, not forced by this work).

## 5. Risks

- **R1 — Triangle ordering / golden drift.** `buildSceneBVH` (one merged SAH) and
  `packSceneFromCore` (per-primitive BLAS-concat SAH) produce **different
  triangle orderings** (`sceneBvhFromCore.ts:119` says so explicitly). Any
  builder swap that feeds an order-sensitive consumer (the emitter CDF, the
  per-tri `bvhIndex.w` packing) must keep the orderings *consistent within a
  track*, or A/B will diverge. Mitigation: the world-space merge in §3 must
  match `buildSceneBVH`'s merge+SAH for the merged-track consumers, validated by
  golden parity test. **Radiometric — must be GPU-A/B'd**, hence step 6 is gated.
- **R2 — Material dedup semantics.** `snapshotPreBuildMaterials` value-dedups
  THREE materials (React re-render churn). The core path doesn't have that churn
  (one `MaterialSpec` per primitive), but DDGI's 64-slot cap + RC's packing
  assume a deduped list. Mitigation: dedup `MaterialSpec` by structural
  signature in the core merge (mirror the existing `matSig`).
- **R3 — `userData.surfaceTextureId` / `skipEmitter`.** ReSTIR reads
  `mat.userData.surfaceTextureId` (`packingHelpers.ts:125`) and
  `mat.userData.skipEmitter` (`emitterList.ts:72`). On core these live in
  `material.extensions`. Mitigation: read `extensions['surfaceTextureId']` /
  `extensions['skipEmitter']`; confirm `sceneFromThreeJS` / `vitrumSceneToThree`
  actually carry them (verify before relying).
- **R4 — `mergedGeometry` consumers.** If anything beyond `dispose()` reads the
  `THREE.BufferGeometry`, the THREE-free path must still hand back something
  shaped like it (or the field is deleted). **Verify zero functional reads
  before deletion.**
- **R5 — Escape-hatch THREE scene.** The raw-`threeScene` path (`host.threeScene`,
  no core meshes — `HybridEngineLifecycle.ts:326`) **genuinely needs** THREE
  ingestion. The decouple makes core-scene ingestion THREE-free; it does **not**
  delete `buildSceneBVH`. Keep both; route by `coreSceneSuppliesMeshes()`.
- **R6 — `three/webgpu` `StorageBufferAttribute` (RC).** RC wraps BVH buffers in
  `StorageBufferAttribute` for the TSL renderer backend. That is a **render**
  coupling; removing it is the render-graph decouple wave, not this one. The
  ingestion decouple stops at producing the raw typed arrays.

## 6. The first increment (implemented this session)

**`coreMaterialToMaterialEntry(material: MaterialSpec): MaterialEntryInput`**
added to `@vitrum/shared-bvh/src/materialEntry.ts` (+ export in `index.ts`), with
unit tests.

- **What it decouples.** The material half of ingestion. RC
  (`rc/bvhCompute.ts:threeToMaterialEntryInput`) and DDGI
  (`ddgi/probeUpdateMaterials.ts:threeToMaterialEntryInput`) **reconstruct**
  `MaterialEntryInput` by reading a `THREE.Material` through
  `extractThreePbrScalars`. When those packers are fed a core `Scene`, the
  material is already a `MaterialSpec` — this adapter maps it **directly**, no
  THREE round-trip. It is the canonical "core `MaterialSpec` → GPU material
  struct" bridge that steps 1/5 of §4 build on.
- **Why it's safe.** Strictly **additive**: a new pure function + a new export.
  **No existing function or call site is modified.** The THREE adapters stay
  exactly as they are. It has **no THREE import** (lives in a file that already
  has none) and adds **no dependency** (`MaterialSpec` comes from `@vitrum/core`,
  already a `shared-bvh` dep). CPU-only, fully unit-verifiable, no GPU, no
  radiometric A/B needed (it's a field-mapping; the bytes it ultimately produces
  via `packMaterials` are golden-pinned by the existing `materialEntry.test.ts`).
- **Field mapping** (`MaterialSpec` → `MaterialEntryInput`): `baseColor`→
  `baseColor`, `roughness`→`roughness`, `metallic`→`metalness`,
  `emissive·emissiveIntensity`→`emissive` (pre-multiplied, matching the RC
  adapter at `rc/bvhCompute.ts:96`), `ior`→`ior`, `transmission`→`transmission`,
  `attenuationColor`→`attenuationColor`, `attenuationDistance`→
  `attenuationDistance`, `thickness`→`thickness`. `packMaterials` then applies
  the canonical defaults for any field left `undefined` — so an
  `emissiveIntensity` of `undefined` defaults to ×1 (matching
  `extractThreePbrScalars`' `PBR_DEFAULTS.emissiveIntensity = 1`).
- **Deliberate parity note.** The RC adapter floors `thickness` to `0.1` when
  zero/absent (its per-tri Beer-Lambert needs a non-zero numerator). DDGI does
  **not**. That floor is an **RC-policy** decision, not a property of the core
  material, so `coreMaterialToMaterialEntry` does **not** bake it in — it passes
  `thickness` through faithfully (the canonical default is `0`, matching DDGI).
  When RC adopts this adapter it applies its `0.1` floor on top, exactly as it
  does today on the `extractThreePbrScalars` output. Documented in-code so the
  RC-vs-DDGI divergence stays intentional and visible.

## 7. Recommended next increments (in order)

1. **Emitter/material classification overloads** (§4 step 2) — `MaterialSpec`
   variants of `materialEmissiveLe`, the Beer-Lambert tint, and the emitter
   classifier. Pure data, unit-testable against the THREE versions for a handful
   of materials. Low risk.
2. **`mergeWorldSpaceFromCore(scene)`** (§4 step 3) — the keystone world-space
   tri-stream/merged-BVH builder. Golden-parity test vs `buildSceneBVH` output
   (positions/normals/indices/triMaterialId, allowing for the documented SAH
   ordering difference where it's order-insensitive). **Medium risk (R1).**
3. **`buildEmitterListFromCore`** (§4 step 4) — wire (1)+(2)+core RectAreaLight
   emitters into the **TLAS** path's emitter build, removing `buildSharedBVH`
   from `buffersFromScenePack`. GPU A/B against the current TLAS emitter output
   (deterministic-seed converged ReSTIR-DI frame).
4. **`SceneBvhFromCore` (DDGI)** + **`buildRCSceneBVHFromCore` (RC)** (§4 step 5)
   on (1)+(2). DDGI/RC are merged-only, so this is the cleanest place to retire
   `buildSceneBVH` for those two engines. GPU A/B per engine.
5. **Switch the ReSTIR merged path** (§4 step 6), then **drop `import * as THREE`
   from the ingestion files** (§4 step 7), leaving `buildSceneBVH` /
   `sceneBvh.ts` as the **escape-hatch-only** raw-THREE-scene ingest.

Each of 2–5 is an independent, GPU-A/B-gated radiometric change — sequence them
one engine/path at a time, never as one sweeping rewrite.
