# Path-Tracer Library Readiness — Complexity Sweep Findings + Refactor Sequence

**Date:** 2026-05-07 | **Branch:** `main` (commit 9ee2ce8) | **Author:** complexity-sweep
**Anti-cheese rule:** code WORKS as of 9ee2ce8 on Lovelace. Every refactor must hardware-validate via Chrome MCP before commit.
**Out of scope:** noise, GPU load, color vibrance, came-white, glass-edge enclosure (polish backlog, tracked elsewhere).

## TL;DR

The layered hybrid is structurally complete: all 8 GI terms have live data sources, every bind-group binding is consumed, the routing dispatches Path-B correctly. **6 real bugs** were found (one cosmetic, five hidden). Library extraction needs **~12 surface contamination fixes** (app-specific naming, hardcoded scene constants, missing JSDoc, stage coupling). Iteration churn left **6 duplication clusters** and **~15 stale-comment sites**. Plus **8 truly-dead Path-A files** that the user previously kept for reference (now obsolete after Path-B shipped).

---

## A. Hybrid completeness — verified

| Term                                         | Producer                                              | Consumer                          | Status |
| -------------------------------------------- | ----------------------------------------------------- | --------------------------------- | ------ |
| `Lo_emit`                                    | emitter CDF (engines/restir/bvhCompute)               | shade.wgsl emitter sampling       | ✓      |
| `Lo_direct + Lo_sunCaustic + Lo_skyAperture` | ReSTIR RIS                                            | shade.wgsl                        | ✓      |
| `Lo_fill`                                    | ReSTIR temporal+spatial                               | shade.wgsl                        | ✓      |
| `Lo_ddgi`                                    | useDDGI compute → atlas → setDDGIInputs               | shade.wgsl ddgiSampleFromBindings | ✓      |
| `Lo_rc`                                      | dispatchCascadePasses → cascade buffers → setRCInputs | shade.wgsl rcSampleSunVisibility  | ✓      |
| `Lo_indirect`                                | ReSTIR GI (single-bounce)                             | shade.wgsl                        | ✓      |

All bind-group bindings consumed. Routing correct. Frame-loop ordering safe via WebGPU queue submission semantics.

---

## B. Real bugs (Hybrid-Incomplete category)

### B1. fillCascadeDebug doesn't reach the GPU

`cascadePyramid.ts:fillCascadeDebug` writes to `b.cascades[k]` but never sets `needsUpdate=true` on `b.gpuCascades[k]`. The CPU array is filled; the GPU StorageBufferAttribute retains stale content. The smoke-test path (non-WebGPU backend OR explicit debugFill) silently renders whatever was last in the buffer (zeros on first frame, last real compute thereafter).
**Severity:** Medium — non-Lovelace fallback path is broken; Lovelace itself is fine.
**Fix:** After `fillCascadeDebug(...)` in `dispatchCascadePasses` (cascadeDispatch.ts:246–260), iterate `cascadeBuffers.gpuCascades` and set `needsUpdate=true`. ~3 lines.

### B2. walkaroundDiffuseLighting.updateBuffers no-ops

`walkaroundDiffuseLighting.ts:161–172` reassigns the local `c0Storage` variable, but the compiled `Fn(...)()` closure has already captured the original storage node by reference. Buffer-swap calls silently render against the stale buffer.
**Severity:** Medium — RC standalone (RcStage) path is affected when scene topology changes; Lovelace appearance is fine because the buffer instance is stable across frames.
**Fix (option A):** Trigger full `buildWalkaroundLightingNode` rebuild on buffer change (caller re-attaches material).
**Fix (option B):** Mutate the existing GPU buffer in place rather than allocating a new StorageBufferAttribute.
**Recommendation:** Option B — matches the "never allocate a second StorageBufferAttribute over the same Float32Array" invariant in cascadePyramid.ts.

### B3. rcSampleSunVisibility uses wrong distance metric

`shade.wgsl:rcSampleSunVisibility` selects cascade level by `length(worldPos - cameraPos)`. RC theory says level should be selected by the probe-to-receiver interval distance, not camera distance. With the current scene (camera ~140 inches from receiver) it accidentally picks a reasonable cascade for typical viewpoints, but rotating the camera produces incorrect cascade selection.
**Severity:** Medium-high — physically wrong but visually plausible at default camera. Caustic crispness will jitter as user moves camera.
**Fix:** Replace `camDist` with distance from `worldPos` to nearest probe origin (or to `rcParams.roomOrigin`). One-line shader change. Hardware-validate caustic stability under walk-camera.

### B4. Two divergent RC samplers + two divergent DDGI samplers

- RC: TSL `walkaroundDiffuseLighting.buildWalkaroundLightingNode` (RcStage standalone path) does trilinear+full-cosine-bin-integration on cascade C0 only. WGSL `rcSampleSunVisibility` (hybrid path) does nearest-probe+single-bin with multi-cascade selection.
- DDGI: TSL `ddgiSampleWgsl.DDGI_SAMPLE_WGSL` (WalkaroundStage standalone) and WGSL `ddgiSampleFromBindings` (hybrid). Math currently matches (verified line-by-line) but no shared atlas-layout constants — drift inevitable.

**Severity:** High for library extraction (the public sampler API needs a single source of truth). Medium for hybrid correctness today (samplers each work for their own engine).
**Fix:** Extract atlas-layout constants (IRR_CELL=8, VIS_CELL=16, BORDER=2 → IRR_STRIDE=10, VIS_STRIDE=18) to a shared TS module, template-substitute into both WGSL strings. For RC: extract a canonical `rcSampleCascade(worldPos, queryDir)` WGSL fragment used by both consumer paths; update walkaroundDiffuseLighting to invoke it via TSL `wgslFn`.

### B5. Visibility atlas format mismatch

`WalkaroundGPUPipeline._ddgiPlaceholderRg16f` is `'rg16float'`; live atlas (`probeUpdatePass._getOrCreateAtlasTexture`) is `'rgba16float'`. WebGPU's `'unfilterable-float'` sample type accepts both, so today this silently works. But format-mismatch in the same BGL slot is a latent hazard if either side changes.
**Severity:** Low — currently masked by isDDGIWired gate.
**Fix:** Change `_ddgiPlaceholderRg16f` allocation to `'rgba16float'`. One-character diff (variable rename optional).

### B6. Edge beads enter the BVH via DEFAULT_FILTER

`lib/bvhCommon.ts:DEFAULT_FILTER` accepts `MeshStandardMaterial`. `edgeMaterialFactory.createEdgeMaterial` returns `MeshStandardMaterial`. All came/solder bead meshes (3–8 mm thin tubes) are pulled into the BVH for both DDGI and RC — bloating the BVH and risking shadow-ray self-intersection at panel edges.
**Severity:** Medium — invisible on render output but contributes to BVH bloat and known floor-caustics issue (task #27 in tracking).
**Fix:** Stamp `userData.excludeFromBVH=true` in `createEdgeMaterial` (or on the meshes via `LeadCameBead3D`/`SolderBead3D`); update `DEFAULT_FILTER` to reject objects with that flag. Two callsite changes + filter update.

### B7. Double-mount FaceRenderer + EdgeLines — verified reachable (2026-05-07)

StudioScene mounts `<FaceRenderer />` and `<EdgeLines />` when `showPanelLayers = !showLamp && !isRoomMode` (StudioScene.tsx:69, 243, 253). HybridLayeredStage mounts both unconditionally (HybridLayeredStage.tsx:581, 587).

**Reachability verified:** In 3D + panel mode (not room, not lamp) + exploreEnabled + walkaroundEngine='hybrid', BOTH mount sets are present simultaneously.

**Visual impact:** Likely benign. HybridLayeredStage drives `WalkaroundGPUPipeline.renderFrame` which writes the swap chain directly via WebGPU. R3F's normal raster pass doesn't run when ReSTIR owns the frame, so the StudioScene-mounted meshes appear in the scene graph for BVH traversal but don't paint pixels. The duplicate glass + came geometry IS in the scene graph, which means ReSTIR's BVH builder (`engines/restir/bvhCompute.buildSceneBVH`) sees doubled glass triangles — bloating BVH leaf count and potentially the emitter list.

**Fix:** Gate HybridLayeredStage's mounts on `(space.kind === 'room') || (showLamp)` — i.e., only mount the panel layers when StudioScene's outer mounts are suppressed. Or equivalently, accept a `showPanelLayers: boolean` prop from StudioScene.

**Severity:** Medium. BVH bloat is real but masked. Wave 2 live-path batch.

---

## C. Library-extraction surface contamination

### C1. App-specific naming on public API

| Site                                                                   | Current            | Library-grade                                            |
| ---------------------------------------------------------------------- | ------------------ | -------------------------------------------------------- | ------------ | -------------------------------------- |
| `PipelineFrameInputs.sunDirection`, `.sunIntensity`                    | "sun"              | `primaryLightDir`, `primaryLightIntensity` (+ JSDoc)     |
| `bvhCompute.buildSceneBVH({ sunDirection, sunIntensity, sunColor })`   | "sun"              | same rename; **delete `sunColor`** (declared but unused) |
| `giReceiver.isGIReceiver` hardcodes `userData.glassPiece`              | hardcoded sentinel | `isExcluded?: (mesh) => boolean` opt                     |
| `probeUpdatePass._uploadLights` switches on `'sun'                     | 'fixture'          | 'teaLight'`                                              | string union | accept caller-normalized `DDGILight[]` |
| `cascadePyramid.ts` header comments "panel pieces", "stained-glass"    | scene-specific     | document `s_0` generically                               |
| `probeUpdateRays.wgsl` "0.5 inch step", "stained-glass slab thickness" | hardcoded          | `FrameParams.maxSlabCrossings`                           |

### C2. Hardcoded scene constants

| Site                                         | Current                                                     | Library-grade                                       |
| -------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| `useCascadeBuffers.DEFAULT_BOUNDS`           | `±96 × ±64,44 × 0,168` (stained-glass room)                 | remove default; require caller to pass bounds       |
| `engines/restir/bvhCompute.PROXY_MESH_NAMES` | `surface_floor_living`/`surface_ceiling_living`             | `proxyMeshNames` opt parameter (default to current) |
| `HybridLayeredStage.isSceneReady`            | searches by `surface_floor_living` + `GLASS_FACE_MESH_NAME` | caller-supplied `isSceneReady` predicate            |
| `HybridLayeredStage` sun discovery           | `scene.getObjectByName('sun')`                              | sun direction as prop                               |
| `HybridLayeredStage` Redux selectors         | `selectMount`/`selectSpace`/`selectGraph`                   | accept inputs as props                              |

### C3. Missing/wrong JSDoc on exports

- `PipelineFrameInputs` — no JSDoc on any field
- `SceneBVHBuffers.materialColors` — JSDoc claims per-tri colors; actually a sentinel (real colors live in `bvhIndex.w`). Missing `@deprecated` warning.
- `lib/bvhCommon.buildSceneBVH` — no `@throws`; `BvhWithRoots` private `_roots` field has no version pin
- `gpuDetection.__resetGpuDetectionForTests` — production bundle includes test infra; move to a `.test-utils.ts` adjacent file

### C4. Stage coupling (HybridLayeredStage → app primitives)

`FaceRenderer`, `EdgeLines`, `MountDispatch`, `LightSourceList`, `RoomLoader`, Redux selectors, `lightboxDimsFor`, `GLASS_FACE_MESH_NAME`, `FLOOR_Y` — all stained-glass-specific. For library extraction, refactor to:

- `useHybridLayeredGI({ scene, bvh, sunDirection, ddgiEnabled, rcEnabled, device })` hook returns no JSX, just owns the compute pipeline + WebGPU pipeline lifecycle
- Stage component shell stays app-specific; library only ships the hook + `WalkaroundGPUPipeline` + WGSL strings

### C5. WebGPU device-limit declaration belongs to the library

`StudioScene` gl factory has `requiredLimits: { maxStorageBuffersPerShaderStage: 16 }`. Library consumers must know this. Export `HYBRID_WEBGPU_REQUIRED_LIMITS = { maxStorageBuffersPerShaderStage: 16 }` from `engines/restir/WalkaroundGPUPipeline` (or a sibling capability file).

### C6. Engine routing not data-driven

StudioScene's `walkaroundActive` guard hardcodes 4 engine strings; the render-switch is a 4-way nested ternary. For extension by library consumers, replace with a `WALKAROUND_ENGINES` record keyed by string → React.ComponentType. Today's 4 engines populate it; consumers extend via Object.assign.

---

## D. Iteration-churn duplications

### D1. Two `useSceneBVH` hooks

- `walkaround/useSceneBVH.ts` (53 LOC) — RC's old, no opts, returns RC's `SceneBVH`
- `lib/useSceneBVH.ts` (79 LOC) — generalized, opts param + optsRef pattern, returns `SceneBVHCommonResult`

**Migration:** Replace `walkaround/useSceneBVH.ts` with a thin RC-typed wrapper that calls `lib/useSceneBVH` then converts via `bvhCompute.buildSceneBVH` adapter. Update RcStage + HybridLayeredStage to consume the wrapper. Bonus: fix `lib/useSceneBVH.ts:75` — remove `opts` from useEffect deps (defeats the optsRef pattern; causes inline-object-literal callers to thrash).

### D2. Two `upgradeToNodeMaterial` copies

- `applyDDGIShading.ts:68–95` — private (legacy)
- `lib/nodeMaterialUpgrade.ts` — shared (preferred)
  Fix: `applyDDGIShading.ts` imports from lib; delete its local copy.

### D3. Two `getDDGISampleFn` singletons

- `applyDDGIShading.ts:101–106` (live consumer)
- `layers/DDGILayer.ts:47–52` (orphan — resolves with T1)
  Fix: After T1 decided. If orphans deleted → no-op. If kept → move singleton into `ddgiSampleWgsl.ts` so module graph has one cache.

### D4. Three-copy `invertMat4` + `generatePrimaryRay`

Both `ris.wgsl` and `shade.wgsl` define local `invertMat4`/`generatePrimaryRay` despite `common.wgsl` providing `_common` versions, AND `common.wgsl` is prepended to both at compile time (`WalkaroundGPUPipeline.ts:546`).
Fix: Delete locals in both files; rename calls to `_common`. Removes ~120 lines of dead WGSL.

### D5. `octDecode` in three places

- Canonical: `wgsl/octahedral.wgsl.ts`
- Inline in `ddgiSampleWgsl.ts` (intentional — TSL parser bug, documented)
- Inline in `wgsl/probeRayCast.wgsl.ts` (RC) — should import OCTAHEDRAL_WGSL
  Fix: RC's `probeRayCast.wgsl.ts` imports `OCTAHEDRAL_WGSL`. Inline copy in `ddgiSampleWgsl.ts` keeps documentation comment cross-linking to the canonical module.

### D6. RC vs ReSTIR `bvhCompute.ts` (CLEAN — no merge)

Both delegate to `lib/bvhCommon`; their post-packing diverges intentionally (RC: 16-float MaterialEntry SSBO; ReSTIR: bvhIndex.w RGBA8 packing + emitter list). No fix.

---

## E. Stale comments + dead exports

### E1. Plan-section refs in lib/\* headers

Cleanse `§3.1 §3.4 §3.8 §3.9 §6.1 §6.2 §4.2`, `M0-M5`, `branch-parity-inventory.md`, worktree paths (`stainedGlass-{ddgi,rc,restir}`), commit hashes from headers of: `lib/bvhCommon.ts`, `lib/useSceneBVH.ts`, `lib/nodeMaterialUpgrade.ts`, `lib/wgpuSupport.ts`, `walkaround/useSceneBVH.ts`, `walkaround/bvhCompute.ts`, `engines/restir/bvhCompute.ts:507–511` (tombstone block).

### E2. Stale phase / "not yet" comments

- `WalkaroundGPUPipeline.ts:91, 257–261, 384–385` — "Phase 1.2B will switch shade.wgsl..." (already done)
- `shade.wgsl.ts:804` — "Reinhard + sRGB" (now ACES — open question: is ACES correct, or should it be Khronos PBR Neutral per memory D7=C?)
- `ddgiSampleWgsl.ts:80, 108` — "1px border" (BORDER=2, stride is 8+2=10, 16+2=18)
- `cascadePyramid.ts: line 29` — "Previous (paper-derived) dimensions" (OK — historical context is fine)

### E3. Per-branch terminology

`lib/bvhCommon.ts:35–42` "Per-branch consumers" → "Per-engine consumers"

### E4. 12 unused exports (de-export, don't delete bodies)

`resetDDGIShading`, `buildCascadeDispatch`, `cascadeTotalRays`, `cascadeBufferSize`, `makeGIReceiverMaterial`, `getCachedGpuDetection`, `readAdapterInfo`, `classifyAdapter`, `isWebGPUSupported`, `recommendedResolutionScale` (delete this — no caller anywhere), `StorageBufferHandle`, `DDGIMaterialEntry`, `HardwareVerdict`, `setPtDeviceLost`/`setExploreEnabled`/`setWalkaroundEngine` re-exports in `uiSlice` barrel.

### E5. Stale eslint-disable comments

`RcStage.tsx:428, 468, 578` — three directives where the underlying violation no longer exists.

---

## F. Test gaps

### F1. Hybrid full-composition spec missing

`14-walkaround-hybrid.spec.ts` only tests `{ddgi:true, rc:false, restir:false}` and all-off. Add a third case `{ddgi:true, rc:true, restir:true}` asserting chromaStdDev ≥ 0.02 (the "glorious target" composition).

### F2. scriptedCameraPath duplicated

Byte-for-byte identical between `12-walkaround-ddgi.spec.ts` and `14-walkaround-hybrid.spec.ts`. Extract to `src/__tests__/fixtures/walkaroundCameraPath.ts`.

### F3. Stale test header history

`13-walkaround-restir.spec.ts:1–33` carries v1–v8 version history. Trim to current invariants.

---

## G. Path-A orphan cluster (8 files) — DECISION REQUIRED

I4 + G4 + dead-code agree: 7 files are TRULY DEAD (no live importers anywhere); 1 (`HybridStage.tsx`) is CROSS-REFERENCED ONLY (named in StudioScene comment + e2e spec descriptions, never imported).

| File                                   | Status         | Last touched                                  |
| -------------------------------------- | -------------- | --------------------------------------------- |
| `walkaround/HybridStage.tsx`           | cross-ref-only | `5a1bacb` (collapse to WalkaroundStage clone) |
| `walkaround/HybridContext.ts`          | dead           | `263e534`                                     |
| `walkaround/useHybridFrameLoop.ts`     | dead           | `263e534`                                     |
| `walkaround/applyHybridShading.ts`     | dead           | `263e534`                                     |
| `walkaround/layers/DDGILayer.ts`       | dead           | `c9de48e`                                     |
| `walkaround/layers/RCLayerStub.ts`     | dead           | `263e534`                                     |
| `walkaround/layers/ReSTIRLayerStub.ts` | dead           | `263e534`                                     |
| `walkaround/layers/types.ts`           | dead           | `263e534`                                     |

**User memory says** keep these as Path-A reference for "future Phase 2+ layered re-introduction." But Path-B's single-shade-pass design fundamentally invalidates that rationale (the architecture problem that killed Path-A — `outputNode` reassignment crashing render-object cache — is a property of TSL re-injection, not of layering itself; Path-B sidesteps it via raw WGSL without re-injection).

Active code carries stale cross-references that mislead future readers:

- `StudioScene.tsx:21–24` describes the cluster as "kept in tree as Path-A reference"
- `HybridStage.tsx:3` JSDoc claims it mounts when `walkaroundEngine === 'hybrid'` — false; LayeredStage mounts
- `DDGILayer.ts:251` says "called by HybridStage on unmount" — false; HybridStage doesn't import DDGILayer
- `HybridContext.ts:91` says "Callers (typically HybridStage) stash this..." — false
- `e2e/14-walkaround-hybrid.spec.ts:81, 88` test descriptions say "HybridStage" — should say "HybridLayeredStage"

Decision options listed in §K below.

---

## H. Open question: tone-map (ACES vs Khronos PBR Neutral)

Memory `project_layered_hybrid_milestone.md` and `project_hybrid_renderer.md` D7=C says **Khronos PBR Neutral, NOT ACES**. But `composite.wgsl.ts:69` uses `acesFilm()` and `shade.wgsl.ts:804` comment was just updated to mention ACES. Either:

- the memory is stale (D7 was changed during merge but memory wasn't updated), or
- composite.wgsl was changed but is wrong

Need user verification before any tone-map-related cleanup proceeds.

---

## I. Files confirmed clean

`probeGrid`, `sceneBvh`, `hammersley.wgsl`, `octahedral.wgsl`, `probeUpdateBlend.wgsl`, `cascadePyramid` (functional logic — doc nit in C1), `composite.wgsl`, `atrous.wgsl`, `WalkaroundDebugBridge`, `walkaroundBridgeTypes`, `gpuDetection.test`, `bvhCommon.test`, `ddgiSampleWgsl.test`, `edges/cameDimensions`, `edgeConstants`, `useEdgeMaterial`, `SolderBead3D`, `EdgeLines` (no GI coupling — but B6 BVH-filter bug), `WalkaroundStage`, `RcStage`, `RestirStage`, `gpuDetection`, `lib/bvhCommon` (great test coverage; E1+C3 nits only), `e2e/12-ddgi`, `e2e/13-rc`, `e2e/13-restir` (E2-stale-header only).

---

## J. Sequencing (dependency-ordered, NOT priority)

The user's standing direction: every issue gets fixed; only sequencing decision is "in what order to unblock the next." Library extraction is the long-game target — most impactful refactors (C1–C6) need decisions resolved first.

**Wave 1 — Decisions + safe quick-wins (no Lovelace risk)**

- §G decision (orphan cluster: keep/delete/archive)
- §H decision (tone-map: ACES or PBR Neutral)
- E4 de-export sweep (12 exports, no body changes)
- E2 + E3 stale comment cleanse
- E5 eslint-disable removal
- D2 + D4 + D5 (WGSL/TSL helper consolidation — common.wgsl is already prepended; deleting locals is provably safe)

**Wave 2 — Real bugs (Lovelace validation each)**

- B1 fillCascadeDebug needsUpdate
- B5 visibility format mismatch
- B6 edge beads excluded from BVH
- B3 RC cascade level distance metric
- B2 walkaroundDiffuseLighting buffer-swap
- B7 verify double-mount reachability + fix if reachable

**Wave 3 — Hybrid-completeness coherence**

- B4 shared atlas-layout constants for DDGI samplers + canonical WGSL `rcSampleCascade`
- F1 add full-composition e2e spec (validates B3 + B4 fixes)
- F2 fixture extraction
- F3 stale spec header trim

**Wave 4 — Library-extraction surface (longest, highest decision-weight)**

- C5 export device-limit constant
- C6 engine registry (data-driven dispatch)
- C2 hardcoded scene constants → opt parameters
- C3 JSDoc completion + sentinel warnings
- C1 naming rename (sun → primaryLight) — coordinated rename across PipelineFrameInputs + bvhCompute opts + probeUpdatePass + JSDoc
- C4 HybridLayeredStage → useHybridLayeredGI hook + thin app shell

**Wave 5 — Iteration churn cleanup**

- D1 useSceneBVH consolidation (with type-bridge wrapper)
- D3 getDDGISampleFn singleton (depends on §G outcome)

**Wave 6 — Path-A cluster execution**

- Per §G outcome.

Each wave commits separately; each ships a Lovelace MCP screenshot in commit message.

---

## K. Decisions to confirm with user (one at a time)

K1 — Path-A orphan cluster (§G)
K2 — Tone-map canonical answer (§H)
K3 — Library extraction scope: full extraction to npm (Wave 4 in full) vs internal cleanup only (Wave 4 limited to in-tree refactor without API rename) vs defer entirely
K4 — Wave ordering: linear (Wave 1 → 6) vs parallel (Wave 1 + 2 + 3 in parallel since they're independent)

§§ K1–K4 will be asked in turn as AskUserQuestion prompts. The plan file remains a living document — answers update §J sequencing.
