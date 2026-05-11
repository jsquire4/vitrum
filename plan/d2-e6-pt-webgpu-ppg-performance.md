# Plan: D2 (pt-webgpu trace dedup) + E6 (PPG spatial acceleration)

**Goal:** Reduce maintenance risk and shader cost for two deferred remediation items from `plan/remediation-checklist.md`.

---

## Part D2 — `@vitrum/pt-webgpu`: composable trace bodies

### Current state

In `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts`:

- `traceClosest` (from ~725) walks the mesh BVH with a **shrinking** far slab (`intersectAabb(..., tMin, hit.dist)`), updates a full `SceneHit` (barycentric-ish normal, `triIndex`, etc.).
- `traceAny` (from ~859) duplicates the same stack traversal and leaf triangle loops, but uses a **fixed** `tMax` for both AABB and triangle tests and returns early on first hit.
- The **analytic primitives** block (loop over `analyticHeaders`, world↔local transforms, shape dispatch, world ray-parameter test) is also duplicated with the same structural difference: `worldT < hit.dist` vs `worldT < tMax`.

The only semantic differences are (1) the ray interval used for AABB clipping and (2) whether a triangle/analytic hit updates `SceneHit` vs returns `true` immediately.

### Risk if left duplicated

Future fixes (UVs, alpha test, instance offsets, BVH layout changes) must be applied twice; divergent behavior between shadow/visibility rays and primary closest-hit is a common source of light leaks and self-intersection bugs.

### Recommended architecture

**Single traversal core with a small “mode” contract** — WGSL has no closures, so use an explicit flag or a `ptr<function, f32>` for the active far bound.

1. **Extract mesh BVH traversal** into one internal helper, e.g. `fn traceMeshBvhInterval(ray: Ray, tMin: f32, tFar: ptr<function, f32>, closest: bool, hit: ptr<function, SceneHit>) -> bool`
   - `closest == true`: after each closer triangle hit, write `SceneHit` and set `*tFar = hit.dist` (matching current `traceClosest` AABB behavior).
   - `closest == false`: read `*tFar` once as the fixed shadow/light bound; never shrink the slab for AABB tests; on triangle hit in `(tMin, *tFar)`, return `true` immediately (or set `hit.didHit` and return — pick one convention and document it).
   - Keep the existing fixed-size stack (`array<u32, 64>`), node bounds fetch, and leaf triangle indexing **verbatim** on first refactor; only deduplicate structure.

2. **Extract analytic intersection** into `fn traceAnalyticInterval(...)` with the same `(tMin, tFar, closest)` pattern and shared world-distance check `worldT > tMin && worldT < *tFar`.

3. **Thin public wrappers** (preserve call sites and shader readability):

   ```wgsl
   fn traceClosest(ray: Ray, tMin: f32, tMax: f32) -> SceneHit { ... }
   fn traceAny(ray: Ray, tMin: f32, tMax: f32) -> bool { ... }
   ```

4. **Optional packaging:** if the main module string becomes unwieldy, move the extracted bodies into `packages/pt-webgpu/src/wgsl/traceScene.wgsl.ts` (or `bvhTraverse.wgsl.ts`) and concatenate from `pathTraceBruteforce.wgsl.ts` (same pattern as `common.wgsl.ts`). Keep **one** WGSL module entry point for the path tracer so bind layouts stay unchanged.

### Verification / DoD

- `npm run typecheck` and `npm test` for `@vitrum/pt-webgpu` (including `wgslContract.test.ts`).
- Per project protocol: capture a **before** reference render and an **after** for a representative scene (e.g. under `tools/reference-renders/`), A/B visually; numerical pixel diff acceptable only if explained (e.g. reorder of co-planar hits should not change).
- Grep confirms only one copy of the BVH leaf inner loop and one copy of the analytic shape dispatch block.

### Sequencing

1. Refactor only (no algorithm change), merge.
2. Optional follow-up: specialize `traceAny` further (cheaper normal, skip barycentric) once behavior is unified — **separate** change with its own reference pass.

---

## Part E6 — `@vitrum/walkaround-hybrid`: PPG cell lookup acceleration

### Current state

`packages/walkaround-hybrid/src/ppg/wgsl/ppgSample.wgsl.ts` — `ppgFindCellIndex` does a linear scan over `arrayLength(&ppgCells)` (up to ~10K). That is intentional for Sprint 11 structure but **not** viable for full-frame dispatch (see the blocking comment in-file and `plan/sprint-11-ppg-integration.md`).

Both `ppgSampleDirection` and `ppgPDF` call `ppgFindCellIndex` → every acceleration improvement applies twice (same as D2 lesson: centralize lookup).

### Target complexity

Replace **O(N)** per lookup with **O(log N)** (kd-tree on cell centers, as in the sprint doc) or **O(1)** amortized (uniform spatial grid / hashing). Pick one primary strategy and document the other as fallback.

### Option A — **Kd-tree on cell positions** (aligns with existing sprint note)

**CPU / host (TypeScript):**

- When `ppgCells` buffer is (re)built or defragmented, run a deterministic builder over active cells only:
  - Input: list of `(cellIndexInBuffer, position)` for cells that participate in guiding.
  - Output: flat `ppgKdNodes` storage buffer: per node, `splitAxis` (u32 0|1|2), `splitPos` (f32), `leftChild`, `rightChild`, `cellIndexOrSentinel` for leaves — exact layout to match WGSL struct packing (watch padding; mirror `PPGSpatialCell` discipline).
- Reuse or cite a small in-repo kd-build (no new heavy deps) — median split on longest axis is enough for v1.

**GPU / WGSL:**

- Replace `ppgFindCellIndex` with iterative descent from root index 0 (explicit stack or pointer-walking if heap layout is implicit).
- **Nearest-neighbor** vs **point containment:** current shader uses **nearest cell center** in Euclidean distance. A kd-tree built on centers supports exact NN search but GPU-friendly variant is often:
  - **Approximate:** descent to leaf containing query point, then check a small constant set of neighbor cells; or
  - **Exact NN:** maintain a best-so-far distance and traverse with prune (more logic, still O(log N) typical for low dimension).

For v1, specify in the plan implementation: **exact NN on 3D kd-tree** with a small fixed stack (depth ≤ ceil(log2(N)) + small reserve for backtracking if you implement full NN; simpler MVP is “descend to bucket + linear scan of bucket-sized subset” if you use **implicit grid** hybrid).

**Bindings:**

- New storage buffer + bind group entry in the eventual PPG bind group (`pipelineCompiler` / `resourceManager`); version `PPGSpatialCell` consumers if needed.

### Option B — **Uniform grid / spatial hash** (often faster to ship on GPU)

- Host precomputes AABB of all cell positions + cell size estimate.
- WGSL maps `worldPos` to `(i,j,k)`; looks up `head` index in a chain or a fixed small list of candidate cell indices in the neighborhood; computes argmin distance over **bounded** candidates (e.g. 8–27 buckets).
- O(1) lookup when density is roughly uniform; may need tuning when cells cluster.

**Recommendation:** Implement **Option A** if staying aligned with `plan/sprint-11-ppg-integration.md`; spike **Option B** only if kd NN proves too branchy on real hardware (measure with timestamp queries already present in walkaround).

### Verification / DoD

- **Correctness:** Unit test on CPU: for random queries and fixed cell sets, `findCellIndex_accel(worldPos) == findCellIndex_bruteforce(worldPos)` (allow tie-break rule: smallest index on equal distance — document).
- **Performance:** Micro-benchmark or timestamp-query note: lookup time vs N (e.g. N=1K, 4K, 10K).
- **Integration:** Keep shader structurally valid when PPG dispatch is still off; no regression in packages that only typecheck-include PPG strings.
- Update or remove the **blocking** banner comment in `ppgSample.wgsl.ts` once complexity is acceptable.

### Sequencing

1. Land **CPU builder + buffer layout** + tests (no WGSL change) behind a feature flag or dead buffer until bindings wired.
2. Swap `ppgFindCellIndex` implementation; retain brute-force behind `#ifdef`-style comment or `const USE_BRUTE_PPG_LOOKUP = false` in TS-injected prefix if you need A/B.
3. When Sprint 11 dispatch is wired, validate end-to-end variance reduction (reference frame optional).

### Dependencies

- **E6** is gated on PPG buffer lifecycle (`resourceManager.ts`, eventual `setScene`/`reset` hooks) — confirm rebuild triggers whenever spatial cell positions change.

---

## Cross-cutting notes

- Both items benefit from **one internal lookup / traversal function** and **thin public APIs** so PDF and sampling (E6) or closest vs any-hit (D2) cannot drift.
- Neither item requires npm publish; workspace `file:` links only.

When both are done, update `plan/remediation-checklist.md` to check off **D2** and **E6** and add a short blurb to `CHANGELOG.md`.
