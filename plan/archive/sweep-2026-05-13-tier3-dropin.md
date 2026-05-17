# Tier 3 — Drop-in productization

**Date:** 2026-05-13 | **Branch:** off `feat/sweep-2026-05-12-followup` (after H1–H6 land)

This plan resolves the 12 friction points that block "drop-in" adoption,
turning vitrum from a research-grade reference into a library that an
external developer can install and integrate in 5 lines.

**Target user (assumed):** existing THREE.js developers who want to add GI
to their app. API mirrors three.js conventions where reasonable; THREE.Scene
is a first-class input; vitrum's own Scene type is the internal canonical
form, exposed for advanced use.

If the target user is actually (b) new-from-scratch 3D-on-web devs or
(c) researchers, the architecture changes — flag now to redirect.

---

## Phase ordering and dependencies

```
T3.A (unified createEngine) ─┬─ T3.B (README) ─┬─ T3.C (hero examples)
                             ├─ T3.D (glTF loader) ┘
                             ├─ T3.E (telemetry hooks)
                             ├─ T3.F (lifecycle helpers + <VitrumCanvas>)
                             └─ T3.H (drop threeScene direct path)

T3.G (debug package) — independent, parallel
T3.I (animation validation) — independent, parallel
T3.J (npm publish prep) — gating; comes last
```

**Wave 1 (parallel):** T3.A, T3.G, T3.I
**Wave 2 (after T3.A):** T3.B, T3.D, T3.E, T3.F, T3.H
**Wave 3 (after T3.A + T3.D):** T3.C
**Wave 4 (gating):** T3.J

---

## Phase T3.A — Unified `createEngine()` factory

**File:** `packages/core/src/createEngine.ts` (new)

Replace the divergent `createPTEngine_WebGL2(canvas, opts)` and
`createWalkaroundHybridEngine({...30 fields...})` factories with a single
top-level factory that auto-picks the backend, derives sensible defaults
from scene + canvas, and exposes the engine via the existing `Engine`
contract.

```ts
export interface CreateEngineOptions {
  /** Canvas the engine renders into. Engine attaches a ResizeObserver. */
  canvas: HTMLCanvasElement;

  /** Scene (vitrum or THREE; auto-converted via three-bindings). */
  scene: Scene | THREE.Scene;

  /** Camera (per-frame updated; or supply a static initial). */
  camera: Camera | THREE.Camera;

  /** Hint at quality vs speed tradeoff:
   *   'realtime'     — prefer WalkaroundHybrid (WebGPU GI; 60fps target)
   *   'quality'      — prefer PathTracer (WebGL2 PT; converges over seconds)
   *   'auto' (def)   — pick based on capabilities + scene complexity */
  prefer?: 'realtime' | 'quality' | 'auto';

  /** Optional advanced overrides — most users leave empty. */
  advanced?: Partial<HybridEngineOptions> | Partial<PTEngineOptions>;

  /** Debug overlay opt-in. */
  debug?: boolean;
}

export async function createEngine(opts: CreateEngineOptions): Promise<Engine>;
```

Internal:
- **Capability detection:** `navigator.gpu?.requestAdapter()` for WebGPU;
  fall back to WebGL2.
- **Backend pick:**
  - `prefer: 'realtime'` + WebGPU → walkaround-hybrid.
  - `prefer: 'quality'` + WebGL2 → pt-webgl.
  - `prefer: 'auto'` (default) → walkaround-hybrid if WebGPU + scene
    triangle count < 500k; else pt-webgl.
- **Default derivation from scene AABB diagonal `D`:**
  - `cameraMoveResetThresholdSq = (D × 0.001)²`
  - `temporalAccumAlpha = 0.01` (frame-rate dependent; document)
  - `emitterDist2Floor = (D × 0.0001)²`
  - `triIntersectEpsilon = D × 1e-6`
  - directional light defaults: `[0.3, -0.7, 0.6]`, intensity 1.0
  - sky defaults: `[0.5, 0.7, 1.0]` tint, irradiance 0.3
- **Auto-resize:** `new ResizeObserver(...)` on canvas;
  call `engine.setCanvasSize(w, h)` automatically.
- **Auto-dispose chain:** `engine.dispose()` cleans up the ResizeObserver,
  GPU device (if engine-owned), and any persistent textures.

**Acceptance:**
- `await createEngine({ canvas, scene: threeScene, camera: threeCamera })`
  produces a working engine with no other config.
- Switching `prefer: 'quality'` swaps to PT without changing other code.
- `engine.dispose()` followed by `engine.dispose()` is idempotent.

**Tests:** `packages/core/__tests__/createEngine.test.ts`:
- Default-construction smoke test (mocked canvas + GPU).
- Capability-based pick (mock WebGPU absent → PT chosen).
- AABB-derived defaults match the formula.
- Idempotent dispose.

**Effort:** 3 days.

---

## Phase T3.B — README rewrite + capability matrix + quick start

**File:** `README.md` (rewrite from "package architecture overview" to
"what is vitrum + 5-line hello world + when-to-use-which-engine").

Sections:

1. **What is vitrum** — 1-paragraph elevator pitch.
2. **5-line hello world**:
   ```ts
   import { createEngine } from '@vitrum/core';
   const engine = await createEngine({ canvas, scene, camera });
   function frame() {
     engine.renderFrame({ camera });
     requestAnimationFrame(frame);
   }
   frame();
   ```
3. **Capability matrix** — table of "feature × engine" (PT, walkaround):
   - GI quality: realtime / converged
   - Bounce count: 1 / unlimited
   - Light types: point/area/dir/sky
   - Materials: PBR / +clearcoat / +transmission / +volumetric
   - Caustics: none / approximate / true
   - Animation: static / camera / dynamic geometry
   - Hardware: WebGPU / WebGL2 / both
4. **When to use which engine** — decision tree.
5. **API reference** — `createEngine`, `Engine`, `Scene`, `Camera`,
   `Material`, `Light` types with one-line descriptions + link to the
   per-package README for detail.
6. **Lifecycle** — Canvas remount, route change, tab visibility (with
   working `<VitrumCanvas>` snippet from T3.F).
7. **Examples** — link to `examples/hero-*`.
8. **Performance** — per-scene-complexity budget (1k tris@30fps integrated;
   100k tris@30fps dGPU; reference renders).

**Acceptance:**
- A graphics dev can read the README in 10 minutes and have a working
  vitrum integration.
- The 5-line example actually works copy-pasted into a Vite project
  (verify by following the README on a clean machine).

**Effort:** 1 day.

---

## Phase T3.C — Hero examples

**Files:** `examples/hero-viewer/`, `examples/hero-lighting-designer/`,
`examples/hero-product-viz/` (3 new directories).

Each is a standalone Vite project (the existing `examples/cornell-box`
is the template). Each MUST run `npm install && npm run dev` to a working
demo without further setup.

**`examples/hero-viewer/`** — drop-in glTF viewer:
- 5-line setup (matches the README hello-world).
- Drag-drop a .glb file → renders.
- Camera orbit controls (use `three/examples/jsm/controls/OrbitControls`).
- Engine pick toggle (PT / walkaround) in UI.

**`examples/hero-lighting-designer/`** — interactive lighting:
- Static scene (small architectural interior).
- 3–5 draggable lights; UI panel for color, intensity, type.
- Real-time GI updates (forces walkaround-hybrid).
- Frame-time HUD (uses T3.E telemetry).

**`examples/hero-product-viz/`** — PT product render:
- Static jewelry / glass scene (commission a small one or use a free CC0
  glTF; cite source).
- Progressive PT render with "samples accumulated" counter.
- Material editor: roughness, metallic, transmission, IOR sliders.
- "Save high-res render" button (renders at 4K, saves .png).

**Acceptance:**
- Each demo runs cleanly on first try.
- Each is <200 LoC of host code (the engine handles the heavy lift).
- Each demonstrates a feature that the README capability matrix calls out.

**Effort:** 1 week.

---

## Phase T3.D — `@vitrum/three-bindings` glTF loader pipeline

**File:** `packages/three-bindings/src/gltfLoader.ts` (new).

```ts
export async function loadGltfScene(
  url: string | File | Blob,
): Promise<{ scene: Scene; camera?: Camera; lights?: Light[] }>;
```

Internally:
1. Load the glTF/glb via `three/examples/jsm/loaders/GLTFLoader`.
2. Convert THREE.Scene → vitrum Scene via existing `sceneFromThreeJS`.
3. Extract embedded cameras and lights if present; surface as defaults.

**Tests:** unit-test with a small CC0 glTF fixture in
`packages/three-bindings/__tests__/fixtures/`.

**Acceptance:** the hero-viewer example uses this as its scene loader.

**Effort:** 2 days.

---

## Phase T3.E — Telemetry hooks

**Files:** `packages/core/src/engine.ts` (extend Engine interface),
each engine implementation.

```ts
export interface Engine {
  // ...existing methods...

  /** Subscribe to per-frame stats. Returns unsubscribe fn. */
  onFrame(cb: (stats: FrameStats) => void): () => void;

  /** Subscribe to long-running progress (PT spp, denoiser convergence). */
  onProgress(cb: (progress: ProgressStats) => void): () => void;
}

export interface FrameStats {
  frameTimeMs: number;
  gpuTimeMs?: number;       // if timestamp queries supported
  passTimings?: Record<string, number>;
  spp?: number;             // for PT
  bvhDepth?: number;        // diagnostic
  estimatedGpuMemoryBytes?: number;
}

export interface ProgressStats {
  kind: 'pt-spp' | 'denoiser-converge' | 'ddgi-warmup';
  current: number;
  target: number;
  fraction: number;
}
```

**Implementation:**
- `WalkaroundHybridEngine`: emits `frameTimeMs` from rAF deltas; `passTimings`
  from existing `timestampQueries.ts`; estimated memory from
  `resourceManager.ts` totals.
- `PTEngine_WebGL2`: emits `spp` progress.

**Acceptance:**
- Hero-lighting-designer's frame-time HUD subscribes via `onFrame()`.
- Hero-product-viz's progress bar subscribes via `onProgress()`.

**Effort:** 1 day.

---

## Phase T3.F — Lifecycle helpers + `<VitrumCanvas>` React component

**Files:**
- `packages/three-bindings/src/react/VitrumCanvas.tsx` (new — opt-in React
  helper; package gets `react` as a peer dep marked optional).
- `packages/three-bindings/src/lifecycle/vanilla.ts` (vanilla equivalent
  for non-React hosts).

```tsx
// React
import { VitrumCanvas } from '@vitrum/three-bindings/react';
<VitrumCanvas
  scene={scene}
  camera={camera}
  prefer="realtime"
  onFrame={(stats) => setFps(1000 / stats.frameTimeMs)}
/>
```

The component:
- Creates a `<canvas>` ref.
- `useEffect` calls `createEngine()` on mount, `engine.dispose()` on unmount.
- Survives Canvas remount (via key-stable engine instance in a ref).
- Pauses the rAF loop when `document.visibilityState === 'hidden'`.
- Resizes via the engine's auto-attached ResizeObserver.

```ts
// Vanilla
import { attachVitrum } from '@vitrum/three-bindings/lifecycle';
const handle = await attachVitrum({ canvas, scene, camera, prefer: 'auto' });
handle.dispose();  // unmount
```

**Tests:** mount/unmount cycle test (jsdom + mocked GPU); visibility-pause
test.

**Acceptance:** hero-viewer uses `<VitrumCanvas>` in <30 LoC including imports.

**Effort:** 3 days.

---

## Phase T3.G — `@vitrum/dev` debug package

**New package:** `packages/dev/`

Opt-in dev-only debug overlays. Hosts add as a devDependency.

**Components:**

1. **`<FrameTimeHUD>`** — top-right corner; live frame time + 60-frame
   moving average + pass breakdown. Uses `engine.onFrame()`.

2. **`<DDGIAtlasViewer>`** — pop-out panel showing the irradiance + visibility
   atlases live. Click a probe to highlight in 3D.

3. **`<BVHVisualizer>`** — overlay the BVH bounding boxes color-coded by
   depth. Toggle via key.

4. **`<GISignalSplit>`** — view direct, indirect, AO, total separately
   (split-screen 2×2).

5. **`<DenoiserABToggle>`** — keyboard `D` toggles denoiser on/off so the
   user can compare.

6. **`<MaterialInspector>`** — click a mesh, see its vitrum Material params,
   live-edit them.

These are React components that pair with `<VitrumCanvas>`. The package
also exposes a vanilla `attachDebugOverlays(engine, canvas)` for non-React.

**Acceptance:** hero-lighting-designer uses `<FrameTimeHUD>` and
`<MaterialInspector>`; documented in their READMEs.

**Effort:** 1 week.

---

## Phase T3.H — Drop the `threeScene` direct path on HybridEngine

**File:** `packages/walkaround-hybrid/src/HybridEngine.ts`.

Currently the constructor takes `threeScene: THREE.Scene` directly AND
exposes `setScene(vitrumScene)`. Two paths into the same engine.

After T3.A lands, the unified `createEngine()` does the THREE.Scene →
vitrum Scene conversion in the factory, so the engine itself only needs
the `setScene()` path.

Steps:
1. Mark `threeScene` option as `@deprecated` in HybridEngineOptions.
2. Internally, `createEngine()` calls `sceneFromThreeJS(threeScene)`
   then `engine.setScene(vitrumScene)`. Construction no longer plumbs
   THREE through the engine.
3. After 1 sprint of deprecation, remove the `threeScene` field.

**Acceptance:** the engine has one canonical scene-input path; the host
never directly passes THREE.Scene to the engine constructor.

**Effort:** ½ day.

---

## Phase T3.I — Animation/dynamic-geometry validation + docs

**File:** new `plan/animation-support-status.md`.

The engine assumes static scenes. Camera animation works (camera is
per-frame in `FrameInput`). Dynamic geometry: untested.

Steps:
1. Author 3 test cases:
   - **Camera-only animation:** orbit camera around a static scene; verify
     temporal accumulator resets cleanly on motion.
   - **Light animation:** move a directional light; verify GI updates
     (DDGI atlas should track; ReSTIR-DI should re-converge).
   - **Mesh transform animation:** translate a mesh per frame; verify the
     BVH rebuild path doesn't leak memory or stall the frame.

2. Document what works, what doesn't, and the costs:
   - Camera animation: full support; uses `prevViewMatrix` for ReSTIR temporal.
   - Light animation: limited — DDGI re-converges over ~30 frames; ReSTIR
     immediate.
   - Mesh transform animation: requires BVH rebuild (currently full re-build;
     incremental TLAS not implemented).
   - Skinning: not supported; needs a `BoneSkinning` helper that vitrum
     doesn't yet have.

3. File 1–2 specific gap items into the backlog with effort estimates.

**Acceptance:** `animation-support-status.md` accurately captures the
current state. Hero-lighting-designer demonstrates light animation
working.

**Effort:** 2 days.

---

## Phase T3.J — npm publish prep

This is gating; everything else has to land first.

Steps:
1. Verify each `packages/*/package.json` has correct `main`, `types`,
   `exports`, `files`, `repository`, `license`, `description`.
2. Run `npm pack --dry-run` per package to confirm shipped contents
   (no fixtures, no internal docs leaking).
3. Pin peer dependencies to versions that exist in npm (currently many
   `file:../...` workspace links; needs publishable form).
4. Author a short `RELEASING.md` for the user (the publish itself is
   their explicit-instruction call per CLAUDE.md).
5. Set up an npm scope `@vitrum/*` (user creates the npm org).
6. Tag a `0.1.0-alpha.1` release on the branch.
7. **Stop short of `npm publish`** — that's the user's explicit-instruction
   step.

**Acceptance:** the user can run `npm publish --workspaces --tag alpha`
and it succeeds. Until they do, vitrum is published-ready but not
published.

**Effort:** 1 day after everything else lands.

---

## Tier 3 totals

- T3.A: 3 days
- T3.B: 1 day
- T3.C: 5 days
- T3.D: 2 days
- T3.E: 1 day
- T3.F: 3 days
- T3.G: 5 days
- T3.H: ½ day
- T3.I: 2 days
- T3.J: 1 day

**Total: ~23.5 days, compressible to ~2 weeks with parallelism.**

Wave-by-wave:
- Wave 1 (parallel): T3.A (3d), T3.G (5d), T3.I (2d) — 5d wall-time
- Wave 2 (after T3.A): T3.B + T3.D + T3.E + T3.F + T3.H — ~3d wall-time
  (parallel)
- Wave 3 (after T3.A + T3.D): T3.C — 5d
- Wave 4: T3.J — 1d

≈ 14 days wall-time if parallelized; ≈ 24 days serial.

---

## Open decisions before execution

**Target user.** Plan assumes (a) THREE.js developers per project history
(stainedGlass + three-bindings precedent + threeScene direct path on
HybridEngine). If actually (b) new-from-scratch 3D-on-web devs, the
React component naming, glTF loader location, and example boilerplate all
shift. Confirm before T3.B/F/G land.

**npm scope.** Plan assumes `@vitrum/*` scope exists or will be created.
If a different name is needed (e.g., scope already taken), tell me.

**Hero scenes — sourcing.** The hero examples need 1–2 polished glTF
scenes (interior + jewelry). Options:
- Commission small ones (slow + costs).
- Use Khronos sample assets (free CC0; but generic-looking).
- Use Sketchfab CC0 scenes (curated, cite source).
- Build minimal procedural scenes (cheap; less impressive).
Recommend Sketchfab CC0 + cite for time/cost balance.

**`<VitrumCanvas>` React-only?** The plan ships React + vanilla. If you
want Vue/Svelte/Solid wrappers too, they're each ~½ day extra. Recommend
React + vanilla only (covers 80% of hosts).
