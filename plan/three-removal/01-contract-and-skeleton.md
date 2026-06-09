# WS1 — Package skeleton + `Engine` contract implementation

> The foundation. Unblocks every other workstream. All types here are quoted from `@vitrum/core` (verified `file:line`). `pt-webgl`'s `ptEngineWebGL2.ts` and pt-webgpu's `index.ts` are the two reference implementations to copy patterns from.

## 1. Package layout

```
packages/pt-webgl2/
  package.json            # name @vitrum/pt-webgl2; deps: @vitrum/core, @vitrum/shared-bvh,
                          #   @vitrum/shared-samplers, @vitrum/shared-denoisers. NO three, NO three-mesh-bvh.
  tsconfig.json           # extends ../../tsconfig.base.json (composite project ref to the 3 shared pkgs + core)
  vitest.config.ts        # jsdom + a webgl2 polyfill stub for unit tests (mirror pt-webgl's)
  src/
    index.ts              # createPTEngine_WebGL2 factory + PTEngineWebGL2 class (unexported) + PTEngineWebGL2Surface
    options.ts            # PTEngineWebGL2Options extends EngineOptions (device: WebGL2RenderingContext)
    state.ts              # StateSlot + makeStateSlot (copy verbatim from pt-webgl:442-455)
    capabilities.ts       # PT_WEBGL2_SUPPORT (Required<SupportSets>) + buildCapabilities()
    gl/                   # WS2 — the raw-WebGL2 framework (GlResources, programBuilder, fbo, fullscreenQuad, blend)
    scene/               # WS3 — packers + BVH adapter + uploadSceneTextures
    glsl/                # WS4 — ported kernels + composeTraceGlsl + the BVHShaderGLSL port
    frameParamsPacker.ts  # WS5 — the per-frame uniform packer
    traceTier.ts          # WS5 — WebGL2 capability-tier gate
    __tests__/
      packageBoundary.test.ts   # asserts no `from 'three'` anywhere (mirror pt-webgl legacyThreeBoundary)
```

## 2. The `Engine` contract — what we MUST implement

Verified required (non-optional) members (`core/src/engine/index.ts`): `state:52`, `capabilities:53`, `setScene:59`, `renderFrame:196`, `reset:201`, `pause:250`, `resume:254`, `dispose:261`. Everything else is `?`-gated. We additionally implement `getScene`, `onFrame`, `onProgress`, and the incremental-patch methods (`updatePrimitive`/`updateEmitter`/`updateEnvironment`/`updateLighting`) to match pt-webgl's promise row.

Exact signatures to satisfy (verbatim, `core/src/engine/index.ts`):
```ts
readonly state: EngineState;                                   // EngineState = 'uninitialized'|'initializing'|'ready'|'paused'|'error'|'disposed'
readonly capabilities: EngineCapabilities;
setScene(scene: Scene): void;
getScene?(): Scene | null;
updatePrimitive?(id: string, patch: Partial<ScenePrimitive>): void;
updateEmitter?(id: string, patch: Partial<SceneEmitter>): void;
updateEnvironment?(env: SceneEnvironment | null): void;
updateLighting?(opts: Readonly<Record<string, unknown>>): void;
renderFrame(input: FrameInput): FrameOutput;       // FrameOutput = FrameSkipped | FrameRendered
reset(): void;
pause(): void; resume(): void; dispose(): void;
onFrame?(cb: (stats: FrameStats) => void): () => void;
onProgress?(cb: (progress: ProgressStats) => void): () => void;
```

## 3. `state.ts` — copy verbatim (the `StateSlot` pattern is backend-local, not a core type)

```ts
import type { EngineState } from '@vitrum/core';

export interface StateSlot { readonly get: () => EngineState; readonly set: (s: EngineState) => void; }
export function makeStateSlot(initial: EngineState = 'initializing'): StateSlot {
  let s: EngineState = initial;
  return { get: () => s, set: (v) => { s = v; } };
}
```
State machine (mirror pt-webgl): `initializing` (in factory, pre-`set('ready')`) → `ready` ↔ `paused` (`pause`/`resume`) → `disposed` (terminal). Every mutator opens with `if (this.#slot.get() === 'disposed') throw new Error('<method>: engine is disposed');`. `dispose()` sets `'disposed'` **last**, after freeing GL resources.

## 4. `options.ts`

```ts
import type { EngineOptions } from '@vitrum/core';

export interface PTEngineWebGL2Options extends EngineOptions {
  /** Host-owned WebGL2 context. The engine allocates against it but NEVER loses/destroys it. */
  readonly device: WebGL2RenderingContext;
  /** Optional forced tier (mirror pt-webgpu traceTier): 'full' | 'lite'. */
  readonly traceTier?: 'full' | 'lite';
  readonly spectral?: boolean;
  readonly bdpt?: boolean;
}
```
Note `EngineOptions.device` is typed `unknown` in core (`factory.ts:49`); the package narrows it. Shared option fields available: `maxBounces`, `maxSamplesPerPixel`, `denoiser`, `causticStrategy`, `causticOptions`, `extensions` (`factory.ts:46-153`).

## 5. `index.ts` — the class skeleton (mirror pt-webgpu's `index.ts` structure)

```ts
class PTEngineWebGL2 implements Engine {
  readonly #slot: StateSlot;
  readonly #gl: WebGL2RenderingContext;
  readonly #maxBouncesLimit: number;
  readonly #maxSamplesLimit: number;
  readonly #causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly #traceTier: 'full' | 'lite';

  #scene: Scene | null = null;
  #sceneTextures: UploadedSceneTextures | null = null;   // WS3 — the GL texture bundle (analog of UploadedSceneBuffers)
  #geoPack: ScenePackResult | null = null;               // from @vitrum/shared-bvh, for incremental patches
  #samplesAccumulated = 0;
  #activeBounces = 1;
  #lastFrameInput: FrameInput | null = null;

  readonly #gpu: GlResources;                            // WS2 — owns FBOs/textures/programs (analog of GpuResources)
  #onFrameSubs = new Set<(s: FrameStats) => void>();
  #onProgressSubs = new Set<(p: ProgressStats) => void>();
  readonly #spectralEnabled: boolean;
  readonly #bdpt: boolean;
  readonly #postDenoiser: OIDNFinalDispatcher | null;    // reuse @vitrum/shared-denoisers OIDNDispatcherCore

  constructor(opts: PTEngineWebGL2Options, slot: StateSlot, traceTier: 'full' | 'lite') {
    this.#slot = slot; this.#gl = opts.device;
    this.#maxBouncesLimit = clampBounces(opts.maxBounces);
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SPP;
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    this.#traceTier = traceTier; this.#spectralEnabled = opts.spectral ?? false; this.#bdpt = opts.bdpt ?? false;
    this.#gpu = new GlResources(this.#gl, traceTier, this.#bdpt);
    this.#postDenoiser = opts.denoiser === 'oidn-final' ? new OIDNFinalDispatcher(/* ...glReadback */) : null;
  }

  get state(): EngineState { return this.#slot.get(); }
  get capabilities(): EngineCapabilities { return buildCapabilities(this.#causticStrategy, this.#maxBouncesLimit, this.#maxSamplesLimit); }

  setScene(scene: Scene): void {
    if (this.#slot.get() === 'disposed') throw new Error('setScene: engine is disposed');
    this.#repackScene(scene);   // WS3 — partition → packSceneFromCore → pack textures → upload → reset
  }
  getScene(): Scene | null { return this.#scene; }     // capability-filtered retained scene (NOT a copy)

  renderFrame(input: FrameInput): FrameOutput { /* WS5 — see 05-frame-loop-and-features.md §1 */ }
  reset(): void { this.#samplesAccumulated = 0; this.#gpu.clearAccum(); }
  pause(): void { this.#guardLive('pause'); this.#slot.set('paused'); }
  resume(): void { this.#guardLive('resume'); this.#slot.set('ready'); }

  dispose(): void {
    if (this.#slot.get() === 'disposed') return;
    this.#postDenoiser?.dispose();
    this.#gpu.dispose();                 // deletes all FBOs/textures/programs
    this.#sceneTextures?.destroy();
    this.#sceneTextures = null; this.#scene = null; this.#geoPack = null;
    this.#onFrameSubs.clear(); this.#onProgressSubs.clear();
    this.#slot.set('disposed');          // last
  }

  onFrame(cb: (s: FrameStats) => void): () => void { this.#onFrameSubs.add(cb); return () => this.#onFrameSubs.delete(cb); }
  onProgress(cb: (p: ProgressStats) => void): () => void { this.#onProgressSubs.add(cb); return () => this.#onProgressSubs.delete(cb); }
  // updatePrimitive/updateEmitter/updateEnvironment/updateLighting — WS5 incremental-patch (mirror pt-webgl scenePatch)

  #guardLive(m: string): void { if (this.#slot.get() === 'disposed') throw new Error(`${m}: engine is disposed`); }
}

export interface PTEngineWebGL2Surface { /* e.g. getDenoisedFrame(): BackendTexture | null */ }

export const createPTEngine_WebGL2: EngineFactory<PTEngineWebGL2Options, Engine & PTEngineWebGL2Surface> =
  async (opts) => {
    const gl = opts.device;
    if (gl == null || typeof gl.createFramebuffer !== 'function')
      throw new TypeError('createPTEngine_WebGL2: device must be a WebGL2RenderingContext');
    if (opts.maxBounces !== undefined && opts.maxBounces < 1)
      throw new RangeError(`createPTEngine_WebGL2: maxBounces must be >= 1 (got ${opts.maxBounces})`);
    const traceTier = resolveWebGl2TraceTier(gl, opts.traceTier);   // WS5 traceTier.ts
    const slot = makeStateSlot();              // 'initializing'
    const engine = new PTEngineWebGL2(opts, slot, traceTier);
    slot.set('ready');
    return engine;
  };
```

## 6. `capabilities.ts` — the single-source pattern (mirror `PT_WEBGPU_SUPPORT`)

The packer and the advertised capabilities must read the **same** support object so they can't diverge (the verified pt-webgpu invariant, `index.ts:391-399`).

```ts
import type { SupportSets, ScenePrimitive, SceneEmitter, SceneEnvironment, AnalyticShape, EngineCapabilities } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';

export const PT_WEBGL2_SUPPORT: Required<SupportSets> = {
  supportedPrimitiveKinds: new Set<ScenePrimitive['kind']>(['mesh', 'instanced-mesh', 'analytic', 'skinned-mesh']),
  supportedEmitterKinds: new Set<SceneEmitter['kind']>(['directional', 'point', 'spot', 'rect-area', 'disc-area', 'mesh-area']),
  supportedAnalyticShapes: new Set<AnalyticShape>([/* WebGL2 analytic set; subset of pt-webgpu's */]),
  supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri', 'procedural-sky']),
};

export function buildCapabilities(causticStrategy: EngineCapabilities['causticStrategy'], maxBounces: number, maxSpp: number): EngineCapabilities {
  return {
    supportsIncrementalScene: true,
    incrementalPatchSupport: { transform: true, positions: true, material: true, emitter: true, topology: false },
    supportsAddRemovePrimitive: false,
    supportsAuxBuffers: true,            // MRT G-buffer (gNormalDepth/gAlbedo) — degrade to false if no MRT
    accumulates: true,
    maxSamplesPerPixel: maxSpp, maxBounces,
    supportedAnalyticShapes: new Set(PT_WEBGL2_SUPPORT.supportedAnalyticShapes),
    supportedEmitterKinds:  new Set(PT_WEBGL2_SUPPORT.supportedEmitterKinds),
    supportedPrimitiveKinds: new Set(PT_WEBGL2_SUPPORT.supportedPrimitiveKinds),
    supportedEnvironmentKinds: new Set(PT_WEBGL2_SUPPORT.supportedEnvironmentKinds),
    presentationMode: 'offscreen-texture',
    causticStrategy,                     // REQUIRED field
    supportDetails: BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails,   // by reference, post §5-master core edit
  };
}
```
`setScene` then runs `partitionSceneBySupport(inputScene, this.capabilities)` (`core/src/scene/partitionSceneBySupport.ts:63`) to warn-and-skip unsupported nodes — exactly pt-webgl `ptEngineWebGL2.ts:900` / pt-webgpu `uploadSceneBuffers.ts:258`.

## 7. Texture branding (the FrameOutput contract)

Every output handle is branded at the boundary: `asBackendTexture<'pt-webgl2', WebGLTexture>(tex)` (`core/src/frame.ts:226`). `FrameRendered` requires `primaryRadiance: BackendTexture` and optionally `normalDepth`/`albedo`/`variance`/`motionVectors` (`frame.ts:178-204`). There is no per-channel helper — the four brand helpers (`asBackendTexture`, `asBackendTextureFormat`, `narrowToBackendTexture`, `narrowToBackendTextureFormat`) are the complete set.

## 8. WS1 done-when
- The skeleton typechecks against `@vitrum/core` with all 8 required members + the contract subset above.
- `createPTEngine_WebGL2(stubGl)` returns an engine in state `'ready'`; `dispose()` → `'disposed'`; mutators throw post-dispose.
- `packageBoundary.test.ts` green (no `from 'three'`).
- `renderFrame` returns `{ kind: 'skipped', samplesAccumulated: 0, isConverged: false }` until WS2/WS3/WS5 land (a legal `FrameSkipped`).
