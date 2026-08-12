// <VitrumCanvas> — drop-in React component.
//
// Wraps the attachVitrum() vanilla helper so React-app hosts can render
// a vitrum-driven canvas with a single component. Lifecycle is honest:
//
//   - useEffect creates the engine on mount, disposes on unmount.
//   - The engine instance lives in a ref, so Strict-Mode-re-mount doesn't
//     end up with two engines for the same canvas (we tear down + recreate
//     in one effect cycle).
//   - The component pauses the loop when document.visibilityState !==
//     'visible' (delegated to attachVitrum's pauseOnHidden default).
//   - The component re-creates the engine when `scene` or `prefer`
//     changes (those are creation-time decisions on the Engine contract;
//     no mutate-in-place path). For `quality` changes we push the latest
//     value into the rAF tick via a getter closure over a ref — no engine
//     churn, and prop updates propagate on the very next frame.
//
// `react` is an OPTIONAL peer dep of @vitrum/engine; this file does not
// load unless the host explicitly imports `@vitrum/engine/react`. Non-React
// hosts never pay the cost.

import * as React from 'react';
import type {
  AdapterProfile,
  EngineError,
  EngineWarning,
  Scene,
  FrameInput,
  FrameStats,
  ProgressStats,
} from '@vitrum/core';
import {
  releaseGltfResources,
  type GltfAssetInput,
  type GltfAssetResult,
  type GltfForEngineResult,
} from '@vitrum/gltf-adapter';
import type {
  AttachVitrumHandle,
  AttachVitrumBackendChangedEvent,
  AttachVitrumCamera,
  AttachVitrumRecreateEngineFactory,
  AttachVitrumSceneControllerPlayback,
} from '../lifecycle/vanilla.js';
import { attachVitrum } from '../lifecycle/vanilla.js';
import type {
  CreateEngineErrorEvent,
  EngineWithBackendId,
  EnginePreference,
  CreateEngineOptions,
} from '../createEngine.js';
import {
  loadGltfWithEngine,
  loadGltfWithProgressiveEngine,
  type GltfCreateProgressiveEngineOptions,
  type GltfProgressiveEngineResult,
  type LoadGltfWithEngineOptions,
} from '../gltf.js';
import { createEngineGltfAssetHint } from '../gltfAssetHint.js';
import { createProgressiveEngine, progressiveHandleAsEngine } from '../createProgressiveEngine.js';

export type VitrumCanvasGltfOptions = Omit<
  LoadGltfWithEngineOptions,
  'engine' | 'engineOptions' | 'attachScene'
>;

export type VitrumCanvasProgressiveGltfOptions = Omit<GltfCreateProgressiveEngineOptions, 'canvas'>;

type VitrumCanvasGltfResult =
  | GltfForEngineResult<EngineWithBackendId>
  | GltfProgressiveEngineResult;

function isProgressiveGltfResult(
  result: VitrumCanvasGltfResult | undefined,
): result is GltfProgressiveEngineResult {
  return result?.engine != null && 'coordinator' in result.engine;
}

function disposePendingGltfEngine(result: VitrumCanvasGltfResult | undefined): void {
  try {
    result?.engine?.dispose();
  } catch {
    // The attach path is already being abandoned; cleanup stays best-effort.
  } finally {
    releaseGltfResources(result);
  }
}

function releasePendingGltfResources(result: VitrumCanvasGltfResult | undefined): void {
  releaseGltfResources(result);
}

function vitrumCanvasPlaybackEnabled(
  playback: AttachVitrumSceneControllerPlayback | undefined,
): boolean {
  if (playback === false) return false;
  if (playback === undefined) return true;
  return playback === true || typeof playback === 'object';
}

function vitrumCanvasPlaybackLoop(
  playback: AttachVitrumSceneControllerPlayback | undefined,
): boolean | undefined {
  return playback != null && typeof playback === 'object' ? playback.loop : undefined;
}

export interface VitrumCanvasProps {
  /** Scene description in the host-agnostic @vitrum/core contract. Required
   *  unless `gltf` is supplied. */
  scene?: Scene;
  /** glTF/GLB input accepted by `@vitrum/gltf-adapter`. When supplied,
   *  VitrumCanvas loads it on mount / prop identity change, passes the imported
   *  scene to attachVitrum, and forwards the asset recommendation to
   *  createEngine so `prefer:"auto"` can skip the progressive viewer when
   *  the planner named WebGL2, and otherwise use it as the single-engine
   *  fallback. */
  gltf?: GltfAssetInput;
  /** Adapter options for `gltf`: buffers, decoder hooks, texture decode hooks,
   *  baseUri, fetch, compatibility policy inputs, etc. Identity changes recreate
   *  the engine, matching `scene` creation-time semantics. */
  gltfOptions?: VitrumCanvasGltfOptions;
  /** Use the progressive walkaround→pt-webgpu facade for `gltf` instead of a
   *  single backend selected by `loadGltfWithEngine`. */
  gltfProgressive?: boolean;
  /** Progressive-engine options for `gltfProgressive`; `canvas`, `scene`, and
   *  `controller` are owned by VitrumCanvas / the glTF loader. */
  gltfProgressiveOptions?: VitrumCanvasProgressiveGltfOptions;
  /** Opt-in RAF-driven playback for glTF animations loaded through `gltf`. */
  gltfPlayback?: AttachVitrumSceneControllerPlayback;
  /** Called after a glTF asset loads successfully and before attachVitrum
   *  resolves. Callback updates do not recreate the engine. */
  onGltfLoaded?: (asset: GltfAssetResult, result: VitrumCanvasGltfResult) => void;
  /** Camera the engine reads each frame. Host mutates this (orbit
   *  controls, scripted animation); the canvas pushes its matrices into
   *  renderFrame on every RAF tick. Omit on the glTF path to use
   *  `asset.cameras[0]` / `controller.cameras[0]` after clip advance. */
  camera?: AttachVitrumCamera;
  /** Quality vs speed hint. See CreateEngineOptions.prefer. */
  prefer?: EnginePreference;
  /** Per-frame quality dials. Prop changes propagate live: the latest
   *  value is read by `attachVitrum`'s rAF tick every frame via a getter
   *  closure over a mutable ref, so updates never trigger engine teardown
   *  + recreate. */
  quality?: NonNullable<FrameInput['quality']>;
  /** Frame-level telemetry (forwards engine.onFrame). */
  onFrame?: (stats: FrameStats) => void;
  /** Progress telemetry (forwards engine.onProgress; PT engines only). */
  onProgress?: (progress: ProgressStats) => void;
  /** Pause RAF loop on tab-hidden (default true). */
  pauseOnHidden?: boolean;
  /** Backend-specific createEngine overrides. */
  advanced?: CreateEngineOptions['advanced'];
  /** Explicit backend target for legacy `advanced` overrides. */
  advancedBackend?: CreateEngineOptions['advancedBackend'];
  /** Backend-keyed createEngine overrides for deterministic auto/fallback paths. */
  advancedByBackend?: CreateEngineOptions['advancedByBackend'];
  /** Named research niche. Forwarded to createEngine / attachVitrum. */
  experimentalPreset?: CreateEngineOptions['experimentalPreset'];
  /** Enable backend debug surfaces. */
  debug?: boolean;
  /** Called for nonfatal create/attach/backend capability warnings. */
  onWarning?: (warning: EngineWarning) => void;
  /** Called with the probed WebGPU adapter profile before walkaround device creation. */
  onAdapterProfile?: (profile: AdapterProfile) => void;
  /** Called for recoverable create/attach/runtime engine errors. */
  onError?: (error: unknown, event: CreateEngineErrorEvent) => void;
  /** GPU/runtime engine errors forwarded from the engine's onError subscription
   *  (device-lost, validation errors, WebGL context-lost).  See
   *  {@link AttachVitrumOptions.onEngineError}. */
  onEngineError?: (error: EngineError) => void;
  /** Called when attachVitrum fails. */
  onAttachError?: (error: unknown) => void;
  /**
   * Automatically recreate the engine after a fatal device-lost or context-lost
   * error.  Forwarded to {@link attachVitrum}'s `autoRecreateOnDeviceLoss` option.
   * See that option's documentation for the full recreate sequence, GI-state
   * preservation, and retry-cap behaviour.  Default: `false`.
   */
  autoRecreateOnDeviceLoss?: boolean;
  /** Called when fatal-loss recovery installs a different backend or profile. */
  onBackendChanged?: (event: AttachVitrumBackendChangedEvent) => void;
  /** Forwarded to the underlying canvas element. */
  style?: React.CSSProperties;
  /** Forwarded to the underlying canvas element. */
  className?: string;
}

export const VitrumCanvas = React.forwardRef<HTMLCanvasElement, VitrumCanvasProps>(
  function VitrumCanvas(props, externalRef) {
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const handleRef = React.useRef<AttachVitrumHandle | null>(null);
    // Structural prop changes can arrive while async glTF decode / engine
    // construction is still in flight. Serialize effect generations so one
    // canvas is never concurrently owned by two attachVitrum calls.
    const attachSerialRef = React.useRef<Promise<void>>(Promise.resolve());
    const qualityRef = React.useRef<VitrumCanvasProps['quality']>(props.quality);
    const onFrameRef = React.useRef<VitrumCanvasProps['onFrame']>(props.onFrame);
    const onProgressRef = React.useRef<VitrumCanvasProps['onProgress']>(props.onProgress);
    const onWarningRef = React.useRef<VitrumCanvasProps['onWarning']>(props.onWarning);
    const onAdapterProfileRef = React.useRef<VitrumCanvasProps['onAdapterProfile']>(
      props.onAdapterProfile,
    );
    const onErrorRef = React.useRef<VitrumCanvasProps['onError']>(props.onError);
    const onEngineErrorRef = React.useRef<VitrumCanvasProps['onEngineError']>(props.onEngineError);
    const onBackendChangedRef = React.useRef<VitrumCanvasProps['onBackendChanged']>(
      props.onBackendChanged,
    );
    const onGltfLoadedRef = React.useRef<VitrumCanvasProps['onGltfLoaded']>(props.onGltfLoaded);
    // H31 — ref-stabilize `onAttachError` so inline callbacks do not cause full
    // engine teardown+recreate on every parent render. `advanced` is a creation-time
    // backend option, so identity changes intentionally recreate the engine.
    const onAttachErrorRef = React.useRef<VitrumCanvasProps['onAttachError']>(props.onAttachError);

    // Keep the latest dynamic props in refs so the engine sees them
    // without having to be torn down + recreated when they change.
    React.useEffect(() => {
      qualityRef.current = props.quality;
    }, [props.quality]);
    React.useEffect(() => {
      onFrameRef.current = props.onFrame;
    }, [props.onFrame]);
    React.useEffect(() => {
      onProgressRef.current = props.onProgress;
    }, [props.onProgress]);
    React.useEffect(() => {
      onWarningRef.current = props.onWarning;
    }, [props.onWarning]);
    React.useEffect(() => {
      onAdapterProfileRef.current = props.onAdapterProfile;
    }, [props.onAdapterProfile]);
    React.useEffect(() => {
      onErrorRef.current = props.onError;
    }, [props.onError]);
    React.useEffect(() => {
      onEngineErrorRef.current = props.onEngineError;
    }, [props.onEngineError]);
    React.useEffect(() => {
      onBackendChangedRef.current = props.onBackendChanged;
    }, [props.onBackendChanged]);
    React.useEffect(() => {
      onGltfLoadedRef.current = props.onGltfLoaded;
    }, [props.onGltfLoaded]);
    React.useEffect(() => {
      onAttachErrorRef.current = props.onAttachError;
    }, [props.onAttachError]);

    // Effect dependency captures the structural inputs only. Mutable refs
    // bypass the dep array for dynamic props.
    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      let cancelled = false;
      let attached: AttachVitrumHandle | null = null;
      let pendingGltfResult: VitrumCanvasGltfResult | undefined;
      const abortController = typeof AbortController !== 'undefined' ? new AbortController() : null;

      const attach = async (): Promise<void> => {
        if (cancelled) return;
        const gltfSignal = composeAbortSignal(props.gltfOptions?.signal, abortController?.signal);
        let gltfResult: VitrumCanvasGltfResult | undefined;
        if (props.gltf !== undefined) {
          if (props.gltfProgressive === true) {
            const progressiveOptions = props.gltfProgressiveOptions ?? {};
            const playbackEnabled = vitrumCanvasPlaybackEnabled(props.gltfPlayback);
            const playbackLoop = vitrumCanvasPlaybackLoop(props.gltfPlayback);
            gltfResult = await loadGltfWithProgressiveEngine(props.gltf, {
              ...(props.gltfOptions ?? {}),
              ...(gltfSignal != null ? { signal: gltfSignal } : {}),
              engineOptions: {
                canvas,
                ...(playbackEnabled ? {} : { controllerDeltaSeconds: 0 }),
                ...(playbackLoop !== undefined ? { controllerLoop: playbackLoop } : {}),
                ...progressiveOptions,
                ...(props.debug != null ? { debug: props.debug } : {}),
                onAdapterProfile: (profile) => {
                  try {
                    onAdapterProfileRef.current?.(profile);
                  } catch {
                    /* host callback must not propagate — ignore */
                  }
                },
                onError: (error, event) => {
                  onErrorRef.current?.(error, event);
                },
                onWarning: (warning) => {
                  try {
                    onWarningRef.current?.(warning);
                  } catch {
                    /* host callback must not propagate — ignore */
                  }
                },
              },
            });
          } else {
            gltfResult = await loadGltfWithEngine(props.gltf, {
              ...(props.gltfOptions ?? {}),
              ...(gltfSignal != null ? { signal: gltfSignal } : {}),
              attachScene: false,
              engineOptions: {
                canvas,
                ...(props.prefer ? { prefer: props.prefer } : {}),
                ...(props.advanced != null ? { advanced: props.advanced } : {}),
                ...(props.advancedBackend != null
                  ? { advancedBackend: props.advancedBackend }
                  : {}),
                ...(props.advancedByBackend != null
                  ? { advancedByBackend: props.advancedByBackend }
                  : {}),
                ...(props.experimentalPreset != null
                  ? { experimentalPreset: props.experimentalPreset }
                  : {}),
                ...(props.debug != null ? { debug: props.debug } : {}),
                onWarning: (warning) => {
                  try {
                    onWarningRef.current?.(warning);
                  } catch {
                    /* host callback must not propagate — ignore */
                  }
                },
                onAdapterProfile: (profile) => {
                  try {
                    onAdapterProfileRef.current?.(profile);
                  } catch {
                    /* host callback must not propagate — ignore */
                  }
                },
                onError: (error, event) => {
                  onErrorRef.current?.(error, event);
                },
              },
            });
          }
        }
        pendingGltfResult = gltfResult;
        if (cancelled) {
          disposePendingGltfEngine(pendingGltfResult);
          pendingGltfResult = undefined;
          return;
        }
        if (gltfResult !== undefined) {
          if (gltfResult.engine == null) {
            throw new Error('[VitrumCanvas] glTF loader did not return an engine.');
          }
          try {
            onGltfLoadedRef.current?.(gltfResult.asset, gltfResult);
          } catch {
            /* host callback must not propagate — ignore */
          }
        }
        const scene = gltfResult?.asset.scene ?? props.scene;
        const gltfEngine = isProgressiveGltfResult(gltfResult)
          ? progressiveHandleAsEngine(gltfResult.engine)
          : gltfResult?.engine;
        const progressiveRecreateEngine: AttachVitrumRecreateEngineFactory | undefined =
          isProgressiveGltfResult(gltfResult)
            ? async ({ scene: recreateScene }) => {
                const progressiveOptions = props.gltfProgressiveOptions ?? {};
                const playbackEnabled = vitrumCanvasPlaybackEnabled(props.gltfPlayback);
                const playbackLoop = vitrumCanvasPlaybackLoop(props.gltfPlayback);
                const nextHandle = await createProgressiveEngine({
                  canvas,
                  scene: recreateScene,
                  controller: gltfResult.controller,
                  ...(playbackEnabled ? {} : { controllerDeltaSeconds: 0 }),
                  ...(playbackLoop !== undefined ? { controllerLoop: playbackLoop } : {}),
                  ...progressiveOptions,
                  ...(props.debug != null ? { debug: props.debug } : {}),
                  onAdapterProfile: (profile) => {
                    try {
                      onAdapterProfileRef.current?.(profile);
                    } catch {
                      /* host callback must not propagate — ignore */
                    }
                  },
                  onError: (error, event) => {
                    onErrorRef.current?.(error, event);
                  },
                  onWarning: (warning) => {
                    try {
                      onWarningRef.current?.(warning);
                    } catch {
                      /* host callback must not propagate — ignore */
                    }
                  },
                });
                gltfResult.controller.attachEngine(nextHandle.coordinator, { setScene: false });
                return progressiveHandleAsEngine(nextHandle);
              }
            : undefined;
        if (scene == null) {
          throw new TypeError('[VitrumCanvas] either `scene` or `gltf` must be supplied.');
        }
        if (props.gltf === undefined && props.camera == null) {
          throw new TypeError('[VitrumCanvas] `camera` is required when `gltf` is not supplied.');
        }
        const h = await attachVitrum({
          canvas,
          scene,
          ...(gltfResult !== undefined
            ? { gltfAsset: createEngineGltfAssetHint(gltfResult.asset) }
            : {}),
          ...(gltfEngine !== undefined ? { engine: gltfEngine } : {}),
          ...(progressiveRecreateEngine !== undefined
            ? { recreateEngine: progressiveRecreateEngine }
            : {}),
          ...(gltfResult?.controller !== undefined && !isProgressiveGltfResult(gltfResult)
            ? {
                sceneController: gltfResult.controller,
                sceneControllerPlayback: props.gltfPlayback ?? true,
              }
            : {}),
          ...(props.camera != null ? { camera: props.camera } : {}),
          ...(props.prefer ? { prefer: props.prefer } : {}),
          ...(props.pauseOnHidden != null ? { pauseOnHidden: props.pauseOnHidden } : {}),
          ...(props.advanced != null ? { advanced: props.advanced } : {}),
          ...(props.advancedBackend != null ? { advancedBackend: props.advancedBackend } : {}),
          ...(props.advancedByBackend != null
            ? { advancedByBackend: props.advancedByBackend }
            : {}),
          ...(props.experimentalPreset != null
            ? { experimentalPreset: props.experimentalPreset }
            : {}),
          ...(props.debug != null ? { debug: props.debug } : {}),
          ...(props.autoRecreateOnDeviceLoss != null
            ? { autoRecreateOnDeviceLoss: props.autoRecreateOnDeviceLoss }
            : {}),
          // Live-propagation: pass a getter so the rAF tick reads the latest
          // `props.quality` each frame (qualityRef.current is updated by the
          // useEffect above whenever props.quality changes).
          quality: () => qualityRef.current,
          onFrame: (stats) => {
            onFrameRef.current?.(stats);
          },
          onProgress: (progress) => {
            onProgressRef.current?.(progress);
          },
          onWarning: (warning) => {
            try {
              onWarningRef.current?.(warning);
            } catch {
              /* host callback must not propagate — ignore */
            }
          },
          onAdapterProfile: (profile) => {
            try {
              onAdapterProfileRef.current?.(profile);
            } catch {
              /* host callback must not propagate — ignore */
            }
          },
          onError: (error, event) => {
            onErrorRef.current?.(error, event);
          },
          onEngineError: (err) => {
            try {
              onEngineErrorRef.current?.(err);
            } catch {
              /* host callback must not propagate — ignore */
            }
          },
          onBackendChanged: (event) => {
            try {
              onBackendChangedRef.current?.(event);
            } catch {
              /* host callback must not propagate — ignore */
            }
          },
        });
        if (cancelled) {
          h.dispose();
          releasePendingGltfResources(pendingGltfResult);
          pendingGltfResult = undefined;
          return;
        }
        handleRef.current = h;
        attached = h;
      };

      const attachCycle = attachSerialRef.current
        .catch(() => undefined)
        .then(attach);
      attachSerialRef.current = attachCycle.catch(() => undefined);

      void attachCycle.catch((err) => {
        disposePendingGltfEngine(pendingGltfResult);
        pendingGltfResult = undefined;
        if (cancelled) return;
        // H31 — call via ref so inline onAttachError props don't appear in
        // the effect dep array (which would cause teardown per parent render).
        try {
          onAttachErrorRef.current?.(err);
        } catch {
          /* host callback must not propagate — ignore */
        }
        try {
          onErrorRef.current?.(err, { phase: 'attach:initial', recoverable: false });
        } catch {
          /* host callback must not propagate — ignore */
        }
        console.error('[VitrumCanvas] attachVitrum failed:', err);
      });

      return () => {
        cancelled = true;
        try {
          abortController?.abort();
        } catch {
          /* best-effort cancel — ignore */
        }
        try {
          attached?.dispose();
        } catch {
          // Unmount must still release imported resources when lifecycle disposal fails.
        } finally {
          if (attached !== null) {
            releasePendingGltfResources(pendingGltfResult);
            pendingGltfResult = undefined;
          }
          handleRef.current = null;
        }
      };
      // H31 — `onAttachError` is intentionally NOT in the dep array; it is accessed
      // via a stable ref to prevent inline callbacks from tearing down the engine.
      // `advanced` stays in the dep array because backend options are construction
      // inputs and prop identity changes must apply.
    }, [
      props.scene,
      props.gltf,
      props.gltfOptions,
      props.gltfProgressive,
      props.gltfProgressiveOptions,
      props.gltfPlayback,
      props.camera,
      props.prefer,
      props.pauseOnHidden,
      props.advanced,
      props.advancedBackend,
      props.advancedByBackend,
      props.experimentalPreset,
      props.debug,
      props.autoRecreateOnDeviceLoss,
    ]);

    // forwardRef plumbing.
    React.useImperativeHandle(externalRef, () => canvasRef.current as HTMLCanvasElement, []);

    return (
      <canvas
        ref={canvasRef}
        style={props.style}
        {...(props.className ? { className: props.className } : {})}
      />
    );
  },
);

function composeAbortSignal(
  external: AbortSignal | undefined,
  internal: AbortSignal | undefined,
): AbortSignal | undefined {
  if (external != null && internal != null) {
    const abortSignalCtor = globalThis.AbortSignal as
      | (typeof AbortSignal & { any?: (signals: AbortSignal[]) => AbortSignal })
      | undefined;
    const nativeAny = abortSignalCtor?.any;
    if (nativeAny != null) return nativeAny([external, internal]);

    const AbortControllerCtor = globalThis.AbortController;
    if (AbortControllerCtor == null) return external;
    const composite = new AbortControllerCtor();
    const abort = (): void => composite.abort();
    if (external.aborted || internal.aborted) {
      abort();
    } else {
      external.addEventListener('abort', abort, { once: true });
      internal.addEventListener('abort', abort, { once: true });
    }
    return composite.signal;
  }
  return external ?? internal;
}
