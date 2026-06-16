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
import type { EngineError, Scene, FrameInput, FrameStats, ProgressStats } from '@vitrum/core';
import {
  type GltfAssetInput,
  type GltfAssetResult,
  type GltfForEngineResult,
} from '@vitrum/gltf-adapter';
import type {
  AttachVitrumHandle,
  AttachVitrumSceneControllerPlayback,
  CameraLike,
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
  type LoadGltfWithEngineOptions,
} from '../gltf.js';

export type VitrumCanvasGltfOptions = Omit<
  LoadGltfWithEngineOptions,
  'engine' | 'engineOptions' | 'attachScene'
>;

export interface VitrumCanvasProps {
  /** Scene description in the host-agnostic @vitrum/core contract. Required
   *  unless `gltf` is supplied. */
  scene?: Scene;
  /** glTF/GLB input accepted by `@vitrum/gltf-adapter`. When supplied,
   *  VitrumCanvas loads it on mount / prop identity change, passes the imported
   *  scene to attachVitrum, and forwards the asset recommendation to
   *  createEngine so `prefer:"auto"` is feature-aware instead of triangle-only. */
  gltf?: GltfAssetInput;
  /** Adapter options for `gltf`: buffers, decoder hooks, texture decode hooks,
   *  baseUri, fetch, compatibility policy inputs, etc. Identity changes recreate
   *  the engine, matching `scene` creation-time semantics. */
  gltfOptions?: VitrumCanvasGltfOptions;
  /** Opt-in RAF-driven playback for glTF animations loaded through `gltf`. */
  gltfPlayback?: AttachVitrumSceneControllerPlayback;
  /** Called after a glTF asset loads successfully and before attachVitrum
   *  resolves. Callback updates do not recreate the engine. */
  onGltfLoaded?: (
    asset: GltfAssetResult,
    result: GltfForEngineResult<EngineWithBackendId>,
  ) => void;
  /** Camera the engine reads each frame. Host mutates this (orbit
   *  controls, scripted animation); the canvas pushes its matrices into
   *  renderFrame on every RAF tick. */
  camera: CameraLike;
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
  /** Enable backend debug surfaces. */
  debug?: boolean;
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
  /** Forwarded to the underlying canvas element. */
  style?: React.CSSProperties;
  /** Forwarded to the underlying canvas element. */
  className?: string;
}

export const VitrumCanvas = React.forwardRef<HTMLCanvasElement, VitrumCanvasProps>(
  function VitrumCanvas(props, externalRef) {
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const handleRef = React.useRef<AttachVitrumHandle | null>(null);
    const qualityRef = React.useRef<VitrumCanvasProps['quality']>(props.quality);
    const onFrameRef = React.useRef<VitrumCanvasProps['onFrame']>(props.onFrame);
    const onProgressRef = React.useRef<VitrumCanvasProps['onProgress']>(props.onProgress);
    const onErrorRef = React.useRef<VitrumCanvasProps['onError']>(props.onError);
    const onEngineErrorRef = React.useRef<VitrumCanvasProps['onEngineError']>(props.onEngineError);
    const onGltfLoadedRef = React.useRef<VitrumCanvasProps['onGltfLoaded']>(props.onGltfLoaded);
    // H31 — ref-stabilize `onAttachError` so inline callbacks do not cause full
    // engine teardown+recreate on every parent render. `advanced` is a creation-time
    // backend option, so identity changes intentionally recreate the engine.
    const onAttachErrorRef = React.useRef<VitrumCanvasProps['onAttachError']>(props.onAttachError);

    // Keep the latest dynamic props in refs so the engine sees them
    // without having to be torn down + recreated when they change.
    React.useEffect(() => { qualityRef.current = props.quality; },         [props.quality]);
    React.useEffect(() => { onFrameRef.current = props.onFrame; },         [props.onFrame]);
    React.useEffect(() => { onProgressRef.current = props.onProgress; },   [props.onProgress]);
    React.useEffect(() => { onErrorRef.current = props.onError; },         [props.onError]);
    React.useEffect(() => { onEngineErrorRef.current = props.onEngineError; }, [props.onEngineError]);
    React.useEffect(() => { onGltfLoadedRef.current = props.onGltfLoaded; }, [props.onGltfLoaded]);
    React.useEffect(() => { onAttachErrorRef.current = props.onAttachError; }, [props.onAttachError]);

    // Effect dependency captures the structural inputs only. Mutable refs
    // bypass the dep array for dynamic props.
    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      let cancelled = false;
      let attached: AttachVitrumHandle | null = null;
      const abortController = typeof AbortController !== 'undefined'
        ? new AbortController()
        : null;

      const attach = async (): Promise<void> => {
        const gltfSignal = composeAbortSignal(props.gltfOptions?.signal, abortController?.signal);
        const gltfResult = props.gltf !== undefined
          ? await loadGltfWithEngine(props.gltf, {
              ...(props.gltfOptions ?? {}),
              ...(gltfSignal != null ? { signal: gltfSignal } : {}),
              attachScene: false,
              engineOptions: {
                canvas,
                ...(props.prefer ? { prefer: props.prefer } : {}),
                ...(props.advanced != null ? { advanced: props.advanced } : {}),
                ...(props.debug != null ? { debug: props.debug } : {}),
                onError: (error, event) => { onErrorRef.current?.(error, event); },
              },
            })
          : undefined;
        if (cancelled) return;
        if (gltfResult !== undefined) {
          if (gltfResult.engine == null) {
            throw new Error('[VitrumCanvas] loadGltfWithEngine did not return an engine.');
          }
          try {
            onGltfLoadedRef.current?.(gltfResult.asset, gltfResult);
          } catch { /* host callback must not propagate — ignore */ }
        }
        const scene = gltfResult?.asset.scene ?? props.scene;
        if (scene == null) {
          throw new TypeError('[VitrumCanvas] either `scene` or `gltf` must be supplied.');
        }
        const h = await attachVitrum({
          canvas,
          scene,
          ...(gltfResult !== undefined ? { gltfAsset: gltfResult.asset } : {}),
          ...(gltfResult?.engine !== undefined ? { engine: gltfResult.engine } : {}),
          ...(gltfResult?.controller !== undefined ? { sceneController: gltfResult.controller } : {}),
          ...(gltfResult?.controller !== undefined && props.gltfPlayback !== undefined
            ? { sceneControllerPlayback: props.gltfPlayback }
            : {}),
          camera: props.camera,
          ...(props.prefer ? { prefer: props.prefer } : {}),
          ...(props.pauseOnHidden != null ? { pauseOnHidden: props.pauseOnHidden } : {}),
          ...(props.advanced != null ? { advanced: props.advanced } : {}),
          ...(props.debug != null ? { debug: props.debug } : {}),
          ...(props.autoRecreateOnDeviceLoss != null
            ? { autoRecreateOnDeviceLoss: props.autoRecreateOnDeviceLoss }
            : {}),
          // Live-propagation: pass a getter so the rAF tick reads the latest
          // `props.quality` each frame (qualityRef.current is updated by the
          // useEffect above whenever props.quality changes).
          quality: () => qualityRef.current,
          onFrame: (stats) => { onFrameRef.current?.(stats); },
          onProgress: (progress) => { onProgressRef.current?.(progress); },
          onError: (error, event) => { onErrorRef.current?.(error, event); },
          onEngineError: (err) => { try { onEngineErrorRef.current?.(err); } catch { /* host callback must not propagate — ignore */ } },
        });
        if (cancelled) {
          h.dispose();
          return;
        }
        handleRef.current = h;
        attached = h;
      };

      void attach()
        .catch((err) => {
          if (cancelled) return;
          // H31 — call via ref so inline onAttachError props don't appear in
          // the effect dep array (which would cause teardown per parent render).
          onAttachErrorRef.current?.(err);
          console.error('[VitrumCanvas] attachVitrum failed:', err);
        });

      return () => {
        cancelled = true;
        try { abortController?.abort(); } catch { /* best-effort cancel — ignore */ }
        attached?.dispose();
        handleRef.current = null;
      };
    // H31 — `onAttachError` is intentionally NOT in the dep array; it is accessed
    // via a stable ref to prevent inline callbacks from tearing down the engine.
    // `advanced` stays in the dep array because backend options are construction
    // inputs and prop identity changes must apply.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
      props.scene,
      props.gltf,
      props.gltfOptions,
      props.gltfPlayback,
      props.camera,
      props.prefer,
      props.pauseOnHidden,
      props.advanced,
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
    return abortSignalCtor?.any?.([external, internal]) ?? external;
  }
  return external ?? internal;
}
