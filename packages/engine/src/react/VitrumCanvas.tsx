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
//     no mutate-in-place path). For `quality` changes we just push a
//     fresh FrameInput.quality on the next frame — no engine churn.
//
// `react` is an OPTIONAL peer dep of @vitrum/engine; this file does not
// load unless the host explicitly imports `@vitrum/engine/react`. Non-React
// hosts never pay the cost.

import * as React from 'react';
import type { Scene as ThreeScene, PerspectiveCamera, OrthographicCamera } from 'three';
import type { Scene, FrameInput, FrameStats, ProgressStats } from '@vitrum/core';
import type { AttachVitrumHandle } from '../lifecycle/vanilla.js';
import { attachVitrum } from '../lifecycle/vanilla.js';
import type { EnginePreference, CreateEngineOptions } from '../createEngine.js';

export interface VitrumCanvasProps {
  /** Scene description (vitrum or THREE). */
  scene: Scene | ThreeScene;
  /** Camera the engine reads each frame. Host mutates this (orbit
   *  controls, scripted animation); the canvas pushes its matrices into
   *  renderFrame on every RAF tick. */
  camera: PerspectiveCamera | OrthographicCamera;
  /** Quality vs speed hint. See CreateEngineOptions.prefer. */
  prefer?: EnginePreference;
  /** Per-frame quality dials. Changing this prop does NOT reconstruct
   *  the engine; the new value is read on the next RAF tick. */
  quality?: NonNullable<FrameInput['quality']>;
  /** Frame-level telemetry (forwards engine.onFrame). */
  onFrame?: (stats: FrameStats) => void;
  /** Progress telemetry (forwards engine.onProgress; PT engines only). */
  onProgress?: (progress: ProgressStats) => void;
  /** Pause RAF loop on tab-hidden (default true). */
  pauseOnHidden?: boolean;
  /** Optional backend debug toggle forwarded to createEngine/attachVitrum. */
  debug?: CreateEngineOptions['debug'];
  /** Optional backend-specific advanced options forwarded as-is. */
  advanced?: CreateEngineOptions['advanced'];
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
    const cameraRef = React.useRef<VitrumCanvasProps['camera']>(props.camera);
    const onFrameRef = React.useRef<VitrumCanvasProps['onFrame']>(props.onFrame);
    const onProgressRef = React.useRef<VitrumCanvasProps['onProgress']>(props.onProgress);

    // Keep the latest dynamic props in refs so the engine sees them
    // without having to be torn down + recreated when they change.
    React.useEffect(() => { qualityRef.current = props.quality; },     [props.quality]);
    React.useEffect(() => { cameraRef.current = props.camera; },       [props.camera]);
    React.useEffect(() => { onFrameRef.current = props.onFrame; },     [props.onFrame]);
    React.useEffect(() => { onProgressRef.current = props.onProgress; }, [props.onProgress]);

    // Effect dependency captures the structural inputs only. Mutable refs
    // bypass the dep array for dynamic props.
    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      let cancelled = false;
      let attached: AttachVitrumHandle | null = null;

      attachVitrum({
        canvas,
        scene: props.scene,
        camera: cameraRef.current,
        cameraSource: () => cameraRef.current,
        qualitySource: () => qualityRef.current,
        ...(props.prefer ? { prefer: props.prefer } : {}),
        ...(props.advanced != null ? { advanced: props.advanced } : {}),
        ...(props.debug != null ? { debug: props.debug } : {}),
        ...(props.pauseOnHidden != null ? { pauseOnHidden: props.pauseOnHidden } : {}),
        ...(qualityRef.current ? { quality: qualityRef.current } : {}),
        onFrame: (stats) => { onFrameRef.current?.(stats); },
        onProgress: (progress) => { onProgressRef.current?.(progress); },
      })
        .then((h) => {
          if (cancelled) {
            h.dispose();
            return;
          }
          handleRef.current = h;
          attached = h;
        })
        .catch((err) => {
          console.error('[VitrumCanvas] attachVitrum failed:', err);
        });

      return () => {
        cancelled = true;
        attached?.dispose();
        handleRef.current = null;
      };
    }, [props.scene, props.prefer, props.pauseOnHidden, props.advanced, props.debug]);

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
