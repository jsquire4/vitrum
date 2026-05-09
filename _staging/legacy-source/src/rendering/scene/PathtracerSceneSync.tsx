import { useEffect, useRef } from 'react';
import { useSelector } from 'react-redux';
import { useThree } from '@react-three/fiber';
import { usePathtracer } from '@react-three/gpu-pathtracer';
import { selectActiveTimeOfDay, selectBackdropMode, selectCameraMode, selectEdgeProperties, selectFaceProperties, selectLightsById, selectMount, selectSpace, selectGraph } from '@/store/selectors';
import { debounceMsForEditRate } from './ptDebounce';

/**
 * Bridge between Redux state changes and the path tracer's internal
 * scene/material/light tables. Without this component, mutations
 * inside React (face added, glass type changed, light added) re-render
 * the React-Three subtree but the path tracer keeps using its
 * first-mount snapshot — the canvas appears to "lag behind" until the
 * user toggles PT off and on.
 *
 * The @react-three/gpu-pathtracer wrapper calls setScene only on
 * scene/camera reference changes, which never fire for normal Redux
 * mutations. This component closes that gap by watching Redux state
 * directly and calling pathtracer.setScene() with a 50ms debounce so
 * rapid edits coalesce into one BVH rebuild instead of N.
 *
 * Mount inside the <Pathtracer> subtree (PathTracingLayer) so
 * usePathtracer() resolves to the same WebGLPathTracer instance the
 * wrapper instantiated. ptActive gating happens at the parent — when
 * the wrapper isn't mounted, this component isn't either.
 */
export function PathtracerSceneSync() {
  const { scene, camera } = useThree();
  const { pathtracer } = usePathtracer();

  // Every Redux signal that should trigger a PT scene refresh:
  //   graph             — vertex / edge / face topology
  //   faceProperties    — glass type / colour / texture / iridescent /
  //                       thickness; baked into a fresh material at
  //                       React render time but BVH still needs the new
  //                       material instance.
  //   edgeProperties    — came / foil width / patina / cameType — drives
  //                       H-channel rail dimensions in EdgeLines.
  //   lighting.byId     — every LightSource toggle / add / remove / edit
  //                       (S4 architecture pillar 2). Subscribing to byId
  //                       is enough — allIds tracks lockstep.
  //   scene.mount       — windowInWall ↔ lightbox ↔ lamp ↔ freestanding
  //                       swaps the entire mount subtree (geometry change).
  //   scene.space       — void ↔ room toggles enclosure geometry.
  //   cameraMode        — 2D/3D toggle.
  //   timeOfDay         — sun position + intensity track this; PT must
  //                       resample on change.
  //   backdropMode      — toggles env IBL between procedural sky bake and
  //                       drei <Environment> HDR; PT picks up the change
  //                       via usePTEnvironment but BVH still needs setScene.
  const graph = useSelector(selectGraph);
  const faceProps = useSelector(selectFaceProperties);
  const edgeProps = useSelector(selectEdgeProperties);
  const lightingById = useSelector(selectLightsById);
  const mount = useSelector(selectMount);
  const space = useSelector(selectSpace);
  const cameraMode = useSelector(selectCameraMode);
  // Sweep finding (correctness 2026-05-08 Bug 18): active selector so
  // PT scene-sync re-fires when room-mode timeOfDay changes.
  const timeOfDay = useSelector(selectActiveTimeOfDay);
  const backdropMode = useSelector(selectBackdropMode);

  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Slice 3.4 — ring buffer of recent mutation timestamps (DOMHighResTimeStamp).
  // debounceMsForEditRate stretches the debounce window from 50ms to 250ms when
  // SceneControl edit rate exceeds the threshold, reducing the device-hang risk
  // during rapid slider scrubs.
  const editTimestampsRef = useRef<number[]>([]);

  useEffect(() => {
    if (!pathtracer) return;
    if (timeoutRef.current) clearTimeout(timeoutRef.current);

    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    editTimestampsRef.current.push(now);
    // Cap ring buffer length so it doesn't grow unbounded over a session.
    if (editTimestampsRef.current.length > 16) editTimestampsRef.current.shift();
    const debounceMs = debounceMsForEditRate(editTimestampsRef.current, now);

    timeoutRef.current = setTimeout(() => {
      pathtracer.setScene(scene, camera);
      timeoutRef.current = null;
    }, debounceMs);
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [
    pathtracer, scene, camera,
    graph, faceProps, edgeProps,
    lightingById, mount, space,
    cameraMode, timeOfDay, backdropMode,
  ]);

  return null;
}
