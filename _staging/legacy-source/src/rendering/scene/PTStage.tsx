import { useEffect, useRef } from 'react';
import { Environment } from '@react-three/drei';
import { useThree } from '@react-three/fiber';
import { useSelector } from 'react-redux';
import { lightboxDimsFor } from './mounts/lightboxDims';
import { LightSourceList } from './lighting/renderLightSource';
import { SunPathTraced } from './lighting/renderers/sunPathTraced';
import type { SunLight } from './lighting/lightSourceTypes';
import { usePTEnvironment } from './ptEnvironment';
import { useStageCommonState } from './useStageCommonState';
import { StageMountTree } from './StageMountTree';
import type { SkyParams } from './skyParams';
import type { BackdropMode } from '@/store/uiSlice';
import { selectActiveTimeOfDay, selectLightAllIds, selectLightsById, selectOutdoorScenePreset } from '@/store/selectors';
import { outdoorHdriForPreset } from './outdoorScenePresets';
import { PTPostProcessing } from './PTPostProcessing';

/**
 * Path-traced 3D stage — sibling of <RasterStage>. Mounts only when
 * `pathTracerEnabled && cameraMode === '3d'`. The component is the
 * boundary that guarantees no PT-incompatible JSX (Sky shader,
 * GizmoHelper internals) ever enters the path tracer's setScene walk.
 *
 * Render-mode invariant (S4-T20): PT is the verifier, not the editor.
 * Edits trigger 50ms-debounced setScene rebuilds (PathtracerSceneSync);
 * each rebuild + sample-converge cycle costs 60–120s. Real-time editing
 * during PT is impossible by physics — users toggle PT off to edit.
 *
 * Backdrop strategy:
 *   sky / night → bake procedural sky into scene.environment via
 *     usePTEnvironment + ptIblBaker (the procedural Sky shader can't
 *     be sampled by PT directly).
 *   studio / sunset → drei <Environment preset> loads an HDR
 *     asynchronously; usePTEnvironment's frame-watcher catches the
 *     load and calls pathtracer.updateEnvironment() once.
 *   none → null env / null bg, intensity 0.
 */

interface PTStageProps {
  backdropMode: BackdropMode;
  skyParams: SkyParams;
  nightSkyParams: SkyParams;
  frameLayout: { cx: number; cy: number; w: number; h: number };
}

export function PTStage({
  backdropMode,
  skyParams,
  nightSkyParams,
  frameLayout,
}: PTStageProps) {
  // Slice 5.1 — common selectors + room bounds derivation pulled into
  // a shared hook so RasterStage + PTStage agree on the contract.
  // Slice 5.3 — OrbitControls lifted to StudioScene; this stage no
  // longer needs roomOrbitRadius / roomBoundsMax.
  const { mount, space, isRoomMode, roomKey, suppressSun } = useStageCommonState(backdropMode);
  // Sweep finding (correctness 2026-05-08 Bug 18): active selector
  // tracks roomDoc.scene.timeOfDay when editing a .sroom.
  const timeOfDay = useSelector(selectActiveTimeOfDay);
  const outdoorScenePreset = useSelector(selectOutdoorScenePreset);
  const camera = useThree(state => state.camera);

  // Active sun light source (if any). PT replaces THREE.DirectionalLight
  // with ShapedAreaLight (disc) so NEE samples it as a real area emitter
  // → physical penumbra everywhere instead of knife-edge shadows. The
  // raster path keeps the directional sun via SunRenderer; in PT mode we
  // suppress the directional and mount <SunPathTraced /> in its place.
  const lightAllIds = useSelector(selectLightAllIds);
  const lightsById = useSelector(selectLightsById);
  const activeSun: SunLight | null = (() => {
    for (const id of lightAllIds) {
      const l = lightsById[id];
      if (l?.kind === 'sun' && l.on) return l;
    }
    return null;
  })();
  // suppressSun semantics: studio/sunset HDRIs carry a baked sun lobe so
  // the directional sun would double-count. The same gate applies to
  // the PT area-light sun. Phase 1.2: when an outdoor HDRI is loaded
  // for room mode, that HDRI ALSO carries a sun (Polyhaven HDRIs come
  // with their sun-disc baked into the equirect map). Adding the
  // ShapedAreaLight on top would produce two suns — exactly the
  // "sunlight from above the roof" symptom that originally prompted
  // the outdoor-HDRI disable in 2026-05-08. The fix: suppress the
  // area-light sun whenever an outdoor HDRI is active.
  const hasOutdoorHdri = space.kind === 'room' && outdoorHdriForPreset(outdoorScenePreset, timeOfDay) !== undefined;
  const ptSunSuppressed = suppressSun || hasOutdoorHdri;

  // Auto-frame the camera ONCE per mount when PT activates inside a ROOM
  // scene. The default editor camera (panel-design pose) sits at z≈36
  // looking at the panel at z≈0; with the noon sun at 22° elevation
  // (skyParams Y×0.4) the sun-through-panel caustic projects onto the
  // floor at z≈47–140 — entirely BEHIND the camera, so the user sees a
  // featureless wall-shadow region and assumes caustics are broken.
  // Mirror RestirStage's room pose (z=160 looking at z=0 from camera y=0,
  // target y=-30) so PT and walkaround share an identical default framing
  // of the through-glass projection. framedRef keeps subsequent
  // OrbitControls drags authoritative.
  //
  // Gate on space.kind === 'room' (NOT activeDocument === 'room' aka
  // isRoomMode). The user can be editing the panel document while
  // inhabiting a room scene; we want the framing to fire whenever the
  // scene we're rendering IS a room.
  const isRoomSpace = space.kind === 'room';
  const framedRef = useRef(false);
  useEffect(() => {
    if (framedRef.current) return;
    if (!isRoomSpace) return;
    camera.position.set(frameLayout.cx, 0, 160);
    camera.lookAt(frameLayout.cx, -30, 0);
    camera.updateMatrixWorld();
    framedRef.current = true;
  }, [camera, isRoomSpace, frameLayout.cx]);

  // PT-tuned IBL: bakes sky/night procedurally, observes drei's
  // async HDR for studio/sunset, sets scene.environmentIntensity per
  // backdrop, and calls pathtracer.updateEnvironment() on every
  // change. ptActive is implicitly true here — PTStage only mounts
  // when PT is enabled. When space.kind === 'room' and an outdoor HDRI
  // URL is defined for the current timeOfDay bucket, that takes priority
  // over the procedural sky path (auto-activates once LFS assets ship).
  usePTEnvironment(backdropMode, skyParams, nightSkyParams, true, space, timeOfDay, outdoorScenePreset);
  const lightboxDims = lightboxDimsFor(mount, frameLayout.w, frameLayout.h);

  return (
    <>
      {/* Studio/sunset HDR via drei <Environment> is PT-safe (it sets
          scene.environment / scene.background as textures, no in-scene
          mesh). Sky/night use the bakeSkyEquirect → scene.environment
          path inside usePTEnvironment instead of a procedural mesh. */}
      {backdropMode === 'studio' && <Environment preset="studio" background />}
      {backdropMode === 'sunset' && <Environment preset="sunset" background />}

      {/* PT skips ambient/hemisphere fill — env IBL handles diffuse fill.
       *  Always pass suppressSun=true so the directional sun from
       *  SunRenderer doesn't mount under the path tracer (PT uses the
       *  area-light replacement below). */}
      <LightSourceList ctx={{ lightbox: lightboxDims, suppressSun: true }} />

      {/* PT-specific area-light sun (ShapedAreaLight disc). Replaces
       *  the directional sun for path-traced NEE so shadow boundaries
       *  get physical penumbra. Suppressed when the active HDRI carries
       *  its own baked sun (studio/sunset). */}
      {activeSun && !ptSunSuppressed && <SunPathTraced src={activeSun} />}

      {/* Slice 5.2 — shared mount tree component (PT never wants the
       *  raster backboard regardless of backdrop). */}
      <StageMountTree
        mode="pt"
        isRoomMode={isRoomMode}
        roomKey={roomKey}
        frameLayout={frameLayout}
        showBackboard={false}
      />
      {/* CausticsReceiver intentionally not mounted under PT — its
          receiver plane occludes the camera's transmitted view of any
          backlight, and its cookie spotLight double-projects fake
          colour onto a panel that PT computes real caustics for. PT
          resolves caustics natively via BVH. */}

      {/* Slice 5.3 — OrbitControls lifted to StudioScene so the
       *  controls persist across stage swaps. */}

      {/* Note: <GizmoHelper> intentionally absent — its internal helper
          materials (some lacking `.color`) would crash gpu-pathtracer's
          MaterialsTexture.updateFrom. The raster preview retains the
          gizmo via <RasterStage>. */}

      {/* Phase 2 — camera-response post-processing. EffectComposer
       *  runs on the PT-accumulated radiance buffer once per converged
       *  frame. The CameraLook preset (documentary / cinematic /
       *  architectural) drives Bloom + DoF + CA + Vignette + Grain. */}
      <PTPostProcessing />
    </>
  );
}
