/**
 * HybridLayeredStage — app-specific React shell around the layered hybrid
 * path tracer. The GI compute + render lifecycle lives in the
 * `useHybridLayeredGI` hook (see that file for the pipeline architecture);
 * this component owns:
 *   - Redux selectors that pick up scene/mount/space/lighting state
 *   - SwiftShader refusal banner gating
 *   - Camera framing on first mount
 *   - JSX shell (Sky/Environment backdrop, RoomLoader, FaceRenderer +
 *     EdgeLines in room mode, MountDispatch, OrbitControls,
 *     WalkaroundDebugBridge)
 *   - __HYBRID_LAYERS__ DevTools toggle polling
 *
 * Library-extraction split: the hook is engine-agnostic (drop into any
 * R3F scene with the right inputs); this stage is the stained-glass-app
 * adapter that supplies those inputs.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { Sky, Environment, Html, OrbitControls } from '@react-three/drei';
import { useThree } from '@react-three/fiber';
import { useSelector } from 'react-redux';
import { MountDispatch } from '../mounts/MountDispatch';
import { lightboxDimsFor } from '../mounts/lightboxDims';
import { LightSourceList } from '../lighting/renderLightSource';
import { RoomLoader } from '../../assets/RoomLoader';
import { FaceRenderer } from '../../glass/FaceRenderer';
import { GLASS_FACE_MESH_NAME } from '../../glass/GlassMesh';
import { EdgeLines } from '../../edges/EdgeLines';
import { WalkaroundDebugBridge } from './engines/restir/WalkaroundDebugBridge';
import type { WgWalkaroundBridge } from './engines/restir/walkaroundBridgeTypes';
import { useHybridLayeredGI } from './useHybridLayeredGI';
import { detectGpu, type GpuDetection } from './gpuDetection';
import { FLOOR_Y } from '../../assets/livingRoomShellGeometry';
import type { BackdropMode } from '@/store/uiSlice';
import type { SkyParams } from '../skyParams';
import { computeLightingState } from '../lightingState';
import type { RootState } from '@/store';
import { selectMount, selectSpace, selectGraph } from '@/store/selectors';
import { selectProperties } from '@/store/selectors/properties';

interface HybridLayeredStageProps {
  backdropMode: BackdropMode;
  skyParams: SkyParams;
  nightSkyParams: SkyParams;
  frameLayout: { cx: number; cy: number; w: number; h: number };
  orbitTarget: [number, number, number];
}

// DEV-only: module-level mount counter so we can tell apart
// effect re-runs from full component reinstantiation. The
// previous instrumentation showed `rebuild-debounce effect fire #1`
// resetting to #1 mid-session, which only happens if React unmounted
// the component and remounted a fresh instance. This counter
// confirms (or refutes) that hypothesis numerically.
let HYBRID_MOUNT_SEQ = 0;

// DEV-only: unified state accessor. Call `window.__HYBRID_STATE__()`
// from the dev console for a one-shot snapshot of every relevant
// runtime field — Redux viewport/scene, GPU adapter info, DDGI/RC
// state, walkaround pipeline status, frame timings, mount counter.
// Read-only; safe to call from anywhere. Installed once per session
// (idempotent — last write wins, value is stable since the function
// closes over no module state besides HYBRID_MOUNT_SEQ).
function installHybridStateAccessor(): void {
  if (!import.meta.env.DEV) return;
  const w = window as unknown as Record<string, unknown>;
  if (typeof w['__HYBRID_STATE__'] === 'function') return;
  w['__HYBRID_STATE__'] = () => {
    const win = window as Window & typeof globalThis & {
      __DDGI__?: { ready?: boolean; probeCount?: number; lastFrameMs?: number };
    };
    const wg  = win.__WG__;
    const wgpu = win.__WGPU__;
    const wa   = wgpu?.walkaround as undefined | {
      passes?: { pipeline?: { lastGpuTimings?: Record<string, number>; lastGpuTimingsFrame?: number }; bvh?: unknown };
      frameTimings?: Array<{ t: number; ms: number }>;
    };
    const ddgi = win.__DDGI__;
    const store = (win as unknown as { __REDUX_STORE__?: { getState: () => Record<string, unknown> } }).__REDUX_STORE__;
    const state = store?.getState() ?? {};
    const viewport = state['viewport'] as Record<string, unknown> | undefined;
    const scene    = state['scene']    as Record<string, unknown> | undefined;
    const graph    = state['graph']    as { faces?: object; edges?: object; vertices?: object } | undefined;
    const props    = state['properties'] as { faceProperties?: object; edgeProperties?: object } | undefined;
    const ft = wa?.frameTimings;
    const recentFrames = Array.isArray(ft) ? ft.slice(-30).map((f) => f.ms) : [];
    const recentAvgMs = recentFrames.length
      ? recentFrames.reduce((a: number, b: number) => a + b, 0) / recentFrames.length
      : null;
    return {
      mountSeq: HYBRID_MOUNT_SEQ,
      gpu: wg ? {
        isWebGPU: wg.isWebGPU,
        isHardwareGpu: wg.isHardwareGpu,
        vendor: wg.adapterVendor,
        arch: wg.adapterArchitecture,
      } : null,
      walkaround: wa ? {
        passesBound: !!wa.passes,
        pipelineSet: !!wa.passes?.pipeline,
        bvhSet: !!wa.passes?.bvh,
        frameTimingsLen: ft?.length ?? 0,
        recentAvgDispatchMs: recentAvgMs?.toFixed(2) ?? null,
        latestFrameAgoMs: ft?.length
          ? Math.round(performance.now() - ft[ft.length - 1].t)
          : null,
        // Real GPU per-pass time (ms) from timestamp queries. Populated
        // every ~1-2 frames once the readback completes. Empty {} if the
        // adapter doesn't support 'timestamp-query'. The sum (`total`)
        // is the canonical per-frame GPU cost — compare against
        // 1000/targetFps to know if you have headroom.
        gpuTimingsMs: wa.passes?.pipeline?.lastGpuTimings ?? {},
        gpuTimingsFrame: wa.passes?.pipeline?.lastGpuTimingsFrame ?? -1,
      } : null,
      ddgi: ddgi ? {
        ready: ddgi.ready,
        probeCount: ddgi.probeCount,
        lastFrameMs: ddgi.lastFrameMs?.toFixed(2),
      } : null,
      redux: {
        cameraMode: viewport?.['cameraMode'],
        backdropMode: viewport?.['backdropMode'],
        timeOfDay: viewport?.['timeOfDay'],
        walkaroundEngine: viewport?.['walkaroundEngine'],
        exploreEnabled: viewport?.['exploreEnabled'],
        spaceKind: (scene?.['space'] as { kind?: string } | undefined)?.kind,
        roomKey:   (scene?.['space'] as { roomKey?: string } | undefined)?.roomKey,
        graphFaces: graph?.faces ? Object.keys(graph.faces).length : 0,
        graphEdges: graph?.edges ? Object.keys(graph.edges).length : 0,
        propertiesFaces: props?.faceProperties ? Object.keys(props.faceProperties).length : 0,
      },
      caveat: 'recentAvgDispatchMs = JS submit time only, not GPU completion. Real GPU time needs timestamp queries.',
    };
  };
  console.log('[hybrid:debug] window.__HYBRID_STATE__() accessor installed — call it for a state snapshot');
}

export function HybridLayeredStage({
  backdropMode,
  skyParams,
  nightSkyParams,
  frameLayout,
}: HybridLayeredStageProps) {
  // ── DEV-only mount/unmount tracking ──────────────────────────────
  // Empty-deps useEffect fires once on mount + cleanup on unmount.
  // Distinguishes "effect re-ran" from "whole component
  // reinstantiated" — the latter wipes the temporal accumulator and
  // is the user-visible "cells go grey" symptom.
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    HYBRID_MOUNT_SEQ++;
    const seq = HYBRID_MOUNT_SEQ;
    const t = performance.now();
    installHybridStateAccessor();
    console.log(`[hybrid:debug] HybridLayeredStage MOUNTED #${seq}`,
      { t: t.toFixed(0) });
    return () => {
      const lifeMs = performance.now() - t;
      console.log(`[hybrid:debug] HybridLayeredStage UNMOUNTED #${seq}`,
        { lifeMs: lifeMs.toFixed(0) });
    };
  }, []);

  const mount        = useSelector(selectMount);
  const space        = useSelector(selectSpace);
  const graphFaces   = useSelector(selectGraph);
  const properties   = useSelector(selectProperties);
  const timeOfDay    = useSelector((s: RootState) => s.viewport.timeOfDay);

  const lightboxDims  = lightboxDimsFor(mount, frameLayout.w, frameLayout.h);
  const roomKey       = space.kind === 'room' ? space.roomKey : null;
  const suppressSun   = backdropMode === 'studio' || backdropMode === 'sunset';
  const skyBackdrop   = backdropMode === 'sky' || backdropMode === 'night';
  const showBackboard = !skyBackdrop;

  const { scene, camera } = useThree();

  // ── SwiftShader refusal gate ──────────────────────────────────────
  const [gpu, setGpu] = useState<GpuDetection | null>(null);
  useEffect(() => {
    let cancelled = false;
    detectGpu().then((res) => { if (!cancelled) setGpu(res); });
    return () => { cancelled = true; };
  }, []);
  const refuseToMount = gpu !== null && gpu.isWebGPU && !gpu.isHardwareGpu;

  // ── __HYBRID_LAYERS__ DevTools toggles ───────────────────────────
  // RC toggle dropped step 4 of restructure — RC is no longer wired
  // into the hybrid shade pass.
  const [ddgiOn, setDdgiOn] = useState(true);
  useEffect(() => {
    const w = window as unknown as { __HYBRID_LAYERS__?: { ddgi?: boolean } };
    setDdgiOn(w.__HYBRID_LAYERS__?.ddgi !== false);
    const id = window.setInterval(() => {
      setDdgiOn(w.__HYBRID_LAYERS__?.ddgi !== false);
    }, 500);
    return () => window.clearInterval(id);
  }, []);

  // ── WebGPU device handle bridge ──────────────────────────────────
  const wgGate = useRef<WgWalkaroundBridge | null>(null);
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const check = () => {
      if (cancelled) return;
      const g = window.__WG__;
      if (g) {
        wgGate.current = g;
        return;
      }
      // Yield to the event loop. The previous `Promise.resolve().then(check)`
      // form spun in the microtask queue and starved animation frames + layout
      // work while waiting for the WebGPU bridge to publish.
      timer = setTimeout(check, 16);
    };
    check();
    return () => {
      cancelled = true;
      if (timer != null) clearTimeout(timer);
    };
  }, []);

  const wgpuDevice: GPUDevice | null = (() => {
    const w = window as Window & typeof globalThis;
    return w.__WGPU__?.device ?? null;
  })();

  // ── Default room-aware framing (re-runs when space.kind toggles) ─
  // framedRef tracks LAST-FRAMED space kind; clearing it on transition
  // ensures the room-mode default position takes effect when entering
  // room mode (was previously stuck at the panel-mode position because
  // framedRef.current was set on first mount and never cleared).
  const framedRef = useRef<string | null>(null);
  useEffect(() => {
    const targetSpaceKind = space.kind === 'room' ? 'room' : 'panel';
    if (framedRef.current === targetSpaceKind) return;
    if (refuseToMount) return;
    const w = window as Window & { __CHROMA_TEST_CAMERA__?: boolean };
    if (w.__CHROMA_TEST_CAMERA__ === true) {
      camera.position.set(0, 0, 80);
      camera.lookAt(0, FLOOR_Y, -40);
      camera.updateMatrixWorld();
      framedRef.current = targetSpaceKind;
      return;
    }
    const isRoomSpace = space.kind === 'room';
    if (isRoomSpace) {
      // Camera INSIDE the room — verified against actual geometry
      // 2026-05-08 after a prior comment had the coordinate convention
      // exactly backwards:
      //
      //   livingRoomShellGeometry.ts:19-20 defines BACK_Z = -0.05,
      //   FRONT_Z = +167.95 — so the room interior is z ∈ [0, +168],
      //   NOT z<0. The panel sits in the back wall at z≈0 with its
      //   front face normal +Z (pointing INTO the room).
      //
      //   skyParams.ts:27 puts the sun at sunPos.z = -1 (i.e., on the
      //   -Z side of the panel, OUTSIDE the room). That's the correct
      //   backlight orientation: light enters the panel from -Z, exits
      //   into the room at +Z, where the camera now lives.
      //
      // Camera at (cx, 30, +100) places it ~100" deep into the room
      // interior, ~3' off the floor, looking down-and-toward at the
      // panel below the centerline. Primary rays go -Z toward the
      // panel, hit the front face (normal +Z), Lo_emit's bidirectional
      // sunDot test fires (|dot(sunDir, +Z)| ≈ 0.707 at noon).
      //
      // The pre-fix camera was at z=-100 — that put the camera BEHIND
      // the back wall (BACK_Z = -0.05), looking through the wall at
      // the panel from the OUTSIDE/sun side. Hence the user-reported
      // "dark/gray room" + every wall-illumination diagnostic
      // pointing at zero contribution: the camera couldn't see the
      // room interior.
      camera.position.set(frameLayout.cx, 30, 100);
      camera.lookAt(frameLayout.cx, -30, 0);
    } else {
      camera.position.set(frameLayout.cx, frameLayout.cy, 60);
      camera.lookAt(frameLayout.cx, frameLayout.cy, 0);
    }
    camera.updateMatrixWorld();
    framedRef.current = targetSpaceKind;
  }, [camera, frameLayout.cx, frameLayout.cy, refuseToMount, space.kind]);

  // ── Stained-glass-app scene-readiness predicate ──────────────────
  // TODO(extract): duplicated in walkaround/engines/restir/RestirStage.tsx.
  // When the host app extraction resumes, move both copies into a shared
  // `createIsSceneReadyPredicate(scene, roomKey)` utility in lib/.
  const isSceneReady = useMemo(() => () => {
    let hasFaceGeo = false;
    let hasRoomFloor = false;
    let glassTriCount = 0;
    let totalTriCount = 0;
    scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh || !mesh.geometry) return;
      const idx = mesh.geometry.index;
      const tris = idx ? idx.count / 3 : (mesh.geometry.attributes.position?.count ?? 0) / 3;
      totalTriCount += tris;
      if (obj.name === 'surface_floor_living') hasRoomFloor = true;
      if (obj.name === GLASS_FACE_MESH_NAME && mesh.geometry.attributes.position) {
        hasFaceGeo = true;
        glassTriCount += tris;
      }
    });
    if (roomKey) {
      return hasRoomFloor && hasFaceGeo && glassTriCount >= 60;
    }
    return hasFaceGeo && totalTriCount >= 200;
  }, [scene, roomKey]);

  // ── Lighting state derived from shared computeLightingState ──────
  // Single source-of-truth for sun direction, sun intensity, sky tint.
  // PT/raster read identical values; their three.js DirectionalLight +
  // scene.environment derive from the same skyParams + timeOfDay.
  //
  // Step 3 of the restructure (2026-05-08): peakOverride dropped now
  // that the tonemap is the standard per-channel ACES at exposure 1.0
  // (parity with three.js R3F default). Sun intensity tracks the same
  // getSunIntensity(timeOfDay) bucket that PT uses — π at noon, etc.
  const isNight = backdropMode === 'night';
  const lightingState = useMemo(
    () => computeLightingState({
      timeOfDay,
      skyParams: isNight ? nightSkyParams : skyParams,
      isNight,
    }),
    [timeOfDay, skyParams, nightSkyParams, isNight],
  );
  const primaryLightDir = lightingState.sunDirection;
  const primaryLightIntensity = lightingState.sunIntensity;
  const skyTint = lightingState.skyTint;
  const skyIrradiance = lightingState.skyIrradiance;

  // Pipeline rebuild trigger — debounced so heavy edit storms (e.g.
  // applying materials to 300 cells in one tick during fixture load)
  // collapse into a single rebuild instead of 300. Re-fires when:
  //   - roomKey changes (entering/exiting room mode)
  //   - graph reference changes (topology edit) AFTER 600ms of no further changes
  //   - properties reference changes (material edit) AFTER 600ms of no further changes
  // 600ms is long enough to swallow a full stress-fixture dispatch but
  // short enough to feel responsive on single material edits.
  const [rebuildEpoch, setRebuildEpoch] = useState(0);
  // DEV-only instrumentation: track which dep CHANGED on each effect
  // fire — distinguishes graphFaces-driven churn from properties-driven
  // churn. The effect itself doesn't cause init storms (it only sets a
  // 600ms timer), but if its DEP REFERENCES tick faster than 600ms we
  // never fire the timer at all → forever-stalled pipeline. See
  // [hybrid:debug] logs in console.
  //
  // The ref allocation must run unconditionally to satisfy React's hooks
  // rules (useRef may not be conditionally called). Mutation + logging is
  // DEV-guarded below; the prod cost of this ref is a single empty object
  // per stage instance.
  const debugDepsRef = useRef<{ graph: unknown; properties: unknown; fires: number }>({
    graph: graphFaces, properties, fires: 0,
  });
  useEffect(() => {
    if (import.meta.env.DEV) {
      const d = debugDepsRef.current;
      d.fires++;
      const graphChanged = d.graph !== graphFaces;
      const propsChanged = d.properties !== properties;
      console.log('[hybrid:debug] rebuild-debounce effect fire #' + d.fires, {
        graphChanged, propsChanged,
        graphFaceCount: graphFaces ? Object.keys(graphFaces).length : 0,
        propertyCount: properties?.faceProperties
          ? Object.keys(properties.faceProperties).length : 0,
      });
      d.graph = graphFaces;
      d.properties = properties;
    }
    const id = window.setTimeout(() => {
      if (import.meta.env.DEV) {
        console.log('[hybrid:debug] rebuildEpoch++ — pipelineRebuildKey will change');
      }
      setRebuildEpoch((e) => e + 1);
    }, 600);
    return () => window.clearTimeout(id);
  }, [graphFaces, properties]);
  const pipelineRebuildKey = `${roomKey ?? 'panel'}|${rebuildEpoch}`;

  // ── The single GI orchestration call ─────────────────────────────
  const { passes } = useHybridLayeredGI({
    enabled: !refuseToMount,
    device: wgpuDevice,
    primaryLightDir,
    primaryLightIntensity,
    skyTint,
    skyIrradiance,
    ddgiOn,
    isSceneReady,
    pipelineRebuildKey,
  });

  // Refuse-to-mount overlay
  if (refuseToMount) {
    const vendor = wgGate.current?.adapterVendor || '(unknown)';
    const architecture = wgGate.current?.adapterArchitecture || '(unknown)';
    return (
      <Html center style={{ pointerEvents: 'none' }}>
        <div role="alert" style={{
          background: '#1a1a2e', color: '#ffe066',
          border: '2px solid #ffe066', padding: '32px',
          borderRadius: 8, font: '14px/1.5 system-ui, sans-serif',
          maxWidth: 640, textAlign: 'center',
          boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>
            Layered hybrid GI requires a hardware GPU
          </div>
          <div style={{ fontSize: 14, marginBottom: 12 }}>
            SwiftShader (software rasterizer) detected. Hybrid GI is misleading
            on software fallback — refusing to mount.
          </div>
          <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
            adapter.info.vendor = {vendor}<br/>
            adapter.info.architecture = {architecture}
          </div>
        </div>
      </Html>
    );
  }

  return (
    <>
      {/* Backdrop */}
      {backdropMode === 'sky' && (
        <Sky
          distance={450000}
          sunPosition={skyParams.sunPosition}
          turbidity={skyParams.turbidity}
          rayleigh={skyParams.rayleigh}
          mieCoefficient={skyParams.mieCoefficient}
          mieDirectionalG={skyParams.mieDirectionalG}
        />
      )}
      {backdropMode === 'night' && (
        <Sky
          distance={450000}
          sunPosition={nightSkyParams.sunPosition}
          turbidity={nightSkyParams.turbidity}
          rayleigh={nightSkyParams.rayleigh}
          mieCoefficient={nightSkyParams.mieCoefficient}
          mieDirectionalG={nightSkyParams.mieDirectionalG}
        />
      )}
      {backdropMode === 'studio' && <Environment preset="studio" background />}
      {backdropMode === 'sunset' && <Environment preset="sunset" background />}

      <LightSourceList ctx={{ lightbox: lightboxDims, suppressSun }} />

      {/* RoomLoader: mode="raster" → walls DoubleSide. */}
      {roomKey && <RoomLoader roomKey={roomKey} mode="raster" />}

      {/* Glass + came beads — mounted ONLY in room mode. In panel mode
          StudioScene already mounts FaceRenderer + EdgeLines (audit B7). */}
      {roomKey && <FaceRenderer />}
      {roomKey && <EdgeLines />}

      <MountDispatch
        centerX={frameLayout.cx}
        centerY={frameLayout.cy}
        width={frameLayout.w}
        height={frameLayout.h}
        mode="pt"
        showBackboard={showBackboard}
      />

      <OrbitControls
        target={[frameLayout.cx, frameLayout.cy, 0]}
        enablePan
        enableZoom
        enableRotate
        minDistance={20}
        maxDistance={400}
      />

      {passes !== null && <WalkaroundDebugBridge passes={passes} />}
    </>
  );
}
