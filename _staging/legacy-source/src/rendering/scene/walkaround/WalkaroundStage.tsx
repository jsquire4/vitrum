/**
 * WalkaroundStage — the 3D stage mounted when
 * `viewport.exploreEnabled && viewport.walkaroundEngine === 'ddgi'`
 * (locked-decision #3, branch-parity layered Redux shape).
 *
 * Structurally mirrors RasterStage but:
 * - Runs on WebGPU renderer (no <Sky> drei component — uses drei Environment
 *   or a simple ambient as backdrop until the TSL Sky port lands).
 * - Drives the DDGI probe update via useDDGI().
 * - Injects DDGI diffuse indirect into all scene materials via
 *   applyDDGIShading().
 * - Exposes window.__DDGI__ + window.__SET_CAMERA__ for Playwright tests.
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Sky, Environment, Html } from '@react-three/drei';
import { useSelector } from 'react-redux';
import { useThree } from '@react-three/fiber';
import { MountDispatch } from '../mounts/MountDispatch';
import { lightboxDimsFor } from '../mounts/lightboxDims';
import { LightSourceList } from '../lighting/renderLightSource';
import { RoomLoader } from '../../assets/RoomLoader';
import { useDDGI } from './useDDGI';
import { applyDDGIShading } from './applyDDGIShading';
import { detectGpu, type GpuDetection } from './gpuDetection';
import type { SkyParams } from '../skyParams';
import type { BackdropMode } from '@/store/uiSlice';
import { selectMount, selectSpace, selectLightsById } from '@/store/selectors';

interface WalkaroundStageProps {
  backdropMode: BackdropMode;
  skyParams: SkyParams;
  nightSkyParams: SkyParams;
  frameLayout: { cx: number; cy: number; w: number; h: number };
  orbitTarget: [number, number, number];
}

export function WalkaroundStage({
  backdropMode,
  skyParams,
  nightSkyParams,
  frameLayout,
}: WalkaroundStageProps) {
  const mount       = useSelector(selectMount);
  const space       = useSelector(selectSpace);
  const lightingById = useSelector(selectLightsById);

  const lightboxDims  = lightboxDimsFor(mount, frameLayout.w, frameLayout.h);
  const roomKey       = space.kind === 'room' ? space.roomKey : null;
  const suppressSun   = backdropMode === 'studio' || backdropMode === 'sunset';

  // Hardware-GPU gate (Option F). On first mount, request a WebGPU adapter
  // and check `adapter.info` for SwiftShader. On software rasterizer we
  // refuse to mount the DDGI scene — silent SwiftShader fallbacks have
  // historically produced low-chroma "almost passes" in validation, so
  // failing fast prevents future false-positive renders.
  // The detection publishes window.__WG__ even before this state updates,
  // so the e2e precondition gate sees the flag immediately on resolve.
  const [gpu, setGpu] = useState<GpuDetection | null>(null);
  useEffect(() => {
    let cancelled = false;
    detectGpu().then((res) => { if (!cancelled) setGpu(res); });
    return () => { cancelled = true; };
  }, []);

  const refuseToMount = gpu !== null && gpu.isWebGPU && !gpu.isHardwareGpu;

  // DDGI hook — owns BVH, probe grid, and runs compute passes each frame.
  // Also installs window.__SET_CAMERA__ + window.__DDGI__ for e2e tests.
  // Disabled on SwiftShader so probeUpdatePass doesn't spin trying to
  // re-detect each frame; the refuse-to-mount overlay below carries the
  // user-visible signal.
  const ddgi = useDDGI({ enabled: !refuseToMount });

  // Inject DDGI into scene materials after each scene-graph change.
  const { scene, gl, camera } = useThree();
  useEffect(() => {
    if (ddgi.probeGrid.irradianceA) {
      applyDDGIShading(scene, ddgi.probeGrid, true);
    }
  }, [scene, ddgi.probeGrid, lightingById, mount, space]);

  // Panel-framing default camera. The chroma e2e test owns the camera
  // entirely via `window.__SET_CAMERA__` (called every sample step) and
  // sets a final straight-down floor view at `[0, 0, 60]`/`[0, -60, 60]`.
  // For the dev/.sglass-loaded path no test bridge fires, so the camera
  // would otherwise stay at StudioCameraRig's default `[0, 0, 30]` aimed
  // at the origin — with a real loaded scene (panel center near
  // `(frameLayout.cx, frameLayout.cy, 0)`, e.g. ~(6.5, 4.5, 0) for the
  // 4×5 honeycomb fixture) the panel sits outside the frustum and the
  // user sees mostly empty space. This effect frames the panel head-on
  // from a "viewer in the room" distance on first mount only, leaving
  // subsequent `__SET_CAMERA__` calls authoritative. Setting
  // `window.__CHROMA_TEST_CAMERA__ = true` before mount opts out of the
  // framing setup entirely (the chroma test does not need this since it
  // overrides the camera every step, but the flag is provided so any
  // future test path can preserve the StudioCameraRig default).
  const framedRef = useRef(false);
  useEffect(() => {
    if (framedRef.current) return;
    if (refuseToMount) return;
    const w = window as unknown as { __CHROMA_TEST_CAMERA__?: boolean };
    if (w.__CHROMA_TEST_CAMERA__ === true) {
      framedRef.current = true;
      return;
    }
    // Position: head-on, 60" back from the panel along +Z, at panel
    // center height. Target: panel center on the back wall (z=0). With
    // a 45° FOV perspective camera this frames a panel up to ~50"
    // wide/tall comfortably with surrounding wall + floor visible —
    // i.e. "I'm standing in a living room looking at the suncatcher in
    // the window."
    camera.position.set(frameLayout.cx, frameLayout.cy, 60);
    camera.lookAt(frameLayout.cx, frameLayout.cy, 0);
    camera.updateMatrixWorld();
    framedRef.current = true;
  }, [camera, frameLayout.cx, frameLayout.cy, refuseToMount]);

  // Defense-in-depth for the WebGPU transmission-compositor format mismatch.
  // StudioScene's `flat` + `linear` Canvas props set NoToneMapping +
  // LinearSRGBColorSpace via R3F at mount time, which is what we need for
  // the WebGPU `Renderer._getFrameBufferTarget()` to return null and avoid
  // the rgba16float ↔ rgba8unorm copy mismatch in PhysicalLightingModel's
  // `viewportMipTexture()`. R3F has been observed to clobber these on later
  // re-configure passes; this effect re-asserts the values directly on the
  // live WebGPURenderer so nothing else can downstream-toggle the HDR
  // framebuffer back on.
  useEffect(() => {
    const renderer = gl as unknown as THREE.WebGLRenderer;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const anyR = renderer as any;
    if (anyR.backend?.isWebGPUBackend === true) {
      // The react-hooks/immutability rule flags assignment through `gl`
      // (it treats hook return values as read-only). The intent here is
      // exactly to mutate the live renderer's tone-map config; gl is the
      // authoritative renderer instance and there's no other handle.
      // eslint-disable-next-line react-hooks/immutability
      renderer.toneMapping = THREE.NoToneMapping;
      renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    }
  }, [gl]);

  // Refuse-to-mount path: render only the SwiftShader warning so that the
  // canvas area shows an unambiguous error instead of a deceptive scene.
  if (refuseToMount) {
    return (
      <Html center style={{ pointerEvents: 'none' }}>
        <div
          style={{
            background: '#220000',
            color: '#ff6b6b',
            border: '2px solid #ff6b6b',
            padding: '24px 32px',
            borderRadius: 8,
            font: '14px/1.5 system-ui, sans-serif',
            maxWidth: 480,
            textAlign: 'left',
            boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
            Walk-around mode requires a hardware GPU
          </div>
          <div style={{ marginBottom: 8 }}>
            Detected SwiftShader (software rasterizer). DDGI output on
            SwiftShader is misleading — refusing to mount the 3D scene.
          </div>
          <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: 12, color: '#ffaaaa' }}>
            adapter.info.vendor = {gpu?.adapterVendor || '(unknown)'}<br />
            adapter.info.architecture = {gpu?.adapterArchitecture || '(unknown)'}
          </div>
          <div style={{ marginTop: 12, fontSize: 12, color: '#ddaaaa' }}>
            Re-launch Chrome with hardware acceleration enabled (chrome://gpu must
            show a real vendor) and reload.
          </div>
        </div>
      </Html>
    );
  }

  return (
    <>
      {/* Backdrop — same as RasterStage for now. drei Sky works on WebGPU
          renderer because Sky's RawShaderMaterial compiles via three's
          WebGL2 backend fallback path. We accept this for v1; a full TSL
          Sky port is a P7 polish item. */}
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

      {/* Room + lights + mount — same API as RasterStage. */}
      {roomKey && <RoomLoader roomKey={roomKey} mode="raster" />}
      <LightSourceList ctx={{ lightbox: lightboxDims, suppressSun }} />
      <MountDispatch
        centerX={frameLayout.cx}
        centerY={frameLayout.cy}
        width={frameLayout.w}
        height={frameLayout.h}
        mode="raster"
        showBackboard={false}
      />

      {/* Ambient fill — ensures scene isn't pitch-black while probes warm up. */}
      <ambientLight intensity={0.15} />

      {/* Camera control note: WalkaroundStage does NOT mount its own
       *  OrbitControls (slice-5.3 lifted those to StudioScene; StudioScene
       *  gates them off when walkaroundActive (= exploreEnabled &&
       *  walkaroundEngine === 'ddgi') to avoid conflicting with the
       *  walk camera). The e2e DDGI test drives the camera explicitly via
       *  `window.__SET_CAMERA__` (installed by useDDGI). Wiring useWalkCamera
       *  in for human users is a follow-up — it currently mutates the camera
       *  every useFrame, which would fight __SET_CAMERA__ in the test. */}

      {/* Gizmo intentionally absent in walkaround. drei's <GizmoHelper> /
       *  <GizmoViewport> mount internal helpers (e.g. <AxisHead>) that
       *  call `gl.capabilities.getMaxAnisotropy()` — a method on
       *  WebGLRenderer's `capabilities` that does NOT exist on
       *  WebGPURenderer. With the explore-mode WebGPU canvas, this
       *  throws `TypeError: Cannot read properties of undefined (reading
       *  'getMaxAnisotropy')` from inside drei, CanvasErrorBoundary
       *  catches it, and the canvas never mounts (no DDGI, no panel,
       *  no walkaround). Same root cause as why <EdgeLines> is gated
       *  off in StudioScene under walkaroundActive (drei's <Line> uses
       *  LineMaterial / ShaderMaterial which is also WebGL-only). The
       *  raster-mode RasterStage still mounts GizmoHelper unaffected. */}
    </>
  );
}
