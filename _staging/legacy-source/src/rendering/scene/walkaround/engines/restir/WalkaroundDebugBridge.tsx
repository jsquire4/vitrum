/**
 * WalkaroundDebugBridge — exposes the live WebGPU pipeline state on
 * `window.__WGPU__.walkaround` for the e2e test (§9.3).
 *
 * Sibling of PathtracerDebugBridge. Mounts inside WalkaroundStage so
 * it can access the pass handles and camera set by that component.
 *
 * DEV-only: the conditional is a static boolean replaced by Vite at build
 * time, so the entire component DCEs out of production builds.
 */
import { useEffect } from 'react';
import { useThree } from '@react-three/fiber';

interface WalkaroundDebugBridgeProps {
  /** The compiled pass handles from WalkaroundStage, or null while building. */
  passes: unknown;
}

export function WalkaroundDebugBridge({ passes }: WalkaroundDebugBridgeProps) {
  const { camera } = useThree();

  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const w = window as Window & typeof globalThis;
    if (!w.__WGPU__) return;

    // WalkaroundStage owns walkaround initialization (set only after
    // pipeline.initialize() completes). The bridge just updates passes +
    // camera on the existing object so the test can observe them.
    // Never create the walkaround sub-object here — that would let
    // heartbeat frames register before ReSTIR is online.
    if (w.__WGPU__.walkaround) {
      w.__WGPU__.walkaround.passes = passes;
      w.__WGPU__.walkaround.camera = camera;
    }
  }, [passes, camera]);

  return null;
}
