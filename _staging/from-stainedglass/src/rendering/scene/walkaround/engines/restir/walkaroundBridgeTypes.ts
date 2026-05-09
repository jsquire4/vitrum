/**
 * ReSTIR-only Window augmentation for the `__WGPU__` test bridge.
 *
 * `__WG__` is declared canonically in `walkaround/gpuDetection.ts` and
 * shared across every engine; only ReSTIR has its own pipeline-state
 * bridge (`__WGPU__`), so the type lives next to its consumers.
 */

import type { GpuDetection } from '../../gpuDetection';

/**
 * ReSTIR debug bridge. `WalkaroundGPUPipeline` populates
 * `walkaround.passes`, `.camera`, `.frameTimings`, `.renderOneFrame`, and
 * `.captureFrame` after the ReSTIR pipeline finishes initializing; the gl
 * factory in StudioScene populates the rest at `renderer.init()` time so
 * the adapter handles are observable before any frames render.
 */
export interface WgpuBridge {
  renderer: unknown;
  device: GPUDevice | null;
  adapter: GPUAdapter | null;
  isHardwareGpu?: boolean;
  adapterInfo?: { vendor: string; architecture: string };
  frameTimings: { t: number; ms: number }[];
  walkaround?: {
    passes: unknown;
    camera?: unknown;
    frameTimings: { t: number; ms: number }[];
    renderOneFrame?: () => Promise<number>;
    captureFrame?: () => Promise<{ pixels: Float32Array; width: number; height: number }>;
  };
}

/**
 * Backwards-compatible alias for the cross-engine `__WG__` contract.
 * Re-exports `GpuDetection` so RestirStage's existing import surface
 * keeps working.
 */
export type WgWalkaroundBridge = GpuDetection;

declare global {
  interface Window {
    __WGPU__?: WgpuBridge;
  }
}
