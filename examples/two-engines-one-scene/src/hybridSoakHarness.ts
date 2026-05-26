/**
 * PR-6 hybrid lifecycle soak hooks (Playwright-driven).
 */

import type { GpuMemoryBreakdown, Scene, ScenePrimitive } from '@vitrum/core';
import type { HybridEngine } from '@vitrum/walkaround-hybrid';

export interface HybridSoakFlags {
  readonly hybridSoakAuto: boolean;
  readonly hybridSoakFrames: number;
  readonly hybridSoakMaterialEvery: number;
  readonly hybridSoakEmitterEvery: number;
}

export interface HybridSoakResult {
  readonly ok: boolean;
  readonly framesPolled: number;
  readonly materialPatches: number;
  readonly emitterPatches: number;
  readonly engineState: string;
  readonly lastWalkFrame: number;
  readonly estimatedGpuMemoryBytes?: number;
  readonly gpuMemoryBreakdown?: GpuMemoryBreakdown | null;
  readonly error?: string;
}

function firstMeshPrimitive(scene: Scene): ScenePrimitive | null {
  return scene.primitives.find((p) => p.kind === 'mesh' || p.kind === 'skinned-mesh') ?? null;
}

export function installHybridSoakApi(engine: HybridEngine, scene: Scene): {
  run: (flags: HybridSoakFlags) => Promise<HybridSoakResult>;
} {
  const run = async (flags: HybridSoakFlags): Promise<HybridSoakResult> => {
    const prim = firstMeshPrimitive(scene);
    const emitter = scene.emitters[0];
    let materialPatches = 0;
    let emitterPatches = 0;
    let framesPolled = 0;
    let lastWalkFrame = 0;

    const deadline = performance.now() + Math.max(flags.hybridSoakFrames * 50, 30_000);
    while (framesPolled < flags.hybridSoakFrames && performance.now() < deadline) {
      if (engine.state === 'disposed') break;
      const telemetry = (globalThis as unknown as { __vitrum?: { walkaround?: { frame: number } } })
        .__vitrum?.walkaround;
      const frame = telemetry?.frame ?? framesPolled;
      lastWalkFrame = frame;

      if (prim != null && flags.hybridSoakMaterialEvery > 0 && frame % flags.hybridSoakMaterialEvery === 0) {
        const hue = (frame * 0.07) % 1;
        engine.updatePrimitive(String(prim.id), {
          material: { baseColor: [0.4 + hue * 0.5, 0.35, 0.55], roughness: 0.45, metallic: 0 },
        });
        materialPatches += 1;
      }
      if (
        emitter != null
        && flags.hybridSoakEmitterEvery > 0
        && frame % flags.hybridSoakEmitterEvery === 0
      ) {
        engine.updateEmitter(String(emitter.id), { intensity: 0.8 + (frame % 10) * 0.05 });
        emitterPatches += 1;
      }

      framesPolled += 1;
      // eslint-disable-next-line no-await-in-loop
      await new Promise<void>((r) => requestAnimationFrame(() => r()));
    }

    const mem = engine.debug.estimatedGpuMemoryBytes?.() ?? null;
    return {
      ok: engine.state === 'ready' && framesPolled >= Math.min(8, flags.hybridSoakFrames),
      framesPolled,
      materialPatches,
      emitterPatches,
      engineState: engine.state,
      lastWalkFrame,
      ...(mem != null ? { estimatedGpuMemoryBytes: mem.total, gpuMemoryBreakdown: mem } : {}),
      ...(framesPolled < 8 ? { error: `only ${framesPolled} frames polled` } : {}),
    };
  };

  const api = { run };
  (globalThis as unknown as { __vitrumHybridSoak: typeof api }).__vitrumHybridSoak = api;
  return api;
}

export async function maybeAutoRunHybridSoak(
  engine: HybridEngine,
  scene: Scene,
  flags: HybridSoakFlags,
  api: ReturnType<typeof installHybridSoakApi>,
): Promise<HybridSoakResult | null> {
  if (!flags.hybridSoakAuto) return null;
  const result = await api.run(flags);
  (globalThis as unknown as { __vitrumHybridSoakLast: HybridSoakResult }).__vitrumHybridSoakLast = result;
  console.log(`VITRUM_HYBRID_SOAK_RESULT=${JSON.stringify(result)}`);
  return result;
}
