/**
 * PR-6 hybrid benchmark hooks for Playwright (`run-pr-hybrid-bench.mjs`).
 *
 * Exposes `window.__vitrumPrBench` on the walkaround-hybrid example page.
 */

import type { Scene, ScenePrimitive } from '@vitrum/core';
import type { HybridEngine } from '@vitrum/walkaround-hybrid';

export type PrBenchMode = 'material-churn' | 'emitter-churn' | 'frame-sample';

export interface PrBenchFlags {
  readonly prBench: PrBenchMode | null;
  readonly prBenchIters: number;
  readonly prBenchFrames: number;
  readonly prBenchAuto: boolean;
}

export interface PrBenchResult {
  readonly scenario: string;
  readonly ok: boolean;
  readonly elapsedMs: number;
  readonly iterations?: number;
  readonly p95FrameMs?: number;
  readonly sampleCount?: number;
  readonly engineState: string;
  readonly error?: string;
}

function firstMeshPrimitive(scene: Scene): ScenePrimitive | null {
  return scene.primitives.find((p) => p.kind === 'mesh' || p.kind === 'skinned-mesh') ?? null;
}

function p95(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95));
  return sorted[idx] ?? 0;
}

async function waitForFrameSamples(
  engine: HybridEngine,
  targetCount: number,
  timeoutMs: number,
): Promise<number[]> {
  const times: number[] = [];
  const unsub = engine.onFrame((stats) => {
    times.push(stats.frameTimeMs);
  });
  const deadline = performance.now() + timeoutMs;
  while (times.length < targetCount && performance.now() < deadline) {
    if (engine.state === 'disposed') break;
    await new Promise<void>((r) => requestAnimationFrame(() => r()));
  }
  unsub();
  return times;
}

export function installPrBenchApi(
  engine: HybridEngine,
  scene: Scene,
  flags: PrBenchFlags,
): {
  runMaterialChurn: (iterations?: number) => Promise<PrBenchResult>;
  runEmitterChurn: (iterations?: number) => Promise<PrBenchResult>;
  runFrameSample: (frames?: number) => Promise<PrBenchResult>;
  run: (mode: PrBenchMode) => Promise<PrBenchResult>;
} {
  const runMaterialChurn = async (iterations = flags.prBenchIters): Promise<PrBenchResult> => {
    const prim = firstMeshPrimitive(scene);
    if (prim == null) {
      return {
        scenario: 'PR-hybrid-material-churn',
        ok: false,
        elapsedMs: 0,
        engineState: engine.state,
        error: 'no mesh primitive in scene',
      };
    }
    const t0 = performance.now();
    try {
      for (let i = 0; i < iterations; i += 1) {
        const t = (i % 100) / 100;
        engine.updatePrimitive(String(prim.id), {
          material: {
            baseColor: [0.2 + t * 0.6, 0.35, 0.45],
            roughness: 0.25 + (i % 5) * 0.05,
            metallic: 0,
          },
        });
      }
      return {
        scenario: 'PR-hybrid-material-churn',
        ok: engine.state === 'ready',
        elapsedMs: performance.now() - t0,
        iterations,
        engineState: engine.state,
      };
    } catch (e) {
      return {
        scenario: 'PR-hybrid-material-churn',
        ok: false,
        elapsedMs: performance.now() - t0,
        iterations,
        engineState: engine.state,
        error: e instanceof Error ? e.message : String(e),
      };
    }
  };

  const runEmitterChurn = async (iterations = flags.prBenchIters): Promise<PrBenchResult> => {
    const emitter = scene.emitters[0];
    if (emitter == null) {
      return {
        scenario: 'PR-hybrid-emitter-churn',
        ok: false,
        elapsedMs: 0,
        engineState: engine.state,
        error: 'no emitters in scene',
      };
    }
    const t0 = performance.now();
    try {
      for (let i = 0; i < iterations; i += 1) {
        engine.updateEmitter(String(emitter.id), {
          intensity: 1 + (i % 20) * 0.1,
        });
      }
      return {
        scenario: 'PR-hybrid-emitter-churn',
        ok: engine.state === 'ready',
        elapsedMs: performance.now() - t0,
        iterations,
        engineState: engine.state,
      };
    } catch (e) {
      return {
        scenario: 'PR-hybrid-emitter-churn',
        ok: false,
        elapsedMs: performance.now() - t0,
        iterations,
        engineState: engine.state,
        error: e instanceof Error ? e.message : String(e),
      };
    }
  };

  const runFrameSample = async (frames = flags.prBenchFrames): Promise<PrBenchResult> => {
    const t0 = performance.now();
    const times = await waitForFrameSamples(engine, frames, Math.max(frames * 200, 30_000));
    return {
      scenario: 'PR-hybrid-frame-sample',
      ok: times.length >= Math.min(frames, 8) && engine.state === 'ready',
      elapsedMs: performance.now() - t0,
      p95FrameMs: p95(times),
      sampleCount: times.length,
      engineState: engine.state,
      ...(times.length < 8 ? { error: `only ${times.length} frame samples` } : {}),
    };
  };

  const run = async (mode: PrBenchMode): Promise<PrBenchResult> => {
    if (mode === 'material-churn') return runMaterialChurn();
    if (mode === 'emitter-churn') return runEmitterChurn();
    return runFrameSample();
  };

  const api = { runMaterialChurn, runEmitterChurn, runFrameSample, run };
  (globalThis as unknown as { __vitrumPrBench: typeof api }).__vitrumPrBench = api;
  return api;
}

export async function maybeAutoRunPrBench(
  engine: HybridEngine,
  scene: Scene,
  flags: PrBenchFlags,
  api: ReturnType<typeof installPrBenchApi>,
): Promise<PrBenchResult | null> {
  if (!flags.prBenchAuto || flags.prBench == null) return null;
  const result = await api.run(flags.prBench);
  (globalThis as unknown as { __vitrumPrBenchLast: PrBenchResult }).__vitrumPrBenchLast = result;
  console.log(`VITRUM_PR_BENCH_RESULT=${JSON.stringify(result)}`);
  return result;
}
