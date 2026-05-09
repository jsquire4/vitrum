import { useEffect, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { usePathtracer } from '@react-three/gpu-pathtracer';
import { PT_TARGET_SAMPLES } from './pathtracerConstants';

/**
 * Dev-only bridge: exposes the live `WebGLPathTracer` to `window.__PT__`
 * (and a per-sample timestamp ring) so e2e timing tests can read
 * `samples` and per-sample wall time without instrumenting React.
 *
 * Stripped from production by Vite: `import.meta.env.DEV` is a static
 * boolean replaced at build time, so the whole branch DCEs out.
 *
 * Mounts inside the `<Pathtracer>` subtree (PathTracingLayer) so
 * `usePathtracer()` resolves to the same WebGLPathTracer instance the
 * wrapper instantiated.
 */
export function PathtracerDebugBridge({ targetSamples }: { targetSamples: number }) {
  const { pathtracer } = usePathtracer();
  const lastSamples = useRef(0);
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const w = window as unknown as { __PT__?: unknown };
    w.__PT__ = {
      pathtracer,
      sampleStamps: [] as { sample: number; t: number }[],
      mountedAt: performance.now(),
      targetSamples,
    };
    lastSamples.current = 0;
    return () => { delete (w as { __PT__?: unknown }).__PT__; };
  }, [pathtracer, targetSamples]);
  useFrame(() => {
    if (!import.meta.env.DEV) return;
    // Once we've reached the convergence target, samples stops
    // changing, so the comparison below would never fire — but the
    // useFrame callback would still run every frame in dev. Bail
    // out early once we've recorded the final sample.
    const dbg = (window as unknown as {
      __PT__?: { targetSamples?: number; sampleStamps: { sample: number; t: number }[] };
    }).__PT__;
    const target = dbg?.targetSamples ?? PT_TARGET_SAMPLES;
    if (lastSamples.current >= target) return;
    if (!dbg) return;
    const cur = (pathtracer as unknown as { samples: number }).samples;
    if (cur !== lastSamples.current) {
      dbg.sampleStamps.push({ sample: cur, t: performance.now() });
      lastSamples.current = cur;
    }
  });
  return null;
}
