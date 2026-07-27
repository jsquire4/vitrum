// stubEngine.ts — shared test fixture (T5-A / D2-7).
//
// Several engine proxy/lifecycle tests hand-rolled the same minimal `Engine`
// scaffold — an `offscreen-texture` capabilities object plus the eight required
// lifecycle methods (setScene/renderFrame/reset/pause/resume/dispose + the two
// telemetry accessors) — differing only in the feature methods the test under
// scrutiny probes. This fixture centralizes that scaffold so each test supplies
// ONLY its feature-specific overrides. Assertions are unchanged; the fixture is
// a pure test-only consolidation.

import { vi } from 'vitest';
import type { EngineCapabilities, EngineState, FrameOutput, Engine } from '@vitrum/core';

/**
 * Minimal `offscreen-texture` capabilities scaffold used by the proxy/lifecycle
 * tests. `overrides` shallow-merge over the base so a caller can flip a single
 * capability (e.g. `supportsAddRemovePrimitive`) or add a nested field (e.g.
 * `incrementalPatchSupport`) without re-declaring the whole object.
 */
export function stubCapabilities(
  overrides: Partial<Record<string, unknown>> = {},
): EngineCapabilities {
  return {
    supportsIncrementalScene: false,
    supportsAddRemovePrimitive: false,
    supportsAuxBuffers: false,
    accumulates: true,
    maxSamplesPerPixel: 1,
    maxBounces: 1,
    supportedAnalyticShapes: new Set(),
    supportedEmitterKinds: new Set(),
    supportedPrimitiveKinds: new Set(),
    supportedEnvironmentKinds: new Set(),
    presentationMode: 'offscreen-texture',
    causticStrategy: 'none',
    ...overrides,
  } as unknown as EngineCapabilities;
}

const SKIPPED_FRAME: FrameOutput = {
  kind: 'skipped',
  samplesAccumulated: 0,
  isConverged: false,
} as unknown as FrameOutput;

/**
 * Base `Engine` scaffold with `vi.fn()` lifecycle stubs and the given (or a
 * fresh `offscreen-texture`) capabilities. Spread the returned object and layer
 * feature methods on top:
 *
 * ```ts
 * const engine: Engine = { ...stubEngine(), addPrimitive: (p) => add(p) };
 * ```
 */
export function stubEngine(caps: EngineCapabilities = stubCapabilities()): Engine {
  return {
    get state(): EngineState { return 'ready'; },
    get capabilities() { return caps; },
    setScene: vi.fn(),
    renderFrame: vi.fn(() => SKIPPED_FRAME),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
  };
}
