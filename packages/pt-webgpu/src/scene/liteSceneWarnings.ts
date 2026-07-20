// Lite-tier setScene warning emitter (T3-B god-file split, 2026-07-20).
//
// Extracted verbatim from `PTEngineWebGPU.setScene`'s `traceTier === 'lite'`
// block in `index.ts`. Emits the three structured lite-tier content warnings
// (analytic primitives, unsupported emitters, non-bakeable vertex colors) in the
// SAME order via the caller-supplied `warn` callback (the engine's `#warn`). No
// logic change — only the home moved and the emit channel is now a parameter.

import type { EngineWarning, Scene } from '@vitrum/core';
import { PT_WEBGPU_LITE_UNSUPPORTED_EMITTER_KINDS } from '../capabilities.js';
import { collectLiteUnsupportedVertexColorPrimitiveIds } from './liteSceneAnalysis.js';

/**
 * Emit the lite-tier setScene warnings for content the lite kernel cannot render
 * faithfully. Ordered exactly as the former inline block: analytic primitives →
 * unsupported emitters (mesh-area) → non-bakeable vertex colors. `warn` is the
 * engine's structured-warning sink.
 */
export function emitLiteSceneWarnings(
  scene: Scene,
  warn: (warning: EngineWarning) => void,
): void {
  // Warn when the scene contains content the lite tier cannot handle.
  // B12 — point/spot/rect-area emitters and HDRI environments are now
  // supported via texture packing (liteLightTex, liteEnvTex, liteEnvCdfTex).
  // Remaining unsupported: analytic primitives (group-1 absent) and mesh-area
  // emitters (no NEE path in lite kernel). Static
  // mesh/skinned/instanced primitives, including non-identity transforms, are
  // baked into the lite tier's single world-space BLAS at pack time.
  const analyticPrimitives = scene.primitives.filter((p) => p.kind === 'analytic');
  if (analyticPrimitives.length > 0) {
    const primitiveIds = analyticPrimitives.map((p) => p.id);
    warn({
      code: 'pt-webgpu.lite-analytic-primitive',
      backend: 'pt-webgpu',
      phase: 'setScene',
      method: 'setScene',
      message:
        `[vitrum/pt-webgpu] Lite tier: scene contains ${analyticPrimitives.length} analytic primitive(s) — ` +
        'analytic shape rendering requires the full tier (group-1 bindings are absent on lite). ' +
        'They are ignored by the lite renderer after this structured warning.',
      details: {
        count: analyticPrimitives.length,
        primitiveIds,
        primitiveKinds: ['analytic'],
        requiredTier: 'full',
        fallback: 'ignore-unsupported-lite-primitive',
      },
    });
  }
  // B12 — only mesh-area is unsupported on lite; point/spot/rect/disc-area
  // are handled via texture-packed NEE (liteLightTex).
  // D8.14: kind-set sourced from PT_WEBGPU_LITE_UNSUPPORTED_EMITTER_KINDS
  // (derived from PT_WEBGPU_SUPPORT diff) rather than inline literals.
  const unsupportedEmitters = scene.emitters.filter(
    (e) => PT_WEBGPU_LITE_UNSUPPORTED_EMITTER_KINDS.has(e.kind),
  );
  if (unsupportedEmitters.length > 0) {
    const kinds = [...new Set(unsupportedEmitters.map((e) => e.kind))].join(', ');
    const emitterIds = unsupportedEmitters.map((e) => e.id);
    const emitterKinds = unsupportedEmitters.map((e) => e.kind);
    warn({
      code: 'pt-webgpu.lite-unsupported-emitters',
      backend: 'pt-webgpu',
      phase: 'setScene',
      method: 'setScene',
      message:
        `[vitrum/pt-webgpu] Lite tier: scene contains emitters of kind(s) [${kinds}] — ` +
        'mesh-area emitters are not supported on the lite tier (no NEE path in lite kernel). ' +
        'They are ignored by the lite renderer after this structured warning.',
      details: {
        kinds,
        count: unsupportedEmitters.length,
        emitterIds,
        emitterKinds,
        requiredTier: 'full',
        fallback: 'ignore-unsupported-lite-emitter',
      },
    });
  }
  // B12 follow-up — HDRI environments and all directional emitters are now
  // supported via texture packing; no first-directional truncation warning.
  const vertexColorPrimitiveIds = collectLiteUnsupportedVertexColorPrimitiveIds(scene);
  if (vertexColorPrimitiveIds.length > 0) {
    warn({
      code: 'pt-webgpu.lite-unsupported-vertex-colors',
      backend: 'pt-webgpu',
      phase: 'setScene',
      method: 'setScene',
      message:
        `[vitrum/pt-webgpu] Lite tier: scene contains ${vertexColorPrimitiveIds.length} primitive(s) with ` +
        'non-constant or alpha-bearing vertex colors (COLOR_0). Constant RGB vertex colors are baked into ' +
        'the lite material base color, but arbitrary COLOR_0 still needs the full-tier vertex-color binding.',
      details: {
        primitiveIds: vertexColorPrimitiveIds,
        bakedWhen: 'constant-rgb-alpha-one',
        requiredTier: 'full',
      },
    });
  }
}
