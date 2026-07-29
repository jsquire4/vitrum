// Lite-tier strict scene validation.
//
// The lite shader omits analytic bindings, explicit mesh-area-light sampling,
// and arbitrary active COLOR_n streams. Rendering after dropping any of those
// inputs is a lossy scene substitution, so ingestion fails before
// packing/allocation.

import type { Scene } from '@vitrum/core';
import { collectLiteUnsupportedVertexColorPrimitiveIds } from './liteSceneAnalysis.js';

/** Reject content the lite shader cannot reproduce exactly. */
export function assertLiteSceneSupported(scene: Scene): void {
  const analyticPrimitives = scene.primitives.filter((p) => p.kind === 'analytic');
  const meshAreaEmitters = scene.emitters.filter((e) => e.kind === 'mesh-area');
  const vertexColorPrimitiveIds = collectLiteUnsupportedVertexColorPrimitiveIds(scene);
  if (
    analyticPrimitives.length === 0 &&
    meshAreaEmitters.length === 0 &&
    vertexColorPrimitiveIds.length === 0
  ) return;

  const details: string[] = [];
  if (analyticPrimitives.length > 0) {
    details.push(
      `analytic primitives [${analyticPrimitives.map((p) => p.id).join(', ')}]`,
    );
  }
  if (meshAreaEmitters.length > 0) {
    details.push(
      `mesh-area emitters [${meshAreaEmitters.map((e) => e.id).join(', ')}]`,
    );
  }
  if (vertexColorPrimitiveIds.length > 0) {
    details.push(
      `non-bakeable active vertex-color primitives [${vertexColorPrimitiveIds.join(', ')}]`,
    );
  }
  throw new TypeError(
    `[vitrum/pt-webgpu] Lite tier cannot render ${details.join('; ')}. ` +
      'Use the full trace tier or remove the unsupported scene content.',
  );
}
