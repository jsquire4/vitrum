/**
 * ShadePass — re-traces primary ray, evaluates ReSTIR, computes DI + indirect bounce.
 *
 * Reads the final DI reservoir (post-spatial-2) and the final GI reservoir
 * (post-gi-spatial-2). Writes hdrColorTexture (direct), hdrIndirectTexture
 * (indirect), hdrTotalTexture (direct+indirect), gNormalDepthTexture, and
 * the albedoTexture for downstream demodulation.
 *
 * Full-res dispatch with the hybrid-layers (DDGI) group bound at slot 3;
 * the dispatch body is the shared {@link SharedBindGroupPass}.
 */

import { SharedBindGroupPass } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class ShadePass extends SharedBindGroupPass {
  readonly id = 'shade' as const;
  readonly dependencies: readonly string[] = ['spatial-2', 'gi-spatial-2'];
  readonly passLabels: readonly PassLabel[] = ['shade'];

  protected override readonly useHybridLayers = true;
}
