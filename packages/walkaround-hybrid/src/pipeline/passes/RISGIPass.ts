/**
 * RISGIPass — Sprint 16 ReSTIR-GI RIS pass.
 *
 * Half-resolution dispatch (W/2 × H/2). Reuses the shared frame/scene/ubo
 * bind groups + the hybrid-layers (DDGI) bind group at slot 3.
 *
 * Runs after the DI spatial passes (consumes the spatially-fused DI
 * reservoir from `bgFrame`) so the GI reservoir is built on the
 * variance-reduced primary visibility.
 *
 * Dispatch body is the shared {@link SharedBindGroupPass}; this pass only
 * flips on `useHybridLayers` (slot 3) and `halfRes`.
 */

import { SharedBindGroupPass } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class RISGIPass extends SharedBindGroupPass {
  readonly id = 'gi-ris' as const;
  readonly dependencies: readonly string[] = ['spatial-2'];
  readonly passLabels: readonly PassLabel[] = ['gi-ris'];

  protected override readonly useHybridLayers = true;
  protected override readonly halfRes = true;
}
