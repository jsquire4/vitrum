/**
 * RISPass — ReSTIR-DI primary-ray-cast + initial candidate sampling.
 *
 * Casts primary rays through the BVH, samples emitter candidates via
 * importance sampling, and writes the current-frame reservoir. Uses the
 * shared frame/scene/ubo bind groups; no pass-private bind groups —
 * the dispatch body lives in {@link SharedBindGroupPass}.
 */

import { SharedBindGroupPass } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class RISPass extends SharedBindGroupPass {
  readonly id = 'ris' as const;
  readonly dependencies: readonly string[] = ['sample-budget'];
  readonly passLabels: readonly PassLabel[] = ['ris'];
}
