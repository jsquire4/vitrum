/**
 * TemporalReservoirPass — ReSTIR-DI temporal reuse (merge with previous-frame reservoir).
 *
 * Reuses the shared frame/scene/ubo bind groups; no pass-private state —
 * the dispatch body lives in {@link SharedBindGroupPass}.
 */

import { SharedBindGroupPass } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class TemporalReservoirPass extends SharedBindGroupPass {
  readonly id = 'temporal' as const;
  readonly dependencies: readonly string[] = ['ris'];
  readonly passLabels: readonly PassLabel[] = ['temporal'];
}
