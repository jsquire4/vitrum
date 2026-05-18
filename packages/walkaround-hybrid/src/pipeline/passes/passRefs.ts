/**
 * Mutable wrappers for state shared between the orchestrator and individual
 * Pass implementations. The orchestrator owns the storage so values persist
 * across frames; each Pass receives the ref via its constructor and reads /
 * writes through `.value`.
 *
 * Mirrors the {@link UboRef} pattern already used by `bindGroupBuilders.ts`
 * for the lazy-allocated UBO buffers.
 */

/** Mutable wrapper around an integer ping-pong index (0 or 1). */
export interface PingPongRef {
  value: number;
}
