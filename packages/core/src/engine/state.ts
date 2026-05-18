// Engine lifecycle state.
//
// Split from the original `engine.ts` (sweep A-7) so each banded concern of
// the engine contract lives in its own file. The state union is intentionally
// minimal — every transition has well-defined host semantics documented at
// the methods that drive it (see `Engine.pause`/`resume`/`dispose` in
// `./index.ts`).

export type EngineState =
  | 'uninitialized'
  | 'initializing'
  | 'ready'
  | 'paused'
  /** Unrecoverable init/runtime failure — GPU resources torn down; recreate the engine. */
  | 'error'
  | 'disposed';
