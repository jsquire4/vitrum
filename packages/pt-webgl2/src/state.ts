import type { EngineState } from '@vitrum/core';

/**
 * Backend-local state token (NOT a core type — copied from the pt-webgl/pt-webgpu
 * pattern). Keeps the `set('ready')` transition private to the factory: the engine
 * exposes only `get state()` reading `slot.get()`.
 */
export interface StateSlot {
  readonly get: () => EngineState;
  readonly set: (s: EngineState) => void;
}

export function makeStateSlot(initial: EngineState = 'initializing'): StateSlot {
  let s: EngineState = initial;
  return {
    get: () => s,
    set: (v) => {
      s = v;
    },
  };
}
