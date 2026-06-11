// A no-op WebGL2RenderingContext mock — rich enough to drive the full pt-webgl2
// pipeline (FullscreenQuad/program build, texture uploads, FBO setup, draw) WITHOUT
// a GPU. It verifies ENGINE ORCHESTRATION (the accumulation loop, convergence,
// FrameOutput shape, resource lifecycle) — NOT pixel correctness, which is the
// real-GPU capture-host A/B (plan 06). create* return fresh fake handles; status
// queries return the matching "complete/success" sentinel; everything else no-ops.

const REAL_ENUMS: Record<string, number> = {
  MAX_DRAW_BUFFERS: 0x8824,
  MAX_TEXTURE_IMAGE_UNITS: 0x8872,
  MAX_TEXTURE_SIZE: 0x0d33,
  MAX_ARRAY_TEXTURE_LAYERS: 0x88ff,
  COMPILE_STATUS: 0x8b81,
  LINK_STATUS: 0x8b82,
  ACTIVE_UNIFORMS: 0x8b86,
  INVALID_INDEX: 0xffffffff,
};

export interface MockGlOptions {
  /** Recording map for uniform-set tracking (upload-gap guard tests). */
  record?: Map<string, unknown>;
  /**
   * Override MAX_TEXTURE_SIZE returned by `getParameter`. Default: 16384.
   * Set to a small value (e.g. 4) to test the size-validation guards.
   */
  maxTexSize?: number;
  /**
   * Override MAX_ARRAY_TEXTURE_LAYERS returned by `getParameter`. Default: 256.
   * Set to a small value (e.g. 2) to test the array-layer guards.
   */
  maxArrayLayers?: number;
  /**
   * When true, `isContextLost()` returns true from the start — simulates an
   * already-lost context for testing the context-loss robustness paths.
   */
  contextLost?: boolean;
}

/**
 * @param recordOrOpts - Either a recording Map (legacy positional form for
 *   back-compat) or a `MockGlOptions` bag.
 *   - When a Map is supplied: name-tags every uniform location and records
 *     scalar/array uniform-set calls (upload-gap GUARD tests, §H H1–H3).
 *   - When `MockGlOptions` is supplied: full control over tex-size limits and
 *     context-loss simulation.
 *   - Omit for the default no-op mock (anonymous locations, 16384 tex limit).
 */
export function createMockGl(recordOrOpts?: Map<string, unknown> | MockGlOptions): WebGL2RenderingContext {
  // Normalise the overloaded argument.
  let record: Map<string, unknown> | undefined;
  let maxTexSizeOverride: number | undefined;
  let maxArrayLayersOverride: number | undefined;
  let contextLostOverride = false;
  if (recordOrOpts instanceof Map) {
    record = recordOrOpts;
  } else if (recordOrOpts != null) {
    record = recordOrOpts.record;
    maxTexSizeOverride = recordOrOpts.maxTexSize;
    maxArrayLayersOverride = recordOrOpts.maxArrayLayers;
    contextLostOverride = recordOrOpts.contextLost ?? false;
  }
  let nextEnum = 0x9000;
  const byName = new Map<string, number>(Object.entries(REAL_ENUMS));
  const byValue = new Map<number, string>();
  for (const [k, v] of byName) byValue.set(v, k);

  const enumOf = (name: string): number => {
    let v = byName.get(name);
    if (v == null) {
      v = nextEnum++;
      byName.set(name, v);
      byValue.set(v, name);
    }
    return v;
  };

  const nameOf = (loc: unknown): string | null =>
    loc != null && typeof loc === 'object' && '__u' in loc ? String((loc).__u) : null;
  const rec = (loc: unknown, value: unknown): void => {
    if (record == null) return;
    const n = nameOf(loc);
    if (n != null) record.set(n, value);
  };

  const handlers: Record<string, (...args: unknown[]) => unknown> = {
    getExtension: () => ({}),
    isContextLost: () => contextLostOverride,
    getParameter: (p) => {
      switch (byValue.get(p as number)) {
        case 'MAX_DRAW_BUFFERS': return 8;
        case 'MAX_TEXTURE_IMAGE_UNITS': return 32;
        case 'MAX_TEXTURE_SIZE': return maxTexSizeOverride ?? 16384;
        case 'MAX_ARRAY_TEXTURE_LAYERS': return maxArrayLayersOverride ?? 256;
        default: return 0;
      }
    },
    getProgramParameter: (_prog, pname) =>
      byValue.get(pname as number) === 'LINK_STATUS' ? true : 0,
    getShaderParameter: () => true,
    getProgramInfoLog: () => '',
    getShaderInfoLog: () => '',
    // Name-tag locations only when recording (so GlProgram's per-name #loc cache
    // carries the name through to the setters below); else opaque, as before.
    getUniformLocation: record == null ? () => ({}) : (_prog, name) => ({ __u: name }),
    getUniformBlockIndex: () => 0,
    getActiveUniform: () => null,
    checkFramebufferStatus: () => enumOf('FRAMEBUFFER_COMPLETE'),
    // Recording uniform setters (no-op when `record` is absent — the value is just
    // dropped, same as the catch-all). Cover the scalar + array forms the engine uses.
    uniform1ui: (loc, v) => rec(loc, v),
    uniform1i: (loc, v) => rec(loc, v),
    uniform1f: (loc, v) => rec(loc, v),
    uniform1fv: (loc, v) => rec(loc, v),
    // vec2/vec3 recorders (resolution, u_jakobCoeffs, …). Stored as a tuple array.
    uniform2f: (loc, x, y) => rec(loc, [x, y]),
    uniform3f: (loc, x, y, z) => rec(loc, [x, y, z]),
    // H6: matrix uniform recorder (environmentRotation + camera matrices).
    uniformMatrix4fv: (loc, _transpose, v) => rec(loc, v),
  };

  // Minimal mock canvas — supports addEventListener/removeEventListener so the
  // engine's context-loss listener wiring does not throw. `dispatchEvent` lets
  // tests fire synthetic webglcontextlost / webglcontextrestored events.
  const canvasListeners = new Map<string, Set<EventListenerOrEventListenerObject>>();
  const mockCanvas = {
    addEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
      let s = canvasListeners.get(type);
      if (s == null) { s = new Set(); canvasListeners.set(type, s); }
      s.add(listener);
    },
    removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
      canvasListeners.get(type)?.delete(listener);
    },
    /** Test helper: fire all registered listeners for `type` with `event`. */
    dispatchEvent(type: string, event: Event): void {
      for (const l of canvasListeners.get(type) ?? []) {
        if (typeof l === 'function') l(event);
        else l.handleEvent(event);
      }
    },
  };

  const target: Record<string, unknown> = { canvas: mockCanvas };
  return new Proxy(target, {
    get(t, prop): unknown {
      if (typeof prop !== 'string') return undefined;
      if (prop in t) return t[prop];
      if (prop in handlers) return handlers[prop];
      if (byName.has(prop)) return byName.get(prop);
      if (/^[A-Z][A-Z0-9_]*$/.test(prop)) return enumOf(prop); // a GL constant
      if (prop.startsWith('create')) return () => ({});        // fresh fake handle
      return () => undefined;                                  // any other GL call → no-op
    },
  }) as unknown as WebGL2RenderingContext;
}
