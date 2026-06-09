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
  COMPILE_STATUS: 0x8b81,
  LINK_STATUS: 0x8b82,
  ACTIVE_UNIFORMS: 0x8b86,
  INVALID_INDEX: 0xffffffff,
};

/**
 * @param record - When supplied, the mock name-tags every uniform location
 *   (`getUniformLocation` returns `{ __u: name }`) and records each scalar/array
 *   uniform-set call into the map keyed by uniform name. This is what makes the
 *   upload-gap GUARD test (uploadGapGuard.test.ts) possible — the default no-arg
 *   mock returns anonymous `{}` locations and swallows the setters, so it is BLIND
 *   to "was `lights.count` / the CMF tables / `backgroundAlpha` ever uploaded?"
 *   (the §H H1–H3 bug class). Omit `record` to keep the original no-op behaviour.
 */
export function createMockGl(record?: Map<string, unknown>): WebGL2RenderingContext {
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
    loc != null && typeof loc === 'object' && '__u' in loc ? String((loc as { __u: unknown }).__u) : null;
  const rec = (loc: unknown, value: unknown): void => {
    if (record == null) return;
    const n = nameOf(loc);
    if (n != null) record.set(n, value);
  };

  const handlers: Record<string, (...args: unknown[]) => unknown> = {
    getExtension: () => ({}),
    getParameter: (p) => {
      switch (byValue.get(p as number)) {
        case 'MAX_DRAW_BUFFERS': return 8;
        case 'MAX_TEXTURE_IMAGE_UNITS': return 32;
        case 'MAX_TEXTURE_SIZE': return 16384;
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
  };

  const target: Record<string, unknown> = {};
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
