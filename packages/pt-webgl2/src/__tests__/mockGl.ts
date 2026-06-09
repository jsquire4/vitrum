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

export function createMockGl(): WebGL2RenderingContext {
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
    getUniformLocation: () => ({}),
    getUniformBlockIndex: () => 0,
    getActiveUniform: () => null,
    checkFramebufferStatus: () => enumOf('FRAMEBUFFER_COMPLETE'),
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
