import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';
import {
  selectWebGl2TraceTier,
  resolveWebGl2TraceTier,
  type WebGl2TraceTier,
} from './traceTier.js';

// Minimal WebGL2 enum constants the gate reads (real values from the spec).
const GL = {
  MAX_DRAW_BUFFERS: 0x8824,
  MAX_COLOR_ATTACHMENTS: 0x8cdf,
  MAX_TEXTURE_IMAGE_UNITS: 0x8872,
  MAX_TEXTURE_SIZE: 0x0d33,
} as const;

interface StubOpts {
  drawBuffers?: number;
  colorAttachments?: number;
  texUnits?: number;
  maxTexSize?: number;
  floatColor?: boolean;
}

/** Build a stub that quacks like the slice of WebGL2RenderingContext the gate touches. */
function stubGl(opts: StubOpts = {}): WebGL2RenderingContext {
  const {
    drawBuffers = 8,
    colorAttachments = 8,
    texUnits = 16,
    maxTexSize = 16384,
    floatColor = true,
  } = opts;
  const gl = {
    MAX_DRAW_BUFFERS: GL.MAX_DRAW_BUFFERS,
    MAX_COLOR_ATTACHMENTS: GL.MAX_COLOR_ATTACHMENTS,
    MAX_TEXTURE_IMAGE_UNITS: GL.MAX_TEXTURE_IMAGE_UNITS,
    MAX_TEXTURE_SIZE: GL.MAX_TEXTURE_SIZE,
    getExtension(name: string): object | null {
      return name === 'EXT_color_buffer_float' && floatColor ? {} : null;
    },
    getParameter(pname: number): number {
      switch (pname) {
        case GL.MAX_DRAW_BUFFERS:
          return drawBuffers;
        case GL.MAX_COLOR_ATTACHMENTS:
          return colorAttachments;
        case GL.MAX_TEXTURE_IMAGE_UNITS:
          return texUnits;
        case GL.MAX_TEXTURE_SIZE:
          return maxTexSize;
        default:
          return 0;
      }
    },
  };
  return gl as unknown as WebGL2RenderingContext;
}

describe('selectWebGl2TraceTier', () => {
  it("returns 'full' when the shared NEE framebuffer requirements are met", () => {
    expect(selectWebGl2TraceTier(stubGl())).toBe('full');
    expect(
      selectWebGl2TraceTier(stubGl({
        drawBuffers: 4,
        colorAttachments: 4,
        texUnits: 16,
        maxTexSize: 2_048,
      })),
    ).toBe('full');
  });

  it('fails closed when either shared NEE framebuffer limit is too small', () => {
    expect(() => selectWebGl2TraceTier(stubGl({ drawBuffers: 3 }))).toThrow(
      /requires at least 4 draw buffers, 4 color attachments/,
    );
    expect(() =>
      selectWebGl2TraceTier(stubGl({ colorAttachments: 3 })),
    ).toThrow(/requires at least 4 draw buffers, 4 color attachments/);
    expect(() => selectWebGl2TraceTier(stubGl({ texUnits: 15 }))).toThrow(
      /16 fragment texture units/,
    );
  });

  it('does not use scene-specific texture dimensions as an output-tier switch', () => {
    expect(selectWebGl2TraceTier(stubGl({ maxTexSize: 2_048 }))).toBe('full');
  });

  it('throws when EXT_color_buffer_float is absent', () => {
    expect(() => selectWebGl2TraceTier(stubGl({ floatColor: false }))).toThrow(
      /EXT_color_buffer_float required/,
    );
  });
});

describe('resolveWebGl2TraceTier', () => {
  it('auto-selects when no force is given', () => {
    expect(resolveWebGl2TraceTier(stubGl())).toBe('full');
  });

  it("force: 'lite' always returns lite even on a full-capable context", () => {
    const forced: WebGl2TraceTier = 'lite';
    expect(resolveWebGl2TraceTier(stubGl(), forced)).toBe('lite');
  });

  it("force: 'full' succeeds when the context supports it", () => {
    expect(resolveWebGl2TraceTier(stubGl(), 'full')).toBe('full');
  });

  it('both explicit profiles reject a context below the shared kernel floor', () => {
    expect(() => resolveWebGl2TraceTier(stubGl({ drawBuffers: 3 }), 'full')).toThrow(
      /requires at least 4 draw buffers/,
    );
    expect(() => resolveWebGl2TraceTier(stubGl({ drawBuffers: 3 }), 'lite')).toThrow(
      /requires at least 4 draw buffers/,
    );
  });

  it('throws (even with force) when EXT_color_buffer_float is absent', () => {
    expect(() => resolveWebGl2TraceTier(stubGl({ floatColor: false }), 'lite')).toThrow(
      /EXT_color_buffer_float required/,
    );
  });
});

describe('Road D9 trace-tier contract', () => {
  it('keeps WebGl2TraceTier owned by traceTier.ts and re-exported from options.ts', () => {
    const traceTierSource = readFileSync(new URL('./traceTier.ts', import.meta.url), 'utf8');
    const optionsSource = readFileSync(new URL('./options.ts', import.meta.url), 'utf8');

    expect(traceTierSource).toContain("export type WebGl2TraceTier = 'full' | 'lite'");
    expect(optionsSource).toContain("import type { WebGl2TraceTier } from './traceTier.js'");
    expect(optionsSource).toContain('export type { WebGl2TraceTier }');
    expect(optionsSource).not.toContain("type WebGl2TraceTier = 'full' | 'lite'");
  });

  it('documents lite as an explicit aux-output profile, not a hardware-limit bypass', () => {
    const traceTierSource = readFileSync(new URL('./traceTier.ts', import.meta.url), 'utf8');
    const readme = readFileSync(new URL('../README.md', import.meta.url), 'utf8');

    expect(traceTierSource).toContain('The path-tracing kernel itself');
    expect(traceTierSource).toContain('keeps the same bounce count');
    expect(traceTierSource).toContain('optional BSDF lobes');
    expect(traceTierSource).toContain('only aux-buffer products are');
    expect(traceTierSource).toContain('explicit lower-memory/output profile');
    expect(readme).toContain('The path-tracing kernel');
    expect(readme).toContain('runs **unchanged**');
  });
});
