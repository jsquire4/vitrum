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
  MAX_TEXTURE_IMAGE_UNITS: 0x8872,
  MAX_TEXTURE_SIZE: 0x0d33,
} as const;

interface StubOpts {
  drawBuffers?: number;
  texUnits?: number;
  maxTexSize?: number;
  floatColor?: boolean;
}

/** Build a stub that quacks like the slice of WebGL2RenderingContext the gate touches. */
function stubGl(opts: StubOpts = {}): WebGL2RenderingContext {
  const {
    drawBuffers = 8,
    texUnits = 16,
    maxTexSize = 16384,
    floatColor = true,
  } = opts;
  const gl = {
    MAX_DRAW_BUFFERS: GL.MAX_DRAW_BUFFERS,
    MAX_TEXTURE_IMAGE_UNITS: GL.MAX_TEXTURE_IMAGE_UNITS,
    MAX_TEXTURE_SIZE: GL.MAX_TEXTURE_SIZE,
    getExtension(name: string): object | null {
      return name === 'EXT_color_buffer_float' && floatColor ? {} : null;
    },
    getParameter(pname: number): number {
      switch (pname) {
        case GL.MAX_DRAW_BUFFERS:
          return drawBuffers;
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
  it("returns 'full' when all limits meet the full-tier minimums", () => {
    expect(selectWebGl2TraceTier(stubGl())).toBe('full');
    // exactly at the thresholds
    expect(
      selectWebGl2TraceTier(stubGl({ drawBuffers: 3, texUnits: 12, maxTexSize: 8192 })),
    ).toBe('full');
  });

  it("returns 'lite' when MAX_DRAW_BUFFERS is below 3", () => {
    expect(selectWebGl2TraceTier(stubGl({ drawBuffers: 1 }))).toBe('lite');
  });

  it("returns 'lite' when MAX_TEXTURE_IMAGE_UNITS is below 12", () => {
    expect(selectWebGl2TraceTier(stubGl({ texUnits: 8 }))).toBe('lite');
  });

  it("returns 'lite' when MAX_TEXTURE_SIZE is below 8192", () => {
    expect(selectWebGl2TraceTier(stubGl({ maxTexSize: 4096 }))).toBe('lite');
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
    expect(resolveWebGl2TraceTier(stubGl({ drawBuffers: 1 }))).toBe('lite');
  });

  it("force: 'lite' always returns lite even on a full-capable context", () => {
    const forced: WebGl2TraceTier = 'lite';
    expect(resolveWebGl2TraceTier(stubGl(), forced)).toBe('lite');
  });

  it("force: 'full' succeeds when the context supports it", () => {
    expect(resolveWebGl2TraceTier(stubGl(), 'full')).toBe('full');
  });

  it("force: 'full' throws when the context cannot meet full-tier limits", () => {
    expect(() => resolveWebGl2TraceTier(stubGl({ drawBuffers: 1 }), 'full')).toThrow(
      /traceTier=full requested but context reports/,
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

  it('documents lite tier as aux-buffer-only degradation, not a hidden trace-kernel cap', () => {
    const traceTierSource = readFileSync(new URL('./traceTier.ts', import.meta.url), 'utf8');
    const readme = readFileSync(new URL('../README.md', import.meta.url), 'utf8');

    expect(traceTierSource).toContain('The path-tracing kernel itself');
    expect(traceTierSource).toContain('keeps the same bounce count');
    expect(traceTierSource).toContain('optional BSDF lobes');
    expect(traceTierSource).toContain('only aux-buffer products are');
    expect(readme).toContain('The path-tracing kernel');
    expect(readme).toContain('runs **unchanged**');
  });
});
