import { describe, expect, it } from 'vitest';
import { GlProgram, buildPreamble, buildVertexSource, buildFragmentSource } from './glProgram.js';

// ─────────────────────────────────────────────────────────────────────────────
// GPU-free source-assembly and parallel-link state-machine gate.
// Runtime feature changes build a staged replacement program graph; GlProgram
// defines are immutable construction inputs.
// ─────────────────────────────────────────────────────────────────────────────

describe('buildPreamble / source assembly', () => {
  it('emits #version 300 es then the #define lines in insertion order', () => {
    const pre = buildPreamble(new Map([['FEATURE_MIS', 1], ['RANDOM_TYPE', 2]]));
    expect(pre).toBe('#version 300 es\n#define FEATURE_MIS 1\n#define RANDOM_TYPE 2\n');
  });

  it('vertex source = preamble + body', () => {
    const defs = new Map([['CAMERA_TYPE', 0]]);
    const vert = buildVertexSource(defs, 'void main() {}');
    expect(vert.startsWith('#version 300 es\n#define CAMERA_TYPE 0\n')).toBe(true);
    expect(vert.endsWith('void main() {}')).toBe(true);
  });

  it('fragment source adds the THREE-injected precision + pc_fragColor header', () => {
    const defs = new Map([['FEATURE_FOG', 1]]);
    const frag = buildFragmentSource(defs, 'void main() {}');
    expect(frag).toContain('#version 300 es');
    expect(frag).toContain('#define FEATURE_FOG 1');
    expect(frag).toContain('precision highp float;');
    expect(frag).toContain('precision highp int;');
    expect(frag).toContain('layout(location = 0) out vec4 pc_fragColor;');
    // The header must come AFTER the preamble (defines) and BEFORE the body.
    expect(frag.indexOf('#define FEATURE_FOG 1')).toBeLessThan(frag.indexOf('pc_fragColor'));
    expect(frag.indexOf('pc_fragColor')).toBeLessThan(frag.indexOf('void main()'));
  });
});

function createParallelCompileMock(): {
  readonly gl: WebGL2RenderingContext;
  setComplete(value: boolean): void;
  readonly linkStatusQueries: () => number;
  readonly deletedPrograms: readonly WebGLProgram[];
} {
  const COMPLETION_STATUS_KHR = 0x91b1;
  let complete = false;
  let linkStatusQueries = 0;
  let nextId = 1;
  const deletedPrograms: WebGLProgram[] = [];
  const gl = {
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b30,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    ACTIVE_UNIFORMS: 0x8b86,
    createShader: () => ({ id: nextId++ }),
    createProgram: () => ({ id: nextId++ }),
    shaderSource: () => {},
    compileShader: () => {},
    attachShader: () => {},
    detachShader: () => {},
    linkProgram: () => {},
    deleteShader: () => {},
    deleteProgram: (program: WebGLProgram) => { deletedPrograms.push(program); },
    getExtension: (name: string) =>
      name === 'KHR_parallel_shader_compile' ? { COMPLETION_STATUS_KHR } : null,
    getShaderParameter: () => true,
    getShaderInfoLog: () => '',
    getProgramInfoLog: () => '',
    getProgramParameter: (_program: WebGLProgram, pname: number) => {
      if (pname === COMPLETION_STATUS_KHR) return complete;
      if (pname === 0x8b82) {
        linkStatusQueries += 1;
        return true;
      }
      if (pname === 0x8b86) return 0;
      return 0;
    },
    getActiveUniform: () => null,
    useProgram: () => {},
  } as unknown as WebGL2RenderingContext;
  return {
    gl,
    setComplete(value) { complete = value; },
    linkStatusQueries: () => linkStatusQueries,
    deletedPrograms,
  };
}

describe('GlProgram parallel link readiness', () => {
  it('never queries blocking LINK_STATUS until COMPLETION_STATUS_KHR is true', () => {
    const mock = createParallelCompileMock();
    const prog = new GlProgram(mock.gl, 'void main() {}', 'void main() {}', {});

    expect(prog.prepare()).toBe(false);
    expect(mock.linkStatusQueries()).toBe(0);

    mock.setComplete(true);
    expect(prog.prepare()).toBe(true);
    expect(mock.linkStatusQueries()).toBe(1);
    expect(prog.use()).toBe(true);
  });

  it('times out an incomplete link and recovers only after dispose', async () => {
    const mock = createParallelCompileMock();
    const prog = new GlProgram(
      mock.gl,
      'void main() {}',
      'void main() {}',
      {},
      { linkTimeoutMs: 1 },
    );
    expect(prog.prepare()).toBe(false);
    await new Promise((resolve) => setTimeout(resolve, 5));
    expect(() => prog.prepare()).toThrow(/did not complete within 1 ms/);
    expect(mock.deletedPrograms).toHaveLength(1);
    expect(() => prog.prepare()).toThrow(/did not complete within 1 ms/);

    prog.dispose();
    mock.setComplete(true);
    expect(prog.prepare()).toBe(true);
  });
});
