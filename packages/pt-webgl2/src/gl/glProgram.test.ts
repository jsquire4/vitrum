import { describe, expect, it } from 'vitest';
import { GlProgram, buildPreamble, buildVertexSource, buildFragmentSource } from './glProgram.js';

// ─────────────────────────────────────────────────────────────────────────────
// GPU-FREE gate. We never call use()/relink (which would need a real GL context),
// so the only GL the GlProgram constructor touches is none — it just stores state.
// We verify:
//   1. setDefine is CHANGE-GATED (returns true only on an actual value change), the
//      property that prevents per-frame recompiles (which reset the accumulator).
//   2. The built preamble carries '#version 300 es' + the expected '#define' lines,
//      and the fragment source adds the THREE-injected header (pc_fragColor @ loc 0).
// ─────────────────────────────────────────────────────────────────────────────

// A minimal stand-in for WebGL2RenderingContext — the constructor never dereferences
// it (relink is the only consumer, and we don't call use()). Cast through unknown.
const fakeGl = {} as unknown as WebGL2RenderingContext;

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

describe('GlProgram.setDefine change-gating', () => {
  it('returns false (no-op) when set to the value it was constructed with', () => {
    const prog = new GlProgram(fakeGl, 'V', 'F', { FEATURE_MIS: 1 });
    expect(prog.setDefine('FEATURE_MIS', 1)).toBe(false);
  });

  it('returns true on an actual change, false on a repeat of the new value', () => {
    const prog = new GlProgram(fakeGl, 'V', 'F', { FEATURE_MIS: 1 });
    expect(prog.setDefine('FEATURE_MIS', 0)).toBe(true); // 1 → 0 is a real change
    expect(prog.setDefine('FEATURE_MIS', 0)).toBe(false); // 0 → 0 is a no-op
  });

  it('returns true for a brand-new define name', () => {
    const prog = new GlProgram(fakeGl, 'V', 'F', {});
    expect(prog.setDefine('CAMERA_TYPE', 2)).toBe(true);
    expect(prog.setDefine('CAMERA_TYPE', 2)).toBe(false);
  });

  it('does not relink on construction (program is null until first use)', () => {
    const prog = new GlProgram(fakeGl, 'V', 'F', { FEATURE_MIS: 1 });
    expect(prog.program).toBeNull();
  });

  it('an unchanged define never marks dirty — samplerUnit/program stay untouched', () => {
    const prog = new GlProgram(fakeGl, 'V', 'F', { RANDOM_TYPE: 2 });
    // Repeated no-op sets must not throw (no relink path is entered).
    for (let i = 0; i < 5; i += 1) expect(prog.setDefine('RANDOM_TYPE', 2)).toBe(false);
    expect(prog.samplerUnit('anything')).toBeNull();
  });
});
