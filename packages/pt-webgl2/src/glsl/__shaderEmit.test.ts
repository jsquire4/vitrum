import { writeFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { composeTraceGlsl } from './composeTraceGlsl.js';
import { DEFAULT_TRACE_FEATURES, featureDefines } from '../featureTypes.js';
import { buildFragmentSource, buildVertexSource } from '../gl/glProgram.js';
import { FULLSCREEN_VERT } from '../gl/fullscreenQuad.js';

// Dev utility (run via vitest): emit the EXACT vertex + fragment GLSL the engine's
// GlProgram would compile, so a browser probe can compile it on a real GL driver
// (llvmpipe) — the validation the no-GPU mock cannot do. Writes to /tmp.
describe('emit composed shaders for real-driver compile probe', () => {
  it('writes the diffuse-default vert+frag to /tmp', () => {
    const f = DEFAULT_TRACE_FEATURES;
    const defines = new Map(Object.entries(featureDefines(f)));
    // The EXACT sources GlProgram compiles (incl. the THREE-compat GLSL3 bridges).
    const frag = buildFragmentSource(defines, composeTraceGlsl(f));
    const vert = buildVertexSource(defines, FULLSCREEN_VERT);
    writeFileSync('/tmp/ptwebgl2-frag.glsl', frag);
    writeFileSync('/tmp/ptwebgl2-vert.glsl', vert);
    expect(frag.length).toBeGreaterThan(5000);
    expect(vert.length).toBeGreaterThan(20);
  });
});
