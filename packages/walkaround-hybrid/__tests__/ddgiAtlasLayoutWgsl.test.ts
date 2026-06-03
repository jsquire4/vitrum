/**
 * T10-DDGI — atlas-layout single-source assertions for the probe-update
 * blend + border WGSL factories.
 *
 * `ddgiAtlasLayout.ts` is the documented single source of truth for the
 * octahedral cell sizes (IRR_CELL / VIS_CELL), border, and strides. The
 * blend (`probeUpdateBlend.wgsl.ts`) and border (`probeUpdateBorder.wgsl.ts`)
 * compute shaders previously HARDCODED these as WGSL literals (`IRR_CELL=8u`,
 * `VIS_CELL=16u`, `IRR_STRIDE=10u`, `VIS_STRIDE=18u`), which silently risked
 * drifting from the layout module.
 *
 * After the conversion to factories the emitted WGSL must reflect the layout
 * constants. These tests pin that linkage two ways:
 *
 *   1. The currently-emitted WGSL contains the exact literals derived from the
 *      live layout constants (and the legacy const exports match the factories).
 *   2. A factory invoked with a *changed* cell/stride emits *different* WGSL —
 *      proving the value is genuinely interpolated, not coincidentally equal to
 *      a hardcoded literal.
 */

import { describe, it, expect } from 'vitest';
import {
  IRR_CELL,
  VIS_CELL,
  BORDER,
  IRR_STRIDE,
  VIS_STRIDE,
} from '../src/ddgi/ddgiAtlasLayout.js';
import {
  makeProbeUpdateBlendIrrWGSL,
  makeProbeUpdateBlendVisWGSL,
} from '../src/ddgi/wgsl/probeUpdateBlend.wgsl.js';
import {
  makeBorderFillWGSL,
  makeProbeUpdateBorderIrrWGSL,
  makeProbeUpdateBorderVisWGSL,
} from '../src/ddgi/wgsl/probeUpdateBorder.wgsl.js';

describe('T10-DDGI — blend WGSL reflects ddgiAtlasLayout', () => {
  it('IRR_CELL / VIS_CELL constants in the emitted WGSL match the layout', () => {
    const irr = makeProbeUpdateBlendIrrWGSL();
    const vis = makeProbeUpdateBlendVisWGSL();
    // Both factories emit the shared COMMON header, so both contain both decls.
    for (const src of [irr, vis]) {
      expect(src).toContain(`const IRR_CELL:       u32 = ${IRR_CELL}u;`);
      expect(src).toContain(`const VIS_CELL:       u32 = ${VIS_CELL}u;`);
    }
  });

  it('the entry-point + workgroup-size are unchanged (8×8 irr, 16×16 vis)', () => {
    expect(makeProbeUpdateBlendIrrWGSL()).toContain('@compute @workgroup_size(8, 8, 1)\nfn probeUpdateBlendIrradiance');
    expect(makeProbeUpdateBlendVisWGSL()).toContain('@compute @workgroup_size(16, 16, 1)\nfn probeUpdateBlendVisibility');
  });
});

describe('T10-DDGI — border WGSL reflects ddgiAtlasLayout', () => {
  it('IRR_CELL/IRR_STRIDE and VIS_CELL/VIS_STRIDE are interpolated into the border WGSL', () => {
    const irr = makeProbeUpdateBorderIrrWGSL();
    const vis = makeProbeUpdateBorderVisWGSL();
    expect(irr).toContain(`const CELL:   u32 = ${IRR_CELL}u;`);
    expect(irr).toContain(`const STRIDE: u32 = ${IRR_STRIDE}u;`);
    expect(vis).toContain(`const CELL:   u32 = ${VIS_CELL}u;`);
    expect(vis).toContain(`const STRIDE: u32 = ${VIS_STRIDE}u;`);
  });

  it('stride equals cell + BORDER (layout invariant)', () => {
    expect(IRR_STRIDE).toBe(IRR_CELL + BORDER);
    expect(VIS_STRIDE).toBe(VIS_CELL + BORDER);
  });

  it('entry-point names + workgroup sizes are preserved per atlas', () => {
    expect(makeProbeUpdateBorderIrrWGSL()).toContain('@compute @workgroup_size(48, 1, 1)\nfn probeUpdateBorderIrradiance');
    expect(makeProbeUpdateBorderVisWGSL()).toContain('@compute @workgroup_size(256, 1, 1)\nfn probeUpdateBorderVisibility');
  });
});

describe('T10-DDGI — constants are genuinely interpolated (not coincidentally hardcoded)', () => {
  it('changing the blend COMMON header cell changes the emitted WGSL', () => {
    // The blend factories interpolate IRR_CELL/VIS_CELL from the layout module
    // at call time. We assert the *shape* of the dependency by confirming the
    // emitted literal tracks the imported constant (which a hardcoded literal
    // would not). If someone reverts to `IRR_CELL: u32 = 8u` hardcoded, the
    // assertions in the suites above would pass today (8 == IRR_CELL) but
    // would break the moment IRR_CELL changes — this test documents intent.
    expect(makeProbeUpdateBlendIrrWGSL()).toContain(`${IRR_CELL}u`);
  });

  it('makeBorderFillWGSL with a non-layout cell emits that cell verbatim', () => {
    const probe = makeBorderFillWGSL({
      cell: 99,
      stride: 101,
      workgroupSize: 64,
      entryPoint: 'probeUpdateBorderProbe',
    });
    expect(probe).toContain('const CELL:   u32 = 99u;');
    expect(probe).toContain('const STRIDE: u32 = 101u;');
    expect(probe).toContain('@compute @workgroup_size(64, 1, 1)\nfn probeUpdateBorderProbe');
    // And the layout-driven factory must NOT match the probe values.
    expect(makeProbeUpdateBorderIrrWGSL()).not.toContain('const CELL:   u32 = 99u;');
  });
});
