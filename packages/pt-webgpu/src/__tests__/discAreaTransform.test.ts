import { describe, expect, it } from 'vitest';
import { discArea } from '../bdpt/flatEmitterWalk.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_WGSL } from '../wgsl/pathTrace/connect.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL } from '../wgsl/pathTrace/connectLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL } from '../wgsl/pathTrace/kernelLite.wgsl.js';
import { PT_WEBGPU_MEDIUM_NEE_WGSL } from '../wgsl/pathTrace/mediumNee.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';

describe('affine transformed disc-area measure', () => {
  it('uses the affine unit-disc Jacobian for nonuniform and sheared axes', () => {
    const u: [number, number, number] = [2, 0, 0];
    const v: [number, number, number] = [1, 3, 0];

    // |u×v| = 6. The former π|u|² formula gives 4π, while the also-stale
    // π|u||v| formula gives 2π√10; this fixture distinguishes all three.
    expect(discArea(u, v)).toBeCloseTo(6 * Math.PI, 12);
    expect(discArea(u, v)).not.toBeCloseTo(4 * Math.PI, 12);
    expect(discArea(u, v)).not.toBeCloseTo(2 * Math.sqrt(10) * Math.PI, 12);
  });

  it('pins every pt-webgpu disc sampling and connection branch to π|u×v|', () => {
    const sites: ReadonlyArray<readonly [string, string, string]> = [
      ['BDPT light subpath', PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL, 'areaS = PI * length(cross(ru, rv));'],
      ['MNEE caustic emitter', PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL, 'out.area = PI * length(cross(u, v));'],
      ['full BSDF connection', PT_WEBGPU_PATH_TRACE_CONNECT_WGSL, 'PI * sqrt(axisCrossLen2),'],
      ['lite BSDF connection', PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL, 'PI * sqrt(axisCrossLen2),'],
      ['full NEE', PT_WEBGPU_PATH_TRACE_KERNEL_WGSL, 'area = PI * length(cross(ru, rv));'],
      ['lite NEE', PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL, 'area = PI * length(cross(ru, rv));'],
      ['medium NEE', PT_WEBGPU_MEDIUM_NEE_WGSL, 'area = PI * length(cross(uAxis, vAxis));'],
      ['one-edge suffix NEE', RESTIR_PT_PRODUCER_WGSL, 'area = PI * length(cross(ru, rv));'],
      ['SPPM photon emission', SPPM_PHOTON_PASS_WGSL, 'area = PI * normalLen;'],
    ];

    for (const [label, source, expected] of sites) {
      expect(source, label).toContain(expected);
    }

    const allSources = sites.map(([, source]) => source).join('\n');
    expect(allSources).not.toMatch(/PI\s*\*\s*dot\((?:uAxis|ru),\s*(?:uAxis|ru)\)/);
    expect(allSources).not.toMatch(/PI\s*\*\s*length\(u\)\s*\*\s*length\(v\)/);
    expect(allSources).not.toMatch(/PI\s*\*\s*(?:r|rrad|rradL)\s*\*\s*(?:r|rrad|rradL)/);
  });

  it('solves affine light coordinates with the full Gram matrix', () => {
    const expected = [
      'let uv = dot(uAxis, vAxis);',
      'let uCoord = (relU * vLen2 - relV * uv) / axisCrossLen2;',
      'let vCoord = (relV * uLen2 - relU * uv) / axisCrossLen2;',
    ];
    for (const source of [
      PT_WEBGPU_PATH_TRACE_CONNECT_WGSL,
      PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL,
    ]) {
      for (const line of expected) expect(source).toContain(line);
      expect(source).not.toContain('dot(rel, uAxis) / uLen2');
      expect(source).not.toContain('dot(rel, vAxis) / vLen2');
    }

    // Recover q=(0.25,-0.5) from p=q.x*u+q.y*v in a sheared basis.
    const u = [2, 0, 0] as const;
    const v = [1, 3, 0] as const;
    const p = [
      0.25 * u[0] - 0.5 * v[0],
      0.25 * u[1] - 0.5 * v[1],
      0.25 * u[2] - 0.5 * v[2],
    ] as const;
    const uu = u[0] ** 2 + u[1] ** 2 + u[2] ** 2;
    const uv = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const vv = v[0] ** 2 + v[1] ** 2 + v[2] ** 2;
    const pu = p[0] * u[0] + p[1] * u[1] + p[2] * u[2];
    const pv = p[0] * v[0] + p[1] * v[1] + p[2] * v[2];
    const determinant = uu * vv - uv * uv;

    expect((pu * vv - pv * uv) / determinant).toBeCloseTo(0.25, 12);
    expect((pv * uu - pu * uv) / determinant).toBeCloseTo(-0.5, 12);
  });
});
