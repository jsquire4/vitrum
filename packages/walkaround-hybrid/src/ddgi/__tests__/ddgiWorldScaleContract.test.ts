import { describe, expect, it } from 'vitest';
import { PROBE_RAY_CAST_WGSL } from '@vitrum/walkaround-rc';
import {
  classifyAndRelocateProbe,
  DDGI_PROBE_MIN_HIT_DISTANCE_NORMALIZED,
  type ProbeClassificationRay,
} from '../probeState.js';
import {
  makeProbeUpdateBlendIrrWGSL,
  makeProbeUpdateBlendVisWGSL,
} from '../wgsl/probeUpdateBlend.wgsl.js';
import { PROBE_CLASSIFY_RELOCATE_WGSL } from '../wgsl/probeClassifyRelocate.wgsl.js';
import { makeProbeUpdateRaysWGSL } from '../wgsl/probeUpdateRays.wgsl.js';

function scaleRays(
  rays: readonly ProbeClassificationRay[],
  scale: number,
): ProbeClassificationRay[] {
  return rays.map(({ direction, hitDistance }) => ({
    direction,
    hitDistance:
      hitDistance >= 1.0e19 ? hitDistance : hitDistance * scale,
  }));
}

describe('DDGI world-scale contract', () => {
  it('classifies and relocates proportionally below five centimetres', () => {
    const normalizedRays: readonly ProbeClassificationRay[] = [
      { direction: [1.0e-30, 0, 0], hitDistance: 0.001 },
      { direction: [-1.0e-30, 0, 0], hitDistance: 1.0e20 },
    ];
    const unit = classifyAndRelocateProbe(
      [0, 0, 0],
      normalizedRays,
      1,
    );
    const tinySpacing = 0.01;
    const tiny = classifyAndRelocateProbe(
      [0, 0, 0],
      scaleRays(normalizedRays, tinySpacing),
      tinySpacing,
    );

    expect(tiny.active).toBe(unit.active);
    expect(tiny.validRayCount).toBe(unit.validRayCount);
    expect(tiny.backfaceCount).toBe(unit.backfaceCount);
    expect(tiny.offset.map((lane) => lane / tinySpacing)).toEqual(
      unit.offset,
    );
  });

  it('counts a finite hit beyond five percent of a one-centimetre cell', () => {
    const spacing = 0.01;
    const result = classifyAndRelocateProbe(
      [0, 0, 0],
      [
        {
          direction: [0, 1, 0],
          hitDistance:
            spacing * (DDGI_PROBE_MIN_HIT_DISTANCE_NORMALIZED + 0.01),
        },
      ],
      spacing,
    );
    expect(result.validRayCount).toBe(1);
  });

  it('derives blend validity, classification clearance, and traversal progress from spacing', () => {
    const irr = makeProbeUpdateBlendIrrWGSL();
    const vis = makeProbeUpdateBlendVisWGSL();
    const rays = makeProbeUpdateRaysWGSL(4);

    for (const blend of [irr, vis]) {
      expect(blend).toContain(
        'gridParams.spacing * DDGI_PROBE_MIN_HIT_DISTANCE_NORMALIZED',
      );
      expect(blend).not.toMatch(/ray\.hitDistance < 0\.05/);
    }
    expect(vis).toContain('gridParams.spacing * 16.0');
    expect(vis).not.toContain('max(1.0, gridParams.spacing * 16.0)');

    expect(PROBE_CLASSIFY_RELOCATE_WGSL).toContain(
      'gridParams.spacing * MIN_HIT_DISTANCE_NORMALIZED',
    );
    expect(PROBE_CLASSIFY_RELOCATE_WGSL).not.toContain(
      'max(MIN_HIT_DISTANCE',
    );
    expect(rays).toContain(
      'gridParams.spacing * normalizedDistance',
    );
    expect(rays).toContain(
      'ddgiProbeDistance(DDGI_TRACE_T_MIN_NORMALIZED)',
    );
    expect(rays).toContain(
      'ddgiProbeDistance(DDGI_SURFACE_STEP_NORMALIZED)',
    );
    expect(rays).toContain(
      'ddgiProbeDistance(DDGI_GLASS_BOUNDARY_STEP_NORMALIZED)',
    );
    expect(rays).not.toContain('const DDGI_TRI_EPSILON');
    expect(rays).not.toMatch(/max\(1e-4,\s*DDGI_/);
    expect(rays).not.toContain(
      'hitPos / max(gridParams.spacing, 1e-4)',
    );
    expect(rays).toContain('let dist = ddgiLengthOrZero(toL);');
    expect(rays).not.toContain(
      'let dist2   = max(dot(toL, toL), 1e-8);',
    );
  });

  it('uses max-component-equilibrated direction normalization in DDGI and RC', () => {
    const rays = makeProbeUpdateRaysWGSL(4);
    expect(rays).toContain(
      'let maxComponent = max(abs(v.x), max(abs(v.y), abs(v.z)));',
    );
    expect(rays).not.toContain('if (len2 < 1e-20)');
    expect(PROBE_CLASSIFY_RELOCATE_WGSL).toContain(
      'let maxComponent = max(abs(direction.x)',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let maxComponent = max(abs(v.x), max(abs(v.y), abs(v.z)));',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let scaledFallback = fallback / fallbackMaxComponent;',
    );
  });
});
