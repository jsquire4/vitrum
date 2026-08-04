import { describe, expect, it } from 'vitest';

import {
  PT_WEBGPU_TRACE_WGSL,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';

function acceptsSidedHit(
  frontFace: boolean,
  doubleSided: boolean,
  transmission: number,
): boolean {
  return frontFace || doubleSided || transmission > 0;
}

describe('pt-webgpu double-sided traversal contract', () => {
  it('culls only opaque one-sided back faces and preserves dielectric exits', () => {
    expect(acceptsSidedHit(true, false, 0)).toBe(true);
    expect(acceptsSidedHit(false, false, 0)).toBe(false);
    expect(acceptsSidedHit(false, true, 0)).toBe(true);
    expect(acceptsSidedHit(false, false, 1e-12)).toBe(true);
    expect(acceptsSidedHit(false, false, 1)).toBe(true);

    for (const wgsl of [PT_WEBGPU_TRACE_WGSL, PT_WEBGPU_TRACE_LITE_WGSL]) {
      expect(wgsl).toContain('doubleSided: bool,');
      expect(wgsl).toContain(
        'let materialFlagsDecoded = materialRecordExactU32(m26.w, 8u);',
      );
      expect(wgsl).toContain('mat.doubleSided = (materialFlags & 4u) != 0u;');
      expect(wgsl).toContain('return mat.doubleSided || mat.transmission > 0.0;');
      expect(wgsl).toContain('materialAcceptsSidedHit(matId, hit.frontFace)');
      expect(wgsl).toContain('!materialShadowCastDisabled(matId)');
      expect(wgsl).toContain('fn nextSidedTraversalCursor(cursor: f32, hitDist: f32) -> f32');
      expect(wgsl).not.toContain('MATERIAL_SIDED_TRAVERSAL_LIMIT');
    }
  });

  it('records geometric winding independently of authored shading normals', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('frontFace: bool,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'var shadeNormal = triHit.normal * triHit.side;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let frontFace = triHit.side > 0.0;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let orientationPreserving = transformLinearOrientationSign(l2w0, l2w1, l2w2) > 0.0;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      '(*hit).frontFace = select(!localHit.frontFace, localHit.frontFace, orientationPreserving);',
    );
  });

  it('keeps the CWBVH closest path on the same parity-correct sidedness contract', () => {
    const cwbvh = composePtWebgpuTraceWgsl(false, { cwbvhClosest: true });
    expect(cwbvh).toContain('(*hit).frontFace = exactHit.side > 0.0;');
    expect(cwbvh).toContain('fn traceClosestRaw(ray: Ray, tMin: f32, tMax: f32) -> SceneHit');
    expect(cwbvh).toContain('_ = traceTlasClosestCwbvh(ray, tMin, tMax, &hit);');
    expect(cwbvh).toContain(
      '(*hit).frontFace = select(!localHit.frontFace, localHit.frontFace, orientationPreserving);',
    );
    expect(cwbvh).toContain('materialAcceptsSidedHit(matId, hit.frontFace)');
  });

  it('uses the geometric side for every advanced transport path', () => {
    for (const source of [
      PT_WEBGPU_TRACE_WGSL,
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL,
      RESTIR_PT_PRODUCER_WGSL,
      SPPM_PHOTON_PASS_WGSL,
    ]) {
      expect(source).toContain('hit.frontFace');
      expect(source).not.toContain('dot(hit.normal, ray.direction) < 0.0');
      expect(source).not.toContain('dot(ray.direction, hit.normal) < 0.0');
    }
  });
});
