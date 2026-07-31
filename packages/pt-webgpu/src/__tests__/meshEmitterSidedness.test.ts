import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_CONNECT_WGSL } from '../wgsl/pathTrace/connect.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_MEDIUM_NEE_WGSL } from '../wgsl/pathTrace/mediumNee.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';

describe('pt-webgpu mesh-emitter sidedness closure', () => {
  it('uses the packed material flag in direct, continuation, volume, and ReSTIR estimators', () => {
    expect(PT_WEBGPU_PATH_TRACE_CONNECT_WGSL).toContain(
      'fn meshAreaLightIsTwoSided(index: u32) -> bool',
    );
    expect(PT_WEBGPU_PATH_TRACE_CONNECT_WGSL).toContain(
      'meshAreaLightCosineTowardReceiver(',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'meshAreaLightCosineTowardReceiver(',
    );
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      'meshAreaLightCosineTowardReceiver(',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'meshAreaLightCosineTowardReceiver(',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'out.emissionAdmitted = isFrontFace || mat.doubleSided;',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'if (!prevAllowsAreaMis && sm.emissionAdmitted) {',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      '(isFrontFace || mat.doubleSided)',
    );
  });

  it('carries the same flag through MNEE, BDPT launch/MIS, and SPPM photons', () => {
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'out.twoSided = select(0u, 1u, meshAreaLightIsTwoSided(index));',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'select(0.0, 1.0, meshAreaLightIsTwoSided(mi))',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'pdfScatter = hemi.pdf * select(1.0, 0.5, twoSidedEmitter);',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'lvMatId == BDPT_LV_AREA_EMITTER_MATID && lv4.y > 0.5',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'abs(dot(lightNormal, lcToE)) * 0.5 * INV_PI',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'sidedPowerScale = 2.0;',
    );
  });
});
