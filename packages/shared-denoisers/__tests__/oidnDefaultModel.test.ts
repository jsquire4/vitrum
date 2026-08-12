import { describe, expect, it } from 'vitest';
import {
  DEFAULT_OIDN_RT_HDR_ALB_NRM_MODEL_URL,
  oidnModelUrlIsHostProvided,
  resolveOidnModelUrl,
} from '../src/oidnDefaultModel.js';

describe('OIDN default model URL', () => {
  it('returns the pinned alb+nrm URL when the host omits modelUrl', () => {
    expect(resolveOidnModelUrl(undefined)).toBe(DEFAULT_OIDN_RT_HDR_ALB_NRM_MODEL_URL);
    expect(resolveOidnModelUrl('')).toBe(DEFAULT_OIDN_RT_HDR_ALB_NRM_MODEL_URL);
    expect(resolveOidnModelUrl('   ')).toBe(DEFAULT_OIDN_RT_HDR_ALB_NRM_MODEL_URL);
    expect(oidnModelUrlIsHostProvided(undefined)).toBe(false);
  });

  it('keeps a host-provided URL', () => {
    expect(resolveOidnModelUrl('/models/custom.onnx')).toBe('/models/custom.onnx');
    expect(oidnModelUrlIsHostProvided('/models/custom.onnx')).toBe(true);
  });
});
