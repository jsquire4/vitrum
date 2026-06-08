import { describe, expect, it } from 'vitest';
import { isSoftwareGlRenderer } from '../legacy/three/bdpt/isSoftwareGlRenderer.js';

describe('isSoftwareGlRenderer', () => {
  it('detects SwiftShader ANGLE strings', () => {
    expect(
      isSoftwareGlRenderer(
        'ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)',
      ),
    ).toBe(true);
  });

  it('does not flag discrete GPU ANGLE strings', () => {
    expect(isSoftwareGlRenderer('ANGLE (NVIDIA, NVIDIA GeForce RTX 4090 Direct3D11 vs_5_0 ps_5_0)')).toBe(
      false,
    );
  });
});
