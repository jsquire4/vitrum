import { describe, expect, it } from 'vitest';
import { pickBackend } from '../createEngineScale.js';

describe('createEngine backend selection', () => {
  it('uses a glTF recommended backend when prefer is auto', () => {
    expect(pickBackend('auto', true, 12, false, 'pt-webgpu')).toBe('pt-webgpu');
    expect(pickBackend('auto', true, 1_000_000, false, 'walkaround-hybrid')).toBe('walkaround-hybrid');
    expect(pickBackend('auto', true, 12, false, 'pt-webgl2')).toBe('pt-webgl2');
  });

  it('keeps explicit preference stronger than the glTF recommendation', () => {
    expect(pickBackend('quality', true, 12, false, 'walkaround-hybrid')).toBe('pt-webgl2');
    expect(pickBackend('realtime', true, 1_000_000, false, 'pt-webgpu')).toBe('walkaround-hybrid');
  });

  it('falls back to pt-webgl2 for WebGPU recommendations on WebGL-only hosts', () => {
    expect(pickBackend('auto', false, 12, false, 'pt-webgpu')).toBe('pt-webgl2');
    expect(pickBackend('auto', false, 12, false, 'walkaround-hybrid')).toBe('pt-webgl2');
  });
});
