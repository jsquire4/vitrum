import { describe, expect, it } from 'vitest';
import { convertAnimations, type GltfAnimationImportDiagnostic } from './animations.js';
import type { GltfJson } from './gltfTypes.js';

function unsupportedPointerFixture(): GltfJson {
  return {
    asset: { version: '2.0' },
    animations: [{
      name: 'unsupported-mesh-pointer',
      samplers: [{ input: 0, output: 1 }],
      channels: [{
        sampler: 0,
        target: {
          path: 'pointer',
          extensions: {
            KHR_animation_pointer: {
              pointer: '/meshes/0/weights',
            },
          },
        },
      }],
    }],
  };
}

describe('recoverable animation degradation', () => {
  it('skips an unsupported KHR_animation_pointer channel even when the diagnostic observer throws', () => {
    const warnings: string[] = [];
    const diagnostics: GltfAnimationImportDiagnostic[] = [];

    expect(() => {
      const clips = convertAnimations(
        unsupportedPointerFixture(),
        new Map(),
        warnings,
        (diagnostic) => {
          diagnostics.push(diagnostic);
          throw new Error('host diagnostic observer failed');
        },
      );
      expect(clips).toEqual([]);
    }).not.toThrow();

    expect(diagnostics).toEqual([
      expect.objectContaining({
        severity: 'warning',
        code: 'unsupported-animation-target-path',
        path:
          'animations[0].channels[0].target.extensions.' +
          'KHR_animation_pointer.pointer',
      }),
      expect.objectContaining({
        severity: 'warning',
        code: 'dropped-animation',
        path: 'animations[0]',
      }),
    ]);
    expect(warnings).toEqual([
      expect.stringContaining('/meshes/0/weights'),
      expect.stringContaining('has no importable channels and was skipped'),
    ]);
  });
});
