// Type-shape + import smoke for the lifecycle helpers.
//
// Full attachVitrum() behaviour (RAF loop, ResizeObserver, visibility
// pause, engine dispose) requires a DOM + WebGPU/WebGL device, which the
// node test environment doesn't provide. That coverage lives in the
// shader-compile-ci end-to-end smoke + the cornell-box example.
//
// What we CAN test here without a browser:
//   - The module imports without throwing.
//   - The public type surface is what the README documents.

import { describe, it, expect } from 'vitest';

describe('@vitrum/engine/lifecycle', () => {
  it('exports attachVitrum + types from the main entry', async () => {
    const mod = await import('../src/index.js');
    expect(typeof mod.attachVitrum).toBe('function');
  });

  it('exports attachVitrum + types from the dedicated entry', async () => {
    const mod = await import('../src/lifecycle/index.js');
    expect(typeof mod.attachVitrum).toBe('function');
  });

  it('the React entry exports VitrumCanvas (H54 — tsx transform wired in vitest.config.ts)', async () => {
    // vitest.config.ts now sets esbuild: { jsx: 'automatic' } so .tsx files
    // transform correctly. Verify the export resolves at runtime, not just
    // at file-system level.
    const reactEntry = await import('../src/react/index.js');
    expect(typeof reactEntry.VitrumCanvas).toBe('object'); // React.forwardRef returns an object
  });
});
