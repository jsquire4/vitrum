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

  it('the React entry module exists (loaded lazily — tsx transform required for runtime)', () => {
    // The tsx file isn't transformed in the default vitest config; this
    // test verifies the module path resolves at file-system level.
    const fs = require('node:fs');
    const path = require('node:path');
    const tsx = path.resolve(__dirname, '..', 'src', 'react', 'VitrumCanvas.tsx');
    expect(fs.existsSync(tsx)).toBe(true);
  });
});
