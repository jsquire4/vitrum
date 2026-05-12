/**
 * imports.test.ts — Smoke test: every export from @vitrum/dev loads without
 * throwing at import time.
 *
 * These tests do NOT render components (no JSDOM required) — they just verify
 * that each module can be imported and the exported values are defined.
 * This is the minimum quality gate: if any component has a top-level syntax
 * error or a bad import path, this test catches it immediately.
 */

import { describe, it, expect } from 'vitest';

describe('@vitrum/dev — import-time smoke tests', () => {
  it('index re-exports are importable', async () => {
    const mod = await import('../src/index.js');
    // Types are erased at runtime; verify runtime exports.
    expect(typeof mod.RingBuffer).toBe('function');
    expect(typeof mod.attachDebugOverlays).toBe('function');
    // React components are functions (React.FC is a function type).
    expect(typeof mod.FrameTimeHUD).toBe('function');
    expect(typeof mod.DDGIAtlasViewer).toBe('function');
    expect(typeof mod.BVHVisualizer).toBe('function');
    expect(typeof mod.GISignalSplit).toBe('function');
    expect(typeof mod.DenoiserABToggle).toBe('function');
    expect(typeof mod.MaterialInspector).toBe('function');
  });

  it('FrameTimeHUD module imports cleanly', async () => {
    const mod = await import('../src/react/FrameTimeHUD.js');
    expect(typeof mod.FrameTimeHUD).toBe('function');
    expect(typeof mod.RingBuffer).toBe('function');
  });

  it('DDGIAtlasViewer module imports cleanly', async () => {
    const mod = await import('../src/react/DDGIAtlasViewer.js');
    expect(typeof mod.DDGIAtlasViewer).toBe('function');
  });

  it('BVHVisualizer module imports cleanly', async () => {
    const mod = await import('../src/react/BVHVisualizer.js');
    expect(typeof mod.BVHVisualizer).toBe('function');
  });

  it('GISignalSplit module imports cleanly', async () => {
    const mod = await import('../src/react/GISignalSplit.js');
    expect(typeof mod.GISignalSplit).toBe('function');
  });

  it('DenoiserABToggle module imports cleanly', async () => {
    const mod = await import('../src/react/DenoiserABToggle.js');
    expect(typeof mod.DenoiserABToggle).toBe('function');
  });

  it('MaterialInspector module imports cleanly', async () => {
    const mod = await import('../src/react/MaterialInspector.js');
    expect(typeof mod.MaterialInspector).toBe('function');
  });

  it('vanilla attachDebugOverlays module imports cleanly', async () => {
    const mod = await import('../src/vanilla.js');
    expect(typeof mod.attachDebugOverlays).toBe('function');
  });
});
