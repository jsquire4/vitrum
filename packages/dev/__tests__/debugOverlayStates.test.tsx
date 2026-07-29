import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

const startGpuTextureBlit = vi.hoisted(() =>
  vi.fn(() => vi.fn()),
);

vi.mock('../src/react/gpuTextureBlit.js', () => ({
  startGpuTextureBlit,
}));

import { BVHVisualizer } from '../src/react/BVHVisualizer.js';
import { DDGIAtlasViewer } from '../src/react/DDGIAtlasViewer.js';
import { GISignalSplit } from '../src/react/GISignalSplit.js';
import type { DebuggableEngine } from '../src/types.js';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let container: HTMLDivElement | null = null;

afterEach(() => {
  if (root != null) act(() => root!.unmount());
  container?.remove();
  root = null;
  container = null;
  vi.restoreAllMocks();
  startGpuTextureBlit.mockClear();
});

function mount(element: React.ReactElement): HTMLDivElement {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root!.render(element));
  return container;
}

function engineWithDebug(
  debug: NonNullable<DebuggableEngine['debug']>,
): DebuggableEngine {
  return { debug } as unknown as DebuggableEngine;
}

describe('debug overlay unavailable and resource-identity states', () => {
  it('does not label a null DDGI atlas as a live readback', () => {
    const host = mount(<DDGIAtlasViewer engine={engineWithDebug({
      device: () => ({}) as GPUDevice,
      atlasTexture: () => null,
      visibilityAtlasTexture: () => null,
    })} />);

    expect(host.textContent).toContain('disabled or has not been initialized');
    expect(host.textContent).not.toContain('Live readback');
    expect(startGpuTextureBlit).not.toHaveBeenCalled();
  });

  it('distinguishes unavailable and malformed BVH debug tables', () => {
    const host = mount(<BVHVisualizer
      engine={engineWithDebug({ bvhNodes: () => null })}
      initiallyVisible
      toggleKey={null}
    />);
    expect(host.textContent).toContain('BVH data unavailable');

    act(() => root!.render(<BVHVisualizer
      engine={engineWithDebug({ bvhNodes: () => new Float32Array([1]) })}
      initiallyVisible
      toggleKey={null}
    />));
    expect(host.textContent).toContain('Malformed BVH debug data');
  });

  it('does not restart GI readbacks when only the returned wrapper object changes', () => {
    const device = {} as GPUDevice;
    const direct = { label: 'direct' } as unknown as GPUTexture;
    const indirect = { label: 'indirect' } as unknown as GPUTexture;
    const ao = { label: 'ao' } as unknown as GPUTexture;
    const total = { label: 'total' } as unknown as GPUTexture;
    const engine = engineWithDebug({
      device: () => device,
      giSignalTextures: () => ({ direct, indirect, ao, total }),
    });
    mount(<GISignalSplit engine={engine} active />);
    expect(startGpuTextureBlit).toHaveBeenCalledTimes(4);

    act(() => root!.render(<GISignalSplit engine={engine} active />));
    expect(startGpuTextureBlit).toHaveBeenCalledTimes(4);
  });
});
