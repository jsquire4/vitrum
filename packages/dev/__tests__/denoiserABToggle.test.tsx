import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { DenoiserABToggle } from '../src/react/DenoiserABToggle.js';
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
  vi.useRealTimers();
});

function mount(engine: DebuggableEngine): HTMLElement {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root!.render(<DenoiserABToggle engine={engine} />));
  return container.firstElementChild as HTMLElement;
}

function engineWithDebug(debug: NonNullable<DebuggableEngine['debug']>): DebuggableEngine {
  return { debug } as unknown as DebuggableEngine;
}

describe('DenoiserABToggle', () => {
  it('renders a backend without a setter as an honest non-interactive status', () => {
    const addEventListener = vi.spyOn(window, 'addEventListener');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const badge = mount(engineWithDebug({ isDenoiserEnabled: () => true }));

    expect(badge.textContent).toBe('denoiser unavailable');
    expect(badge.getAttribute('role')).toBe('status');
    expect(badge.getAttribute('aria-disabled')).toBe('true');
    expect(badge.hasAttribute('aria-pressed')).toBe(false);
    expect(badge.hasAttribute('tabindex')).toBe(false);
    expect(badge.style.cursor).toBe('default');
    expect(addEventListener.mock.calls.some(([type]) => String(type) === 'keydown')).toBe(false);

    act(() => {
      badge.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      badge.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'd' }));
    });
    expect(badge.textContent).toBe('denoiser unavailable');
    expect(warn).not.toHaveBeenCalled();
  });

  it('retains click and keyboard behavior when the backend exposes the setter', () => {
    let enabled = true;
    const setDenoiserEnabled = vi.fn((next: boolean) => { enabled = next; });
    const badge = mount(engineWithDebug({
      isDenoiserEnabled: () => enabled,
      setDenoiserEnabled,
    }));

    expect(badge.getAttribute('role')).toBe('button');
    expect(badge.getAttribute('aria-pressed')).toBe('true');
    act(() => window.dispatchEvent(new KeyboardEvent('keydown', { key: 'd' })));
    expect(setDenoiserEnabled).toHaveBeenLastCalledWith(false);
    expect(badge.textContent).toContain('off');

    act(() => badge.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(setDenoiserEnabled).toHaveBeenLastCalledWith(true);
    expect(badge.textContent).toContain('on');
  });

  it('reconciles the badge with an externally changed engine debug state', () => {
    vi.useFakeTimers();
    let enabled = true;
    const badge = mount(engineWithDebug({
      isDenoiserEnabled: () => enabled,
      setDenoiserEnabled: vi.fn(),
    }));

    enabled = false;
    act(() => vi.advanceTimersByTime(250));

    expect(badge.getAttribute('aria-pressed')).toBe('false');
    expect(badge.textContent).toContain('off');
  });
});
