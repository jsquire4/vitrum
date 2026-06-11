// overlayDenoiser.ts — denoiser toggle badge overlay.

import type { DebuggableEngine } from '../types.js';
import { makeDiv } from './domUtils.js';

export function addDenoiserDiagnostics(
  engine: DebuggableEngine,
  add: (el: HTMLElement) => void,
  cleanupFns: Array<() => void>,
): void {
  const hasSetter = typeof engine.debug?.setDenoiserEnabled === 'function';
  const hasGetter = typeof engine.debug?.isDenoiserEnabled === 'function';
  let enabled = hasGetter ? engine.debug?.isDenoiserEnabled?.() ?? true : true;

  const badge = makeDiv({
    position: 'absolute',
    top: '8px',
    left: '8px',
    background: 'rgba(0,0,0,0.72)',
    color: '#e0e0e0',
    fontFamily: 'monospace',
    fontSize: '11px',
    padding: '4px 8px',
    borderRadius: '4px',
    userSelect: 'none',
    cursor: hasSetter ? 'pointer' : 'default',
    zIndex: '9998',
  });

  const render = (): void => {
    const color = !hasSetter ? '#ffb347' : enabled ? '#7dfa7d' : '#fa7d7d';
    badge.style.borderLeft = `3px solid ${color}`;
    badge.textContent = hasSetter
      ? `denoiser ${enabled ? 'on' : 'off'} [D]`
      : 'denoiser unavailable';
    badge.title = hasSetter
      ? 'Click or press D to toggle engine.debug.setDenoiserEnabled().'
      : 'This backend does not expose engine.debug.setDenoiserEnabled().';
  };

  const toggle = (): void => {
    if (!hasSetter) return;
    enabled = !enabled;
    engine.debug?.setDenoiserEnabled?.(enabled);
    render();
  };

  render();
  add(badge);

  badge.addEventListener('click', toggle);
  const keyHandler = (e: KeyboardEvent): void => {
    if (e.key.toLowerCase() === 'd' && !e.ctrlKey && !e.metaKey && !e.altKey) {
      toggle();
    }
  };
  window.addEventListener('keydown', keyHandler);
  cleanupFns.push(() => {
    badge.removeEventListener('click', toggle);
    window.removeEventListener('keydown', keyHandler);
  });
}
