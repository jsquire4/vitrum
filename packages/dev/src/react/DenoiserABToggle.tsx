// DenoiserABToggle — keyboard 'D' toggles denoiser on/off for A/B comparison.
//
// Implementation mode: PARTIAL REAL (approach (a) for interface, (b) for control).
//
// The toggle UI and keyboard handler are fully implemented. The actual
// enable/disable call uses engine.debug.setDenoiserEnabled() (declared in
// types.ts:EngineDebugSurface). When that method is absent, the component
// renders a warning and the key press is a no-op.
//
// To wire this today: add setDenoiserEnabled(enabled: boolean): void to
// HybridEngine and expose it via engine.debug. The walkaround-hybrid denoiser
// is already gated by a flag in the render pipeline; this just surfaces it.
//
// TODO T3.G followup: implement engine.debug.setDenoiserEnabled() in
// HybridEngine by threading a boolean through the SVGF/atrous dispatch path.

import React, { type FC, useEffect, useState } from 'react';
import type { DebuggableEngine } from '../types.js';

export interface DenoiserABToggleProps {
  /** The engine to control. Must implement engine.debug.setDenoiserEnabled (T3.G followup). */
  engine: DebuggableEngine;
  /**
   * Keyboard key that toggles the denoiser. Default: 'd'.
   * Set to null to disable keyboard control; use the rendered button instead.
   */
  toggleKey?: string | null;
  /** CSS class name applied to the badge. */
  className?: string;
}

const BADGE_STYLE: React.CSSProperties = {
  position: 'absolute',
  top: 8,
  left: 8,
  background: 'rgba(0,0,0,0.65)',
  color: '#e0e0e0',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '4px 8px',
  borderRadius: 4,
  userSelect: 'none',
  cursor: 'pointer',
  zIndex: 9998,
};

const ENABLED_COLOR = '#7dfa7d';
const DISABLED_COLOR = '#fa7d7d';
const STUB_COLOR = '#ffb347';

export const DenoiserABToggle: FC<DenoiserABToggleProps> = ({
  engine,
  toggleKey = 'd',
  className,
}) => {
  // Track local enabled state; seed from engine.debug.isDenoiserEnabled() if available.
  const [enabled, setEnabled] = useState<boolean>(() => {
    return engine.debug?.isDenoiserEnabled?.() ?? true;
  });

  const hasDebug =
    typeof engine.debug?.setDenoiserEnabled === 'function';

  const doToggle = (): void => {
    const next = !enabled;

    if (hasDebug) {
      engine.debug!.setDenoiserEnabled!(next);
      setEnabled(next);
    } else {
      // eslint-disable-next-line no-console
      console.warn(
        '[DenoiserABToggle] engine.debug.setDenoiserEnabled() is not implemented. ' +
        'Implement it in HybridEngine to wire the denoiser toggle. ' +
        'See packages/dev/src/types.ts:EngineDebugSurface for the interface.'
      );
      return;
    }
  };

  // Keyboard handler
  useEffect(() => {
    if (toggleKey === null) return;
    const key = toggleKey.toLowerCase();
    const handler = (e: KeyboardEvent): void => {
      if (e.key.toLowerCase() === key && !e.ctrlKey && !e.metaKey && !e.altKey) {
        doToggle();
      }
    };
    window.addEventListener('keydown', handler);
    return () => {
      window.removeEventListener('keydown', handler);
    };
    // doToggle depends on enabled — useEffect re-registers on each change.
    // This is intentional so the closure captures the current `enabled`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toggleKey, enabled, hasDebug]);

  const stateColor = !hasDebug ? STUB_COLOR : enabled ? ENABLED_COLOR : DISABLED_COLOR;
  const stateLabel = !hasDebug
    ? 'denoiser [stub]'
    : `denoiser ${enabled ? '■ on' : '□ off'}`;
  const keyLabel = toggleKey !== null ? ` [${toggleKey.toUpperCase()}]` : '';

  return (
    <div
      className={className}
      style={{ ...BADGE_STYLE, borderLeft: `3px solid ${stateColor}` }}
      role="button"
      tabIndex={0}
      aria-pressed={enabled}
      aria-label={`Toggle denoiser${keyLabel}`}
      onClick={doToggle}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') doToggle();
      }}
    >
      {stateLabel}{keyLabel}
    </div>
  );
};
