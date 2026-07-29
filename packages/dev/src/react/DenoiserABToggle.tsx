// DenoiserABToggle — keyboard 'D' toggles denoiser on/off for A/B comparison.
//
// Calls engine.debug.setDenoiserEnabled / isDenoiserEnabled (EngineDebugSurface).
// HybridEngine wires this via DenoiserAdapterPass (walkaround-hybrid only).
// Other backends render a non-interactive unavailable status when the debug
// surface is absent. They never install a keyboard shortcut or imply that a
// local-only state change affected the renderer.

import React, { type FC, useCallback, useEffect, useState } from 'react';
import type { DebuggableEngine } from '../types.js';
import { useKeyToggle } from './hooks.js';

export interface DenoiserABToggleProps {
  /** Engine with engine.debug.setDenoiserEnabled (HybridEngine). */
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
  zIndex: 9998,
};

const ENABLED_COLOR = '#7dfa7d';
const DISABLED_COLOR = '#fa7d7d';
const UNAVAILABLE_COLOR = '#ffb347';

export const DenoiserABToggle: FC<DenoiserABToggleProps> = ({
  engine,
  toggleKey = 'd',
  className,
}) => {
  // Local state drives React rendering, but the engine remains authoritative:
  // another debug control or a recreated engine may change this value without
  // remounting the component.
  const [enabled, setEnabled] = useState<boolean>(() => {
    return engine.debug?.isDenoiserEnabled?.() ?? true;
  });

  const hasDebug =
    typeof engine.debug?.setDenoiserEnabled === 'function';

  useEffect(() => {
    if (!hasDebug) return;
    const syncFromEngine = (): void => {
      try {
        const current = engine.debug?.isDenoiserEnabled?.();
        if (typeof current === 'boolean') {
          setEnabled((previous) => previous === current ? previous : current);
        }
      } catch {
        // A transient debug-surface failure must not break the host React tree.
      }
    };
    syncFromEngine();
    const interval = setInterval(syncFromEngine, 250);
    return () => clearInterval(interval);
  }, [engine, hasDebug]);

  const doToggle = useCallback((): void => {
    if (!hasDebug) return;
    try {
      const authoritative = engine.debug?.isDenoiserEnabled?.();
      const next = !(typeof authoritative === 'boolean' ? authoritative : enabled);
      engine.debug!.setDenoiserEnabled!(next);
      setEnabled(next);
    } catch {
      // Keep the displayed state unchanged when the engine rejects the toggle.
    }
  }, [enabled, engine, hasDebug]);

  // Keyboard handler — useKeyToggle re-registers when doToggle identity changes,
  // capturing the current `enabled` closure (same semantics as the original useEffect).
  useKeyToggle(hasDebug ? (toggleKey ?? null) : null, doToggle);

  const stateColor = !hasDebug ? UNAVAILABLE_COLOR : enabled ? ENABLED_COLOR : DISABLED_COLOR;
  const stateLabel = !hasDebug
    ? 'denoiser unavailable'
    : `denoiser ${enabled ? '■ on' : '□ off'}`;
  const keyLabel = hasDebug && toggleKey !== null ? ` [${toggleKey.toUpperCase()}]` : '';

  return (
    <div
      className={className}
      style={{
        ...BADGE_STYLE,
        borderLeft: `3px solid ${stateColor}`,
        cursor: hasDebug ? 'pointer' : 'default',
      }}
      role={hasDebug ? 'button' : 'status'}
      tabIndex={hasDebug ? 0 : undefined}
      aria-pressed={hasDebug ? enabled : undefined}
      aria-disabled={hasDebug ? undefined : true}
      aria-label={hasDebug
        ? `Toggle denoiser${keyLabel}`
        : 'Denoiser toggle unavailable on this backend'}
      onClick={hasDebug ? doToggle : undefined}
      onKeyDown={hasDebug
        ? (e) => {
          if (e.key === 'Enter' || e.key === ' ') doToggle();
        }
        : undefined}
    >
      {stateLabel}{keyLabel}
    </div>
  );
};
