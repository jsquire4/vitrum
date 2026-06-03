/**
 * hooks.ts — shared React hooks for @vitrum/dev overlay components.
 *
 * `useKeyToggle` — keyboard shortcut that calls a callback on a key press
 *   (no ctrl/meta/alt). Extracted from BVHVisualizer + DenoiserABToggle.
 *
 * `useDebugDevice` — reads engine.debug?.device() safely and returns the
 *   GPUDevice (or null). Extracted from DDGIAtlasViewer + GISignalSplit.
 */

import { useEffect } from 'react';
import type { DebuggableEngine } from '../types.js';

/**
 * Register a keyboard toggle on a single key (no modifier keys).
 * No-ops when `key` is null. Cleans up on unmount / key change.
 *
 * @param key - Lowercase key string (e.g. 'b', 'd'). Pass null to disable.
 * @param callback - Called when the key fires without ctrl/meta/alt.
 */
export function useKeyToggle(key: string | null, callback: () => void): void {
  useEffect(() => {
    if (key === null) return;
    const normalized = key.toLowerCase();
    const handler = (e: KeyboardEvent): void => {
      if (e.key.toLowerCase() === normalized && !e.ctrlKey && !e.metaKey && !e.altKey) {
        callback();
      }
    };
    window.addEventListener('keydown', handler);
    return () => {
      window.removeEventListener('keydown', handler);
    };
  }, [key, callback]);
}

/**
 * Read `engine.debug?.device()` safely and return the GPUDevice or null.
 * Returns null when the debug surface is absent or the device method is
 * not a function.
 *
 * @param engine - Any DebuggableEngine; may or may not have a debug surface.
 */
export function useDebugDevice(engine: DebuggableEngine): GPUDevice | null {
  const hasDevice = typeof engine.debug?.device === 'function';
  return hasDevice ? (engine.debug?.device?.() ?? null) : null;
}
