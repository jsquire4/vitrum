// MaterialInspector — click a mesh to see its vitrum Material params; live-edit them.
//
// Click-to-pick uses engine.debug.pickPrimitive(x, y) (T3.G) when a `canvas` prop is
// supplied and the engine exposes it (e.g. HybridEngine). Otherwise drive selection
// via the `selectedPrimitiveId` prop.

import React, {
  type FC,
  type ChangeEvent,
  useEffect,
  useState,
} from 'react';
import type { MaterialSpec, Vec3 } from '@vitrum/core';
import type { DebuggableEngine } from '../types.js';

// ────────────────────────────────────────────────────────────────────────────
// Props
// ────────────────────────────────────────────────────────────────────────────

export interface MaterialInspectorProps {
  /** The engine to read/write materials through. */
  engine: DebuggableEngine;
  /**
   * Scene snapshot — used to look up the material of the selected primitive.
   * Pass the same Scene object you passed to engine.setScene().
   */
  scene: import('@vitrum/core').Scene;
  /**
   * ID of the currently selected primitive. When set, takes precedence over the
   * internal canvas click-pick. When null/undefined (and no canvas pick), the
   * panel is closed.
   */
  selectedPrimitiveId?: string | null;
  /**
   * Optional render canvas. When supplied and the engine exposes
   * `debug.pickPrimitive()`, clicking the canvas selects the primitive under the
   * cursor automatically — no external `selectedPrimitiveId` wiring needed
   * (items_to_fix.md T3.G). Pass the same canvas the engine renders into.
   */
  canvas?: HTMLCanvasElement | null;
  /** CSS class name applied to the panel div. */
  className?: string;
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function vec3ToHex(v: Vec3): string {
  const to255 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 255);
  const hex = (n: number): string => n.toString(16).padStart(2, '0');
  return `#${hex(to255(v[0]))}${hex(to255(v[1]))}${hex(to255(v[2]))}`;
}

function hexToVec3(h: string): Vec3 {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 0xff) / 255, ((n >> 8) & 0xff) / 255, (n & 0xff) / 255];
}

// ────────────────────────────────────────────────────────────────────────────
// Styles
// ────────────────────────────────────────────────────────────────────────────

const PANEL_STYLE: React.CSSProperties = {
  position: 'absolute',
  top: 48,
  right: 8,
  background: 'rgba(15,15,20,0.9)',
  color: '#e0e0e0',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '8px 10px',
  borderRadius: 4,
  zIndex: 9998,
  minWidth: 220,
  maxWidth: 260,
  lineHeight: 1.8,
};

const ROW_STYLE: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  marginBottom: 2,
};

const LABEL_STYLE: React.CSSProperties = { color: '#888', flexShrink: 0, marginRight: 8 };

const INPUT_STYLE: React.CSSProperties = {
  background: '#1e1e2e',
  border: '1px solid #444',
  color: '#e0e0e0',
  fontSize: 11,
  width: 70,
  padding: '1px 4px',
  borderRadius: 2,
};

const WARN_STYLE: React.CSSProperties = { color: '#ffb347', fontSize: 10, marginTop: 4 };

const CLOSE_STYLE: React.CSSProperties = {
  float: 'right',
  cursor: 'pointer',
  color: '#888',
  marginLeft: 8,
};

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export const MaterialInspector: FC<MaterialInspectorProps> = ({
  engine,
  scene,
  selectedPrimitiveId,
  canvas,
  className,
}) => {
  const [draft, setDraft] = useState<MaterialSpec | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [pickedId, setPickedId] = useState<string | null>(null);

  const hasPickAPI = typeof engine.debug?.pickPrimitive === 'function';

  // Effective selection: an explicit `selectedPrimitiveId` prop wins; otherwise
  // the internal canvas click-pick (T3.G).
  const effectiveId = selectedPrimitiveId ?? pickedId;

  // Self-wire click-to-pick when a canvas is supplied and the engine exposes the
  // pick API. Maps the CSS-pixel click into canvas backing-store pixels (what
  // pickPrimitive expects). The host may instead drive `selectedPrimitiveId`.
  useEffect(() => {
    if (canvas == null || !hasPickAPI) return undefined;
    const onClick = (e: MouseEvent): void => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      const x = (e.clientX - rect.left) * (canvas.width / rect.width);
      const y = (e.clientY - rect.top) * (canvas.height / rect.height);
      setPickedId(engine.debug?.pickPrimitive?.(x, y) ?? null);
    };
    canvas.addEventListener('click', onClick);
    return () => canvas.removeEventListener('click', onClick);
  }, [canvas, hasPickAPI, engine]);

  // Sync draft when the effective selection changes.
  useEffect(() => {
    if (!effectiveId) {
      setPanelOpen(false);
      return;
    }
    const prim = scene.primitives.find((p) => p.id === effectiveId);
    if (prim) {
      // Deep-clone the material so edits don't mutate the scene object.
      setDraft({ ...prim.material });
      setPanelOpen(true);
    } else {
      setPanelOpen(false); // clicked empty space or an unknown id
    }
  }, [effectiveId, scene]);

  if (!panelOpen || draft === null || !effectiveId) return null;

  // ── Field update helpers ──────────────────────────────────────────────────

  const updateField = <K extends keyof MaterialSpec>(key: K, value: MaterialSpec[K]): void => {
    const next: MaterialSpec = { ...draft, [key]: value };
    setDraft(next);
    // updatePrimitive is optional on Engine; fall back to nothing if absent.
    if (typeof engine.updatePrimitive === 'function') {
      engine.updatePrimitive(effectiveId, { material: next });
    }
  };

  const updateNumber = (key: keyof MaterialSpec) =>
    (e: ChangeEvent<HTMLInputElement>): void => {
      const v = parseFloat(e.target.value);
      if (!isNaN(v)) updateField(key, v as MaterialSpec[typeof key]);
    };

  const updateColor = (key: keyof MaterialSpec) =>
    (e: ChangeEvent<HTMLInputElement>): void => {
      updateField(key, hexToVec3(e.target.value) as MaterialSpec[typeof key]);
    };

  const emissiveColor = draft.emissive ?? [0, 0, 0] satisfies Vec3;

  return (
    <div className={className} style={PANEL_STYLE} role="dialog" aria-label="Material Inspector">
      <div style={{ fontWeight: 'bold', marginBottom: 6 }}>
        Material Inspector
        <span
          style={CLOSE_STYLE}
          role="button"
          tabIndex={0}
          aria-label="Close inspector"
          onClick={() => { setPanelOpen(false); setPickedId(null); }}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { setPanelOpen(false); setPickedId(null); } }}
        >
          ✕
        </span>
      </div>

      <div style={{ color: '#666', fontSize: 10, marginBottom: 6 }}>
        id: {effectiveId}
      </div>

      {/* baseColor */}
      <div style={ROW_STYLE}>
        <span style={LABEL_STYLE}>baseColor</span>
        <input
          type="color"
          value={vec3ToHex(draft.baseColor)}
          onChange={updateColor('baseColor')}
          style={{ width: 36, height: 20, border: 'none', cursor: 'pointer', background: 'none' }}
          title="Base color"
        />
      </div>

      {/* roughness */}
      <div style={ROW_STYLE}>
        <span style={LABEL_STYLE}>roughness</span>
        <input
          type="number"
          min={0} max={1} step={0.01}
          value={draft.roughness}
          onChange={updateNumber('roughness')}
          style={INPUT_STYLE}
        />
      </div>

      {/* metallic */}
      <div style={ROW_STYLE}>
        <span style={LABEL_STYLE}>metallic</span>
        <input
          type="number"
          min={0} max={1} step={0.01}
          value={draft.metallic}
          onChange={updateNumber('metallic')}
          style={INPUT_STYLE}
        />
      </div>

      {/* emissive */}
      <div style={ROW_STYLE}>
        <span style={LABEL_STYLE}>emissive</span>
        <input
          type="color"
          value={vec3ToHex(emissiveColor)}
          onChange={updateColor('emissive')}
          style={{ width: 36, height: 20, border: 'none', cursor: 'pointer', background: 'none' }}
          title="Emissive color"
        />
      </div>

      {/* emissiveIntensity */}
      {draft.emissive !== undefined && (
        <div style={ROW_STYLE}>
          <span style={LABEL_STYLE}>emissInt.</span>
          <input
            type="number"
            min={0} step={0.1}
            value={draft.emissiveIntensity ?? 1}
            onChange={updateNumber('emissiveIntensity')}
            style={INPUT_STYLE}
          />
        </div>
      )}

      {/* transmission */}
      {draft.transmission !== undefined && (
        <div style={ROW_STYLE}>
          <span style={LABEL_STYLE}>transmission</span>
          <input
            type="number"
            min={0} max={1} step={0.01}
            value={draft.transmission}
            onChange={updateNumber('transmission')}
            style={INPUT_STYLE}
          />
        </div>
      )}

      {/* ior */}
      {draft.ior !== undefined && (
        <div style={ROW_STYLE}>
          <span style={LABEL_STYLE}>ior</span>
          <input
            type="number"
            min={1} max={3} step={0.01}
            value={draft.ior}
            onChange={updateNumber('ior')}
            style={INPUT_STYLE}
          />
        </div>
      )}

      {!hasPickAPI && (
        <div style={WARN_STYLE}>
          This engine does not expose <code>debug.pickPrimitive()</code>.
          <br />
          Drive selection via the <code>selectedPrimitiveId</code> prop.
        </div>
      )}
    </div>
  );
};
