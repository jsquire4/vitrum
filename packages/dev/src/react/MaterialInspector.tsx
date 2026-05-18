// MaterialInspector — click a mesh to see its vitrum Material params; live-edit them.
//
// Implementation mode: PARTIAL REAL (approach (a) for interface, (b) for picking).
//
// What's real:
//   - The UI panel that renders Material fields (baseColor, roughness, metallic,
//     emissive, transmission, ior) with live-edit inputs.
//   - engine.updatePrimitive() is called directly on the Engine contract —
//     this is a real Engine API today.
//   - The panel can be opened programmatically by passing `selectedPrimitiveId`.
//
// What's stubbed:
//   - Click-to-pick: determining which primitive the user clicked requires
//     engine.debug.pickPrimitive(x, y) (declared in types.ts:EngineDebugSurface).
//     Without it, the user must select the primitive by ID via the `selectedPrimitiveId` prop.
//
// TODO T3.G followup: implement engine.debug.pickPrimitive() in HybridEngine.
//   Options:
//     (a) CPU-side ray-AABB test against the BVH (approximate; misses concave geometry).
//     (b) GPU: read the primitive-ID G-buffer pixel at (x,y) after the shade pass.
//   Option (b) is more accurate; requires a primitive-ID attachment in the shade pass.

import React, {
  type FC,
  type ChangeEvent,
  useEffect,
  useState,
} from 'react';
import type { Material, Vec3 } from '@vitrum/core';
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
   * ID of the currently selected primitive. When null/undefined, the panel
   * is closed. Use with canvas click handlers + engine.debug.pickPrimitive()
   * when that API is available, or wire your own selection state.
   */
  selectedPrimitiveId?: string | null;
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
  className,
}) => {
  const [draft, setDraft] = useState<Material | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);

  // W3-D8: top-level engine.debug gated on capabilities.debugSurface; the
  // specific debug method remains optional per EngineDebugSurface.
  const hasPickAPI =
    engine.capabilities.debugSurface && !!engine.debug?.pickPrimitive;

  // Sync draft when selection changes.
  useEffect(() => {
    if (!selectedPrimitiveId) {
      setPanelOpen(false);
      return;
    }
    const prim = scene.primitives.find((p) => p.id === selectedPrimitiveId);
    if (prim) {
      // Deep-clone the material so edits don't mutate the scene object.
      setDraft({ ...prim.material });
      setPanelOpen(true);
    }
  }, [selectedPrimitiveId, scene]);

  if (!panelOpen || draft === null || !selectedPrimitiveId) return null;

  // ── Field update helpers ──────────────────────────────────────────────────

  const updateField = <K extends keyof Material>(key: K, value: Material[K]): void => {
    const next: Material = { ...draft, [key]: value };
    setDraft(next);
    // W3-D8: gate on capabilities.incrementalUpdates rather than the
    // method's typeof — when the flag is true, the method is guaranteed
    // present per the EngineCapabilities invariant.
    if (engine.capabilities.incrementalUpdates) {
      engine.updatePrimitive!(selectedPrimitiveId, { material: next });
    }
  };

  const updateNumber = (key: keyof Material) =>
    (e: ChangeEvent<HTMLInputElement>): void => {
      const v = parseFloat(e.target.value);
      if (!isNaN(v)) updateField(key, v as Material[typeof key]);
    };

  const updateColor = (key: keyof Material) =>
    (e: ChangeEvent<HTMLInputElement>): void => {
      updateField(key, hexToVec3(e.target.value) as Material[typeof key]);
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
          onClick={() => setPanelOpen(false)}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setPanelOpen(false); }}
        >
          ✕
        </span>
      </div>

      <div style={{ color: '#666', fontSize: 10, marginBottom: 6 }}>
        id: {selectedPrimitiveId}
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
          Click-to-pick requires engine.debug.pickPrimitive() — T3.G followup.
          <br />
          Use <code>selectedPrimitiveId</code> prop to select manually.
        </div>
      )}
    </div>
  );
};
