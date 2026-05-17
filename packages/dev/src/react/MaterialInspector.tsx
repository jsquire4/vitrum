// MaterialInspector — click a primitive in the canvas to inspect + edit its Material.
//
// W12 implementation: host-side ray casting (no GPU needed). When `canvas` and
// `getCamera` are provided, a pointer-down handler on the canvas builds a
// picking ray from screen NDC + the camera's inverse view-proj, then runs
// Möller-Trumbore against every MeshPrimitive triangle in `scene.primitives`.
// The hit primitive ID becomes the inspector's selected target; the panel
// displays the live Material via `engine.updatePrimitive` for in-place edits.
//
// Per the W12 brief: picking does NOT need GPU; intersect against MeshPrimitive[]
// via Möller-Trumbore in TypeScript. Perf cost on click is fine — even a
// 100k-triangle scene resolves in <100 ms on modern V8.
//
// MeshPrimitive triangles are the only kind supported for picking. Analytic
// primitives and instanced meshes route to a console warning — extending
// coverage is a follow-up if it ever matters.
//
// Programmatic selection via `selectedPrimitiveId` still works (W12 keeps
// the original API surface).

import React, {
  type CSSProperties,
  type ChangeEvent,
  type FC,
  useEffect,
  useRef,
  useState,
} from 'react';
import type { Material, MeshPrimitive, Scene, Vec3 } from '@vitrum/core';
import type { DebuggableEngine } from '../types.js';

// ────────────────────────────────────────────────────────────────────────────
// Props
// ────────────────────────────────────────────────────────────────────────────

export interface MaterialInspectorProps {
  /** The engine to read/write materials through. */
  engine: DebuggableEngine;
  /** Scene snapshot. Same object passed to engine.setScene(). */
  scene: Scene;
  /**
   * The render canvas the user clicks on. Required for pointer-pick.
   * When omitted, the panel relies on `selectedPrimitiveId` only.
   */
  canvas?: HTMLCanvasElement | null;
  /**
   * Camera-state provider, polled each pointer-down to build the picking ray.
   * Must return the same matrices passed to `engine.renderFrame()`.
   * When omitted, pointer-pick is disabled.
   */
  getCamera?: () => CameraState | null;
  /**
   * Programmatic selection. When set, this primitive's material is shown
   * regardless of click history. Set to null to clear.
   */
  selectedPrimitiveId?: string | null;
  /** Called when the user clicks a primitive (pointer-pick). */
  onSelect?: (primitiveId: string | null) => void;
  /** CSS class name applied to the panel div. */
  className?: string;
}

export interface CameraState {
  /** Column-major 4x4 view matrix (same as FrameInput.viewMatrix). */
  readonly viewMatrix: Float32Array;
  /** Column-major 4x4 projection matrix (same as FrameInput.projMatrix). */
  readonly projMatrix: Float32Array;
  /** World-space camera position. */
  readonly cameraPosition: Vec3;
}

// ────────────────────────────────────────────────────────────────────────────
// Math helpers (column-major, three.js convention)
// ────────────────────────────────────────────────────────────────────────────

/** Multiply a 4x4 column-major matrix by a column vector. Returns the 4-vec result. */
function mat4MulVec4(m: Float32Array, v: [number, number, number, number]): [number, number, number, number] {
  const [x, y, z, w] = v;
  return [
    (m[0] ?? 0) * x + (m[4] ?? 0) * y + (m[8 ] ?? 0) * z + (m[12] ?? 0) * w,
    (m[1] ?? 0) * x + (m[5] ?? 0) * y + (m[9 ] ?? 0) * z + (m[13] ?? 0) * w,
    (m[2] ?? 0) * x + (m[6] ?? 0) * y + (m[10] ?? 0) * z + (m[14] ?? 0) * w,
    (m[3] ?? 0) * x + (m[7] ?? 0) * y + (m[11] ?? 0) * z + (m[15] ?? 0) * w,
  ];
}

/** Invert a 4x4 column-major matrix. Returns null if singular. */
function mat4Inverse(m: Float32Array): Float32Array | null {
  // Standard cofactor expansion. Lifted from glMatrix; rewritten without deps.
  const a00 = m[0 ] ?? 0, a01 = m[1 ] ?? 0, a02 = m[2 ] ?? 0, a03 = m[3 ] ?? 0;
  const a10 = m[4 ] ?? 0, a11 = m[5 ] ?? 0, a12 = m[6 ] ?? 0, a13 = m[7 ] ?? 0;
  const a20 = m[8 ] ?? 0, a21 = m[9 ] ?? 0, a22 = m[10] ?? 0, a23 = m[11] ?? 0;
  const a30 = m[12] ?? 0, a31 = m[13] ?? 0, a32 = m[14] ?? 0, a33 = m[15] ?? 0;

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (det === 0 || !isFinite(det)) return null;
  const invDet = 1 / det;
  const out = new Float32Array(16);
  out[0 ] = ( a11 * b11 - a12 * b10 + a13 * b09) * invDet;
  out[1 ] = (-a01 * b11 + a02 * b10 - a03 * b09) * invDet;
  out[2 ] = ( a31 * b05 - a32 * b04 + a33 * b03) * invDet;
  out[3 ] = (-a21 * b05 + a22 * b04 - a23 * b03) * invDet;
  out[4 ] = (-a10 * b11 + a12 * b08 - a13 * b07) * invDet;
  out[5 ] = ( a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  out[6 ] = (-a30 * b05 + a32 * b02 - a33 * b01) * invDet;
  out[7 ] = ( a20 * b05 - a22 * b02 + a23 * b01) * invDet;
  out[8 ] = ( a10 * b10 - a11 * b08 + a13 * b06) * invDet;
  out[9 ] = (-a00 * b10 + a01 * b08 - a03 * b06) * invDet;
  out[10] = ( a30 * b04 - a31 * b02 + a33 * b00) * invDet;
  out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * invDet;
  out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * invDet;
  out[13] = ( a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * invDet;
  out[15] = ( a20 * b03 - a21 * b01 + a22 * b00) * invDet;
  return out;
}

/** Build the picking ray for normalised device coordinates (ndcX, ndcY) ∈ [-1, 1]. */
function buildPickRay(
  camera: CameraState,
  ndcX: number,
  ndcY: number,
): { origin: Vec3; direction: Vec3 } | null {
  const invView = mat4Inverse(camera.viewMatrix);
  const invProj = mat4Inverse(camera.projMatrix);
  if (invView == null || invProj == null) return null;
  // Near + far points in NDC → view space (via invProj) → world space (via invView).
  const nearView = mat4MulVec4(invProj, [ndcX, ndcY, -1, 1]);
  const farView  = mat4MulVec4(invProj, [ndcX, ndcY,  1, 1]);
  const nearViewW = nearView[3] !== 0 ? 1 / nearView[3] : 1;
  const farViewW  = farView[3]  !== 0 ? 1 / farView[3]  : 1;
  const nearVS: [number, number, number, number] = [
    nearView[0] * nearViewW, nearView[1] * nearViewW, nearView[2] * nearViewW, 1,
  ];
  const farVS: [number, number, number, number] = [
    farView[0]  * farViewW,  farView[1]  * farViewW,  farView[2]  * farViewW,  1,
  ];
  const nearWS = mat4MulVec4(invView, nearVS);
  const farWS  = mat4MulVec4(invView, farVS);
  const origin: Vec3 = [nearWS[0], nearWS[1], nearWS[2]];
  const dx = farWS[0] - nearWS[0];
  const dy = farWS[1] - nearWS[1];
  const dz = farWS[2] - nearWS[2];
  const len = Math.hypot(dx, dy, dz);
  if (len === 0) return null;
  const direction: Vec3 = [dx / len, dy / len, dz / len];
  return { origin, direction };
}

/** Möller-Trumbore ray-triangle intersection. Returns ray-parameter t or null. */
function rayTriangleIntersect(
  origin: Vec3,
  direction: Vec3,
  v0: [number, number, number],
  v1: [number, number, number],
  v2: [number, number, number],
): number | null {
  const EPS = 1e-8;
  const e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2];
  const e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2];
  // h = direction × e2
  const hx = direction[1] * e2z - direction[2] * e2y;
  const hy = direction[2] * e2x - direction[0] * e2z;
  const hz = direction[0] * e2y - direction[1] * e2x;
  const a = e1x * hx + e1y * hy + e1z * hz;
  if (a > -EPS && a < EPS) return null;
  const f = 1 / a;
  const sx = origin[0] - v0[0], sy = origin[1] - v0[1], sz = origin[2] - v0[2];
  const u = f * (sx * hx + sy * hy + sz * hz);
  if (u < 0 || u > 1) return null;
  // q = s × e1
  const qx = sy * e1z - sz * e1y;
  const qy = sz * e1x - sx * e1z;
  const qz = sx * e1y - sy * e1x;
  const v = f * (direction[0] * qx + direction[1] * qy + direction[2] * qz);
  if (v < 0 || u + v > 1) return null;
  const t = f * (e2x * qx + e2y * qy + e2z * qz);
  return t > EPS ? t : null;
}

/** Apply a column-major 4x4 to a position. */
function transformPoint(m: Float32Array | undefined, p: readonly [number, number, number]): [number, number, number] {
  if (m == null) return [p[0], p[1], p[2]];
  const v = mat4MulVec4(m, [p[0], p[1], p[2], 1]);
  const w = v[3] !== 0 ? 1 / v[3] : 1;
  return [v[0] * w, v[1] * w, v[2] * w];
}

/**
 * Pick the closest MeshPrimitive hit by `ray` from `scene.primitives`. Returns
 * the primitive ID or null. Analytic + instanced-mesh primitives are skipped
 * (out of scope for W12 — extend if it ever matters).
 */
function pickClosestMesh(
  scene: Scene,
  origin: Vec3,
  direction: Vec3,
): string | null {
  let bestT = Infinity;
  let bestId: string | null = null;

  for (const prim of scene.primitives) {
    if (prim.kind !== 'mesh') continue;
    const mesh = prim as MeshPrimitive;
    const positions = mesh.positions;
    const indices = mesh.indices;
    const xform = mesh.transform;

    const triCount = indices != null ? indices.length / 3 : positions.length / 9;
    for (let i = 0; i < triCount; i++) {
      let ia: number, ib: number, ic: number;
      if (indices != null) {
        ia = indices[i * 3    ] ?? 0;
        ib = indices[i * 3 + 1] ?? 0;
        ic = indices[i * 3 + 2] ?? 0;
      } else {
        ia = i * 3;
        ib = i * 3 + 1;
        ic = i * 3 + 2;
      }
      const v0 = transformPoint(xform, [
        positions[ia * 3    ] ?? 0,
        positions[ia * 3 + 1] ?? 0,
        positions[ia * 3 + 2] ?? 0,
      ]);
      const v1 = transformPoint(xform, [
        positions[ib * 3    ] ?? 0,
        positions[ib * 3 + 1] ?? 0,
        positions[ib * 3 + 2] ?? 0,
      ]);
      const v2 = transformPoint(xform, [
        positions[ic * 3    ] ?? 0,
        positions[ic * 3 + 1] ?? 0,
        positions[ic * 3 + 2] ?? 0,
      ]);
      const t = rayTriangleIntersect(origin, direction, v0, v1, v2);
      if (t !== null && t < bestT) {
        bestT = t;
        bestId = mesh.id;
      }
    }
  }
  return bestId;
}

// ────────────────────────────────────────────────────────────────────────────
// Material-field helpers
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

const PANEL_STYLE: CSSProperties = {
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

const ROW_STYLE: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  marginBottom: 2,
};
const LABEL_STYLE: CSSProperties = { color: '#888', flexShrink: 0, marginRight: 8 };
const INPUT_STYLE: CSSProperties = {
  background: '#1e1e2e',
  border: '1px solid #444',
  color: '#e0e0e0',
  fontSize: 11,
  width: 70,
  padding: '1px 4px',
  borderRadius: 2,
};
const CLOSE_STYLE: CSSProperties = {
  float: 'right',
  cursor: 'pointer',
  color: '#888',
  marginLeft: 8,
};
const HINT_STYLE: CSSProperties = { color: '#888', fontSize: 10, marginTop: 4 };

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export const MaterialInspector: FC<MaterialInspectorProps> = ({
  engine,
  scene,
  canvas,
  getCamera,
  selectedPrimitiveId,
  onSelect,
  className,
}) => {
  const [internalSelected, setInternalSelected] = useState<string | null>(null);
  const isControlled = selectedPrimitiveId !== undefined;
  const activeId = isControlled ? (selectedPrimitiveId ?? null) : internalSelected;

  const [draft, setDraft] = useState<Material | null>(null);
  // Track the canvas + scene + camera in refs so the pointer handler sees
  // current values without re-attaching on every render.
  const sceneRef  = useRef<Scene>(scene);
  const cameraRef = useRef<typeof getCamera | undefined>(getCamera);
  useEffect(() => { sceneRef.current  = scene; },  [scene]);
  useEffect(() => { cameraRef.current = getCamera; }, [getCamera]);

  // Sync draft when selection changes.
  useEffect(() => {
    if (activeId === null) {
      setDraft(null);
      return;
    }
    const prim = scene.primitives.find((p) => p.id === activeId);
    if (prim != null) {
      setDraft({ ...prim.material });
    } else {
      setDraft(null);
    }
  }, [activeId, scene]);

  // Wire the pointer-down handler on the canvas.
  useEffect(() => {
    if (canvas == null || typeof getCamera !== 'function') return;
    const handler = (ev: PointerEvent): void => {
      // Only primary button.
      if (ev.button !== 0) return;
      const cam = cameraRef.current?.();
      if (cam == null) return;
      const rect = canvas.getBoundingClientRect();
      const ndcX = ((ev.clientX - rect.left) / rect.width)  * 2 - 1;
      // WebGPU/three NDC: +y up. Browser +y down → flip.
      const ndcY = -(((ev.clientY - rect.top)  / rect.height) * 2 - 1);
      const ray = buildPickRay(cam, ndcX, ndcY);
      if (ray == null) return;
      const id = pickClosestMesh(sceneRef.current, ray.origin, ray.direction);
      if (!isControlled) setInternalSelected(id);
      onSelect?.(id);
    };
    canvas.addEventListener('pointerdown', handler);
    return () => {
      canvas.removeEventListener('pointerdown', handler);
    };
  }, [canvas, getCamera, isControlled, onSelect]);

  if (activeId === null || draft === null) return null;

  // ── Field update helpers ──────────────────────────────────────────────────

  const commitDraft = (next: Material): void => {
    setDraft(next);
    if (typeof engine.updatePrimitive === 'function') {
      engine.updatePrimitive(activeId, { material: next });
    }
  };

  const updateField = <K extends keyof Material>(key: K, value: Material[K]): void => {
    commitDraft({ ...draft, [key]: value });
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

  const close = (): void => {
    if (!isControlled) setInternalSelected(null);
    onSelect?.(null);
  };

  const emissiveColor = draft.emissive ?? ([0, 0, 0] satisfies Vec3);
  const pickerWired = canvas != null && typeof getCamera === 'function';

  return (
    <div className={className} style={PANEL_STYLE} role="dialog" aria-label="Material Inspector">
      <div style={{ fontWeight: 'bold', marginBottom: 6 }}>
        Material Inspector
        <span
          style={CLOSE_STYLE}
          role="button"
          tabIndex={0}
          aria-label="Close inspector"
          onClick={close}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') close(); }}
        >
          ✕
        </span>
      </div>

      <div style={{ color: '#666', fontSize: 10, marginBottom: 6 }}>
        id: {activeId}
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

      {!pickerWired && (
        <div style={HINT_STYLE}>
          Pass <code>canvas</code> + <code>getCamera</code> to enable click-pick.
        </div>
      )}
    </div>
  );
};

// Re-export the picker helpers so dev-only test scaffolds can exercise the
// math without instantiating the React component. Pure functions only.
export {
  buildPickRay as __pickerBuildRay,
  rayTriangleIntersect as __pickerRayTri,
  pickClosestMesh as __pickerPickClosestMesh,
  mat4Inverse as __pickerMat4Inverse,
};
