// debugUtils.ts — shared helpers for debug overlay value queries.

import type { ScenePrimitive } from '@vitrum/core';
import type { DebuggableEngine } from '../types.js';

export interface DebugCallResult<T> {
  readonly status: 'unsupported' | 'ready' | 'missing' | 'error';
  readonly value: T | null;
}

export function safeDebugCall<T>(fn: (() => T | null) | undefined): DebugCallResult<T> {
  if (typeof fn !== 'function') return { status: 'unsupported', value: null };
  try {
    const value = fn();
    return value == null
      ? { status: 'missing', value: null }
      : { status: 'ready', value };
  } catch {
    return { status: 'error', value: null };
  }
}

export function debugValueStatus<T>(fn: (() => T | null) | undefined, noun: string): string {
  const result = safeDebugCall<T>(fn);
  if (result.status === 'unsupported') return `${noun} api unavailable`;
  if (result.status === 'missing') return `${noun} missing`;
  return result.status;
}

export function textureAvailability(value: unknown): string {
  return value == null ? 'missing' : 'ready';
}

export function materialPatchStatus(engine: DebuggableEngine): string {
  const support = engine.capabilities.incrementalPatchSupport?.material;
  if (support === true) return 'native';
  if (engine.capabilities.supportsIncrementalScene) return 'check backend';
  return 'unsupported';
}

export function formatMaterial(material: ScenePrimitive['material']): string {
  const color = material.baseColor.map((v) => v.toFixed(2)).join(',');
  return `base [${color}] r ${material.roughness.toFixed(2)} m ${material.metallic.toFixed(2)}`;
}

export function formatPrimitiveDetails(primitive: ScenePrimitive): string {
  if (primitive.kind === 'analytic') return primitive.shape;
  if (primitive.kind === 'instanced-mesh') return `${primitive.instances.length} instances`;
  const vertexCount = Math.floor(primitive.positions.length / 3);
  const triangleCount = primitive.indices != null
    ? Math.floor(primitive.indices.length / 3)
    : Math.floor(vertexCount / 3);
  if (primitive.kind === 'skinned-mesh') return `${vertexCount} verts, ${primitive.bones.length / 16} bones`;
  return `${vertexCount} verts, ${triangleCount} tris`;
}

export function formatSet(values: ReadonlySet<string> | undefined): string {
  if (values == null || values.size === 0) return 'none';
  const entries = Array.from(values);
  return entries.length <= 3 ? entries.join(', ') : `${entries.slice(0, 3).join(', ')} +${entries.length - 3}`;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '-';
  if (bytes < 1024) return `${bytes} B`;
  const kib = bytes / 1024;
  if (kib < 1024) return `${kib.toFixed(1)} KiB`;
  const mib = kib / 1024;
  if (mib < 1024) return `${mib.toFixed(1)} MiB`;
  return `${(mib / 1024).toFixed(2)} GiB`;
}

export function findCanvas(container: HTMLElement): HTMLCanvasElement | null {
  if (typeof HTMLCanvasElement !== 'undefined' && container instanceof HTMLCanvasElement) {
    return container;
  }
  return container.querySelector('canvas');
}
