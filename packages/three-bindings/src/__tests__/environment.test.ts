/**
 * environment.test.ts
 *
 * Verifies that `resolveEnvironment` captures `intensity` and `rotationY` off
 * the THREE.Scene's environment fields (Fix 2 — stainedGlass PT-mode env-update
 * round-trip). Without this, env intensity (timeOfDay scrub) and any HDRI
 * yaw applied via `scene.environmentRotation` are silently dropped on the
 * THREE → vitrum direction and PT mode renders with the wrong sky brightness.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { resolveEnvironment } from '../environment.js';

function makeFakeHdriTexture(): THREE.Texture {
  // The minimum surface resolveEnvironment needs is something with isTexture-ish
  // semantics, but resolveEnvironment only checks `scene.environment != null` —
  // it doesn't introspect the texture. A bare Texture instance is sufficient.
  return new THREE.Texture();
}

describe('resolveEnvironment', () => {
  it('returns { kind: "none" } when scene.environment is null', () => {
    const scene = new THREE.Scene();
    const env = resolveEnvironment(scene);
    expect(env.kind).toBe('none');
  });

  it('returns HDRI env with default intensity=1 and rotationY=0', () => {
    const scene = new THREE.Scene();
    scene.environment = makeFakeHdriTexture();
    const env = resolveEnvironment(scene);
    expect(env.kind).toBe('hdri');
    if (env.kind !== 'hdri') return;
    expect(env.intensity).toBe(1);
    expect(env.rotationY).toBe(0);
  });

  it('captures scene.environmentIntensity', () => {
    const scene = new THREE.Scene();
    scene.environment = makeFakeHdriTexture();
    scene.environmentIntensity = 2.5;
    const env = resolveEnvironment(scene);
    expect(env.kind).toBe('hdri');
    if (env.kind !== 'hdri') return;
    expect(env.intensity).toBe(2.5);
  });

  it('captures scene.environmentRotation.y as rotationY', () => {
    const scene = new THREE.Scene();
    scene.environment = makeFakeHdriTexture();
    scene.environmentRotation.set(0, Math.PI / 3, 0);
    const env = resolveEnvironment(scene);
    expect(env.kind).toBe('hdri');
    if (env.kind !== 'hdri') return;
    expect(env.rotationY).toBeCloseTo(Math.PI / 3);
  });

  it('captures both intensity and rotationY together', () => {
    const scene = new THREE.Scene();
    scene.environment = makeFakeHdriTexture();
    scene.environmentIntensity = 0.4;
    scene.environmentRotation.set(0, -Math.PI / 4, 0);
    const env = resolveEnvironment(scene);
    expect(env.kind).toBe('hdri');
    if (env.kind !== 'hdri') return;
    expect(env.intensity).toBeCloseTo(0.4);
    expect(env.rotationY).toBeCloseTo(-Math.PI / 4);
  });

  it('uses background texture when environment is unset', () => {
    const scene = new THREE.Scene();
    const bg = makeFakeHdriTexture();
    scene.background = bg;
    scene.backgroundIntensity = 0.75;
    scene.backgroundRotation.set(0, Math.PI / 8, 0);
    const env = resolveEnvironment(scene);
    expect(env.kind).toBe('hdri');
    if (env.kind !== 'hdri') return;
    expect(env.hdri).toBe(bg);
    expect(env.intensity).toBeCloseTo(0.75);
    expect(env.rotationY).toBeCloseTo(Math.PI / 8);
  });
});
