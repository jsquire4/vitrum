// Pins quality-option live-propagation behavior for attachVitrum.
//
// Prior to 2026-05-18, `quality` was a static value snapshotted at mount
// time — the RAF tick read `opts.quality` once and never consulted React's
// updated value. <VitrumCanvas>'s qualityRef + useEffect pattern was
// misleading dead code because the engine never re-read the ref. The fix
// added a getter overload so callers (React's <VitrumCanvas> in particular)
// can pass `() => qualityRef.current` and get live propagation without
// engine recreation.
//
// We exercise the extracted `resolveQualityOption` helper directly rather
// than driving a full RAF loop (the RAF loop requires a DOM + GPU device,
// covered by the shader-compile-ci smoke).

import { describe, it, expect } from 'vitest';
import { resolveQualityOption } from '../src/lifecycle/vanilla.js';

describe('resolveQualityOption (attachVitrum live-quality plumbing)', () => {
  it('returns undefined when opts.quality is undefined', () => {
    expect(resolveQualityOption(undefined)).toBeUndefined();
  });

  it('returns the value when opts.quality is a plain object', () => {
    const q = { samplesTarget: 8, bounces: 3 } as const;
    expect(resolveQualityOption(q)).toBe(q);
  });

  it('invokes opts.quality when it is a getter and returns its current value', () => {
    let current: { samplesTarget: number } | undefined = { samplesTarget: 1 };
    const getter = (): { samplesTarget: number } | undefined => current;
    expect(resolveQualityOption(getter)?.samplesTarget).toBe(1);
    current = { samplesTarget: 64 };
    // Critical pin: subsequent calls see the updated value (live-prop).
    expect(resolveQualityOption(getter)?.samplesTarget).toBe(64);
    current = undefined;
    expect(resolveQualityOption(getter)).toBeUndefined();
  });

  it('invokes the getter every call (no caching) — required for React ref pattern', () => {
    let callCount = 0;
    const getter = (): { samplesTarget: number } => {
      callCount += 1;
      return { samplesTarget: callCount };
    };
    expect(resolveQualityOption(getter)?.samplesTarget).toBe(1);
    expect(resolveQualityOption(getter)?.samplesTarget).toBe(2);
    expect(resolveQualityOption(getter)?.samplesTarget).toBe(3);
    expect(callCount).toBe(3);
  });
});
