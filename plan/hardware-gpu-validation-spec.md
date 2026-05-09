# Hardware-GPU Validation: Options Spec

> Research-only. No code changes proposed in this doc.
>
> **Context**: walkaround GI engines (DDGI, RC, ReSTIR, hybrid) all need
> GPU-runtime e2e validation against real hardware to gate the chroma
> tests. The two known blockers — Playwright headless on WSL2 falling
> back to SwiftShader, and MCP-Chrome tabs getting RAF-throttled when
> hidden — are documented in §1. This document surveys the option space
> for a reliable hardware probe.

---

## 1. Problem statement

Three independent agent rounds have hit the same two blockers:

1. **Playwright headless on WSL2** launches Chromium with
   `--use-angle=vulkan --enable-features=Vulkan --enable-unsafe-webgpu
   --ignore-gpu-blocklist` but `navigator.gpu.requestAdapter()` still
   returns `{vendor:'google', architecture:'swiftshader'}` — the
   software fallback. WebGPU partially works on SwiftShader (DDGI runs;
   ReSTIR's `requestDevice()` returns null for compute features). The
   Chromium snap on WSL2 cannot reach the NVIDIA Vulkan ICD on the
   Windows host, so adapter enumeration falls back to SwiftShader
   regardless of flags.

2. **MCP-Chrome (`mcp__claude-in-chrome__*`)** connects to the user's
   real Chrome on Windows. `navigator.gpu.requestAdapter()` returns
   `{vendor:'nvidia', architecture:'lovelace'}` and `requestDevice()`
   succeeds — confirmed by direct probe in this round. **But the
   MCP-controlled tab has `document.visibilityState === 'hidden'`**
   whenever the user has another window foregrounded. Chrome throttles
   `requestAnimationFrame` to ~0 Hz in hidden tabs. R3F's `<Canvas>`
   uses `frameloop="always"` (verified at
   `src/rendering/scene/StudioScene.tsx:124`); with RAF throttled, the
   canvas never auto-sizes past 300×150 and `WalkaroundStage` never
   mounts. A direct probe in this round confirms the freeze: a 1-second
   RAF loop in the hidden tab timed out at the CDP 45 s ceiling
   (effectively zero frames).

Net effect: every available environment can either (a) reach the GPU
but not run RAF, or (b) run RAF but not reach the GPU. Prior reports
of "Hardware GPU verified — RTX 4090 / Lovelace" appear to have come
from runs where the user happened to have the MCP-Chrome window
foregrounded at the right moment, or from FPS-inference on
SwiftShader (which mis-attributes 143 fps DDGI throughput to hardware).

The bug fixes themselves are CPU-test correct and at least partly
algorithmically correct on SwiftShader (RC moved 0/3 → 3/3 caustic
chroma channels post-fix on the *same* software adapter), so the
missing piece is end-to-end runtime confirmation that the fixes
behave correctly on actual GPU at framerate.

---

## 2. Technical landscape

### 2.1 What we have

| Environment | GPU adapter reachable? | RAF runs? | Compute (`requestDevice`)? | Notes |
|---|---|---|---|---|
| Playwright headless (Chromium snap, WSL2) | SwiftShader only | Yes | Yes (limited features) | Current default. Cannot reach NVIDIA ICD. |
| Playwright headless (CI mode flag) | SwiftShader (forced) | Yes | Yes | `--use-angle=swiftshader` baked in, intentional. |
| MCP-Chrome (Windows host, hidden tab) | RTX 4090 / Lovelace ✅ | **No** (throttled) | Yes (probed) | Verified this round. RAF throttled in hidden tab. |
| MCP-Chrome (Windows host, foregrounded tab) | RTX 4090 / Lovelace ✅ | Yes | Yes | Worked in some prior rounds; user-action dependent. |
| User opens app in their own Chrome | RTX 4090 / Lovelace ✅ | Yes | Yes | Manual. No agent automation. |

### 2.2 What CDP / Playwright support that's relevant

- `Page.bringToFront` / `Target.activateTarget` (CDP) — focus a tab
  programmatically. Playwright exposes this as `page.bringToFront()`.
  Some MCP wrappers (e.g. Claude DevTools MCP, by design) block this
  to prevent agents from stealing user focus. **MCP-Chrome's tool
  surface does not expose it.**
- Playwright `headless: false` + `channel: 'chrome'` launches the
  user's installed Chrome (not Chromium snap), which on Windows reaches
  the NVIDIA driver directly. This bypasses the WSL2 ICD problem
  entirely if Playwright runs from PowerShell or the Windows side.
- Chrome flags `--disable-background-timer-throttling
  --disable-renderer-backgrounding
  --disable-backgrounding-occluded-windows` keep RAF running even when
  the tab is occluded. (Note: `--disable-background-timer-throttling`
  was removed around Chrome 78; the other two still work and override
  RAF throttling for hidden/occluded tabs.)
- R3F supports `frameloop="always" | "demand" | "never"` plus
  `useThree(s => s.invalidate)` for manual frame ticking. This lets a
  spec drive frames *without* relying on RAF — but it requires an
  invalidate call from JS, which is straightforward to inject from
  Playwright/CDP.

### 2.3 What's structurally off-limits in this env

- `navigator.gpu` cannot be made available on the Chromium snap +
  WSL2 path without rebuilding Chromium with `skia_use_dawn=true` or
  installing a Vulkan ICD that bridges to the Windows NVIDIA driver
  (e.g., `vulkan-dzn` from Mesa). Reports across the WSL2 community
  are mixed: many users get Vulkan working for compute (CUDA via
  WSL2 NVIDIA driver works fine) but Chromium specifically still
  picks SwiftShader for WebGPU because the ICD bridge isn't on the
  paths Chromium probes. Significant setup work; not a 30-min fix.
- The MCP-Chrome tool surface (verified by reading every tool schema
  in the `mcp__claude-in-chrome__*` namespace) has no
  `Page.bringToFront`, no `chrome.windows.update({focused:true})`, no
  shortcut for tab activation. The closest thing is `computer.key`
  (synthetic keyboard input) — but that types into the focused
  application, which is a user-controlled non-Chrome window when the
  problem actually occurs.

---

## 3. Options

Seven distinct paths, ordered roughly easy→hard:

### Option A — User foregrounds MCP-Chrome window before each round

**Setup**: zero — MCP-Chrome is already configured.
**Per-run cost**: user clicks the MCP-Chrome window, runs validation
within a 2-3 minute wall-clock window, then can switch back. ~30 s
of user attention per round.
**Reliability**: high *during the foregrounded window*. Has worked
in past rounds (commits `995d649`, `8644ec8` referenced "RTX 4090
verified via MCP-Chrome").
**Validates**: full e2e — adapter info, frame rendering, FPS, chroma,
console errors. Same surface as Playwright.
**Requires user action per validation?** Yes.

Trade-off: explicit user gesture per round. Doesn't scale to autonomous
build-loop work, but is the lowest-friction path for the *next* round
of validation right now.

### Option B — Drive R3F manually via `frameloop="never"` + injected invalidate

**Setup**: small code change — add a `?frameloop=manual` URL param or
test-only toggle that sets `<Canvas frameloop="never">` and exposes
`window.__INVALIDATE__` from a `<RenderInvalidator>` child. Leave
production at `frameloop="always"`. ~30 lines.
**Per-run cost**: same as current Playwright runs.
**Reliability**: high *if* the underlying RAF-independent code path
works. WebGPU's `device.queue.submit` doesn't depend on RAF; only the
R3F scheduler does. Driving frames from MCP-Chrome JS would side-step
RAF throttling entirely.
**Validates**: full e2e, including frame rendering and adapter info.
FPS would have to be measured by clock-walling N invalidate cycles,
not by R3F's onFrame instrumentation, which is a slight test
restructure.
**Requires user action per validation?** No.

Trade-off: requires touching the test surface code (the briefing
forbids changing the existing test specs but allows adding new
infrastructure). Might miss issues that *only* show up under sustained
RAF-driven framerate (TDR, GPU memory pressure over many frames).

### Option C — Playwright headed + `channel: 'chrome'` + run from Windows

**Setup**: install Playwright on the Windows side (or run from a
PowerShell shell that talks to the Windows-installed Node), update
`playwright.config.ts` with `headless: false` + `channel: 'chrome'`
+ `launchOptions.args` minus the WSL2 ANGLE flags. Tests run on the
real Chrome with `Page.bringToFront()` available via Playwright API.
~1 hour setup.
**Per-run cost**: same as current Playwright (8-15 s per spec).
**Reliability**: high. This is the canonical path. The user's Chrome
already reaches the RTX 4090 (proven via MCP-Chrome).
**Validates**: full e2e — adapter, FPS, chroma, console errors, in
sustained RAF mode.
**Requires user action per validation?** No, after one-time setup.

Trade-off: requires running Playwright from outside WSL2. Might break
the WSL2-side dev-server lifecycle (`webServer.command` in
`playwright.config.ts`). Workaround: run dev server in WSL2,
Playwright on Windows, baseURL `http://localhost:5173` works because
WSL2 forwards port 5173 to Windows by default.

### Option D — Native Windows Playwright with foreground-suppression flags

**Setup**: same as C, but additionally pass
`--disable-renderer-backgrounding
--disable-backgrounding-occluded-windows` so even if the test window
is partially occluded by another app it still runs at full RAF speed.
~10 min on top of C.
**Per-run cost**: same as C.
**Reliability**: very high — combines the canonical-path advantage of
C with throttle-resistance.
**Validates**: full e2e, robust to user's window arrangement.
**Requires user action per validation?** No.

Trade-off: a strict superset of C, no real downside. Recommended if
C is taken.

### Option E — WSL2 Vulkan ICD bridge (dzn / Lavapipe)

**Setup**: install `mesa-vulkan-drivers` + `vulkan-dzn` on WSL2,
configure `VK_ICD_FILENAMES` to point at the dzn JSON manifest, verify
`vulkaninfo` reports the NVIDIA D3D12 device. Then re-run Playwright
headless on WSL2; Chromium's ANGLE/Vulkan path *might* pick up the
bridge. ~2-4 hours of setup with realistic chance of failure (per
multiple WSL2/NVIDIA forum threads, `dzn` driver detection is finicky
on Ubuntu 24.04).
**Per-run cost**: same as current Playwright (8-15 s per spec).
**Reliability**: medium-low at best. Even if `vulkaninfo` works,
Chromium may still pick SwiftShader because of how ANGLE probes for
adapters. Multiple users on the NVIDIA forums report exactly this
outcome.
**Validates**: full e2e, *if it works*. Adapter would be reported as
`vendor: microsoft, architecture: dzn` (D3D12-bridged) or similar —
still hardware-backed, but not the same code path as native NVIDIA on
Windows. A different runtime than production users will see.
**Requires user action per validation?** No, after one-time setup.

Trade-off: high setup cost, real chance of not working at all, and the
adapter we'd be testing on isn't the same as production. Not
recommended unless Options C/D fall through.

### Option F — Add fail-fast SwiftShader detection at app boot

**Setup**: in `WalkaroundStage` or `WebGPUCanvas`, after
`requestAdapter()`, check `adapter.info.vendor === 'google' &&
adapter.info.architecture === 'swiftshader'` and either (a) refuse to
mount with a clear error overlay or (b) render but log a single
prominent warning. Surface the result through `__WG__.isHardwareGpu`
which the spec already reads. ~20 lines.
**Per-run cost**: zero.
**Reliability**: 100% — string match is deterministic.
**Validates**: nothing new; just makes existing tests fail loudly
instead of silently passing on SwiftShader.
**Requires user action per validation?** No.

Trade-off: orthogonal to Options A-E. **This is a multiplier, not a
substitute.** It guards against the historic failure mode of
"143 fps SwiftShader claimed as hardware-GPU pass." Should be paired
with whichever runtime path is chosen.

### Option G — Dedicated user-driven validation session (manual)

**Setup**: zero. User opens `npm run dev` in their normal Chrome,
clicks through the walkaround viewer, eyeballs the floor for caustics,
checks DevTools for adapter info and console errors.
**Per-run cost**: 5-10 minutes of user time per round per branch (3
branches × 5 min = 15-30 min total).
**Reliability**: highest possible — actual user, actual env.
**Validates**: full e2e *qualitatively*. No machine-readable
chroma std-dev or FPS samples; the user reports impressions.
**Requires user action per validation?** Yes, fully.

Trade-off: doesn't scale and doesn't produce numeric assertions. But
it's the ground-truth check before any merge to main.

---

## 4. Recommendation

### Primary: **Option D (Playwright headed on Windows + throttle-suppression flags)** + **Option F (fail-fast SwiftShader detection)**

D delivers the canonical, sustainable path: real Chrome, real RTX 4090,
full RAF, no user action per round, same test code. F prevents future
silent-SwiftShader regressions across all environments.

The combination addresses every documented failure mode:
- WSL2 SwiftShader fallback → bypassed (test runs from Windows side)
- Hidden-tab RAF throttling → bypassed (foreground flags + headed mode)
- Silent SwiftShader pass → caught at adapter-probe time

One-time setup of ~1-2 hours, then every subsequent round is
zero-touch. Worth the investment given there are three branches that
each need ongoing validation as the build-loop iterates.

### Fallback (if user can't run Playwright from Windows): **Option A + Option F**

A is the lowest-friction path that's been proven to work in past
rounds. Pair with F so SwiftShader-runs fail loudly instead of being
mistaken for hardware passes. Requires the user to foreground the
MCP-Chrome window for ~2-3 min per validation round; acceptable for
the next 1-2 rounds while a longer-term path is set up.

### Why not the others

- **B (manual invalidate)** — clever but introduces a test-only render
  pathway that doesn't match production. Risk of validating something
  that's not what users will see. Reasonable as a third fallback
  if both D and A are blocked, not as primary.
- **C** — strict subset of D; just take D.
- **E (WSL2 Vulkan bridge)** — high effort, real chance of total
  failure, and even on success we'd be testing a non-production
  driver path. Bad ROI.
- **G (manual)** — reserved for pre-merge ground-truth, not for the
  per-round validation cadence.

### Risk-flags on the recommendation

1. Option D needs Node+Playwright on the Windows side. If the user
   prefers to stay WSL2-only, fall back to A + F.
2. Option D's `webServer.reuseExistingServer: true` is already set for
   non-CI in the existing config — running dev server in WSL2 and
   Playwright on Windows should compose cleanly via localhost
   port-forwarding, but verify the first time.
3. Option F changes user-visible app behavior on SwiftShader (refuse
   to mount, or warning overlay). Decide which UX the user wants
   before implementing.

---

## 5. Out of scope

The following were considered but ruled out as outside this
spec:

- **Native Linux runner / dual-boot** — major env change, multi-day
  setup, doesn't help if the user works on Windows day-to-day.
- **Cloud GPU runner (AWS EC2 g4dn / fly.io)** — adds remote-render
  latency (5-15 s per spec on top of cold-start), needs ssh tunnels,
  introduces a non-prod GPU (T4/L4 vs Lovelace) so validation results
  may not translate. Defer unless local options all fail.
- **Rewriting tests in a different framework** (Cypress, WebdriverIO,
  custom Puppeteer harness) — the existing 867 tests + 13 e2e specs
  are working; framework swap is unjustified without a structural
  reason.
- **Forking three-mesh-bvh or three.js** — not a validation problem.
- **Building a custom Chromium with `skia_use_dawn=true`** —
  multi-day; only justified if Options A-D all fail.

---

## 6. What this spec does NOT decide

- Whether to merge any of the three walkaround branches
- Whether the existing thresholds (luma var ≥0.05, chroma std ≥0.02,
  channel std ≥8) are calibrated correctly
- Which technique (DDGI / RC / ReSTIR) should be the production path

These are downstream of having a working validation harness, which is
what this spec exists to unblock.
