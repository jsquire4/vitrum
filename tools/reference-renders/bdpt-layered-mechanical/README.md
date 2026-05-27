# BDPT vs layered-BSDF — mechanical fixtures

Acceptance PNGs for Sprint 10c / 14 (`bdptLayeredAcceptance.gpu.test.ts`).

- `cornell-layered.png` — GPU capture (512×512 quick suite) when promoted from `benchmark:bdpt-layered-refs`.
- `cornell-layered-bdpt.png` — with `?vitrumBdpt=1` the cornell-box example wires BDPT API only (no light-subpath draw pass yet), so the image stays near-black (~12–15 KiB) while layered stays ~500 KiB+. Mechanical harness only requires `bdptDeltaMean > 0.005`.

## Regenerate + gate

```bash
npm run benchmark:bdpt-layered-mechanical
```

## Real captures (GPU host)

```bash
npm run benchmark:bdpt-layered-refs
# or: scripts/capture-cornell-suite.sh --only layered
#     scripts/capture-cornell-suite.sh --only layered --bdpt
```

Compare `cornell-layered.png` vs the BDPT variant visually; promote into dated dirs under `tools/reference-renders/` when refreshing baselines.
