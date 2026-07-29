# Radiometric regression harnesses

These harnesses exercise stable renderer paths against bounded numerical or image-space oracles. They are regression tools, not a second maturity ledger. Current implementation status lives in `plan/road-to-100.md` and `plan/renderer-fidelity-matrix.md`.

## Path-tracer regression suite

| Case | Script | Result | Contract |
|------|--------|--------|----------|
| SPPM | `ab-sppm.mjs` | `results-sppm.json` | Progressive density estimate converges toward the manifold-NEE reference. |
| BDPT | `ab-bdpt.mjs` | `results-bdpt.json` | Bounded multi-vertex BDPT agrees with the unidirectional reference inside the pinned tolerance. |
| ReSTIR-PT | `ab-restir-pt.mjs` | `results-restir-pt.json` | Fresh runs use the current exact seed/capture contract. The retained GPU JSON predates that contract and is historical data, not an active gate. |
| ReSTIR-PT specialty materials | `ab-restir-pt-specialty.mjs` | `results-restir-pt-specialty.json` | Deterministic scalar and map-backed rich-lobe fixtures remain active. |
| Sobol | `ab-sobol.mjs` | `results-sobol.json` | Equal-frame Sobol sampling remains inside the pinned RMSE envelope. |

Run one native WebGPU case:

```bash
npm run radiometric-ab:sppm
npm run radiometric-ab:bdpt
npm run radiometric-ab:restir-pt
npm run radiometric-ab:sobol
```

Run the wrapper across all native WebGPU cases:

```bash
npm run radiometric-ab:pt-host-status
```

The wrapper exits `0` only when every selected case writes a fresh, strictly valid `PASS` result. It exits `2` when the native Deno host cannot expose a suitable WebGPU adapter and exits `1` for test, provenance, or result-validation failures. Select cases with either positional IDs or `VITRUM_PT_RADIOMETRIC_AB_CASES`; set `VITRUM_PT_RADIOMETRIC_AB_TIMEOUT_MS` to change the per-case timeout.
Its gitignored host-status JSON is an ephemeral run diagnostic and is not
implementation authority.

Check the retained numerical baselines, deterministic specialty fixture, and
current source pins without recapturing:

```bash
npm run radiometric-ab:source-check
npm run radiometric-ab:proof-check
npm run radiometric-ab:restir-pt-specialty
```

The source-check also pins a default-test, CPU-only cross-backend rect-area
radiometric A/B. It compares all three renderer wire formats against the core
half-extent contract, including sampled positions, physical area, emitted
power, and emitting-side orientation. This catches convention drift without
requiring a GPU.

`proofs.mjs` and `resultValidation.mjs` define numerical contracts. The checker
accepts three explicitly historical, pre-manifest baselines (SPPM, BDPT, and
Sobol) unless `--source-only` is selected, then checks the ReSTIR-PT specialty
fixture and pins named test-source paths plus production-source identifiers. It
does not execute those named tests. The root `npm run proof-check` uses the
source-only form, so committed host captures are outside that canonical gate.

For each fresh PT capture, `resultProvenance.mjs` hashes the harness script, the
provenance helper, and every eligible runtime file under the source roots listed
in `proofs.mjs`; the PT wrapper recomputes that exact manifest before accepting
the result.

## Walkaround suite

The walkaround harness compares realtime GI configurations for regression diagnosis:

```bash
npm run radiometric-ab:walkaround
npm run radiometric-ab:walkaround-glossy-spp64
npm run radiometric-ab:walkaround-all-spp64
```

`run-walkaround-ab.mjs` owns native-host classification and gitignored local
status files. Those files are ephemeral diagnostics. The committed
`walkaround-ab-*.json` captures are historical baselines; neither kind of file
overrides current production source or the renderer contract.
Fresh walkaround provenance hashes only `run-walkaround-ab.mjs`,
`walkaround-ab.mjs`, and `resultProvenance.mjs`. It proves run/harness identity,
not the complete renderer source tree.

## Result discipline

- A successful script must replace its result file during that run.
- A fresh PT wrapper `PASS` must satisfy its result schema and exact runtime-source manifest.
- A fresh walkaround result must be replaced during the run, satisfy the selected cases' verdict contracts, and match the wrapper/harness/helper hashes above.
- A retained pre-manifest baseline is historical numerical data, never a current-source status claim.
- Host availability is reported separately from estimator correctness.
- A stale status file never opens or closes a code gap.
- When an algorithmic change intentionally moves a numerical baseline, recapture the relevant case and review the metric delta with the source change.
