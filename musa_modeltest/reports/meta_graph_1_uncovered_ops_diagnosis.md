# meta_graph_1 Uncovered Ops Diagnosis

## Summary

- The shaped fallback run before this change assigned `29` nodes to `CPUExecutionProvider`.
- All `29` CPU nodes were ONNX `Reciprocal` nodes.
- After adding MUSA `Reciprocal`, the fallback-enabled rerun placed all executable nodes on `MUSAExecutionProvider`.
- The stricter disabled-fallback run also passed with exit code `0`.

## Root Cause

| OpType | Pre-Fix CPU Nodes | Root Cause | Fix |
|---|---:|---|---|
| `Reciprocal` | 29 | MUSA EP had no `Reciprocal` kernel registration/implementation, so ORT assigned these nodes to CPU when fallback was enabled. | Added MUSA `Reciprocal` kernel using mudnn unary `RECIPROCAL` and registered ONNX versions 6-12 and 13+. |

## Implementation References

- Primary migration reference: `/data/welo/workspace/tensorflow_musa_extension/musa_ext/kernels/math/musa_reciprocal_op.cc`
- Secondary reference checked: `/data/welo/workspace/ort_musa`

## Changed MUSA EP Files

| File | Purpose |
|---|---|
| `onnxruntime/core/providers/musa/math/reciprocal.h` | Declares the typed MUSA `Reciprocal` kernel. |
| `onnxruntime/core/providers/musa/math/reciprocal.cc` | Implements mudnn unary `RECIPROCAL` compute and ONNX kernel registration. |
| `onnxruntime/core/providers/musa/musa_execution_provider.cc` | Adds `Reciprocal` kernel create-info entries to the MUSA EP registry. |

## Verification Evidence

| Check | Result | Evidence |
|---|---|---|
| Build MUSA provider | Pass | `cmake --build build_musa_modeltest/Release --target onnxruntime_providers_musa -- -j64` |
| Fallback enabled placement | Pass | `All nodes placed on [MUSAExecutionProvider]. Number of nodes: 2245` in `musa_modeltest/reports/logs/meta_graph_1_fallback_after_reciprocal_verbose.log` |
| Profile provider events | Pass | `0` CPU node events, `4490` MUSA node events in `musa_modeltest/reports/profiles/meta_graph_1_fallback_after_reciprocal_2026-05-27_19-27-43_293.json` |
| Disabled CPU fallback | Pass | exit code `0` in `musa_modeltest/reports/logs/meta_graph_1_no_fallback_reciprocal.log` |

## Notes

- The first unshaped `-I` attempt is not a coverage signal: unknown symbolic dimensions defaulted to 1 and produced zero-width `Slice` outputs, leading to a downstream GEMM dimension mismatch. The `.spec`-derived `-f` override file fixes the test input shapes.
- The committed reports are the durable evidence. Large profile JSON files are kept in the test workspace but should not be treated as source artifacts unless a reviewer explicitly needs them.
