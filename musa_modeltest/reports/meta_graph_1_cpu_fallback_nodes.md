# meta_graph_1 CPU Fallback Nodes

## Run

- Model: `/data/welo/workspace/tf_test_model/inference/metaGraph/onnx_out/meta_graph_1.onnx`
- Runtime: `/data/welo/workspace/onnxruntime_musa_modeltest/runtime/onnxruntime_perf_test`
- Container: `welo-tf215-test`
- Device: `MUSA_VISIBLE_DEVICES=5`
- Free-dim overrides: `musa_modeltest/reports/meta_graph_1_free_dims.txt`

## Final Fallback Run After Fix

- CPU fallback: enabled
- Result: session initialized and one shaped random-input iteration completed successfully
- Placement: all `2245` executable nodes placed on `MUSAExecutionProvider`
- Profile node events: `4490` total, `0` on CPU, `4490` on MUSA
- Verbose log: `musa_modeltest/reports/logs/meta_graph_1_fallback_after_reciprocal_verbose.log`
- Profile JSON: `musa_modeltest/reports/profiles/meta_graph_1_fallback_after_reciprocal_2026-05-27_19-27-43_293.json`

```bash
./runtime/onnxruntime_perf_test -e musa -v \
  -p musa_modeltest/reports/profiles/meta_graph_1_fallback_after_reciprocal \
  -m times -r 1 -I \
  -f "$(cat musa_modeltest/reports/meta_graph_1_free_dims.txt)" \
  /data/welo/workspace/tf_test_model/inference/metaGraph/onnx_out/meta_graph_1.onnx
```

## Final Provider Placement Summary

| Provider | Node Events |
|---|---:|
| `MUSAExecutionProvider` | 4490 |

## CPU Fallback Summary

| Stage | CPUExecutionProvider Nodes | MUSAExecutionProvider Nodes | Main Finding |
|---|---:|---:|---|
| Before `Reciprocal` kernel | 29 | 2274 | all CPU nodes were `Reciprocal` |
| After `Reciprocal` kernel | 0 | 2245 | no CPU fallback remains in fallback-enabled run |

## Pre-Fix CPU Node List

These nodes came from the initial shaped fallback run and are kept as the node-level migration target that drove the fix.

| # | Node | OpType |
|---:|---|---|
| 1 | `HhPjKqEI6o__2489` | `Reciprocal` |
| 2 | `IC8WtbSqLz__2960` | `Reciprocal` |
| 3 | `9pk7XjMEBK__2718` | `Reciprocal` |
| 4 | `QhXUFR97r0__3095` | `Reciprocal` |
| 5 | `lmbMRuOyvH__3306` | `Reciprocal` |
| 6 | `YzXK9lmxPg__3640` | `Reciprocal` |
| 7 | `ZBkUHTxaKG__3930` | `Reciprocal` |
| 8 | `wv81dbl7rk__4081` | `Reciprocal` |
| 9 | `6zxkXWeuOI__4083` | `Reciprocal` |
| 10 | `YyF0pt5Gkx__4110` | `Reciprocal` |
| 11 | `wfgnRHzuqY__4120` | `Reciprocal` |
| 12 | `JLlcDSgXkW__4105` | `Reciprocal` |
| 13 | `wNnL0b6aAU__4115` | `Reciprocal` |
| 14 | `EAXjzsVR2l__4133` | `Reciprocal` |
| 15 | `fnrOCJzyTE__4128` | `Reciprocal` |
| 16 | `7oDiNAghmJ__4088` | `Reciprocal` |
| 17 | `lk5K2yE3Aj__4141` | `Reciprocal` |
| 18 | `bat2zCulMj__4596` | `Reciprocal` |
| 19 | `Y5SIQLwRAs__4598` | `Reciprocal` |
| 20 | `xdAilwWfDc__4600` | `Reciprocal` |
| 21 | `sUkuv7r4Cn__4683` | `Reciprocal` |
| 22 | `qZDvnfMISc__4685` | `Reciprocal` |
| 23 | `rvNxOqsKRl__4687` | `Reciprocal` |
| 24 | `KaYPkdCpV9__4689` | `Reciprocal` |
| 25 | `LA17uw5tNM__4691` | `Reciprocal` |
| 26 | `Lw8alRPy4c__4693` | `Reciprocal` |
| 27 | `rNl2FIhfP3__4695` | `Reciprocal` |
| 28 | `xThWJzm4Mo__4697` | `Reciprocal` |
| 29 | `nX2YxUyAg0__4699` | `Reciprocal` |

## Disabled Fallback Verification

- CPU fallback: disabled with `session.disable_cpu_ep_fallback|1`
- Result: pass, exit code `0`
- Log: `musa_modeltest/reports/logs/meta_graph_1_no_fallback_reciprocal.log`

```bash
./runtime/onnxruntime_perf_test -e musa \
  -C "session.disable_cpu_ep_fallback|1" \
  -m times -r 1 -I \
  -f "$(cat musa_modeltest/reports/meta_graph_1_free_dims.txt)" \
  /data/welo/workspace/tf_test_model/inference/metaGraph/onnx_out/meta_graph_1.onnx
```
