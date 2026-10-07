# Warning disposition for the selected PK-06 artifacts

No warning filters were used. `selected-build-log-excerpts.json` selects all
warning/error/fail matches with context from 48 logs in the 24 selected entry
directories (12 decode builds plus the 12 unchanged chunk entries). Eight logs
from unused copied cache directories are separately inventoried by hash and
excluded from final artifact counts. Original private logs remain unchanged.

| Diagnostic | Located cause and disposition |
|---|---|
| CCE `-Wcce-compat`: vector loop condition expects uint16 | `atk_tile.h:30`, `advance.h:23`, BF16 `widen.h:10`, `narrow_output.h:12` compare the emitted uint16 induction variable against an int32 `steps` argument. Selected library `ascriptor/backends/cce/emit.py:1461-1472` owns this lowering and requires an upward uint16 loop. Public validation bounds S to 1..16; initial index is 0, step is 1, neither index nor bound can overflow uint16. All 16 legal trip counts were additionally executed for both dtypes (32 cases), with independent states and HV=12. Existing six-bd grid and focused sim/pipesim also passed. Qualification is for these bounded loops; it does not approve unrestricted int32 loop bounds. |
| Python `SyntaxWarning: invalid escape sequence '\d'` | Vendor-generated `cmake/util/ascendc_impl_build.py:194`; tokenizer locates the surrounding `IMPL_API` triple-quoted template at 154..211. The outer ordinary string preserves the backslash, producing the intended raw-string regex in generated Python. No kernel expression is involved. Current source context and hash are recorded in `vendor-source-context.json`. |
| CMake unused variable | `CMAKE_CROSS_PLATFORM_COMPILER`, shown explicitly in the build-log context. CCE compiler/link/package stages completed and their resulting artifacts were independently hash-checked before and after actual NPU calls. This warning does not change the selected target or executable. |
| ATRACE/runtime `drvErr=4`, `halGetDeviceInfo`, `DrvMngGetConsoleLogLevel` | Generated compiler/package steps query device information inside the deliberately device-free compiler container. Compilation and packaging completed; these messages are not a successful device run. Actual NPU acceptance used a separate device-enabled container, fresh health/occupancy checks, shared locks and the selected artifacts. No corresponding execution error occurs in the native logs. |
| `TaskFailCallBackManager` INFO constructor/destructor | Class names matched the broad diagnostic search; these INFO records do not report a failed task. They are retained in the excerpts. |
| `NPUCachingAllocator.cpp:201` 32-padding warning | The allocator announces padding required by this CANN/SoC pairing. Production buffers are framework allocations, never custom unpadded external storage. Main and q/k/v rows own complete 128-element regions; scalar GM reads copy one FP32 value into an owned 32-byte UB row, whose padding is not treated as another scalar. Emitted DMA spans and UB offsets are recorded in `../emission/`; focused pipe models report no hazards or event imbalance. The warning remains visible in every native log excerpt. |
| Five pytest importorskip deprecations | Host-only torch_npu stub intentionally raises ImportError, so existing NPU integration tests skip. Pytest warns about its future default. This is explicitly separated from the actual Docker NPU acceptance, and no unrelated shared tests were changed. |

The private `--reuse-build` attempt failed its artifact-dictionary equality
assertion: removing two trailing blank lines changed the module cache key,
although emitted source bytes stayed identical. New decode entries were compiled
before that assertion. Original failed-attempt tails are retained here. The
subsequent final receipt explicitly reports verification of previously compiled
artifacts, not another compiler invocation and not byte-identical cache reuse.
All final native grids, integration, audit and timing were rerun against the
final source and selected final artifacts. No failed attempt is counted as a pass.

Two private diagnostic script processes were initially classified as foreign by
a module-name-only sampler. Receipt reconciliation checks the saved child PID
against the innermost NSpid plus the exact task Python command. The original
classification count and raw-file hashes are retained; after this reconciliation,
all sampled contexts belong to the task and before/after contexts are empty.
This is a sampled observation, not a guarantee about activity between samples.
