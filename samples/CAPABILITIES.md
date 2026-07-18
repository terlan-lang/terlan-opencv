# Tutorial Capability Contracts

Tutorial coverage follows the package boundary rather than the size of the
upstream OpenCV index. A tutorial is runnable only when every operation it uses
has a reviewed Terlan API, explicit ownership, bounded copied inputs and
outputs, stable failure behavior, and deterministic lifecycle cleanup.

## Capability Decisions

| Tutorial area | Decision | Required contract before a port is accepted |
| --- | --- | --- |
| Video I/O | Blocked | Camera, file, network, timestamp, cancellation, backpressure, and frame ownership belong to explicit host capabilities. `terlan-opencv` may consume an owned frame but must not acquire ambient devices or paths. |
| GUI and HighGUI | Not applicable | Windows, event loops, keyboard input, and trackbars belong to the Terlan UI stack. Tutorials must replace display with exact copied-result assertions instead of wrapping HighGUI. |
| Video analysis | Supported pending API | CPU algorithms may accept bounded owned frame sequences and return copied results or opaque state owners. They must not imply capture, playback, unbounded buffering, or hidden background threads. |
| Object detection | Partially supported | Bounded in-memory DNN output decoding is runnable. Additional detectors require copied model bytes, immutable configuration, bounded detections, deterministic ordering, and no filesystem lookup. |
| Computational photography | Supported pending API | CPU operations may map owned matrices to newly owned matrices. Inputs, allocation bounds, numerical policy, and cleanup must be reviewed per operation. |
| Stitching | Supported pending API | Stitching requires bounded image counts and dimensions, deterministic CPU policy, copied transforms and results, and explicit ownership for any retained estimator state. GUI and file loading remain external. |
| G-API | Blocked | Graph compilation, callbacks, plugins, streaming executors, and backend selection require a Terlan-native graph and callable capability contract. The package will not expose raw OpenCV graph/runtime objects. |
| CUDA and other accelerators | Blocked | Device selection, queues or streams, synchronization, memory budgets, transfer ownership, cancellation, and target availability must be compiler-visible capabilities. CPU `Mat` handles must not be reinterpreted as device storage. |

## Admission Rules

Before changing a row from blocked or pending to runnable:

1. Add the public operation to the reviewed binding metadata and generated
   Terlan surface.
2. Keep native resources opaque and process-local; copy data across the
   package boundary unless an end-to-end lifetime model has been proved.
3. State hard limits for input size, output size, retained state, and work that
   may run synchronously.
4. Use explicit host capabilities for devices, paths, networking, UI, clocks,
   and accelerators. Do not infer them from strings or process-global state.
5. Add package tests for native behavior and a tutorial that executes through
   the guarded worker with exact expected output.
6. Update this contract and the category inventory when the capability is
   promoted.

