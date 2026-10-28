# OpenCV API Inventory

This directory defines the complete public runtime API denominator for pinned
OpenCV 4.13.0.

`opencv_public_api_inventory.cc` includes the umbrella headers for all runtime
modules covered by Terlan parity. The parity tool selects declarations from
216 exact public CPU C++ header origins under the pinned source tree.
Accelerator-specific, private/detail, inline implementation, and legacy C
headers are outside this portable denominator. `compile_commands.json`
provides the maintained Clang extraction context. The normalized, committed
result is `../opencv.api-inventory.json`.

Run the offline mapping report:

```sh
make opencv-api-parity-check
```

Reproduce the inventory from the pinned OpenCV checkout:

```sh
make opencv-api-parity-live-check
```

The strict gate is expected to fail while identities remain unmapped:

```sh
make opencv-api-parity-strict-check
```

Mappings live in `../opencv.api-parity.toml`. A mapping is accepted only when
it names exact upstream identities, a direct or behavior-preserving Terlan
operation, its reshaping rationale, and permanent executable evidence.
