# API Identity Audit

This Rust tool owns declaration identity discovery and implementation mapping
for pinned OpenCV 4.13.0. It reads the committed Clang inventory and reviewed
mapping ledger, reports identity coverage by OpenCV module, and rejects
unknown, duplicate, unsupported, or evidence-free mappings.

Identity coverage is an audit input, not the OpenCV package completion
denominator. C++ overloads, constructors, borrowed forms, and support machinery
can represent the same observable behavior. Product completion is measured by
the module behavioral-parity contract defined in the OpenCV roadmap.

Commands:

```sh
make opencv-api-parity-check
make opencv-api-parity-candidates
make opencv-api-parity-reconcile
make opencv-api-parity-live-check
make opencv-api-parity-strict-check
```

The ordinary check validates and reports mapping progress. The live check
reproduces the inventory from the pinned source checkout. The strict identity
check is a diagnostic that fails until every declaration has an exact mapping;
it is intentionally not a release-completion gate.

The candidate report joins maintained wrapper call metadata to exact inventory
identities. It removes identities already present in the reviewed mapping
ledger and groups the remainder by public Terlan operation. A candidate is a
review queue entry, not parity credit: composed wrappers, fixed policy choices,
and narrower Terlan domains still require an explicit behavioral review and
permanent executable evidence before they enter the ledger.
Mappings may name additional cooperating Terlan operations when one C++ API
uses optional inputs that the typed package exposes as explicit operations.

The reconciliation command regenerates
`bindings/opencv.api-reconciliation.json`. That document classifies every
public package operation as evidence-mapped or as an explicit behavioral gap.
The ordinary parity check rejects a stale reconciliation document.
