# Native Test Fixtures

`minimal_onnx.proto` is the smallest protobuf schema needed to reproduce the
committed ONNX fixtures without an ONNX language package.

`identity_rgb.textproto` and `identity_grayscale.textproto` define fixed
Identity graphs for `[1,3,2,2]` and `[1,1,2,2]` tensors. Their encoded fixtures
replace handwritten Rust model bytes and prove both supported preprocessing
channel layouts against shape-specific models.

`ssd_reshape.textproto` defines a fixed Reshape graph from `[1,3,1,7]` to
`[1,1,3,7]`. `ssd_reshape.onnx` is its encoded form consumed by direct and
generated-helper tests.

`xyxy_reshape.textproto` defines a fixed Reshape graph from `[1,3,1,6]` to
`[1,3,6]` for normalized XYXY detection rows. Its encoded fixture is consumed
by the same direct and generated-helper detection contracts.

`objectness_reshape.textproto` defines a fixed Reshape graph from `[1,3,1,7]`
to `[1,3,7]` for center-XYWH, objectness, and two-class probability rows. Its
encoded fixture drives the direct, generated-helper, package, and consumer
contracts.

`transposed_class_score_reshape.textproto` defines a fixed Reshape graph from
`[1,3,1,6]` to field-major `[1,6,3]` center-XYWH and two-class score planes.
Its encoded fixture proves direct field access without an intermediate
transpose.

`transposed_objectness_reshape.textproto` defines the corresponding fixed
Reshape graph from `[1,3,1,7]` to `[1,7,3]`, adding one objectness plane before
the two class-score planes. Regenerate and compare all committed fixtures with:

```bash
make opencv-onnx-fixture-check
```

The reshape fixtures perform shape conversion only. Test image bytes provide
all SSD, six-field, and objectness row values through the package's fixed
BGR-to-RGB preprocessing contract.
