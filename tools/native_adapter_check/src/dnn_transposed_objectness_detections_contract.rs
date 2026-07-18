//! Inference-backed field-major objectness detection contract.

use super::dnn_field_major_detection_contract_support::{self, FieldMajorDetectionContract};
use super::NativeHelper;

const TRANSPOSED_OBJECTNESS_RESHAPE_ONNX: &[u8] =
    include_bytes!("../../../tests/fixtures/transposed_objectness_reshape.onnx");
const BGR_FIELD_MAJOR_OBJECTNESS: [u8; 21] = [
    128, 102, 64, 204, 51, 102, 26, 77, 204, 128, 102, 64, 51, 51, 102, 229, 204, 204, 128, 229, 77,
];

/// Runs the field-major objectness generated-helper contract.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    dnn_field_major_detection_contract_support::run(
        helper,
        FieldMajorDetectionContract {
            operation: "opencv.dnn.normalized_transposed_objectness_detections",
            model: TRANSPOSED_OBJECTNESS_RESHAPE_ONNX,
            pixels: &BGR_FIELD_MAJOR_OBJECTNESS,
            fields: 7,
            threshold: 0.2,
            filtered_threshold: 0.7,
            label: "field-major objectness",
        },
    )
}
