//! Inference-backed field-major class-score detection contract.

use super::dnn_field_major_detection_contract_support::{self, FieldMajorDetectionContract};
use super::NativeHelper;

const TRANSPOSED_RESHAPE_ONNX: &[u8] =
    include_bytes!("../../../tests/fixtures/transposed_class_score_reshape.onnx");
const BGR_FIELD_MAJOR_ROWS: [u8; 18] = [
    204, 77, 64, 26, 102, 102, 128, 51, 204, 51, 77, 64, 229, 102, 102, 128, 51, 204,
];

/// Runs the field-major class-score generated-helper contract.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    dnn_field_major_detection_contract_support::run(
        helper,
        FieldMajorDetectionContract {
            operation: "opencv.dnn.normalized_transposed_class_score_detections",
            model: TRANSPOSED_RESHAPE_ONNX,
            pixels: &BGR_FIELD_MAJOR_ROWS,
            fields: 6,
            threshold: 0.5,
            filtered_threshold: 0.85,
            label: "field-major class-score",
        },
    )
}
