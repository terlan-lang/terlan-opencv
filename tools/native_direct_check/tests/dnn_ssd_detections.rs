//! Direct deterministic detection projection tests.

mod support;

use support::{
    OBJECTNESS_RESHAPE_ONNX, SSD_RESHAPE_ONNX, TRANSPOSED_CLASS_SCORE_RESHAPE_ONNX,
    TRANSPOSED_OBJECTNESS_RESHAPE_ONNX, XYXY_RESHAPE_ONNX,
};
use terlan_opencv_native::ffi;

const BGR_DETECTION_ROWS: [u8; 21] = [
    0, 0, 0, 0, 255, 255, 242, 204, 229, 0, 51, 26, 0, 51, 26, 255, 153, 128, 255, 153, 128,
];
const BGR_XYXY_ROWS: [u8; 18] = [
    179, 51, 26, 179, 51, 26, 229, 153, 102, 229, 153, 102, 102, 229, 204, 0, 255, 0,
];
const BGR_CENTER_XYWH_ROWS: [u8; 18] = [
    204, 102, 64, 204, 102, 64, 51, 102, 77, 51, 102, 77, 102, 229, 204, 0, 255, 0,
];
const BGR_TOP_LEFT_XYWH_ROWS: [u8; 18] = [
    179, 51, 26, 179, 51, 26, 51, 102, 77, 51, 102, 77, 102, 229, 204, 0, 255, 0,
];
const BGR_OBJECTNESS_ROWS: [u8; 21] = [
    204, 102, 64, 204, 102, 64, 51, 102, 77, 51, 102, 77, 128, 229, 204, 128, 26, 204, 128, 229, 51,
];
const BGR_CLASS_SCORE_ROWS: [u8; 18] = [
    204, 102, 64, 204, 102, 64, 51, 102, 77, 51, 102, 77, 128, 26, 204, 128, 229, 51,
];
const BGR_FIELD_MAJOR_ROWS: [u8; 18] = [
    204, 77, 64, 26, 102, 102, 128, 51, 204, 51, 77, 64, 229, 102, 102, 128, 51, 204,
];
const BGR_FIELD_MAJOR_OBJECTNESS: [u8; 21] = [
    128, 102, 64, 204, 51, 102, 26, 77, 204, 128, 102, 64, 51, 51, 102, 229, 204, 204, 128, 229, 77,
];

/// Runs every detection projection serially against process-global owner counters.
#[test]
fn detection_projection_contracts() {
    decodes_normalized_xyxy_rows();
    decodes_normalized_center_xywh_rows();
    decodes_normalized_top_left_xywh_rows();
    decodes_normalized_objectness_rows();
    decodes_normalized_class_score_rows();
    decodes_normalized_transposed_class_score_rows();
    decodes_normalized_transposed_objectness_rows();
    decodes_normalized_ssd_rows();
}

/// Proves field-major objectness decoding without an intermediate transpose.
fn decodes_normalized_transposed_objectness_rows() {
    assert_eq!(ffi::live_dnn_detections_count(), 0);
    let image = ffi::mat_from_u8(&BGR_FIELD_MAJOR_OBJECTNESS, 1, 7, 3);
    let blob = image
        .as_ref()
        .expect("field-major objectness image")
        .blob_from_bgr_rgb_owned(7, 1);
    let model = ffi::load_onnx_model_owned(TRANSPOSED_OBJECTNESS_RESHAPE_ONNX);
    let output = model
        .as_ref()
        .expect("field-major objectness model")
        .forward_owned(blob.as_ref().expect("field-major objectness blob"));
    let output_ref = output.as_ref().expect("field-major objectness output");
    assert_eq!(copied_shape(output_ref), [1, 7, 3]);

    let detections = output_ref.normalized_transposed_objectness_detections_owned(0.2, 3);
    assert_detections(
        &detections,
        &[1, 0, 0],
        &[
            (229.0 / 255.0) * (229.0 / 255.0),
            (204.0 / 255.0) * (204.0 / 255.0),
            (128.0 / 255.0) * (128.0 / 255.0),
        ],
    );
    let repeated = output_ref.normalized_transposed_objectness_detections_owned(0.2, 3);
    assert_detections(
        &repeated,
        &[1, 0, 0],
        &[
            (229.0 / 255.0) * (229.0 / 255.0),
            (204.0 / 255.0) * (204.0 / 255.0),
            (128.0 / 255.0) * (128.0 / 255.0),
        ],
    );
    let filtered = output_ref.normalized_transposed_objectness_detections_owned(0.7, 3);
    assert_detections(&filtered, &[1], &[(229.0 / 255.0) * (229.0 / 255.0)]);
    let bounded = output_ref.normalized_transposed_objectness_detections_owned(0.0, 1);
    assert_detections(&bounded, &[1], &[(229.0 / 255.0) * (229.0 / 255.0)]);
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.2, 0), (0.2, 10001)] {
        assert!(output_ref
            .normalized_transposed_objectness_detections_owned(threshold, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(output);
    drop(model);
    drop(blob);
    drop(image);
    assert_eq!(
        copied_ints(
            detections
                .as_ref()
                .expect("retained field-major objectness detections")
        ),
        [1, 0, 0]
    );
    drop(bounded);
    drop(filtered);
    drop(repeated);
    drop(detections);
    assert_eq!(ffi::live_dnn_detections_count(), 0);
}

/// Proves field-major class-score decoding without an intermediate transpose.
fn decodes_normalized_transposed_class_score_rows() {
    assert_eq!(ffi::live_dnn_detections_count(), 0);
    let image = ffi::mat_from_u8(&BGR_FIELD_MAJOR_ROWS, 1, 6, 3);
    let blob = image
        .as_ref()
        .expect("field-major image")
        .blob_from_bgr_rgb_owned(6, 1);
    let model = ffi::load_onnx_model_owned(TRANSPOSED_CLASS_SCORE_RESHAPE_ONNX);
    let output = model
        .as_ref()
        .expect("field-major reshape model")
        .forward_owned(blob.as_ref().expect("field-major blob"));
    let output_ref = output.as_ref().expect("field-major output");
    assert_eq!(copied_shape(output_ref), [1, 6, 3]);

    let detections = output_ref.normalized_transposed_class_score_detections_owned(0.5, 3);
    assert_detections(
        &detections,
        &[1, 0, 0],
        &[229.0 / 255.0, 204.0 / 255.0, 128.0 / 255.0],
    );
    let repeated = output_ref.normalized_transposed_class_score_detections_owned(0.5, 3);
    assert_detections(
        &repeated,
        &[1, 0, 0],
        &[229.0 / 255.0, 204.0 / 255.0, 128.0 / 255.0],
    );
    let filtered = output_ref.normalized_transposed_class_score_detections_owned(0.85, 3);
    assert_detections(&filtered, &[1], &[229.0 / 255.0]);
    let bounded = output_ref.normalized_transposed_class_score_detections_owned(0.0, 1);
    assert_detections(&bounded, &[1], &[229.0 / 255.0]);
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        assert!(output_ref
            .normalized_transposed_class_score_detections_owned(threshold, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(output);
    drop(model);
    drop(blob);
    drop(image);
    assert_eq!(
        copied_ints(
            detections
                .as_ref()
                .expect("retained field-major detections")
        ),
        [1, 0, 0]
    );
    drop(bounded);
    drop(filtered);
    drop(repeated);
    drop(detections);
    assert_eq!(ffi::live_dnn_detections_count(), 0);
}

/// Proves variable-class score decoding without an objectness field.
fn decodes_normalized_class_score_rows() {
    assert_eq!(ffi::live_dnn_detections_count(), 0);
    let image = ffi::mat_from_u8(&BGR_CLASS_SCORE_ROWS, 1, 6, 3);
    let blob = image
        .as_ref()
        .expect("class-score image")
        .blob_from_bgr_rgb_owned(6, 1);
    let model = ffi::load_onnx_model_owned(XYXY_RESHAPE_ONNX);
    let output = model
        .as_ref()
        .expect("class-score reshape model")
        .forward_owned(blob.as_ref().expect("class-score blob"));
    let output_ref = output.as_ref().expect("class-score output");
    assert_eq!(copied_shape(output_ref), [1, 3, 6]);

    let detections = output_ref.normalized_class_score_detections_owned(0.5, 3);
    assert_detections(
        &detections,
        &[1, 0, 0],
        &[229.0 / 255.0, 204.0 / 255.0, 128.0 / 255.0],
    );
    let repeated = output_ref.normalized_class_score_detections_owned(0.5, 3);
    assert_detections(
        &repeated,
        &[1, 0, 0],
        &[229.0 / 255.0, 204.0 / 255.0, 128.0 / 255.0],
    );
    let filtered = output_ref.normalized_class_score_detections_owned(0.85, 3);
    assert_detections(&filtered, &[1], &[229.0 / 255.0]);
    let bounded = output_ref.normalized_class_score_detections_owned(0.0, 1);
    assert_detections(&bounded, &[1], &[229.0 / 255.0]);
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        assert!(output_ref
            .normalized_class_score_detections_owned(threshold, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(output);
    drop(model);
    drop(blob);
    drop(image);
    assert_eq!(
        copied_ints(
            detections
                .as_ref()
                .expect("retained class-score detections")
        ),
        [1, 0, 0]
    );
    drop(bounded);
    drop(filtered);
    drop(repeated);
    drop(detections);
    assert_eq!(ffi::live_dnn_detections_count(), 0);
}

/// Proves variable-class objectness decoding and lowest-index class ties.
fn decodes_normalized_objectness_rows() {
    assert_eq!(ffi::live_dnn_detections_count(), 0);
    let image = ffi::mat_from_u8(&BGR_OBJECTNESS_ROWS, 1, 7, 3);
    let blob = image
        .as_ref()
        .expect("objectness image")
        .blob_from_bgr_rgb_owned(7, 1);
    let model = ffi::load_onnx_model_owned(OBJECTNESS_RESHAPE_ONNX);
    let output = model
        .as_ref()
        .expect("objectness reshape model")
        .forward_owned(blob.as_ref().expect("objectness blob"));
    let output_ref = output.as_ref().expect("objectness output");
    assert_eq!(copied_shape(output_ref), [1, 3, 7]);

    let detections = output_ref.normalized_objectness_detections_owned(0.2, 3);
    assert_detections(
        &detections,
        &[1, 0, 0],
        &[
            (229.0 / 255.0) * (229.0 / 255.0),
            (204.0 / 255.0) * (204.0 / 255.0),
            (128.0 / 255.0) * (128.0 / 255.0),
        ],
    );
    let repeated = output_ref.normalized_objectness_detections_owned(0.2, 3);
    assert_detections(
        &repeated,
        &[1, 0, 0],
        &[
            (229.0 / 255.0) * (229.0 / 255.0),
            (204.0 / 255.0) * (204.0 / 255.0),
            (128.0 / 255.0) * (128.0 / 255.0),
        ],
    );
    let filtered = output_ref.normalized_objectness_detections_owned(0.7, 3);
    assert_detections(&filtered, &[1], &[(229.0 / 255.0) * (229.0 / 255.0)]);
    let bounded = output_ref.normalized_objectness_detections_owned(0.0, 1);
    assert_detections(&bounded, &[1], &[(229.0 / 255.0) * (229.0 / 255.0)]);
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.2, 0), (0.2, 10001)] {
        assert!(output_ref
            .normalized_objectness_detections_owned(threshold, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(output);
    drop(model);
    drop(blob);
    drop(image);
    assert_eq!(
        copied_ints(detections.as_ref().expect("retained objectness detections")),
        [1, 0, 0]
    );
    drop(bounded);
    drop(filtered);
    drop(repeated);
    drop(detections);
    assert_eq!(ffi::live_dnn_detections_count(), 0);
}

/// Proves generic rank-three XYXY decoding without SSD background or implicit NMS policy.
fn decodes_normalized_xyxy_rows() {
    assert_eq!(ffi::live_dnn_detections_count(), 0);
    let image = ffi::mat_from_u8(&BGR_XYXY_ROWS, 1, 6, 3);
    let blob = image
        .as_ref()
        .expect("XYXY image")
        .blob_from_bgr_rgb_owned(6, 1);
    let model = ffi::load_onnx_model_owned(XYXY_RESHAPE_ONNX);
    let output = model
        .as_ref()
        .expect("XYXY reshape model")
        .forward_owned(blob.as_ref().expect("XYXY blob"));
    let output_ref = output.as_ref().expect("XYXY output");
    assert_eq!(copied_shape(output_ref), [1, 3, 6]);

    let detections = output_ref.normalized_xyxy_detections_owned(0.5, 3);
    assert_detections(&detections, &[1, 0], &[229.0 / 255.0, 204.0 / 255.0]);
    let repeated = output_ref.normalized_xyxy_detections_owned(0.5, 3);
    assert_detections(&repeated, &[1, 0], &[229.0 / 255.0, 204.0 / 255.0]);
    let filtered = output_ref.normalized_xyxy_detections_owned(0.85, 3);
    assert_detections(&filtered, &[1], &[229.0 / 255.0]);
    let bounded = output_ref.normalized_xyxy_detections_owned(0.0, 1);
    assert_detections(&bounded, &[1], &[229.0 / 255.0]);
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        assert!(output_ref
            .normalized_xyxy_detections_owned(threshold, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(output);
    drop(model);
    drop(blob);
    drop(image);
    assert_eq!(
        copied_ints(detections.as_ref().expect("retained XYXY detections")),
        [1, 0]
    );
    drop(bounded);
    drop(filtered);
    drop(repeated);
    drop(detections);
    assert_eq!(ffi::live_dnn_detections_count(), 0);
}

/// Proves center-XYWH conversion, filtering, ordering, and independent ownership.
fn decodes_normalized_center_xywh_rows() {
    assert_eq!(ffi::live_dnn_detections_count(), 0);
    let image = ffi::mat_from_u8(&BGR_CENTER_XYWH_ROWS, 1, 6, 3);
    let blob = image
        .as_ref()
        .expect("center-XYWH image")
        .blob_from_bgr_rgb_owned(6, 1);
    let model = ffi::load_onnx_model_owned(XYXY_RESHAPE_ONNX);
    let output = model
        .as_ref()
        .expect("center-XYWH reshape model")
        .forward_owned(blob.as_ref().expect("center-XYWH blob"));
    let output_ref = output.as_ref().expect("center-XYWH output");
    assert_eq!(copied_shape(output_ref), [1, 3, 6]);

    let detections = output_ref.normalized_center_xywh_detections_owned(0.5, 3);
    assert_detections(&detections, &[1, 0], &[229.0 / 255.0, 204.0 / 255.0]);
    let boxes = copied_boxes(detections.as_ref().expect("center-XYWH detections"));
    assert!((boxes[0] - 0.2).abs() < 1.0e-6);
    assert!((boxes[1] - 0.2).abs() < 1.0e-6);
    assert!((boxes[2] - 0.6).abs() < 1.0e-6);
    assert!((boxes[3] - 0.6).abs() < 1.0e-6);
    let repeated = output_ref.normalized_center_xywh_detections_owned(0.5, 3);
    assert_detections(&repeated, &[1, 0], &[229.0 / 255.0, 204.0 / 255.0]);
    let filtered = output_ref.normalized_center_xywh_detections_owned(0.85, 3);
    assert_detections(&filtered, &[1], &[229.0 / 255.0]);
    let bounded = output_ref.normalized_center_xywh_detections_owned(0.0, 1);
    assert_detections(&bounded, &[1], &[229.0 / 255.0]);
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        assert!(output_ref
            .normalized_center_xywh_detections_owned(threshold, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(output);
    drop(model);
    drop(blob);
    drop(image);
    assert_eq!(
        copied_ints(
            detections
                .as_ref()
                .expect("retained center-XYWH detections")
        ),
        [1, 0]
    );
    drop(bounded);
    drop(filtered);
    drop(repeated);
    drop(detections);
    assert_eq!(ffi::live_dnn_detections_count(), 0);
}

/// Proves top-left-XYWH conversion, filtering, ordering, and ownership.
fn decodes_normalized_top_left_xywh_rows() {
    assert_eq!(ffi::live_dnn_detections_count(), 0);
    let image = ffi::mat_from_u8(&BGR_TOP_LEFT_XYWH_ROWS, 1, 6, 3);
    let blob = image
        .as_ref()
        .expect("top-left-XYWH image")
        .blob_from_bgr_rgb_owned(6, 1);
    let model = ffi::load_onnx_model_owned(XYXY_RESHAPE_ONNX);
    let output = model
        .as_ref()
        .expect("top-left-XYWH reshape model")
        .forward_owned(blob.as_ref().expect("top-left-XYWH blob"));
    let output_ref = output.as_ref().expect("top-left-XYWH output");
    assert_eq!(copied_shape(output_ref), [1, 3, 6]);

    let detections = output_ref.normalized_top_left_xywh_detections_owned(0.5, 3);
    assert_detections(&detections, &[1, 0], &[229.0 / 255.0, 204.0 / 255.0]);
    let boxes = copied_boxes(detections.as_ref().expect("top-left-XYWH detections"));
    assert!((boxes[0] - 0.2).abs() < 1.0e-6);
    assert!((boxes[1] - 0.2).abs() < 1.0e-6);
    assert!((boxes[2] - 0.6).abs() < 1.0e-6);
    assert!((boxes[3] - 0.6).abs() < 1.0e-6);
    let repeated = output_ref.normalized_top_left_xywh_detections_owned(0.5, 3);
    assert_detections(&repeated, &[1, 0], &[229.0 / 255.0, 204.0 / 255.0]);
    let filtered = output_ref.normalized_top_left_xywh_detections_owned(0.85, 3);
    assert_detections(&filtered, &[1], &[229.0 / 255.0]);
    let bounded = output_ref.normalized_top_left_xywh_detections_owned(0.0, 1);
    assert_detections(&bounded, &[1], &[229.0 / 255.0]);
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        assert!(output_ref
            .normalized_top_left_xywh_detections_owned(threshold, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(output);
    drop(model);
    drop(blob);
    drop(image);
    assert_eq!(
        copied_ints(
            detections
                .as_ref()
                .expect("retained top-left-XYWH detections")
        ),
        [1, 0]
    );
    drop(bounded);
    drop(filtered);
    drop(repeated);
    drop(detections);
    assert_eq!(ffi::live_dnn_detections_count(), 0);
}

/// Proves inference-backed decoding, ordering, filtering, truncation, and independent ownership.
fn decodes_normalized_ssd_rows() {
    assert_eq!(ffi::live_dnn_detections_count(), 0);
    let image = ffi::mat_from_u8(&BGR_DETECTION_ROWS, 1, 7, 3);
    let blob = image
        .as_ref()
        .expect("detection image")
        .blob_from_bgr_rgb_owned(7, 1);
    let model = ffi::load_onnx_model_owned(SSD_RESHAPE_ONNX);
    let output = model
        .as_ref()
        .expect("reshape model")
        .forward_owned(blob.as_ref().expect("detection blob"));
    let output_ref = output.as_ref().expect("SSD-shaped output");
    assert_eq!(copied_shape(output_ref), [1, 1, 3, 7]);
    assert!(output_ref
        .normalized_xyxy_detections_owned(0.5, 3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(output_ref
        .normalized_center_xywh_detections_owned(0.5, 3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(output_ref
        .normalized_top_left_xywh_detections_owned(0.5, 3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(output_ref
        .normalized_objectness_detections_owned(0.5, 3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(output_ref
        .normalized_class_score_detections_owned(0.5, 3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(output_ref
        .normalized_transposed_class_score_detections_owned(0.5, 3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(output_ref
        .normalized_transposed_objectness_detections_owned(0.5, 3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    let detections = ffi::ssd_detections_owned(output_ref, 0.5, 3);
    assert_detections(&detections, &[1, 1], &[229.0 / 255.0, 204.0 / 255.0]);
    let repeated = ffi::ssd_detections_owned(output_ref, 0.5, 3);
    assert_detections(&repeated, &[1, 1], &[229.0 / 255.0, 204.0 / 255.0]);
    let filtered = ffi::ssd_detections_owned(output_ref, 0.85, 3);
    assert_detections(&filtered, &[1], &[229.0 / 255.0]);
    let truncated = ffi::ssd_detections_owned(output_ref, 0.0, 1);
    assert_detections(&truncated, &[1], &[229.0 / 255.0]);
    let empty = ffi::ssd_detections_owned(output_ref, 1.0, 3);
    assert_detections(&empty, &[], &[]);
    let suppressed = detections
        .as_ref()
        .expect("source detections")
        .non_max_suppression_owned(0.3, 3);
    assert_detections(&suppressed, &[1], &[229.0 / 255.0]);
    let retained = detections
        .as_ref()
        .expect("source detections")
        .non_max_suppression_owned(0.5, 3);
    assert_detections(&retained, &[1, 1], &[229.0 / 255.0, 204.0 / 255.0]);
    let bounded = detections
        .as_ref()
        .expect("source detections")
        .non_max_suppression_owned(0.5, 1);
    assert_detections(&bounded, &[1], &[229.0 / 255.0]);
    let repeated_suppression = detections
        .as_ref()
        .expect("source detections")
        .non_max_suppression_owned(0.3, 3);
    assert_detections(&repeated_suppression, &[1], &[229.0 / 255.0]);
    let empty_suppression = empty
        .as_ref()
        .expect("empty detections")
        .non_max_suppression_owned(0.3, 3);
    assert_detections(&empty_suppression, &[], &[]);

    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        assert!(ffi::ssd_detections_owned(output_ref, threshold, maximum).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (threshold, maximum) in [
        (-0.1, 3),
        (1.1, 3),
        (f64::INFINITY, 3),
        (f64::NAN, 3),
        (0.5, 0),
        (0.5, 10001),
    ] {
        assert!(detections
            .as_ref()
            .expect("source detections")
            .non_max_suppression_owned(threshold, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(output);
    drop(model);
    drop(blob);
    drop(image);
    assert_eq!(
        copied_ints(detections.as_ref().expect("retained detections")),
        [1, 1]
    );
    drop(detections);
    assert_eq!(
        copied_ints(suppressed.as_ref().expect("independent suppression")),
        [1]
    );
    drop(empty_suppression);
    drop(repeated_suppression);
    drop(bounded);
    drop(retained);
    drop(suppressed);
    drop(empty);
    drop(truncated);
    drop(filtered);
    drop(repeated);
    assert_eq!(ffi::live_dnn_detections_count(), 0);
}

/// Requires exact classes, approximate confidences, and aligned normalized boxes.
fn assert_detections(
    detections: &cxx::UniquePtr<ffi::OpenCvDnnDetections>,
    classes: &[i64],
    confidences: &[f64],
) {
    let detections = detections.as_ref().expect("SSD detections");
    assert_eq!(detections.count(), classes.len() as i64);
    assert_eq!(copied_ints(detections), classes);
    let actual_confidences = copied_confidences(detections);
    assert_eq!(actual_confidences.len(), confidences.len());
    for (actual, expected) in actual_confidences.iter().zip(confidences) {
        assert!((actual - expected).abs() < 1.0e-6);
    }
    let boxes = copied_boxes(detections);
    assert_eq!(boxes.len(), classes.len() * 4);
    assert!(boxes.iter().all(|value| (0.0..=1.0).contains(value)));
}

/// Copies one generated CXX detection class vector.
fn copied_ints(detections: &ffi::OpenCvDnnDetections) -> Vec<i64> {
    detections
        .class_indices_owned()
        .as_ref()
        .expect("copied class indexes")
        .iter()
        .copied()
        .collect()
}

/// Copies one generated CXX confidence vector.
fn copied_confidences(detections: &ffi::OpenCvDnnDetections) -> Vec<f64> {
    detections
        .confidences_owned()
        .as_ref()
        .expect("copied confidences")
        .iter()
        .copied()
        .collect()
}

/// Copies one generated CXX normalized box vector.
fn copied_boxes(detections: &ffi::OpenCvDnnDetections) -> Vec<f64> {
    detections
        .boxes_owned()
        .as_ref()
        .expect("copied boxes")
        .iter()
        .copied()
        .collect()
}

/// Copies one generated CXX output shape vector.
fn copied_shape(output: &ffi::OpenCvDnnOutput) -> Vec<i64> {
    output
        .shape_owned()
        .as_ref()
        .expect("copied output shape")
        .iter()
        .copied()
        .collect()
}
