//! Inference-backed normalized SSD detection helper contract.

use super::{
    bytes_arg, expect_error_code, expect_floats, expect_int, expect_ints, expect_typed_handle,
    expect_unit, float_arg, int_arg, matrix, NativeHandle, NativeHelper,
};

const SSD_RESHAPE_ONNX: &[u8] = include_bytes!("../../../tests/fixtures/ssd_reshape.onnx");
const BGR_DETECTION_ROWS: [u8; 21] = [
    0, 0, 0, 0, 255, 255, 242, 204, 229, 0, 51, 26, 0, 51, 26, 255, 153, 128, 255, 153, 128,
];

/// Runs exact inference, filtering, ordering, ownership, protocol, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    let image = matrix(helper, &BGR_DETECTION_ROWS, 1, 7, "bgr")?;
    let blob = expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[image.arg(), int_arg(7), int_arg(1)],
        )?,
        "opencv.Dnn.Blob",
    )?;
    let model = expect_typed_handle(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(SSD_RESHAPE_ONNX)])?,
        "opencv.Dnn.OnnxModel",
    )?;
    let output = expect_typed_handle(
        helper.call("opencv.dnn.forward", &[model.arg(), blob.arg()])?,
        "opencv.Dnn.Output",
    )?;
    let detections = decode(helper, &output, 0.5, 3)?;
    assert_detections(helper, &detections, &[1, 1], 2)?;
    let repeated = decode(helper, &output, 0.5, 3)?;
    assert_detections(helper, &repeated, &[1, 1], 2)?;
    let filtered = decode(helper, &output, 0.85, 3)?;
    assert_detections(helper, &filtered, &[1], 1)?;
    let truncated = decode(helper, &output, 0.0, 1)?;
    assert_detections(helper, &truncated, &[1], 1)?;
    let empty = decode(helper, &output, 1.0, 3)?;
    assert_detections(helper, &empty, &[], 0)?;
    let suppressed = suppress(helper, &detections, 0.3, 3)?;
    assert_detections(helper, &suppressed, &[1], 1)?;
    let retained = suppress(helper, &detections, 0.5, 3)?;
    assert_detections(helper, &retained, &[1, 1], 2)?;
    let bounded = suppress(helper, &detections, 0.5, 1)?;
    assert_detections(helper, &bounded, &[1], 1)?;
    let repeated_suppression = suppress(helper, &detections, 0.3, 3)?;
    assert_detections(helper, &repeated_suppression, &[1], 1)?;
    let empty_suppression = suppress(helper, &empty, 0.3, 3)?;
    assert_detections(helper, &empty_suppression, &[], 0)?;
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 10)?;

    expect_error_code(
        helper.call(
            "opencv.dnn.ssd_detections",
            &[
                output.arg_with_owner("foreign-worker"),
                float_arg(0.5),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.ssd_detections",
            &[model.arg(), float_arg(0.5), int_arg(3)],
        )?,
        "handle_type_mismatch",
    )?;
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        expect_error_code(
            helper.call(
                "opencv.dnn.ssd_detections",
                &[output.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.dnn.ssd_detections",
            &[output.arg(), "i:1".to_owned(), int_arg(3)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.detection_count", &[model.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.non_max_suppression",
            &[
                detections.arg_with_owner("foreign-worker"),
                float_arg(0.3),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.non_max_suppression",
            &[model.arg(), float_arg(0.3), int_arg(3)],
        )?,
        "handle_type_mismatch",
    )?;
    for (threshold, maximum) in [
        (-0.1, 3),
        (1.1, 3),
        (f64::INFINITY, 3),
        (f64::NAN, 3),
        (0.5, 0),
        (0.5, 10001),
    ] {
        expect_error_code(
            helper.call(
                "opencv.dnn.non_max_suppression",
                &[detections.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.dnn.non_max_suppression",
            &[detections.arg(), "i:1".to_owned(), int_arg(3)],
        )?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.dnn.dispose_output", &[output.arg()])?)?;
    assert_detections(helper, &detections, &[1, 1], 2)?;
    expect_unit(helper.call("opencv.dnn.dispose_detections", &[detections.arg()])?)?;
    assert_detections(helper, &suppressed, &[1], 1)?;
    expect_error_code(
        helper.call("opencv.dnn.detection_count", &[detections.arg()])?,
        "stale_handle",
    )?;
    for owner in [
        &empty_suppression,
        &repeated_suppression,
        &bounded,
        &retained,
        &suppressed,
        &empty,
        &truncated,
        &filtered,
        &repeated,
    ] {
        expect_unit(helper.call("opencv.dnn.dispose_detections", &[owner.arg()])?)?;
    }
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[model.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose", &[blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)
}

/// Applies class-aware non-maximum suppression to one typed detection owner.
fn suppress(
    helper: &mut NativeHelper,
    detections: &NativeHandle,
    threshold: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.non_max_suppression",
            &[detections.arg(), float_arg(threshold), int_arg(maximum)],
        )?,
        "opencv.Dnn.Detections",
    )
}

/// Decodes one typed SSD detection owner.
fn decode(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    threshold: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.ssd_detections",
            &[output.arg(), float_arg(threshold), int_arg(maximum)],
        )?,
        "opencv.Dnn.Detections",
    )
}

/// Requires aligned copied classes, confidences, and normalized boxes.
fn assert_detections(
    helper: &mut NativeHelper,
    detections: &NativeHandle,
    classes: &[i64],
    count: i64,
) -> Result<(), String> {
    expect_int(
        helper.call("opencv.dnn.detection_count", &[detections.arg()])?,
        count,
    )?;
    let actual_classes =
        expect_ints(helper.call("opencv.dnn.detection_class_indices", &[detections.arg()])?)?;
    if actual_classes != classes {
        return Err(format!("unexpected detection classes: {actual_classes:?}"));
    }
    let confidences =
        expect_floats(helper.call("opencv.dnn.detection_confidences", &[detections.arg()])?)?;
    let boxes = expect_floats(helper.call("opencv.dnn.detection_boxes", &[detections.arg()])?)?;
    if confidences.len() != classes.len()
        || boxes.len() != classes.len() * 4
        || !confidences.iter().all(|value| (0.0..=1.0).contains(value))
        || !boxes.iter().all(|value| (0.0..=1.0).contains(value))
    {
        return Err("detection projections are malformed or unbounded".to_owned());
    }
    if classes.len() == 2
        && (confidences[0] <= confidences[1]
            || (confidences[0] - 229.0 / 255.0).abs() >= 1.0e-6
            || (confidences[1] - 204.0 / 255.0).abs() >= 1.0e-6)
    {
        return Err(format!("unexpected detection confidences: {confidences:?}"));
    }
    Ok(())
}
