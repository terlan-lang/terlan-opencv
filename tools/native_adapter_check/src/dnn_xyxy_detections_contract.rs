//! Inference-backed normalized XYXY detection helper contract.

use super::{
    bytes_arg, expect_error_code, expect_floats, expect_int, expect_ints, expect_typed_handle,
    expect_unit, float_arg, int_arg, matrix, NativeHandle, NativeHelper,
};

const XYXY_RESHAPE_ONNX: &[u8] = include_bytes!("../../../tests/fixtures/xyxy_reshape.onnx");
const BGR_XYXY_ROWS: [u8; 18] = [
    179, 51, 26, 179, 51, 26, 229, 153, 102, 229, 153, 102, 102, 229, 204, 0, 255, 0,
];
const BGR_CENTER_XYWH_ROWS: [u8; 18] = [
    204, 102, 64, 204, 102, 64, 51, 102, 77, 51, 102, 77, 102, 229, 204, 0, 255, 0,
];
const BGR_TOP_LEFT_XYWH_ROWS: [u8; 18] = [
    179, 51, 26, 179, 51, 26, 51, 102, 77, 51, 102, 77, 102, 229, 204, 0, 255, 0,
];
const XYXY_OPERATION: &str = "opencv.dnn.normalized_xyxy_detections";
const CENTER_XYWH_OPERATION: &str = "opencv.dnn.normalized_center_xywh_detections";
const TOP_LEFT_XYWH_OPERATION: &str = "opencv.dnn.normalized_top_left_xywh_detections";

/// Runs inference, class-zero, filtering, ownership, protocol, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    let image = matrix(helper, &BGR_XYXY_ROWS, 1, 6, "bgr")?;
    let blob = expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[image.arg(), int_arg(6), int_arg(1)],
        )?,
        "opencv.Dnn.Blob",
    )?;
    let model = expect_typed_handle(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(XYXY_RESHAPE_ONNX)])?,
        "opencv.Dnn.OnnxModel",
    )?;
    let output = expect_typed_handle(
        helper.call("opencv.dnn.forward", &[model.arg(), blob.arg()])?,
        "opencv.Dnn.Output",
    )?;
    let shape = expect_ints(helper.call("opencv.dnn.output_shape", &[output.arg()])?)?;
    if shape != [1, 3, 6] {
        return Err(format!("unexpected XYXY output shape: {shape:?}"));
    }

    let detections = decode(helper, XYXY_OPERATION, &output, 0.5, 3)?;
    assert_detections(helper, &detections, &[1, 0])?;
    let repeated = decode(helper, XYXY_OPERATION, &output, 0.5, 3)?;
    assert_detections(helper, &repeated, &[1, 0])?;
    let filtered = decode(helper, XYXY_OPERATION, &output, 0.85, 3)?;
    assert_detections(helper, &filtered, &[1])?;
    let bounded = decode(helper, XYXY_OPERATION, &output, 0.0, 1)?;
    assert_detections(helper, &bounded, &[1])?;
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 4)?;

    expect_error_code(
        helper.call(
            XYXY_OPERATION,
            &[
                output.arg_with_owner("foreign-worker"),
                float_arg(0.5),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(XYXY_OPERATION, &[model.arg(), float_arg(0.5), int_arg(3)])?,
        "handle_type_mismatch",
    )?;
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        expect_error_code(
            helper.call(
                XYXY_OPERATION,
                &[output.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            XYXY_OPERATION,
            &[output.arg(), "i:1".to_owned(), int_arg(3)],
        )?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.dnn.dispose_output", &[output.arg()])?)?;
    assert_detections(helper, &detections, &[1, 0])?;
    for owner in [&bounded, &filtered, &repeated, &detections] {
        expect_unit(helper.call("opencv.dnn.dispose_detections", &[owner.arg()])?)?;
    }
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[model.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose", &[blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)?;
    run_center_xywh(helper)
}

/// Runs center-XYWH conversion and generated-helper failure checks.
fn run_center_xywh(helper: &mut NativeHelper) -> Result<(), String> {
    let image = matrix(helper, &BGR_CENTER_XYWH_ROWS, 1, 6, "bgr")?;
    let blob = expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[image.arg(), int_arg(6), int_arg(1)],
        )?,
        "opencv.Dnn.Blob",
    )?;
    let model = expect_typed_handle(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(XYXY_RESHAPE_ONNX)])?,
        "opencv.Dnn.OnnxModel",
    )?;
    let output = expect_typed_handle(
        helper.call("opencv.dnn.forward", &[model.arg(), blob.arg()])?,
        "opencv.Dnn.Output",
    )?;
    let detections = decode(helper, CENTER_XYWH_OPERATION, &output, 0.5, 3)?;
    assert_detections(helper, &detections, &[1, 0])?;
    let boxes = expect_floats(helper.call("opencv.dnn.detection_boxes", &[detections.arg()])?)?;
    if boxes.len() != 8
        || (boxes[0] - 0.2).abs() >= 1.0e-6
        || (boxes[1] - 0.2).abs() >= 1.0e-6
        || (boxes[2] - 0.6).abs() >= 1.0e-6
        || (boxes[3] - 0.6).abs() >= 1.0e-6
    {
        return Err(format!("unexpected center-XYWH conversion: {boxes:?}"));
    }
    let repeated = decode(helper, CENTER_XYWH_OPERATION, &output, 0.5, 3)?;
    assert_detections(helper, &repeated, &[1, 0])?;
    let filtered = decode(helper, CENTER_XYWH_OPERATION, &output, 0.85, 3)?;
    assert_detections(helper, &filtered, &[1])?;
    let bounded = decode(helper, CENTER_XYWH_OPERATION, &output, 0.0, 1)?;
    assert_detections(helper, &bounded, &[1])?;

    expect_error_code(
        helper.call(
            CENTER_XYWH_OPERATION,
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
            CENTER_XYWH_OPERATION,
            &[model.arg(), float_arg(0.5), int_arg(3)],
        )?,
        "handle_type_mismatch",
    )?;
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        expect_error_code(
            helper.call(
                CENTER_XYWH_OPERATION,
                &[output.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            CENTER_XYWH_OPERATION,
            &[output.arg(), "i:1".to_owned(), int_arg(3)],
        )?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.dnn.dispose_output", &[output.arg()])?)?;
    assert_detections(helper, &detections, &[1, 0])?;
    for owner in [&bounded, &filtered, &repeated, &detections] {
        expect_unit(helper.call("opencv.dnn.dispose_detections", &[owner.arg()])?)?;
    }
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[model.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose", &[blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)?;
    run_top_left_xywh(helper)
}

/// Runs top-left-XYWH conversion and generated-helper failure checks.
fn run_top_left_xywh(helper: &mut NativeHelper) -> Result<(), String> {
    let image = matrix(helper, &BGR_TOP_LEFT_XYWH_ROWS, 1, 6, "bgr")?;
    let blob = expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[image.arg(), int_arg(6), int_arg(1)],
        )?,
        "opencv.Dnn.Blob",
    )?;
    let model = expect_typed_handle(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(XYXY_RESHAPE_ONNX)])?,
        "opencv.Dnn.OnnxModel",
    )?;
    let output = expect_typed_handle(
        helper.call("opencv.dnn.forward", &[model.arg(), blob.arg()])?,
        "opencv.Dnn.Output",
    )?;
    let detections = decode(helper, TOP_LEFT_XYWH_OPERATION, &output, 0.5, 3)?;
    assert_detections(helper, &detections, &[1, 0])?;
    let boxes = expect_floats(helper.call("opencv.dnn.detection_boxes", &[detections.arg()])?)?;
    if boxes.len() != 8
        || (boxes[0] - 0.2).abs() >= 1.0e-6
        || (boxes[1] - 0.2).abs() >= 1.0e-6
        || (boxes[2] - 0.6).abs() >= 1.0e-6
        || (boxes[3] - 0.6).abs() >= 1.0e-6
    {
        return Err(format!("unexpected top-left-XYWH conversion: {boxes:?}"));
    }
    let repeated = decode(helper, TOP_LEFT_XYWH_OPERATION, &output, 0.5, 3)?;
    assert_detections(helper, &repeated, &[1, 0])?;
    let filtered = decode(helper, TOP_LEFT_XYWH_OPERATION, &output, 0.85, 3)?;
    assert_detections(helper, &filtered, &[1])?;
    let bounded = decode(helper, TOP_LEFT_XYWH_OPERATION, &output, 0.0, 1)?;
    assert_detections(helper, &bounded, &[1])?;

    expect_error_code(
        helper.call(
            TOP_LEFT_XYWH_OPERATION,
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
            TOP_LEFT_XYWH_OPERATION,
            &[model.arg(), float_arg(0.5), int_arg(3)],
        )?,
        "handle_type_mismatch",
    )?;
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        expect_error_code(
            helper.call(
                TOP_LEFT_XYWH_OPERATION,
                &[output.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            TOP_LEFT_XYWH_OPERATION,
            &[output.arg(), "i:1".to_owned(), int_arg(3)],
        )?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.dnn.dispose_output", &[output.arg()])?)?;
    assert_detections(helper, &detections, &[1, 0])?;
    for owner in [&bounded, &filtered, &repeated, &detections] {
        expect_unit(helper.call("opencv.dnn.dispose_detections", &[owner.arg()])?)?;
    }
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[model.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose", &[blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)
}

/// Decodes one typed normalized XYXY output.
fn decode(
    helper: &mut NativeHelper,
    operation: &str,
    output: &NativeHandle,
    threshold: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            operation,
            &[output.arg(), float_arg(threshold), int_arg(maximum)],
        )?,
        "opencv.Dnn.Detections",
    )
}

/// Requires copied classes, confidence order, and aligned normalized boxes.
fn assert_detections(
    helper: &mut NativeHelper,
    detections: &NativeHandle,
    classes: &[i64],
) -> Result<(), String> {
    expect_int(
        helper.call("opencv.dnn.detection_count", &[detections.arg()])?,
        classes.len() as i64,
    )?;
    let actual =
        expect_ints(helper.call("opencv.dnn.detection_class_indices", &[detections.arg()])?)?;
    let confidence =
        expect_floats(helper.call("opencv.dnn.detection_confidences", &[detections.arg()])?)?;
    let boxes = expect_floats(helper.call("opencv.dnn.detection_boxes", &[detections.arg()])?)?;
    if actual != classes
        || confidence.len() != classes.len()
        || boxes.len() != classes.len() * 4
        || !confidence.windows(2).all(|pair| pair[0] >= pair[1])
        || !boxes.iter().all(|value| (0.0..=1.0).contains(value))
    {
        return Err(format!(
            "malformed XYXY detections: classes={actual:?}, confidence={confidence:?}, boxes={boxes:?}"
        ));
    }
    Ok(())
}
