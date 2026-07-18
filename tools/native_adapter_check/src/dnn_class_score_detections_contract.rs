//! Inference-backed objectness-free class-score detection contract.

use super::{
    bytes_arg, expect_error_code, expect_floats, expect_int, expect_ints, expect_typed_handle,
    expect_unit, float_arg, int_arg, matrix, NativeHandle, NativeHelper,
};

const XYXY_RESHAPE_ONNX: &[u8] = include_bytes!("../../../tests/fixtures/xyxy_reshape.onnx");
const BGR_CLASS_SCORE_ROWS: [u8; 18] = [
    204, 102, 64, 204, 102, 64, 51, 102, 77, 51, 102, 77, 128, 26, 204, 128, 229, 51,
];
const OPERATION: &str = "opencv.dnn.normalized_class_score_detections";

/// Runs class-score decoding, protocol rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    let image = matrix(helper, &BGR_CLASS_SCORE_ROWS, 1, 6, "bgr")?;
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
        return Err(format!("unexpected class-score output shape: {shape:?}"));
    }

    let detections = decode(helper, &output, 0.5, 3)?;
    assert_detections(helper, &detections, &[1, 0, 0])?;
    let repeated = decode(helper, &output, 0.5, 3)?;
    assert_detections(helper, &repeated, &[1, 0, 0])?;
    let filtered = decode(helper, &output, 0.85, 3)?;
    assert_detections(helper, &filtered, &[1])?;
    let bounded = decode(helper, &output, 0.0, 1)?;
    assert_detections(helper, &bounded, &[1])?;

    expect_error_code(
        helper.call(
            OPERATION,
            &[
                output.arg_with_owner("foreign-worker"),
                float_arg(0.5),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(OPERATION, &[model.arg(), float_arg(0.5), int_arg(3)])?,
        "handle_type_mismatch",
    )?;
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        expect_error_code(
            helper.call(
                OPERATION,
                &[output.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(OPERATION, &[output.arg(), "i:1".to_owned(), int_arg(3)])?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.dnn.dispose_output", &[output.arg()])?)?;
    assert_detections(helper, &detections, &[1, 0, 0])?;
    for owner in [&bounded, &filtered, &repeated, &detections] {
        expect_unit(helper.call("opencv.dnn.dispose_detections", &[owner.arg()])?)?;
    }
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[model.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose", &[blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)
}

/// Decodes one objectness-free class-score output through the generated protocol.
fn decode(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    threshold: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            OPERATION,
            &[output.arg(), float_arg(threshold), int_arg(maximum)],
        )?,
        "opencv.Dnn.Detections",
    )
}

/// Requires canonical classes, score confidence, and normalized boxes.
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
            "malformed class-score detections: classes={actual:?}, confidence={confidence:?}, boxes={boxes:?}"
        ));
    }
    Ok(())
}
