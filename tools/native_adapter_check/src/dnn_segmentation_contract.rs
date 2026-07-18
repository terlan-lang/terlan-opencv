//! Deterministic DNN channel-mask helper contract.

use super::dnn_inference_contract::{BGR_PIXELS, IDENTITY_ONNX};
use super::{
    bytes_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit,
    float_arg, int_arg, matrix, NativeHandle, NativeHelper,
};

/// Runs exact bytes, dimensions, owner, type, stale-handle, and input rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    let image = matrix(helper, &BGR_PIXELS, 2, 2, "bgr")?;
    let blob = expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[image.arg(), int_arg(2), int_arg(2)],
        )?,
        "opencv.Dnn.Blob",
    )?;
    let model = expect_typed_handle(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(IDENTITY_ONNX)])?,
        "opencv.Dnn.OnnxModel",
    )?;
    let output = expect_typed_handle(
        helper.call("opencv.dnn.forward", &[model.arg(), blob.arg()])?,
        "opencv.Dnn.Output",
    )?;
    let green = mask(helper, &output, 1, 0.5)?;
    assert_mask(helper, &green, &[0, 255, 0, 255])?;
    let repeated = mask(helper, &output, 1, 0.5)?;
    assert_mask(helper, &repeated, &[0, 255, 0, 255])?;
    let strict = mask(helper, &output, 1, 1.0)?;
    assert_mask(helper, &strict, &[0, 0, 0, 0])?;

    expect_error_code(
        helper.call(
            "opencv.dnn.channel_mask",
            &[
                output.arg_with_owner("foreign-worker"),
                int_arg(1),
                float_arg(0.5),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.channel_mask",
            &[model.arg(), int_arg(1), float_arg(0.5)],
        )?,
        "handle_type_mismatch",
    )?;
    for (channel, threshold) in [(-1, 0.5), (3, 0.5), (1, f64::INFINITY)] {
        expect_error_code(
            helper.call(
                "opencv.dnn.channel_mask",
                &[output.arg(), int_arg(channel), float_arg(threshold)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    expect_unit(helper.call("opencv.mat.dispose", &[green.arg()])?)?;
    expect_error_code(
        helper.call("opencv.mat.to_u8", &[green.arg()])?,
        "stale_handle",
    )?;
    expect_unit(helper.call("opencv.mat.dispose", &[repeated.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[strict.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose_output", &[output.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[model.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose", &[blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)
}

/// Projects one output channel into a typed matrix owner.
fn mask(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    channel: i64,
    threshold: f64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.channel_mask",
            &[output.arg(), int_arg(channel), float_arg(threshold)],
        )?,
        "opencv.Mat.Mat",
    )
}

/// Requires exact two-by-two grayscale dimensions and copied mask bytes.
fn assert_mask(
    helper: &mut NativeHelper,
    mask: &NativeHandle,
    expected: &[u8],
) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.rows", &[mask.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.cols", &[mask.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.channels", &[mask.arg()])?, 1)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, expected)
}
