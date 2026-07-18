//! Shared helper calls for the DNN image-to-blob contract.

use super::{
    expect_error_code, expect_typed_handle, float_list_arg, int_arg, NativeHandle, NativeHelper,
};

/// Generated opaque blob resource type.
pub(super) const BLOB_TYPE: &str = "opencv.Dnn.Blob";
/// Fixed RGB normalization means.
pub(super) const RGB_MEANS: [f64; 3] = [-1.0, 0.5, 1.0];
/// Fixed RGB normalization scales.
pub(super) const RGB_SCALES: [f64; 3] = [2.0, 4.0, 8.0];
/// Fixed grayscale normalization mean.
pub(super) const GRAYSCALE_MEAN: [f64; 1] = [0.5];
/// Fixed grayscale normalization scale.
pub(super) const GRAYSCALE_SCALE: [f64; 1] = [4.0];

/// Creates one typed center-cropped grayscale blob through the helper.
pub(super) fn create_grayscale_center_crop(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_plain(
        helper,
        "opencv.dnn.blob_from_grayscale_center_crop",
        image,
        width,
        height,
    )
}

/// Creates one typed aspect-preserving letterboxed grayscale blob.
pub(super) fn create_grayscale_letterbox(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_plain(
        helper,
        "opencv.dnn.blob_from_grayscale_letterbox",
        image,
        width,
        height,
    )
}

/// Creates one typed normalized center-cropped RGB blob through the helper.
pub(super) fn create_normalized_center_crop(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_normalized_with(
        helper,
        "opencv.dnn.blob_from_bgr_rgb_normalized_center_crop",
        image,
        width,
        height,
        &RGB_MEANS,
        &RGB_SCALES,
    )
}

/// Creates one typed normalized center-cropped grayscale blob.
pub(super) fn create_normalized_grayscale_center_crop(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_normalized_with(
        helper,
        "opencv.dnn.blob_from_grayscale_normalized_center_crop",
        image,
        width,
        height,
        &GRAYSCALE_MEAN,
        &GRAYSCALE_SCALE,
    )
}

/// Creates one typed normalized letterboxed grayscale blob.
pub(super) fn create_normalized_grayscale_letterbox(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_normalized_with(
        helper,
        "opencv.dnn.blob_from_grayscale_normalized_letterbox",
        image,
        width,
        height,
        &GRAYSCALE_MEAN,
        &GRAYSCALE_SCALE,
    )
}

/// Requires both normalized grayscale aspect modes to reject the same input.
pub(super) fn expect_normalized_grayscale_mode_errors(
    helper: &mut NativeHelper,
    image_arg: String,
    width: i64,
    height: i64,
    means: &[f64],
    scales: &[f64],
    expected: &str,
) -> Result<(), String> {
    for operation in [
        "opencv.dnn.blob_from_grayscale_normalized_center_crop",
        "opencv.dnn.blob_from_grayscale_normalized_letterbox",
    ] {
        expect_error_code(
            helper.call(
                operation,
                &[
                    image_arg.clone(),
                    int_arg(width),
                    int_arg(height),
                    float_list_arg(means),
                    float_list_arg(scales),
                ],
            )?,
            expected,
        )?;
    }
    Ok(())
}

/// Creates one typed normalized letterboxed RGB blob through the helper.
pub(super) fn create_normalized_letterbox(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_normalized_with(
        helper,
        "opencv.dnn.blob_from_bgr_rgb_normalized_letterbox",
        image,
        width,
        height,
        &RGB_MEANS,
        &RGB_SCALES,
    )
}

/// Creates one typed normalized RGB blob through the helper.
pub(super) fn create_normalized(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_normalized_with(
        helper,
        "opencv.dnn.blob_from_bgr_rgb_normalized",
        image,
        width,
        height,
        &RGB_MEANS,
        &RGB_SCALES,
    )
}

/// Creates one typed center-cropped RGB blob through the helper.
pub(super) fn create_center_crop(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_plain(
        helper,
        "opencv.dnn.blob_from_bgr_rgb_center_crop",
        image,
        width,
        height,
    )
}

/// Creates one typed aspect-preserving letterboxed RGB blob through the helper.
pub(super) fn create_letterbox(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_plain(
        helper,
        "opencv.dnn.blob_from_bgr_rgb_letterbox",
        image,
        width,
        height,
    )
}

/// Creates one typed grayscale inference-blob handle through the helper.
pub(super) fn create_grayscale(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_plain(
        helper,
        "opencv.dnn.blob_from_grayscale",
        image,
        width,
        height,
    )
}

/// Creates one typed normalized grayscale blob through the helper.
pub(super) fn create_normalized_grayscale(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_normalized_with(
        helper,
        "opencv.dnn.blob_from_grayscale_normalized",
        image,
        width,
        height,
        &GRAYSCALE_MEAN,
        &GRAYSCALE_SCALE,
    )
}

/// Creates one typed inference-blob handle through the helper.
pub(super) fn create(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    create_plain(helper, "opencv.dnn.blob_from_bgr_rgb", image, width, height)
}

/// Calls one dimension-only preprocessing operation and validates its owner.
fn create_plain(
    helper: &mut NativeHelper,
    operation: &str,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(operation, &[image.arg(), int_arg(width), int_arg(height)])?,
        BLOB_TYPE,
    )
}

/// Calls one normalized preprocessing operation and validates its owner.
fn create_normalized_with(
    helper: &mut NativeHelper,
    operation: &str,
    image: &NativeHandle,
    width: i64,
    height: i64,
    means: &[f64],
    scales: &[f64],
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            operation,
            &[
                image.arg(),
                int_arg(width),
                int_arg(height),
                float_list_arg(means),
                float_list_arg(scales),
            ],
        )?,
        BLOB_TYPE,
    )
}
