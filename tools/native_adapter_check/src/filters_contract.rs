//! Fresh-owner image-filter contract.

use super::{
    atom_arg, bytes_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle,
    expect_unit, float_arg, int_arg, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const IMPULSE: &[u8] = &[0, 0, 0, 0, 255, 0, 0, 0, 0];
const ZERO_GRAY: &[u8] = &[0; 9];
const ZERO_BGR: &[u8] = &[0; 27];
const ZERO_BGRA: &[u8] = &[0; 36];
const LAPLACIAN_IMPULSE: &[u8] = &[0, 255, 0, 255, 255, 255, 0, 255, 0];

/// Runs deterministic median and bilateral filtering and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let impulse = matrix(helper, IMPULSE, "gray")?;
    let filtered = filter(helper, &impulse, 3)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[filtered.arg()])?,
        ZERO_GRAY,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[impulse.arg()])?, IMPULSE)?;

    let bgr = matrix(helper, ZERO_BGR, "bgr")?;
    let filtered_bgr = filter(helper, &bgr, 3)?;
    expect_int(
        helper.call("opencv.mat.channels", &[filtered_bgr.arg()])?,
        3,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[filtered_bgr.arg()])?,
        ZERO_BGR,
    )?;
    let bgra = matrix(helper, ZERO_BGRA, "bgra")?;
    let filtered_bgra = filter(helper, &bgra, 3)?;
    expect_int(
        helper.call("opencv.mat.channels", &[filtered_bgra.arg()])?,
        4,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[filtered_bgra.arg()])?,
        ZERO_BGRA,
    )?;
    let bilateral_gray = bilateral(helper, &filtered, 3, 10.0, 10.0)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bilateral_gray.arg()])?,
        ZERO_GRAY,
    )?;
    let bilateral_bgr = bilateral(helper, &bgr, 3, 10.0, 10.0)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bilateral_bgr.arg()])?,
        ZERO_BGR,
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.bilateral_filter",
            &[bgra.arg(), int_arg(3), float_arg(10.0), float_arg(10.0)],
        )?,
        "opencv.invalid_argument",
    )?;
    let laplacian = laplacian_edges(helper, &impulse, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[laplacian.arg()])?,
        LAPLACIAN_IMPULSE,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[impulse.arg()])?, IMPULSE)?;
    expect_error_code(
        helper.call("opencv.imgproc.laplacian_edges", &[bgr.arg(), int_arg(3)])?,
        "opencv.invalid_argument",
    )?;

    for kernel_size in [-3, 0, 1, 2, 4, 257] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.median_blur",
                &[impulse.arg(), int_arg(kernel_size)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for (diameter, sigma_color, sigma_space) in [
        (-1, 1.0, 1.0),
        (0, 1.0, 1.0),
        (2, 1.0, 1.0),
        (257, 1.0, 1.0),
        (3, 0.0, 1.0),
        (3, -1.0, 1.0),
        (3, 65_536.0, 1.0),
        (3, f64::NAN, 1.0),
        (3, f64::INFINITY, 1.0),
        (3, 1.0, 0.0),
        (3, 1.0, -1.0),
        (3, 1.0, 65_536.0),
        (3, 1.0, f64::NAN),
        (3, 1.0, f64::INFINITY),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.bilateral_filter",
                &[
                    impulse.arg(),
                    int_arg(diameter),
                    float_arg(sigma_color),
                    float_arg(sigma_space),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for kernel_size in [-1, 0, 2, 4, 9] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.laplacian_edges",
                &[impulse.arg(), int_arg(kernel_size)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.median_blur",
            &[impulse.arg_with_owner("foreign-worker"), int_arg(3)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.median_blur",
            &[impulse.arg_with_type("other.Package.Resource"), int_arg(3)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.laplacian_edges",
            &[impulse.arg_with_owner("foreign-worker"), int_arg(1)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.laplacian_edges",
            &[impulse.arg_with_type("other.Package.Resource"), int_arg(1)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.bilateral_filter",
            &[
                impulse.arg_with_owner("foreign-worker"),
                int_arg(3),
                float_arg(10.0),
                float_arg(10.0),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.bilateral_filter",
            &[
                impulse.arg_with_type("other.Package.Resource"),
                int_arg(3),
                float_arg(10.0),
                float_arg(10.0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, ZERO_GRAY, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.median_blur", &[stale.arg(), int_arg(3)])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.bilateral_filter",
            &[stale.arg(), int_arg(3), float_arg(10.0), float_arg(10.0)],
        )?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call("opencv.imgproc.laplacian_edges", &[stale.arg(), int_arg(1)])?,
        "stale_handle",
    )?;

    for handle in [
        &laplacian,
        &bilateral_bgr,
        &bilateral_gray,
        &filtered_bgra,
        &bgra,
        &filtered_bgr,
        &bgr,
        &filtered,
        &impulse,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Computes one absolute Laplacian and validates the returned matrix type.
fn laplacian_edges(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    kernel_size: i64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.laplacian_edges",
            &[image.arg(), int_arg(kernel_size)],
        )?,
        MAT_TYPE,
    )
}

/// Applies one bilateral filter and validates the returned matrix type.
fn bilateral(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    diameter: i64,
    sigma_color: f64,
    sigma_space: f64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.bilateral_filter",
            &[
                image.arg(),
                int_arg(diameter),
                float_arg(sigma_color),
                float_arg(sigma_space),
            ],
        )?,
        MAT_TYPE,
    )
}

/// Creates one three-by-three matrix in the requested public layout.
fn matrix(
    helper: &mut NativeHelper,
    bytes: &[u8],
    layout: &str,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[bytes_arg(bytes), int_arg(3), int_arg(3), atom_arg(layout)],
        )?,
        MAT_TYPE,
    )
}

/// Applies one median filter and validates the returned matrix type.
fn filter(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    kernel_size: i64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.median_blur",
            &[image.arg(), int_arg(kernel_size)],
        )?,
        MAT_TYPE,
    )
}
