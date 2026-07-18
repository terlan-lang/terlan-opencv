//! First-order Sobel and Scharr edge-response ownership contract.

use super::{
    atom_arg, bytes_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle,
    expect_unit, int_arg, NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const HORIZONTAL_RAMP: &[u8] = &[0, 10, 20, 0, 10, 20, 0, 10, 20];
const X_RESPONSE: &[u8] = &[40, 80, 40, 40, 80, 40, 40, 80, 40];
const SCHARR_X_RESPONSE: &[u8] = &[160, 255, 160, 160, 255, 160, 160, 255, 160];
const ZERO_GRAY: &[u8] = &[0; 9];

/// Runs first-order Sobel behavior, rejection, and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let ramp = matrix(helper, HORIZONTAL_RAMP, "gray")?;
    let x_edges = sobel(helper, &ramp, 1, 0, 3)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[x_edges.arg()])?,
        X_RESPONSE,
    )?;
    let y_edges = sobel(helper, &ramp, 0, 1, 3)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[y_edges.arg()])?,
        ZERO_GRAY,
    )?;
    let scharr_x = scharr(helper, &ramp, 1, 0)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[scharr_x.arg()])?,
        SCHARR_X_RESPONSE,
    )?;
    let scharr_y = scharr(helper, &ramp, 0, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[scharr_y.arg()])?,
        ZERO_GRAY,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[ramp.arg()])?,
        HORIZONTAL_RAMP,
    )?;

    let bgr = matrix(helper, &[0; 27], "bgr")?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.sobel_edges",
            &[bgr.arg(), int_arg(1), int_arg(0), int_arg(3)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.scharr_edges",
            &[bgr.arg(), int_arg(1), int_arg(0)],
        )?,
        "opencv.invalid_argument",
    )?;
    for (x_order, y_order, kernel_size) in [
        (0, 0, 3),
        (1, 1, 3),
        (2, 0, 3),
        (0, 2, 3),
        (-1, 0, 3),
        (1, 0, 0),
        (1, 0, 2),
        (1, 0, 9),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.sobel_edges",
                &[
                    ramp.arg(),
                    int_arg(x_order),
                    int_arg(y_order),
                    int_arg(kernel_size),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for (x_order, y_order) in [(0, 0), (1, 1), (2, 0), (0, 2), (-1, 0)] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.scharr_edges",
                &[ramp.arg(), int_arg(x_order), int_arg(y_order)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.sobel_edges",
            &[
                ramp.arg_with_owner("foreign-worker"),
                int_arg(1),
                int_arg(0),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.sobel_edges",
            &[
                ramp.arg_with_type("other.Package.Resource"),
                int_arg(1),
                int_arg(0),
                int_arg(3),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.scharr_edges",
            &[
                ramp.arg_with_owner("foreign-worker"),
                int_arg(1),
                int_arg(0),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.scharr_edges",
            &[
                ramp.arg_with_type("other.Package.Resource"),
                int_arg(1),
                int_arg(0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, ZERO_GRAY, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.sobel_edges",
            &[stale.arg(), int_arg(1), int_arg(0), int_arg(3)],
        )?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.scharr_edges",
            &[stale.arg(), int_arg(1), int_arg(0)],
        )?,
        "stale_handle",
    )?;

    for handle in [&bgr, &scharr_y, &scharr_x, &y_edges, &x_edges, &ramp] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Creates one three-by-three matrix in the requested public layout.
fn matrix(helper: &mut NativeHelper, bytes: &[u8], layout: &str) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[bytes_arg(bytes), int_arg(3), int_arg(3), atom_arg(layout)],
        )?,
        MAT_TYPE,
    )
}

/// Computes one Sobel response and validates its result type.
fn sobel(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    x_order: i64,
    y_order: i64,
    kernel_size: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.sobel_edges",
            &[
                image.arg(),
                int_arg(x_order),
                int_arg(y_order),
                int_arg(kernel_size),
            ],
        )?,
        MAT_TYPE,
    )
}

/// Computes one Scharr response and validates its result type.
fn scharr(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    x_order: i64,
    y_order: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.scharr_edges",
            &[image.arg(), int_arg(x_order), int_arg(y_order)],
        )?,
        MAT_TYPE,
    )
}
