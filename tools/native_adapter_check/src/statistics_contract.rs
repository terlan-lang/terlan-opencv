//! Copied per-channel image-statistics contract.

use super::{
    atom_arg, bytes_arg, expect_bytes, expect_error_code, expect_floats, expect_int, expect_ints,
    expect_typed_handle, expect_unit, int_arg, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const BGR: &[u8] = &[0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0];

/// Runs channel-order, copied-result, handle-validation, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let bgr = matrix(helper, BGR, 2, 2, "bgr")?;
    assert_eq!(
        expect_floats(helper.call("opencv.imgproc.mean_channels", &[bgr.arg()])?)?,
        [63.75, 127.5, 127.5]
    );
    expect_close(
        &expect_floats(helper.call("opencv.imgproc.stddev_channels", &[bgr.arg()])?)?,
        &[110.418_238_982_515_93, 127.5, 127.5],
    )?;
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.non_zero_counts", &[bgr.arg()])?)?,
        [1, 2, 2]
    );
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.sum_channels", &[bgr.arg()])?)?,
        [255, 510, 510]
    );
    expect_bytes(helper.call("opencv.mat.to_u8", &[bgr.arg()])?, BGR)?;

    let gray_bytes = [0, 255, 76, 150];
    let gray = matrix(helper, &gray_bytes, 2, 2, "gray")?;
    assert_eq!(
        expect_floats(helper.call("opencv.imgproc.mean_channels", &[gray.arg()])?)?,
        [120.25]
    );
    expect_close(
        &expect_floats(helper.call("opencv.imgproc.stddev_channels", &[gray.arg()])?)?,
        &[94.155_124_661_379_96],
    )?;
    assert_eq!(
        expect_floats(helper.call("opencv.imgproc.min_max_values", &[gray.arg()])?)?,
        [0.0, 255.0]
    );
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.non_zero_counts", &[gray.arg()])?)?,
        [3]
    );
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.sum_channels", &[gray.arg()])?)?,
        [481]
    );
    expect_bytes(helper.call("opencv.mat.to_u8", &[gray.arg()])?, &gray_bytes)?;

    let bgra_bytes = [0, 10, 20, 30, 100, 110, 120, 130];
    let bgra = matrix(helper, &bgra_bytes, 1, 2, "bgra")?;
    assert_eq!(
        expect_floats(helper.call("opencv.imgproc.mean_channels", &[bgra.arg()])?)?,
        [50.0, 60.0, 70.0, 80.0]
    );
    expect_close(
        &expect_floats(helper.call("opencv.imgproc.stddev_channels", &[bgra.arg()])?)?,
        &[50.0, 50.0, 50.0, 50.0],
    )?;
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.non_zero_counts", &[bgra.arg()])?)?,
        [1, 2, 2, 2]
    );
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.sum_channels", &[bgra.arg()])?)?,
        [100, 120, 140, 160]
    );
    let zeros = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(2), int_arg(3), atom_arg("bgr")],
        )?,
        MAT_TYPE,
    )?;
    assert_eq!(
        expect_floats(helper.call("opencv.imgproc.mean_channels", &[zeros.arg()])?)?,
        [0.0, 0.0, 0.0]
    );
    expect_close(
        &expect_floats(helper.call("opencv.imgproc.stddev_channels", &[zeros.arg()])?)?,
        &[0.0, 0.0, 0.0],
    )?;
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.non_zero_counts", &[zeros.arg()])?)?,
        [0, 0, 0]
    );
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.sum_channels", &[zeros.arg()])?)?,
        [0, 0, 0]
    );
    expect_error_code(
        helper.call("opencv.imgproc.min_max_values", &[bgr.arg()])?,
        "opencv.invalid_argument",
    )?;
    let gray_zeros = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(2), int_arg(3), atom_arg("gray")],
        )?,
        MAT_TYPE,
    )?;
    assert_eq!(
        expect_floats(helper.call("opencv.imgproc.min_max_values", &[gray_zeros.arg()])?)?,
        [0.0, 0.0]
    );
    let large_zeros = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(1025), int_arg(1024), atom_arg("gray")],
        )?,
        MAT_TYPE,
    )?;
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.non_zero_counts", &[large_zeros.arg()])?)?,
        [0]
    );
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.sum_channels", &[large_zeros.arg()])?)?,
        [0]
    );
    let bright_bytes = vec![255; 64 * 64];
    let bright = matrix(helper, &bright_bytes, 64, 64, "gray")?;
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.sum_channels", &[bright.arg()])?)?,
        [1_044_480]
    );
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bright.arg()])?,
        &bright_bytes,
    )?;

    for operation in [
        "opencv.imgproc.mean_channels",
        "opencv.imgproc.stddev_channels",
        "opencv.imgproc.min_max_values",
        "opencv.imgproc.non_zero_counts",
        "opencv.imgproc.sum_channels",
    ] {
        expect_error_code(
            helper.call(operation, &[bgr.arg_with_owner("foreign-worker")])?,
            "cross_owner_handle",
        )?;
        expect_error_code(
            helper.call(operation, &[bgr.arg_with_type("other.Package.Resource")])?,
            "handle_type_mismatch",
        )?;
    }
    let stale = matrix(helper, &[255], 1, 1, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.mean_channels", &[stale.arg()])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call("opencv.imgproc.stddev_channels", &[stale.arg()])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call("opencv.imgproc.min_max_values", &[stale.arg()])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call("opencv.imgproc.non_zero_counts", &[stale.arg()])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call("opencv.imgproc.sum_channels", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [
        &bright,
        &large_zeros,
        &gray_zeros,
        &zeros,
        &bgra,
        &gray,
        &bgr,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Requires copied floating-point values to match within native roundoff.
fn expect_close(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(actual, expected)| (actual - expected).abs() > 1.0e-12)
    {
        return Err(format!("expected values {expected:?}, received {actual:?}"));
    }
    Ok(())
}

/// Creates one copied matrix with the requested public layout.
fn matrix(
    helper: &mut NativeHelper,
    bytes: &[u8],
    rows: i64,
    cols: i64,
    layout: &str,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[
                bytes_arg(bytes),
                int_arg(rows),
                int_arg(cols),
                atom_arg(layout),
            ],
        )?,
        MAT_TYPE,
    )
}
