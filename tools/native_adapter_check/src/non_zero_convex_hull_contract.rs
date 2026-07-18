//! Copied nonzero grayscale-mask convex-hull contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_ints, expect_unit, matrix, NativeHelper,
};

/// Runs hull geometry, canonical ordering, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let mask_bytes = [1, 0, 0, 2, 0, 9, 0, 0, 3, 0, 0, 4];
    let mask = matrix(helper, &mask_bytes, 3, 4, "gray")?;
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.non_zero_convex_hull", &[mask.arg()])?)?,
        [0, 0, 3, 0, 3, 2, 0, 2]
    );
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, &mask_bytes)?;

    let line = matrix(helper, &[0, 0, 0, 0, 2, 7, 0, 9], 2, 4, "gray")?;
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.non_zero_convex_hull", &[line.arg()])?)?,
        [0, 1, 3, 1]
    );
    let single = matrix(helper, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, 4, "gray")?;
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.non_zero_convex_hull", &[single.arg()])?)?,
        [3, 2]
    );
    let empty = matrix(helper, &[0; 12], 3, 4, "gray")?;
    assert!(
        expect_ints(helper.call("opencv.imgproc.non_zero_convex_hull", &[empty.arg()])?)?
            .is_empty()
    );

    let bgr = matrix(helper, &[0; 36], 3, 4, "bgr")?;
    let bgra = matrix(helper, &[0; 48], 3, 4, "bgra")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for image in [&bgr, &bgra, &oversized] {
        expect_error_code(
            helper.call("opencv.imgproc.non_zero_convex_hull", &[image.arg()])?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_convex_hull",
            &[mask.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_convex_hull",
            &[mask.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &mask_bytes, 3, 4, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.non_zero_convex_hull", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [&oversized, &bgra, &bgr, &empty, &single, &line, &mask] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}
