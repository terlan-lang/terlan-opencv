//! Bounded copied grayscale circle-detection contract.

use super::{
    drawing_contract_support::draw, expect_bytes, expect_bytes_value, expect_error_code,
    expect_floats, expect_int, expect_typed_handle, expect_unit, float_arg, int_arg, matrix,
    NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Runs geometry, source-integrity, parameter, handle, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let blank = matrix(helper, &[0; 64 * 64], 64, 64, "gray")?;
    let blank_circles = expect_floats(detect(helper, &blank, [10.0, 100.0, 8.0], [8, 12])?)?;
    if !blank_circles.is_empty() {
        return Err(format!("blank image produced circles {blank_circles:?}"));
    }

    let ring = draw(
        helper,
        "opencv.imgproc.draw_circle",
        &blank,
        [32, 32, 10, 255, 2],
    )?;
    let blurred = expect_typed_handle(
        helper.call(
            "opencv.imgproc.gaussian_blur",
            &[ring.arg(), int_arg(5), int_arg(5), float_arg(1.5)],
        )?,
        MAT_TYPE,
    )?;
    let source_bytes = expect_bytes_value(helper.call("opencv.mat.to_u8", &[blurred.arg()])?)?;
    let circles = expect_floats(detect(helper, &blurred, [10.0, 100.0, 8.0], [8, 12])?)?;
    validate_circles(&circles, 64.0, 64.0, 8.0, 12.0)?;
    if !circles.chunks_exact(3).any(|circle| {
        (circle[0] - 32.0).abs() <= 2.0
            && (circle[1] - 32.0).abs() <= 2.0
            && (circle[2] - 10.0).abs() <= 2.0
    }) {
        return Err(format!("expected center circle, received {circles:?}"));
    }
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[blurred.arg()])?,
        &source_bytes,
    )?;

    for parameters in [
        ([0.0, 100.0, 8.0], [8, 12]),
        ([100.0, 100.0, 8.0], [8, 12]),
        ([10.0, 0.0, 8.0], [8, 12]),
        ([10.0, f64::NAN, 8.0], [8, 12]),
        ([10.0, 100.0, 0.0], [8, 12]),
        ([10.0, 100.0, f64::INFINITY], [8, 12]),
        ([10.0, 100.0, 8.0], [0, 12]),
        ([10.0, 100.0, 8.0], [12, 8]),
        ([10.0, 100.0, 8.0], [8, 90]),
    ] {
        expect_error_code(
            detect(helper, &blurred, parameters.0, parameters.1)?,
            "opencv.invalid_argument",
        )?;
    }

    let color = matrix(helper, &[0; 64 * 64 * 3], 64, 64, "bgr")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for invalid in [&color, &oversized] {
        expect_error_code(
            detect(helper, invalid, [10.0, 100.0, 8.0], [8, 12])?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.detect_circles",
            &detect_arguments(
                blank.arg_with_owner("foreign-worker"),
                [10.0, 100.0, 8.0],
                [8, 12],
            ),
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.detect_circles",
            &detect_arguments(
                blank.arg_with_type("other.Package.Resource"),
                [10.0, 100.0, 8.0],
                [8, 12],
            ),
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 64], 8, 8, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        detect(helper, &stale, [2.0, 100.0, 8.0], [1, 3])?,
        "stale_handle",
    )?;

    for handle in [&oversized, &color, &blurred, &ring, &blank] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Calls the generated circle detector with grouped policy parameters.
fn detect(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    thresholds: [f64; 3],
    radii: [i64; 2],
) -> Result<super::protocol::NativeReply, String> {
    helper.call(
        "opencv.imgproc.detect_circles",
        &detect_arguments(image.arg(), thresholds, radii),
    )
}

/// Builds one generated circle-detector argument list.
fn detect_arguments(image: String, thresholds: [f64; 3], radii: [i64; 2]) -> Vec<String> {
    vec![
        image,
        float_arg(thresholds[0]),
        float_arg(thresholds[1]),
        float_arg(thresholds[2]),
        int_arg(radii[0]),
        int_arg(radii[1]),
    ]
}

/// Validates canonical copied center/radius triples.
fn validate_circles(
    values: &[f64],
    width: f64,
    height: f64,
    minimum_radius: f64,
    maximum_radius: f64,
) -> Result<(), String> {
    let valid = !values.is_empty()
        && values.len().is_multiple_of(3)
        && values.chunks_exact(3).all(|circle| {
            circle.iter().all(|value| value.is_finite())
                && circle[0] >= 0.0
                && circle[0] < width
                && circle[1] >= 0.0
                && circle[1] < height
                && circle[2] + 1.0 >= minimum_radius
                && circle[2] <= maximum_radius + 1.0
        })
        && values
            .chunks_exact(3)
            .zip(values[3..].chunks_exact(3))
            .all(|(first, second)| {
                (first[1], first[0], first[2]) <= (second[1], second[0], second[2])
            });
    if valid {
        Ok(())
    } else {
        Err(format!("invalid detected circles {values:?}"))
    }
}
