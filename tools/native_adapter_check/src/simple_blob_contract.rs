//! Canonical copied SimpleBlobDetector contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_unit, float_arg, int_arg, matrix,
    NativeHandle, NativeHelper,
};

/// Runs bright/dark detection, truncation, determinism, and rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    let fixture = blob_fixture(false);
    let image = matrix(helper, &fixture, 160, 160, "gray")?;
    let bright = detect(helper, &image, 255, 64)?;
    validate_blobs(&bright, 2)?;
    let repeated = detect(helper, &image, 255, 64)?;
    if repeated != bright {
        return Err("repeated bright-blob detection was not deterministic".to_owned());
    }
    let truncated = detect(helper, &image, 255, 1)?;
    validate_blobs(&truncated, 1)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, &fixture)?;

    let inverted_fixture = blob_fixture(true);
    let inverted = matrix(helper, &inverted_fixture, 160, 160, "gray")?;
    let dark = detect(helper, &inverted, 0, 64)?;
    validate_blobs(&dark, 2)?;

    let empty_image = matrix(helper, &[0; 160 * 160], 160, 160, "gray")?;
    if !detect(helper, &empty_image, 255, 8)?.is_empty() {
        return Err("uniform image produced bright blobs".to_owned());
    }
    for (blob_value, minimum_area, maximum_area, distance, maximum) in [
        (1, 50, 1000, 5.0, 8),
        (255, 0, 1000, 5.0, 8),
        (255, 50, 50, 5.0, 8),
        (255, 50, 25602, 5.0, 8),
        (255, 50, 1000, 0.0, 8),
        (255, 50, 1000, -1.0, 8),
        (255, 50, 1000, f64::NAN, 8),
        (255, 50, 1000, f64::INFINITY, 8),
        (255, 50, 1000, 227.0, 8),
        (255, 50, 1000, 5.0, 0),
        (255, 50, 1000, 5.0, 4097),
    ] {
        expect_error_code(
            helper.call(
                "opencv.features.simple_blob_keypoints",
                &[
                    empty_image.arg(),
                    int_arg(blob_value),
                    int_arg(minimum_area),
                    int_arg(maximum_area),
                    float_arg(distance),
                    int_arg(maximum),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.features.simple_blob_keypoints",
            &[
                empty_image.arg_with_owner("foreign-worker"),
                int_arg(255),
                int_arg(50),
                int_arg(1000),
                float_arg(5.0),
                int_arg(8),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.features.simple_blob_keypoints",
            &[
                empty_image.arg_with_type("other.Package.Resource"),
                int_arg(255),
                int_arg(50),
                int_arg(1000),
                float_arg(5.0),
                int_arg(8),
            ],
        )?,
        "handle_type_mismatch",
    )?;

    let color = matrix(helper, &[0; 3 * 3 * 3], 3, 3, "bgr")?;
    let too_small = matrix(helper, &[0; 2 * 2], 2, 2, "gray")?;
    let oversized = matrix(helper, &[0; 1025 * 1024], 1025, 1024, "gray")?;
    for source in [&color, &too_small, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.features.simple_blob_keypoints",
                &[
                    source.arg(),
                    int_arg(255),
                    int_arg(1),
                    int_arg(4),
                    float_arg(1.0),
                    int_arg(1),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for source in [
        &oversized,
        &too_small,
        &color,
        &empty_image,
        &inverted,
        &image,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[source.arg()])?)?;
    }
    super::expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Builds two separated circular grayscale blobs.
fn blob_fixture(inverted: bool) -> Vec<u8> {
    let background = if inverted { 255 } else { 0 };
    let foreground = if inverted { 0 } else { 255 };
    let mut pixels = vec![background; 160 * 160];
    for (center_x, center_y, radius) in [(40_i64, 50_i64, 8_i64), (110, 100, 12)] {
        for y in center_y - radius..=center_y + radius {
            for x in center_x - radius..=center_x + radius {
                if (x - center_x).pow(2) + (y - center_y).pow(2) <= radius.pow(2) {
                    pixels[y as usize * 160 + x as usize] = foreground;
                }
            }
        }
    }
    pixels
}

/// Invokes the generated fixed-policy blob operation.
fn detect(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    value: i64,
    maximum: i64,
) -> Result<Vec<f64>, String> {
    expect_floats(helper.call(
        "opencv.features.simple_blob_keypoints",
        &[
            image.arg(),
            int_arg(value),
            int_arg(50),
            int_arg(1000),
            float_arg(5.0),
            int_arg(maximum),
        ],
    )?)
}

/// Validates canonical x/y/diameter/response rows.
fn validate_blobs(values: &[f64], count: usize) -> Result<(), String> {
    if values.len() != count * 4 {
        return Err(format!(
            "expected {count} blobs, received {} scalars",
            values.len()
        ));
    }
    let mut previous: Option<(f64, f64)> = None;
    for item in values.chunks_exact(4) {
        let order = (item[1], item[0]);
        if item.iter().any(|value| !value.is_finite())
            || !(0.0..160.0).contains(&item[0])
            || !(0.0..160.0).contains(&item[1])
            || item[2] <= 0.0
            || item[2] > 453.0
            || item[3] < 0.0
            || previous.is_some_and(|prior| order <= prior)
        {
            return Err("invalid canonical blob row".to_owned());
        }
        previous = Some(order);
    }
    Ok(())
}
