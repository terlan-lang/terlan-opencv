//! Direct copied nonzero-mask fitted-ellipse tests for the CXX bridge.

use terlan_opencv_native::ffi;

/// Proves fitted-ellipse geometry, normalization, validation, and cleanup.
#[test]
fn non_zero_fit_ellipse_is_copied_and_canonical() {
    assert_eq!(ffi::live_mat_count(), 0);
    let horizontal_bytes = ellipse_mask(21, 11, &horizontal_ellipse_points());
    let horizontal = ffi::mat_from_u8(&horizontal_bytes, 11, 21, 1);
    let horizontal_ref = horizontal.as_ref().expect("horizontal ellipse mask");
    assert_ellipse(ellipse(horizontal_ref), [10.0, 5.0, 20.0, 10.0, 0.0]);
    assert_eq!(mat_bytes(horizontal_ref), horizontal_bytes);

    let vertical_bytes = ellipse_mask(11, 21, &vertical_ellipse_points());
    let vertical = ffi::mat_from_u8(&vertical_bytes, 21, 11, 1);
    assert_ellipse(
        ellipse(vertical.as_ref().expect("vertical ellipse mask")),
        [5.0, 10.0, 20.0, 10.0, 90.0],
    );
    let circle_bytes = ellipse_mask(11, 11, &circle_points());
    let circle = ffi::mat_from_u8(&circle_bytes, 11, 11, 1);
    assert_ellipse(
        ellipse(circle.as_ref().expect("circle mask")),
        [5.0, 5.0, 10.0, 10.0, 0.0],
    );

    let insufficient = ffi::mat_from_u8(&[1, 1, 1, 1, 0, 0], 2, 3, 1);
    let collinear = ffi::mat_from_u8(&[1, 1, 1, 1, 1], 1, 5, 1);
    for image in [&insufficient, &collinear] {
        assert!(ellipse(image.as_ref().expect("degenerate ellipse mask")).is_empty());
    }

    let bgr = ffi::mat_zeros(2, 3, 3);
    let bgra = ffi::mat_zeros(2, 3, 4);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for image in [&bgr, &bgra, &oversized] {
        assert!(image
            .as_ref()
            .expect("invalid ellipse mask")
            .non_zero_fit_ellipse_owned()
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(bgra);
    drop(bgr);
    drop(collinear);
    drop(insufficient);
    drop(circle);
    drop(vertical);
    drop(horizontal);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Builds a nonbinary grayscale mask from exact coordinate pairs.
fn ellipse_mask(cols: usize, rows: usize, points: &[(usize, usize)]) -> Vec<u8> {
    let mut bytes = vec![0; cols * rows];
    for (index, &(x, y)) in points.iter().enumerate() {
        bytes[y * cols + x] = (index % 254 + 1) as u8;
    }
    bytes
}

/// Returns exact points on a horizontal ellipse centered at ten, five.
fn horizontal_ellipse_points() -> Vec<(usize, usize)> {
    vec![
        (20, 5),
        (0, 5),
        (10, 10),
        (10, 0),
        (16, 9),
        (16, 1),
        (4, 9),
        (4, 1),
        (18, 8),
        (18, 2),
        (2, 8),
        (2, 2),
    ]
}

/// Returns exact points on a vertical ellipse centered at five, ten.
fn vertical_ellipse_points() -> Vec<(usize, usize)> {
    horizontal_ellipse_points()
        .into_iter()
        .map(|(x, y)| (y, x))
        .collect()
}

/// Returns exact points on a circle centered at five, five.
fn circle_points() -> Vec<(usize, usize)> {
    vec![
        (10, 5),
        (0, 5),
        (5, 10),
        (5, 0),
        (8, 9),
        (8, 1),
        (2, 9),
        (2, 1),
        (9, 8),
        (9, 2),
        (1, 8),
        (1, 2),
    ]
}

/// Copies one native fitted ellipse into a Rust floating-point vector.
fn ellipse(image: &ffi::OpenCvMat) -> Vec<f64> {
    image
        .non_zero_fit_ellipse_owned()
        .as_ref()
        .expect("nonzero fitted ellipse")
        .as_slice()
        .to_vec()
}

/// Requires one copied fitted ellipse to match within native precision.
fn assert_ellipse(actual: Vec<f64>, expected: [f64; 5]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(left, right)| (left - right).abs() <= 1.0e-3));
}

/// Copies one native matrix into a Rust byte vector for exact assertions.
fn mat_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
