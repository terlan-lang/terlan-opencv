//! Direct bounded standard Hough-line tests for the CXX bridge.

use terlan_opencv_native::ffi;

#[rustfmt::skip]
const LINE_MASK: &[u8] = &[
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 1, 2, 3, 4, 5, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
];

/// Proves canonical copied output, validation, immutability, and cleanup.
#[test]
fn detected_lines_are_canonical_and_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);
    let mask = ffi::mat_from_u8(LINE_MASK, 7, 7, 1);
    let mask_ref = mask.as_ref().expect("line mask");
    let lines = detected_lines(mask_ref, 3);
    validate_lines(&lines, 7, 7);
    assert!(
        lines
            .chunks_exact(2)
            .any(|line| (line[0] - 3.0).abs() <= 1.0
                && (line[1] - std::f64::consts::FRAC_PI_2).abs() <= 0.15),
        "expected horizontal source line in {lines:?}"
    );
    assert_eq!(mat_bytes(mask_ref), LINE_MASK);

    let blank = ffi::mat_zeros(7, 7, 1);
    assert!(detected_lines(blank.as_ref().expect("blank mask"), 1).is_empty());
    for threshold in [0, 8] {
        assert!(mask_ref.detect_lines_owned(threshold).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let color = ffi::mat_zeros(7, 7, 3);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for invalid in [&color, &oversized] {
        assert!(invalid
            .as_ref()
            .expect("invalid line image")
            .detect_lines_owned(1)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(color);
    drop(blank);
    drop(mask);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies flattened native lines into Rust-owned floating-point values.
fn detected_lines(image: &ffi::OpenCvMat, threshold: i64) -> Vec<f64> {
    image
        .detect_lines_owned(threshold)
        .as_ref()
        .expect("detected lines")
        .as_slice()
        .to_vec()
}

/// Requires finite bounded rho/theta pairs in canonical theta/rho order.
fn validate_lines(lines: &[f64], rows: i64, columns: i64) {
    assert!(!lines.is_empty());
    assert!(lines.len().is_multiple_of(2));
    let maximum_rho = ((rows - 1).pow(2) + (columns - 1).pow(2)) as f64;
    let maximum_rho = maximum_rho.sqrt() + 1.0;
    for line in lines.chunks_exact(2) {
        assert!(line[0].is_finite() && line[0].abs() <= maximum_rho);
        assert!(line[1].is_finite() && (0.0..std::f64::consts::PI).contains(&line[1]));
    }
    assert!(lines
        .chunks_exact(2)
        .collect::<Vec<_>>()
        .windows(2)
        .all(|pair| {
            pair[0][1] < pair[1][1] || (pair[0][1] == pair[1][1] && pair[0][0] <= pair[1][0])
        }));
}

/// Copies one native matrix into a Rust byte vector.
fn mat_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
