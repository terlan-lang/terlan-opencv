//! Direct bounded grayscale circle-detection tests.

use terlan_opencv_native::ffi;

/// Proves copied circle geometry, validation, source integrity, and cleanup.
#[test]
fn circle_detection_returns_canonical_bounded_triples() {
    assert_eq!(ffi::live_mat_count(), 0);
    let blank = ffi::mat_zeros(64, 64, 1);
    let blank_ref = blank.as_ref().expect("blank circle source");
    assert!(blank_ref
        .detect_circles_owned(10.0, 100.0, 8.0, 8, 12)
        .as_ref()
        .expect("blank circles")
        .is_empty());

    let ring = blank_ref.draw_circle_owned(32, 32, 10, 255, 2);
    let ring_ref = ring.as_ref().expect("circle ring");
    let blurred = ring_ref.gaussian_blur_owned(5, 5, 1.5);
    let blurred_ref = blurred.as_ref().expect("blurred circle ring");
    let source_bytes = mat_bytes(blurred_ref);
    let circles = blurred_ref.detect_circles_owned(10.0, 100.0, 8.0, 8, 12);
    let values = circles.as_ref().expect("detected circles").as_slice();
    assert_circles(values, 64.0, 64.0, 8.0, 12.0);
    assert!(values.chunks_exact(3).any(|circle| {
        (circle[0] - 32.0).abs() <= 2.0
            && (circle[1] - 32.0).abs() <= 2.0
            && (circle[2] - 10.0).abs() <= 2.0
    }));
    assert_eq!(mat_bytes(blurred_ref), source_bytes);

    for parameters in [
        (0.0, 100.0, 8.0, 8, 12),
        (100.0, 100.0, 8.0, 8, 12),
        (10.0, 0.0, 8.0, 8, 12),
        (10.0, f64::NAN, 8.0, 8, 12),
        (10.0, 100.0, 0.0, 8, 12),
        (10.0, 100.0, f64::INFINITY, 8, 12),
        (10.0, 100.0, 8.0, 0, 12),
        (10.0, 100.0, 8.0, 12, 8),
        (10.0, 100.0, 8.0, 8, 90),
    ] {
        assert!(blurred_ref
            .detect_circles_owned(
                parameters.0,
                parameters.1,
                parameters.2,
                parameters.3,
                parameters.4,
            )
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    let color = ffi::mat_zeros(64, 64, 3);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for invalid in [&color, &oversized] {
        assert!(invalid
            .as_ref()
            .expect("invalid circle detector source")
            .detect_circles_owned(10.0, 100.0, 8.0, 8, 12)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(color);
    drop(circles);
    drop(blurred);
    drop(ring);
    drop(blank);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Requires canonical finite center/radius triples within package bounds.
fn assert_circles(values: &[f64], width: f64, height: f64, min: f64, max: f64) {
    assert!(!values.is_empty());
    assert!(values.len().is_multiple_of(3));
    for circle in values.chunks_exact(3) {
        assert!(circle.iter().all(|value| value.is_finite()));
        assert!(circle[0] >= 0.0 && circle[0] < width);
        assert!(circle[1] >= 0.0 && circle[1] < height);
        assert!(circle[2] + 1.0 >= min && circle[2] <= max + 1.0);
    }
    assert!(
        values
            .chunks_exact(3)
            .zip(values[3..].chunks_exact(3))
            .all(|(first, second)| (first[1], first[0], first[2])
                <= (second[1], second[0], second[2]))
    );
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
