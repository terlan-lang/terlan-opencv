//! Direct nonzero-mask convex-hull approximation tests for the CXX bridge.

use terlan_opencv_native::ffi;

const OCTAGON: &[u8] = &[
    0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0,
    0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 8, 0, 0,
];

/// Proves exact, reduced, degenerate, invalid, immutable, and cleanup behavior.
#[test]
fn non_zero_approximate_hull_is_copied_and_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);
    let octagon = ffi::mat_from_u8(OCTAGON, 7, 7, 1);
    let octagon_ref = octagon.as_ref().expect("octagon mask");
    assert_eq!(
        approximate(octagon_ref, 0.0),
        [2, 0, 4, 0, 6, 2, 6, 4, 4, 6, 2, 6, 0, 4, 0, 2]
    );
    let reduced = approximate(octagon_ref, 1.0);
    assert!(reduced.len() >= 6 && reduced.len() < 16 && reduced.len().is_multiple_of(2));
    assert!(reduced
        .chunks_exact(2)
        .all(|point| point[0] >= 0 && point[0] <= 6 && point[1] >= 0 && point[1] <= 6));
    assert_eq!(mat_bytes(octagon_ref), OCTAGON);

    let point = ffi::mat_from_u8(&[0, 5, 0], 1, 3, 1);
    assert_eq!(
        approximate(point.as_ref().expect("point mask"), 1.0),
        [1, 0]
    );
    let line = ffi::mat_from_u8(&[1, 2, 3], 1, 3, 1);
    assert_eq!(
        approximate(line.as_ref().expect("line mask"), 2.0),
        [0, 0, 2, 0]
    );
    let empty = ffi::mat_zeros(3, 3, 1);
    assert!(approximate(empty.as_ref().expect("empty mask"), 0.0).is_empty());

    for epsilon in [-0.1, 9.0, f64::NAN, f64::INFINITY] {
        assert!(octagon_ref
            .non_zero_approximate_hull_owned(epsilon)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let bgr = ffi::mat_zeros(3, 3, 3);
    let bgra = ffi::mat_zeros(3, 3, 4);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for image in [&bgr, &bgra, &oversized] {
        assert!(image
            .as_ref()
            .expect("invalid approximation mask")
            .non_zero_approximate_hull_owned(0.0)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(bgra);
    drop(bgr);
    drop(empty);
    drop(line);
    drop(point);
    drop(octagon);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one approximated hull into a Rust integer vector.
fn approximate(image: &ffi::OpenCvMat, epsilon: f64) -> Vec<i64> {
    image
        .non_zero_approximate_hull_owned(epsilon)
        .as_ref()
        .expect("approximated hull")
        .as_slice()
        .to_vec()
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
