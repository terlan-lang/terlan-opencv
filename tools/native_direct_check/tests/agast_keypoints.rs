//! Direct AGAST keypoint tests for the CXX bridge.

mod support;

use support::{detector_fixture, validate_detector_points};
use terlan_opencv_native::ffi;

/// Proves bounded canonical detection, source ownership, and rejection.
#[test]
fn agast_keypoints_are_bounded_copied_coordinates() {
    assert_eq!(ffi::live_mat_count(), 0);
    let fixture = detector_fixture();
    let image = ffi::mat_from_u8(&fixture, 32, 32, 1);
    let image_ref = image.as_ref().expect("AGAST fixture");
    let points = values(image_ref.agast_keypoints_owned(10, true, 8));
    validate_detector_points("AGAST", &points, 1, 8);
    assert_eq!(values(image_ref.agast_keypoints_owned(10, true, 8)), points);
    let one = values(image_ref.agast_keypoints_owned(10, true, 1));
    validate_detector_points("AGAST", &one, 1, 1);
    let unsuppressed = values(image_ref.agast_keypoints_owned(10, false, 8));
    validate_detector_points("AGAST", &unsuppressed, 1, 8);
    assert_eq!(bytes(image_ref), fixture);

    let empty = ffi::mat_zeros(32, 32, 1);
    assert!(values(
        empty
            .as_ref()
            .expect("empty image")
            .agast_keypoints_owned(10, true, 8)
    )
    .is_empty());
    for (threshold, maximum) in [(0, 1), (256, 1), (10, 0), (10, 4097)] {
        assert!(image_ref
            .agast_keypoints_owned(threshold, true, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let color = ffi::mat_zeros(1, 1, 3);
    assert!(color
        .as_ref()
        .expect("color image")
        .agast_keypoints_owned(10, true, 1)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    assert!(oversized
        .as_ref()
        .expect("oversized image")
        .agast_keypoints_owned(10, true, 1)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(oversized);
    drop(color);
    drop(empty);
    drop(image);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native vector into Rust-owned values.
fn values(values: cxx::UniquePtr<cxx::CxxVector<f64>>) -> Vec<f64> {
    values
        .as_ref()
        .expect("AGAST coordinates")
        .as_slice()
        .to_vec()
}

/// Copies one live matrix into Rust-owned bytes.
fn bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
