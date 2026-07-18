//! Direct automatic Triangle threshold tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves binary modes, parameter bounds, source integrity, and cleanup.
#[test]
fn triangle_threshold_is_automatic_and_grayscale_only() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source_bytes = [0, 0, 0, 0, 255, 255, 255, 255];
    let source = ffi::mat_from_u8(&source_bytes, 2, 4, 1);
    let source_ref = source.as_ref().expect("bimodal grayscale image");
    assert_eq!(
        image_bytes(
            source_ref
                .triangle_threshold_owned(200.0, 0)
                .as_ref()
                .expect("binary Triangle threshold")
        ),
        [0, 0, 0, 0, 200, 200, 200, 200]
    );
    assert_eq!(
        image_bytes(
            source_ref
                .triangle_threshold_owned(200.0, 1)
                .as_ref()
                .expect("inverse Triangle threshold")
        ),
        [200, 200, 200, 200, 0, 0, 0, 0]
    );
    let zero_maximum = source_ref.triangle_threshold_owned(0.0, 0);
    assert_eq!(
        image_bytes(zero_maximum.as_ref().expect("zero-maximum threshold")),
        [0; 8]
    );
    assert_eq!(image_bytes(source_ref), source_bytes);

    let color = ffi::mat_zeros(2, 4, 3);
    assert!(color
        .as_ref()
        .expect("BGR image")
        .triangle_threshold_owned(200.0, 0)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    for (maximum, mode) in [
        (-0.1, 0),
        (255.1, 0),
        (f64::NAN, 0),
        (f64::INFINITY, 0),
        (200.0, -1),
        (200.0, 2),
        (200.0, 4),
    ] {
        assert!(source_ref.triangle_threshold_owned(maximum, mode).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(color);
    drop(zero_maximum);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native image into Rust bytes for deterministic assertions.
fn image_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("threshold bytes")
        .as_slice()
        .to_vec()
}
