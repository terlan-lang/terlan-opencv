//! Direct automatic Otsu threshold tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves binary modes, parameter bounds, source integrity, and cleanup.
#[test]
fn otsu_threshold_is_automatic_and_grayscale_only() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source_bytes = [0, 0, 0, 0, 255, 255, 255, 255];
    let source = ffi::mat_from_u8(&source_bytes, 2, 4, 1);
    let source_ref = source.as_ref().expect("bimodal grayscale image");
    let binary = source_ref.otsu_threshold_owned(200.0, 0);
    assert_eq!(
        binary
            .as_ref()
            .expect("binary Otsu threshold")
            .to_u8()
            .as_ref()
            .expect("binary threshold bytes")
            .as_slice(),
        [0, 0, 0, 0, 200, 200, 200, 200]
    );
    let inverse = source_ref.otsu_threshold_owned(200.0, 1);
    assert_eq!(
        inverse
            .as_ref()
            .expect("inverse Otsu threshold")
            .to_u8()
            .as_ref()
            .expect("inverse threshold bytes")
            .as_slice(),
        [200, 200, 200, 200, 0, 0, 0, 0]
    );
    let zero_maximum = source_ref.otsu_threshold_owned(0.0, 0);
    assert_eq!(
        zero_maximum
            .as_ref()
            .expect("zero-maximum Otsu threshold")
            .to_u8()
            .as_ref()
            .expect("zero-maximum bytes")
            .as_slice(),
        [0; 8]
    );
    assert_eq!(
        source_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source bytes")
            .as_slice(),
        source_bytes
    );

    let color = ffi::mat_zeros(2, 4, 3);
    assert!(color
        .as_ref()
        .expect("BGR image")
        .otsu_threshold_owned(200.0, 0)
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
        assert!(source_ref.otsu_threshold_owned(maximum, mode).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(color);
    drop(zero_maximum);
    drop(inverse);
    drop(binary);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}
