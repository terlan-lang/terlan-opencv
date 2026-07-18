//! Direct bounded pyramid mean-shift tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves constant-color stability, rejection, and fresh ownership.
#[test]
fn mean_shift_filter_is_bgr_only_and_parameter_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);
    let constant_bgr = [
        10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20,
        30, 10, 20, 30,
    ];
    let source = ffi::mat_from_u8(&constant_bgr, 3, 3, 3);
    let source_ref = source.as_ref().expect("constant BGR image");
    let filtered = source_ref.mean_shift_filter_owned(1.0, 1.0, 0);
    let filtered_ref = filtered.as_ref().expect("mean-shift result");
    assert_eq!((filtered_ref.rows(), filtered_ref.cols()), (3, 3));
    assert_eq!(filtered_ref.channels(), 3);
    assert_eq!(
        filtered_ref
            .to_u8()
            .as_ref()
            .expect("mean-shift bytes")
            .as_slice(),
        constant_bgr
    );
    assert_eq!(
        source_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source bytes")
            .as_slice(),
        constant_bgr
    );

    let gray = ffi::mat_zeros(3, 3, 1);
    let bgra = ffi::mat_zeros(3, 3, 4);
    for image in [gray.as_ref().expect("gray"), bgra.as_ref().expect("BGRA")] {
        assert!(image.mean_shift_filter_owned(1.0, 1.0, 0).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (spatial, color, level) in [
        (0.0, 1.0, 0),
        (-1.0, 1.0, 0),
        (65_536.0, 1.0, 0),
        (f64::NAN, 1.0, 0),
        (f64::INFINITY, 1.0, 0),
        (1.0, 0.0, 0),
        (1.0, -1.0, 0),
        (1.0, 65_536.0, 0),
        (1.0, f64::NAN, 0),
        (1.0, f64::INFINITY, 0),
        (1.0, 1.0, -1),
        (1.0, 1.0, 9),
    ] {
        assert!(source_ref
            .mean_shift_filter_owned(spatial, color, level)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert_eq!(ffi::live_mat_count(), 4);

    drop(bgra);
    drop(gray);
    drop(filtered);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}
