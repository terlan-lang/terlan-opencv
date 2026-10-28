//! Direct scale-and-absolute-value tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves exact transforms, layout preservation, finite coefficients, and ownership.
#[test]
fn scale_abs_is_per_channel_finite_and_saturating() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source_bytes = [0, 10, 100, 200, 255];
    let source = ffi::mat_from_u8(&source_bytes, 1, 5, 1);
    let source_ref = source.as_ref().expect("grayscale source");
    let transformed = source_ref.scale_abs_owned(2.0, -20.0);
    assert_eq!(
        transformed
            .as_ref()
            .expect("scaled result")
            .to_u8()
            .as_ref()
            .expect("scaled bytes")
            .as_slice(),
        [20, 0, 180, 255, 255]
    );
    assert_eq!(
        source_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source")
            .as_slice(),
        source_bytes
    );

    let negative = source_ref.scale_abs_owned(-2.0, 0.0);
    assert_eq!(
        negative
            .as_ref()
            .expect("negative-scale result")
            .to_u8()
            .as_ref()
            .expect("negative-scale bytes")
            .as_slice(),
        [0, 20, 200, 255, 255]
    );
    for (layout, channels) in [(3, 3_usize), (4, 4_usize)] {
        let color = ffi::mat_from_u8(&vec![255; channels], 1, 1, layout);
        let constant = color
            .as_ref()
            .expect("color source")
            .scale_abs_owned(0.0, -10.0);
        assert_eq!(
            constant
                .as_ref()
                .expect("constant result")
                .to_u8()
                .as_ref()
                .expect("constant bytes")
                .as_slice(),
            vec![10; channels]
        );
        drop(constant);
        drop(color);
    }

    for (scale, offset) in [
        (65_536.0, 0.0),
        (-65_536.0, 0.0),
        (1.0, 65_536.0),
        (1.0, -65_536.0),
    ] {
        assert!(!source_ref.scale_abs_owned(scale, offset).is_null());
    }
    for (scale, offset) in [
        (f64::NAN, 0.0),
        (f64::INFINITY, 0.0),
        (1.0, f64::NAN),
        (1.0, f64::NEG_INFINITY),
    ] {
        assert!(source_ref.scale_abs_owned(scale, offset).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert_eq!(ffi::live_mat_count(), 3);
    drop(negative);
    drop(transformed);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}
