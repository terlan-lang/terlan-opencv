//! Direct semantic sharpening tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves exact filtering, layout preservation, bounds, and fresh ownership.
#[test]
fn sharpening_is_bounded_centered_and_saturating() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source_bytes = [10, 20, 30];
    let source = ffi::mat_from_u8(&source_bytes, 1, 3, 1);
    let source_ref = source.as_ref().expect("grayscale source");
    let sharpened = source_ref.sharpen_owned(1.0);
    assert_eq!(
        sharpened
            .as_ref()
            .expect("sharpened result")
            .to_u8()
            .as_ref()
            .expect("sharpened bytes")
            .as_slice(),
        [0, 20, 40]
    );
    assert_eq!(
        source_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source")
            .as_slice(),
        source_bytes
    );
    let identity = source_ref.sharpen_owned(0.0);
    assert_eq!(
        identity
            .as_ref()
            .expect("identity result")
            .to_u8()
            .as_ref()
            .expect("identity bytes")
            .as_slice(),
        source_bytes
    );

    for (layout, channels) in [(3, 3_usize), (4, 4_usize)] {
        let color = ffi::mat_from_u8(&vec![10; channels], 1, 1, layout);
        let stable = color
            .as_ref()
            .expect("constant color source")
            .sharpen_owned(16.0);
        assert_eq!(
            stable
                .as_ref()
                .expect("constant sharpened result")
                .to_u8()
                .as_ref()
                .expect("constant sharpened bytes")
                .as_slice(),
            vec![10; channels]
        );
        drop(stable);
        drop(color);
    }

    for amount in [-0.1, 16.1, f64::NAN, f64::INFINITY] {
        assert!(source_ref.sharpen_owned(amount).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert_eq!(ffi::live_mat_count(), 3);
    drop(identity);
    drop(sharpened);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}
