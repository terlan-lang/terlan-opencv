//! Direct immutable adaptive grayscale histogram-equalization tests.

use terlan_opencv_native::ffi;

/// Proves CLAHE shape, validation, source integrity, ownership, and cleanup.
#[test]
fn adaptive_equalization_preserves_shape_and_source() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source_bytes = core::array::from_fn::<_, 64, _>(|index| (index * 4) as u8);
    let source = ffi::mat_from_u8(&source_bytes, 8, 8, 1);
    let source_ref = source.as_ref().expect("adaptive equalization source");
    let equalized = source_ref.adaptive_equalize_hist_owned(2.0, 2, 2);
    let equalized_ref = equalized.as_ref().expect("adaptive equalization result");
    assert_eq!(equalized_ref.rows(), 8);
    assert_eq!(equalized_ref.cols(), 8);
    assert_eq!(equalized_ref.channels(), 1);
    assert_eq!(mat_bytes(equalized_ref).len(), source_bytes.len());
    assert_eq!(mat_bytes(source_ref), source_bytes);

    for parameters in [
        (0.0, 1, 1),
        (-1.0, 1, 1),
        (257.0, 1, 1),
        (f64::NAN, 1, 1),
        (f64::INFINITY, 1, 1),
        (2.0, 0, 1),
        (2.0, 1, 0),
        (2.0, 9, 1),
        (2.0, 1, 9),
        (2.0, 65, 1),
        (2.0, 1, 65),
    ] {
        assert!(source_ref
            .adaptive_equalize_hist_owned(parameters.0, parameters.1, parameters.2)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    let color = ffi::mat_zeros(8, 8, 3);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for invalid in [&color, &oversized] {
        assert!(invalid
            .as_ref()
            .expect("invalid adaptive equalization source")
            .adaptive_equalize_hist_owned(2.0, 1, 1)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(color);
    drop(equalized);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
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
