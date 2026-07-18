//! Direct immutable contained grayscale circle-fill tests.

use terlan_opencv_native::ffi;

/// Proves exact circle fill, validation, ownership, and cleanup.
#[test]
fn fill_circle_clones_before_rasterization() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source_bytes = [1; 25];
    let source = ffi::mat_from_u8(&source_bytes, 5, 5, 1);
    let source_ref = source.as_ref().expect("circle fill source");
    let filled = source_ref.fill_circle_owned(2, 2, 1, 9);
    assert_eq!(
        mat_bytes(filled.as_ref().expect("filled circle")),
        [1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 9, 9, 9, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1,]
    );
    let erased = source_ref.fill_circle_owned(2, 2, 1, 0);
    assert_eq!(
        mat_bytes(erased.as_ref().expect("erased circle fill")),
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,]
    );
    assert_eq!(mat_bytes(source_ref), source_bytes);

    let point_source = ffi::mat_zeros(1, 1, 1);
    let point = point_source
        .as_ref()
        .expect("point circle fill source")
        .fill_circle_owned(0, 0, 0, 7);
    assert_eq!(mat_bytes(point.as_ref().expect("point circle fill")), [7]);

    for parameters in [
        (-1, 2, 0, 0),
        (5, 2, 0, 0),
        (2, -1, 0, 0),
        (2, 5, 0, 0),
        (2, 2, -1, 0),
        (2, 2, 5, 0),
        (0, 2, 1, 0),
        (2, 0, 1, 0),
        (2, 2, 1, -1),
        (2, 2, 1, 256),
    ] {
        assert!(source_ref
            .fill_circle_owned(parameters.0, parameters.1, parameters.2, parameters.3)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let color = ffi::mat_zeros(5, 5, 3);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for invalid in [&color, &oversized] {
        assert!(invalid
            .as_ref()
            .expect("invalid circle fill source")
            .fill_circle_owned(0, 0, 0, 0)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(color);
    drop(point);
    drop(point_source);
    drop(erased);
    drop(filled);
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
