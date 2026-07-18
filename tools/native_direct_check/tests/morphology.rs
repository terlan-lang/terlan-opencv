//! Direct compound-morphology tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves all reviewed operations, bounds, layouts, source integrity, and cleanup.
#[test]
fn compound_morphology_has_exact_bounded_semantics() {
    assert_eq!(ffi::live_mat_count(), 0);
    let bytes = [0, 0, 0, 0, 255, 0, 0, 0, 0];
    let source = ffi::mat_from_u8(&bytes, 3, 3, 1);
    let source_ref = source.as_ref().expect("grayscale impulse");
    let expectations: &[(i64, &[u8])] = &[
        (0, &[0; 9]),
        (1, &[255; 9]),
        (2, &[255; 9]),
        (3, &bytes),
        (4, &[255, 255, 255, 255, 0, 255, 255, 255, 255]),
    ];
    let mut results = Vec::new();
    for (operation, expected) in expectations {
        let result = source_ref.morphology_owned(*operation, 3, 3, 1);
        assert_eq!(
            result
                .as_ref()
                .expect("compound morphology result")
                .to_u8()
                .as_ref()
                .expect("compound morphology bytes")
                .as_slice(),
            *expected
        );
        results.push(result);
    }
    assert_eq!(
        source_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source bytes")
            .as_slice(),
        bytes
    );
    for (operation, width, height, iterations) in [
        (-1, 3, 3, 1),
        (5, 3, 3, 1),
        (0, 0, 3, 1),
        (0, 2, 3, 1),
        (0, 257, 3, 1),
        (0, 3, 0, 1),
        (0, 3, 2, 1),
        (0, 3, 257, 1),
        (0, 3, 3, 0),
        (0, 3, 3, 65),
    ] {
        assert!(source_ref
            .morphology_owned(operation, width, height, iterations)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let bgr = ffi::mat_zeros(3, 3, 3);
    assert!(!bgr
        .as_ref()
        .expect("zero BGR image")
        .morphology_owned(0, 3, 3, 1)
        .is_null());
    let bgra = ffi::mat_zeros(3, 3, 4);
    assert!(!bgra
        .as_ref()
        .expect("zero BGRA image")
        .morphology_owned(1, 3, 3, 1)
        .is_null());

    drop(bgra);
    drop(bgr);
    drop(results);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}
