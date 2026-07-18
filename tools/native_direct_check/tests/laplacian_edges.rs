//! Direct absolute-Laplacian tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves signed-to-absolute conversion, bounds, source integrity, and cleanup.
#[test]
fn laplacian_edges_are_bounded_and_grayscale_only() {
    assert_eq!(ffi::live_mat_count(), 0);
    let bytes = [0, 0, 0, 0, 255, 0, 0, 0, 0];
    let source = ffi::mat_from_u8(&bytes, 3, 3, 1);
    let source_ref = source.as_ref().expect("grayscale impulse");
    let edges = source_ref.laplacian_edges_owned(1);
    let edges_ref = edges.as_ref().expect("absolute Laplacian edges");
    assert_eq!(
        (edges_ref.rows(), edges_ref.cols(), edges_ref.channels()),
        (3, 3, 1)
    );
    assert_eq!(
        edges_ref
            .to_u8()
            .as_ref()
            .expect("absolute Laplacian bytes")
            .as_slice(),
        [0, 255, 0, 255, 255, 255, 0, 255, 0]
    );
    assert_eq!(
        source_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source bytes")
            .as_slice(),
        bytes
    );
    for kernel_size in [3, 5, 7] {
        assert!(!source_ref.laplacian_edges_owned(kernel_size).is_null());
    }
    for kernel_size in [-1, 0, 2, 4, 9] {
        assert!(source_ref.laplacian_edges_owned(kernel_size).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let color = ffi::mat_zeros(3, 3, 3);
    assert!(color
        .as_ref()
        .expect("zero BGR image")
        .laplacian_edges_owned(3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(color);
    drop(edges);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}
