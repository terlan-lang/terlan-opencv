//! Direct immutable full-layout byte-inversion tests.

use terlan_opencv_native::ffi;

/// Proves exact inversion, involution, layout preservation, and cleanup.
#[test]
fn invert_preserves_source_and_layout() {
    assert_eq!(ffi::live_mat_count(), 0);
    let gray_bytes = [0, 1, 127, 128, 254, 255];
    let gray = ffi::mat_from_u8(&gray_bytes, 2, 3, 1);
    let gray_ref = gray.as_ref().expect("gray inversion source");
    let inverted = gray_ref.invert_owned();
    let inverted_ref = inverted.as_ref().expect("gray inversion result");
    assert_eq!(mat_bytes(inverted_ref), [255, 254, 128, 127, 1, 0]);
    let restored = inverted_ref.invert_owned();
    assert_eq!(
        mat_bytes(restored.as_ref().expect("restored gray")),
        gray_bytes
    );
    assert_eq!(mat_bytes(gray_ref), gray_bytes);

    let bgr = ffi::mat_from_u8(&[0, 10, 20, 255, 128, 64], 1, 2, 3);
    let inverted_bgr = bgr.as_ref().expect("BGR source").invert_owned();
    assert_eq!(
        mat_bytes(inverted_bgr.as_ref().expect("BGR inversion")),
        [255, 245, 235, 0, 127, 191]
    );
    let bgra = ffi::mat_from_u8(&[0, 1, 2, 3], 1, 1, 4);
    let inverted_bgra = bgra.as_ref().expect("BGRA source").invert_owned();
    assert_eq!(
        mat_bytes(inverted_bgra.as_ref().expect("BGRA inversion")),
        [255, 254, 253, 252]
    );

    drop(inverted_bgra);
    drop(bgra);
    drop(inverted_bgr);
    drop(bgr);
    drop(restored);
    drop(inverted);
    drop(gray);
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
