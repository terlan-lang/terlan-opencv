//! Direct immutable full-layout binary bitwise tests.

use terlan_opencv_native::ffi;

/// Proves truth tables, self-aliasing, compatibility, layouts, and cleanup.
#[test]
fn bitwise_operations_preserve_inputs_and_layouts() {
    assert_eq!(ffi::live_mat_count(), 0);
    let first_bytes = [0, 15, 240, 170, 85, 255];
    let second_bytes = [255, 51, 15, 85, 170, 0];
    let first = ffi::mat_from_u8(&first_bytes, 2, 3, 1);
    let second = ffi::mat_from_u8(&second_bytes, 2, 3, 1);
    let first_ref = first.as_ref().expect("first bitwise source");
    let second_ref = second.as_ref().expect("second bitwise source");
    let and_result = first_ref.bitwise_and_owned(second_ref);
    let or_result = first_ref.bitwise_or_owned(second_ref);
    let xor_result = first_ref.bitwise_xor_owned(second_ref);
    assert_eq!(
        mat_bytes(and_result.as_ref().expect("bitwise and")),
        [0, 3, 0, 0, 0, 0]
    );
    assert_eq!(
        mat_bytes(or_result.as_ref().expect("bitwise or")),
        [255, 63, 255, 255, 255, 255]
    );
    assert_eq!(
        mat_bytes(xor_result.as_ref().expect("bitwise xor")),
        [255, 60, 255, 255, 255, 255]
    );
    assert_eq!(mat_bytes(first_ref), first_bytes);
    assert_eq!(mat_bytes(second_ref), second_bytes);

    let self_and = first_ref.bitwise_and_owned(first_ref);
    let self_or = first_ref.bitwise_or_owned(first_ref);
    let self_xor = first_ref.bitwise_xor_owned(first_ref);
    assert_eq!(mat_bytes(self_and.as_ref().expect("self and")), first_bytes);
    assert_eq!(mat_bytes(self_or.as_ref().expect("self or")), first_bytes);
    assert_eq!(mat_bytes(self_xor.as_ref().expect("self xor")), [0; 6]);

    let bgr_first = ffi::mat_from_u8(&[0, 15, 240], 1, 1, 3);
    let bgr_second = ffi::mat_from_u8(&[255, 51, 15], 1, 1, 3);
    let bgr_xor = bgr_first
        .as_ref()
        .expect("first BGR")
        .bitwise_xor_owned(bgr_second.as_ref().expect("second BGR"));
    assert_eq!(
        mat_bytes(bgr_xor.as_ref().expect("BGR xor")),
        [255, 60, 255]
    );
    let bgra_first = ffi::mat_from_u8(&[170, 85, 255, 0], 1, 1, 4);
    let bgra_second = ffi::mat_from_u8(&[85, 170, 0, 255], 1, 1, 4);
    let bgra_and = bgra_first
        .as_ref()
        .expect("first BGRA")
        .bitwise_and_owned(bgra_second.as_ref().expect("second BGRA"));
    assert_eq!(mat_bytes(bgra_and.as_ref().expect("BGRA and")), [0; 4]);

    let wrong_shape = ffi::mat_zeros(2, 2, 1);
    let wrong_layout = ffi::mat_zeros(2, 3, 3);
    for invalid in [&wrong_shape, &wrong_layout] {
        let invalid_ref = invalid.as_ref().expect("incompatible bitwise image");
        assert!(first_ref.bitwise_and_owned(invalid_ref).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(first_ref.bitwise_or_owned(invalid_ref).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(first_ref.bitwise_xor_owned(invalid_ref).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(wrong_layout);
    drop(wrong_shape);
    drop(bgra_and);
    drop(bgra_second);
    drop(bgra_first);
    drop(bgr_xor);
    drop(bgr_second);
    drop(bgr_first);
    drop(self_xor);
    drop(self_or);
    drop(self_and);
    drop(xor_result);
    drop(or_result);
    drop(and_result);
    drop(second);
    drop(first);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Proves one mask selects complete multichannel pixels.
#[test]
fn masked_bitwise_operations_select_multichannel_pixels() {
    assert_eq!(ffi::live_mat_count(), 0);
    let first = ffi::mat_from_u8(&[0, 15, 240, 170, 85, 255], 1, 2, 3);
    let second = ffi::mat_from_u8(&[255, 51, 15, 85, 170, 0], 1, 2, 3);
    let mask = ffi::mat_from_u8(&[255, 0], 1, 2, 1);
    let first_ref = first.as_ref().expect("first BGR source");
    let second_ref = second.as_ref().expect("second BGR source");
    let mask_ref = mask.as_ref().expect("BGR pixel mask");

    let inverted = first_ref.invert_masked_owned(mask_ref);
    let conjunction = first_ref.bitwise_and_masked_owned(second_ref, mask_ref);
    let disjunction = first_ref.bitwise_or_masked_owned(second_ref, mask_ref);
    let exclusive_or = first_ref.bitwise_xor_masked_owned(second_ref, mask_ref);

    assert_eq!(
        mat_bytes(inverted.as_ref().expect("masked BGR invert")),
        [255, 240, 15, 0, 0, 0]
    );
    assert_eq!(
        mat_bytes(conjunction.as_ref().expect("masked BGR and")),
        [0, 3, 0, 0, 0, 0]
    );
    assert_eq!(
        mat_bytes(disjunction.as_ref().expect("masked BGR or")),
        [255, 63, 255, 0, 0, 0]
    );
    assert_eq!(
        mat_bytes(exclusive_or.as_ref().expect("masked BGR xor")),
        [255, 60, 255, 0, 0, 0]
    );

    drop(exclusive_or);
    drop(disjunction);
    drop(conjunction);
    drop(inverted);
    drop(mask);
    drop(second);
    drop(first);
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
