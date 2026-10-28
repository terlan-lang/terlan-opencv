//! Direct tests for typed matrix storage and validation.

use std::sync::Mutex;
use terlan_opencv_native::ffi;

/// Serializes tests that assert the process-global native matrix count.
static MATRIX_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Acquires the matrix test lock even when a previous assertion poisoned it.
fn matrix_test_guard() -> std::sync::MutexGuard<'static, ()> {
    MATRIX_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Returns deterministic bytes for a typed matrix with the requested scalar width.
fn fixture_bytes(rows: usize, cols: usize, channels: usize, scalar_width: usize) -> Vec<u8> {
    let length = rows * cols * channels * scalar_width;
    (0..length).map(|index| (index % 251) as u8).collect()
}

/// Asserts that one supported OpenCV scalar depth round-trips exact storage bytes.
fn assert_depth_round_trip(depth: i64, scalar_width: i64) {
    let mut bytes = fixture_bytes(2, 3, 4, scalar_width as usize);
    let expected = bytes.clone();
    let matrix = ffi::mat_from_bytes(&bytes, 2, 3, 4, depth);
    assert!(!matrix.is_null(), "depth {depth} must construct");
    bytes.fill(255);

    let matrix = matrix.as_ref().expect("typed matrix");
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 3);
    assert_eq!(matrix.channels(), 4);
    assert_eq!(matrix.depth_code(), depth);
    assert_eq!(matrix.element_size(), scalar_width);
    assert_eq!(matrix.pixel_size(), 4 * scalar_width);
    assert_eq!(matrix.type_code(), depth + 24);
    assert_eq!(matrix.dimension_count(), 2);
    assert_eq!(matrix.element_count(), 6);
    assert_eq!(matrix.step_scalars(0), 12);
    assert_eq!(matrix.step_scalars(1), 4);
    assert_eq!(matrix.byte_count(), 24 * scalar_width);
    assert!(!matrix.empty());
    assert!(!matrix.is_submatrix());
    assert!(matrix.is_contiguous());
    assert_eq!(
        matrix
            .to_bytes()
            .as_ref()
            .expect("copied typed storage")
            .as_slice(),
        expected
    );
}

/// Verifies exact raw-byte preservation for every supported matrix depth.
#[test]
fn every_supported_depth_round_trips_exact_bytes() {
    let _guard = matrix_test_guard();
    assert_eq!(ffi::live_mat_count(), 0);
    for (depth, scalar_width) in [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 4),
        (5, 4),
        (6, 8),
        (7, 2),
    ] {
        assert_depth_round_trip(depth, scalar_width);
    }
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Verifies typed zero matrices across depth and channel boundaries.
#[test]
fn typed_zeros_report_shape_and_zero_storage() {
    let _guard = matrix_test_guard();
    assert_eq!(ffi::live_mat_count(), 0);
    for (depth, scalar_width, channels) in [(0, 1, 1), (3, 2, 3), (6, 8, 512)] {
        let matrix = ffi::mat_zeros_typed(1, 2, channels, depth);
        assert!(!matrix.is_null());
        let matrix = matrix.as_ref().expect("typed zero matrix");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.channels(), channels);
        assert_eq!(matrix.depth_code(), depth);
        assert_eq!(matrix.element_size(), scalar_width);
        assert_eq!(matrix.byte_count(), 2 * channels * scalar_width);
        assert_eq!(
            matrix
                .to_bytes()
                .as_ref()
                .expect("copied zero storage")
                .as_slice(),
            vec![0; (2 * channels * scalar_width) as usize]
        );
    }
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Verifies depth conversion preserves shape and channels across all scalar depths.
#[test]
fn typed_depth_conversion_preserves_matrix_structure() {
    let _guard = matrix_test_guard();
    assert_eq!(ffi::live_mat_count(), 0);
    let bytes = [0_u8, 10, 100, 200, 255, 64];
    let matrix = ffi::mat_from_bytes(&bytes, 1, 2, 3, 0);
    let matrix = matrix.as_ref().expect("u8 matrix");

    for depth in 0..=7 {
        let converted = matrix.convert_depth_owned(depth, 0.5, 1.0);
        assert!(!converted.is_null(), "conversion to depth {depth}");
        let converted = converted.as_ref().expect("converted matrix");
        assert_eq!(converted.rows(), 1);
        assert_eq!(converted.cols(), 2);
        assert_eq!(converted.channels(), 3);
        assert_eq!(converted.depth_code(), depth);
        assert!(!converted.is_submatrix());
        assert!(converted.is_contiguous());
    }

    for (depth, scale, offset) in [
        (-1, 1.0, 0.0),
        (8, 1.0, 0.0),
        (i64::MAX, 1.0, 0.0),
        (0, f64::NAN, 0.0),
        (0, f64::INFINITY, 0.0),
        (0, 1.0, f64::NEG_INFINITY),
    ] {
        let converted = matrix.convert_depth_owned(depth, scale, offset);
        assert!(converted.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert_eq!(ffi::live_mat_count(), 1);
}

/// Verifies malformed matrix requests fail without allocating a live resource.
#[test]
fn malformed_typed_matrix_requests_are_rejected() {
    let _guard = matrix_test_guard();
    assert_eq!(ffi::live_mat_count(), 0);
    for (bytes, rows, cols, channels, depth) in [
        (&[][..], 1, 1, 1, 0),
        (&[0][..], 0, 1, 1, 0),
        (&[0][..], 1, 0, 1, 0),
        (&[0][..], -1, 1, 1, 0),
        (&[0][..], 1, -1, 1, 0),
        (&[0][..], 1, 1, 0, 0),
        (&[0][..], 1, 1, -1, 0),
        (&[0][..], 1, 1, 513, 0),
        (&[0][..], 1, 1, 1, -1),
        (&[0][..], 1, 1, 1, 8),
        (&[0][..], 1, 1, 1, i64::MAX),
        (&[0][..], i64::MAX, 1, 1, 0),
        (&[0][..], 1, i64::MAX, 1, 0),
        (&[0][..], 1, 1, i64::MAX, 0),
    ] {
        let matrix = ffi::mat_from_bytes(bytes, rows, cols, channels, depth);
        assert!(matrix.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert_eq!(ffi::live_mat_count(), 0);
    }

    for (rows, cols, channels, depth) in [
        (0, 1, 1, 0),
        (1, 0, 1, 0),
        (1, 1, 0, 0),
        (1, 1, 513, 0),
        (1, 1, 1, -1),
        (1, 1, 1, 8),
        (i64::MAX, 1, 1, 0),
        (1, i64::MAX, 1, 0),
        (1, 1, i64::MAX, 0),
    ] {
        let matrix = ffi::mat_zeros_typed(rows, cols, channels, depth);
        assert!(matrix.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert_eq!(ffi::live_mat_count(), 0);
    }

    let matrix = ffi::mat_zeros_typed(2, 3, 4, 0);
    let matrix = matrix.as_ref().expect("valid matrix");
    for dimension in [-1, 2, i64::MAX] {
        assert_eq!(matrix.step_scalars(dimension), 0);
        assert_eq!(ffi::take_last_failure(), 1);
    }
}

/// Verifies every portable OpenCV resize interpolation has a symbolic code.
#[test]
fn every_resize_interpolation_mode_executes() {
    let _guard = matrix_test_guard();
    let matrix = ffi::mat_from_bytes(&[0, 64, 128, 255], 2, 2, 1, 0);
    let matrix = matrix.as_ref().expect("resize source");
    for interpolation in 0..=6 {
        let resized = matrix.resize_owned(3, 3, interpolation);
        let resized = resized.as_ref().expect("supported interpolation");
        assert_eq!((resized.rows(), resized.cols()), (3, 3));
        assert_eq!(resized.type_code(), 0);
    }
    assert!(matrix.resize_owned(3, 3, 7).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
}

/// Returns the exact bytes produced by transposing interleaved matrix pixels.
fn transposed_bytes(source: &[u8], rows: usize, cols: usize, pixel_width: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(source.len());
    for col in 0..cols {
        for row in 0..rows {
            let offset = (row * cols + col) * pixel_width;
            output.extend_from_slice(&source[offset..offset + pixel_width]);
        }
    }
    output
}

/// Returns exact finite scalar storage suitable for arithmetic at one depth.
fn operation_fixture_bytes(depth: i64, scalar_count: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    for index in 0..scalar_count {
        match depth {
            0 | 1 => bytes.push((index % 127) as u8),
            2 | 3 => bytes.extend_from_slice(&((index % 1024) as u16).to_ne_bytes()),
            4 => bytes.extend_from_slice(&(index as i32).to_ne_bytes()),
            5 => bytes.extend_from_slice(&(index as f32).to_ne_bytes()),
            6 => bytes.extend_from_slice(&(index as f64).to_ne_bytes()),
            7 => {
                let half = if index % 2 == 0 {
                    0x0000_u16
                } else {
                    0x3c00_u16
                };
                bytes.extend_from_slice(&half.to_ne_bytes());
            }
            _ => unreachable!("unsupported fixture depth"),
        }
    }
    bytes
}

/// Returns exact bytes produced by a clockwise 90-degree matrix rotation.
fn clockwise_bytes(source: &[u8], rows: usize, cols: usize, pixel_width: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(source.len());
    for col in 0..cols {
        for row in (0..rows).rev() {
            let offset = (row * cols + col) * pixel_width;
            output.extend_from_slice(&source[offset..offset + pixel_width]);
        }
    }
    output
}

/// Copies selected row-major pixels into one exact raw-byte fixture.
fn selected_pixels(
    source: &[u8],
    cols: usize,
    pixel_width: usize,
    coordinates: &[(usize, usize)],
) -> Vec<u8> {
    let mut output = Vec::with_capacity(coordinates.len() * pixel_width);
    for &(row, col) in coordinates {
        let offset = (row * cols + col) * pixel_width;
        output.extend_from_slice(&source[offset..offset + pixel_width]);
    }
    output
}

/// Repeats every row horizontally and then repeats the resulting rows vertically.
fn tiled_bytes(
    source: &[u8],
    rows: usize,
    cols: usize,
    pixel_width: usize,
    vertical: usize,
    horizontal: usize,
) -> Vec<u8> {
    let row_width = cols * pixel_width;
    let mut one_level = Vec::with_capacity(source.len() * horizontal);
    for row in 0..rows {
        let bytes = &source[row * row_width..(row + 1) * row_width];
        for _ in 0..horizontal {
            one_level.extend_from_slice(bytes);
        }
    }
    one_level.repeat(vertical)
}

/// Verifies depth-agnostic matrix operations preserve exact typed storage.
#[test]
fn matrix_preserving_operations_support_every_depth() {
    let _guard = matrix_test_guard();
    assert_eq!(ffi::live_mat_count(), 0);
    for (depth, scalar_width, channels) in [
        (0, 1, 1),
        (1, 1, 3),
        (2, 2, 4),
        (3, 2, 1),
        (4, 4, 3),
        (5, 4, 4),
        (6, 8, 1),
        (7, 2, 3),
    ] {
        let bytes = operation_fixture_bytes(depth, 2 * 3 * channels);
        let matrix = ffi::mat_from_bytes(&bytes, 2, 3, channels as i64, depth);
        let matrix = matrix.as_ref().expect("typed source");
        let mask_bytes = [255_u8, 0, 255, 0, 255, 0];
        let mask = ffi::mat_from_u8(&mask_bytes, 2, 3, 1);
        let mask = mask.as_ref().expect("matrix mask");

        let cloned = matrix.clone_owned();
        let inverted = matrix.invert_owned();
        let inverted_masked = matrix.invert_masked_owned(mask);
        let conjunction = matrix.bitwise_and_owned(matrix);
        let conjunction_masked = matrix.bitwise_and_masked_owned(matrix, mask);
        let disjunction = matrix.bitwise_or_owned(matrix);
        let disjunction_masked = matrix.bitwise_or_masked_owned(matrix, mask);
        let exclusive_or = matrix.bitwise_xor_owned(matrix);
        let exclusive_or_masked = matrix.bitwise_xor_masked_owned(matrix, mask);
        let flipped = matrix.flip_owned(1);
        let rotated = matrix.rotate_owned(0);
        let transposed = matrix.transpose_owned();
        let resized = matrix.resize_owned(3, 2, 0);
        let cropped = matrix.crop_owned(1, 0, 2, 2);
        let padded = matrix.pad_owned(1, 1, 1, 1, 0);
        let horizontal = matrix.concat_horizontal_owned(matrix);
        let vertical = matrix.concat_vertical_owned(matrix);
        let tiled = matrix.tile_owned(2, 2);
        let extracted = matrix.extract_channel_owned(0);
        let extraction_failure = if extracted.is_null() {
            ffi::take_last_failure()
        } else {
            0
        };
        let scaled_absolute = matrix.scale_abs_owned(1.0, 0.0);
        let scale_absolute_failure = if scaled_absolute.is_null() {
            ffi::take_last_failure()
        } else {
            0
        };
        assert!(!flipped.is_null(), "flip depth {depth} channels {channels}");
        assert!(
            !rotated.is_null(),
            "rotate depth {depth} channels {channels}"
        );
        assert!(
            !transposed.is_null(),
            "transpose depth {depth} channels {channels}"
        );
        for (name, result) in [
            ("resize", &resized),
            ("crop", &cropped),
            ("pad", &padded),
            ("horizontal concat", &horizontal),
            ("vertical concat", &vertical),
            ("tile", &tiled),
            ("invert", &inverted),
            ("masked invert", &inverted_masked),
            ("bitwise and", &conjunction),
            ("masked bitwise and", &conjunction_masked),
            ("bitwise or", &disjunction),
            ("masked bitwise or", &disjunction_masked),
            ("bitwise xor", &exclusive_or),
            ("masked bitwise xor", &exclusive_or_masked),
        ] {
            assert!(
                !result.is_null(),
                "{name} depth {depth} channels {channels}"
            );
        }
        if depth == 7 {
            assert_eq!(scale_absolute_failure, 3);
        } else {
            let scaled_absolute = scaled_absolute.as_ref().expect("scale absolute");
            assert_eq!(scaled_absolute.depth_code(), 0);
            assert_eq!(scaled_absolute.channels(), channels as i64);
        }

        assert_eq!(
            cloned
                .as_ref()
                .expect("clone")
                .to_bytes()
                .as_ref()
                .expect("clone bytes")
                .as_slice(),
            bytes
        );
        assert_eq!(
            inverted
                .as_ref()
                .expect("inverted")
                .to_bytes()
                .as_ref()
                .expect("inverted bytes")
                .as_slice(),
            bytes.iter().map(|value| !value).collect::<Vec<_>>()
        );
        for result in [&conjunction, &disjunction] {
            assert_eq!(
                result
                    .as_ref()
                    .expect("bitwise identity result")
                    .to_bytes()
                    .as_ref()
                    .expect("bitwise identity bytes")
                    .as_slice(),
                bytes
            );
        }
        assert_eq!(
            exclusive_or
                .as_ref()
                .expect("exclusive or")
                .to_bytes()
                .as_ref()
                .expect("exclusive-or bytes")
                .as_slice(),
            vec![0; bytes.len()]
        );
        let pixel_width = channels * scalar_width;
        let expected_masked = (0..6)
            .flat_map(|pixel| {
                let offset = pixel * pixel_width;
                if mask_bytes[pixel] == 0 {
                    vec![0; pixel_width]
                } else {
                    bytes[offset..offset + pixel_width].to_vec()
                }
            })
            .collect::<Vec<_>>();
        for result in [&conjunction_masked, &disjunction_masked] {
            assert_eq!(
                result
                    .as_ref()
                    .expect("masked bitwise result")
                    .to_bytes()
                    .as_ref()
                    .expect("masked bitwise bytes")
                    .as_slice(),
                expected_masked
            );
        }
        assert_eq!(
            exclusive_or_masked
                .as_ref()
                .expect("masked exclusive or")
                .to_bytes()
                .as_ref()
                .expect("masked exclusive-or bytes")
                .as_slice(),
            vec![0; bytes.len()]
        );
        let expected_inverted_masked = (0..6)
            .flat_map(|pixel| {
                let offset = pixel * pixel_width;
                if mask_bytes[pixel] == 0 {
                    vec![0; pixel_width]
                } else {
                    bytes[offset..offset + pixel_width]
                        .iter()
                        .map(|value| !value)
                        .collect()
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            inverted_masked
                .as_ref()
                .expect("masked invert")
                .to_bytes()
                .as_ref()
                .expect("masked invert bytes")
                .as_slice(),
            expected_inverted_masked
        );
        if depth != 7 {
            let difference = matrix.abs_diff_owned(matrix);
            let added = matrix.add_saturated_owned(matrix);
            let subtracted = matrix.subtract_saturated_owned(matrix);
            let minimum = matrix.minimum_owned(matrix);
            let maximum = matrix.maximum_owned(matrix);
            let multiplied = matrix.multiply_scaled_owned(matrix, 0.5);
            let divided = matrix.divide_scaled_owned(matrix, 1.0);
            let weighted = matrix.add_weighted_owned(matrix, 0.25, 0.75, 0.0);
            let comparison = matrix.compare_mask_owned(matrix, 0);
            let normalized = matrix.normalize_min_max_owned(-10.0, 10.0);
            assert!(!difference.is_null(), "absdiff depth {depth}");
            assert!(!added.is_null(), "addition depth {depth}");
            assert!(!subtracted.is_null(), "subtraction depth {depth}");
            assert!(!minimum.is_null(), "minimum depth {depth}");
            assert!(!maximum.is_null(), "maximum depth {depth}");
            assert!(!multiplied.is_null(), "multiplication depth {depth}");
            assert!(!divided.is_null(), "division depth {depth}");
            assert!(!weighted.is_null(), "weighted addition depth {depth}");
            assert!(!comparison.is_null(), "comparison depth {depth}");
            assert!(!normalized.is_null(), "normalization depth {depth}");
            assert_eq!(
                difference
                    .as_ref()
                    .expect("difference")
                    .to_bytes()
                    .as_ref()
                    .expect("difference bytes")
                    .as_slice(),
                vec![0; bytes.len()]
            );
            assert_eq!(
                subtracted
                    .as_ref()
                    .expect("subtraction")
                    .to_bytes()
                    .as_ref()
                    .expect("subtraction bytes")
                    .as_slice(),
                vec![0; bytes.len()]
            );
            for result in [
                &added,
                &minimum,
                &maximum,
                &multiplied,
                &divided,
                &weighted,
                &normalized,
            ] {
                let result = result.as_ref().expect("typed arithmetic result");
                assert_eq!(result.depth_code(), depth);
                assert_eq!(result.channels(), channels as i64);
            }
            for result in [&minimum, &maximum, &weighted] {
                assert_eq!(
                    result
                        .as_ref()
                        .expect("identity arithmetic result")
                        .to_bytes()
                        .as_ref()
                        .expect("identity arithmetic bytes")
                        .as_slice(),
                    bytes
                );
            }
            let comparison = comparison.as_ref().expect("comparison mask");
            assert_eq!(comparison.depth_code(), 0);
            assert_eq!(comparison.channels(), channels as i64);
            assert_eq!(
                comparison
                    .to_bytes()
                    .as_ref()
                    .expect("comparison bytes")
                    .as_slice(),
                vec![255; 2 * 3 * channels]
            );
        }

        assert_eq!(
            resized
                .as_ref()
                .expect("resized result")
                .to_bytes()
                .as_ref()
                .expect("resized bytes")
                .as_slice(),
            bytes
        );
        assert_eq!(
            cropped
                .as_ref()
                .expect("cropped result")
                .to_bytes()
                .as_ref()
                .expect("cropped bytes")
                .as_slice(),
            selected_pixels(&bytes, 3, pixel_width, &[(0, 1), (0, 2), (1, 1), (1, 2)])
        );
        let padded = padded.as_ref().expect("padded result");
        assert_eq!((padded.rows(), padded.cols()), (4, 5));
        assert_eq!(padded.depth_code(), depth);
        assert_eq!(padded.channels(), channels as i64);
        assert_eq!(
            horizontal
                .as_ref()
                .expect("horizontal concat")
                .to_bytes()
                .as_ref()
                .expect("horizontal bytes")
                .as_slice(),
            tiled_bytes(&bytes, 2, 3, pixel_width, 1, 2)
        );
        assert_eq!(
            vertical
                .as_ref()
                .expect("vertical concat")
                .to_bytes()
                .as_ref()
                .expect("vertical bytes")
                .as_slice(),
            bytes.repeat(2)
        );
        assert_eq!(
            tiled
                .as_ref()
                .expect("tiled result")
                .to_bytes()
                .as_ref()
                .expect("tiled bytes")
                .as_slice(),
            tiled_bytes(&bytes, 2, 3, pixel_width, 2, 2)
        );

        if depth == 7 {
            assert!(extracted.is_null(), "half-float extraction must fail");
            assert_eq!(extraction_failure, 3);
        } else {
            let extracted = extracted.as_ref().expect("extracted channel");
            assert_eq!(extracted.channels(), 1);
            assert_eq!(extracted.depth_code(), depth);
            let expected_channel = (0..6)
                .flat_map(|pixel| {
                    let offset = pixel * pixel_width;
                    bytes[offset..offset + scalar_width].iter().copied()
                })
                .collect::<Vec<_>>();
            assert_eq!(
                extracted
                    .to_bytes()
                    .as_ref()
                    .expect("extracted bytes")
                    .as_slice(),
                expected_channel
            );
            let replaced = matrix.replace_channel_owned(extracted, 0);
            assert_eq!(
                replaced
                    .as_ref()
                    .expect("replaced channel")
                    .to_bytes()
                    .as_ref()
                    .expect("replaced bytes")
                    .as_slice(),
                bytes
            );

            let merged = extracted.merge_bgr_owned(extracted, extracted);
            let expected_merged = expected_channel
                .chunks_exact(scalar_width)
                .flat_map(|scalar| scalar.iter().copied().cycle().take(scalar_width * 3))
                .collect::<Vec<_>>();
            assert_eq!(
                merged
                    .as_ref()
                    .expect("merged channels")
                    .to_bytes()
                    .as_ref()
                    .expect("merged bytes")
                    .as_slice(),
                expected_merged
            );
        }

        let expected_transpose = transposed_bytes(&bytes, 2, 3, pixel_width);
        let transposed = transposed.as_ref().expect("transposed result");
        assert_eq!((transposed.rows(), transposed.cols()), (3, 2));
        assert_eq!(transposed.depth_code(), depth);
        assert_eq!(
            transposed
                .to_bytes()
                .as_ref()
                .expect("transposed bytes")
                .as_slice(),
            expected_transpose
        );

        let rotated = rotated.as_ref().expect("rotated result");
        assert_eq!((rotated.rows(), rotated.cols()), (3, 2));
        assert_eq!(rotated.depth_code(), depth);
        assert_eq!(
            rotated
                .to_bytes()
                .as_ref()
                .expect("rotated bytes")
                .as_slice(),
            clockwise_bytes(&bytes, 2, 3, pixel_width)
        );

        let flipped = flipped.as_ref().expect("flipped result");
        assert_eq!(flipped.depth_code(), depth);
        for row in 0..2 {
            for col in 0..3 {
                let source_offset = (row * 3 + (2 - col)) * pixel_width;
                let output_offset = (row * 3 + col) * pixel_width;
                assert_eq!(
                    &flipped
                        .to_bytes()
                        .as_ref()
                        .expect("flipped bytes")
                        .as_slice()[output_offset..output_offset + pixel_width],
                    &bytes[source_offset..source_offset + pixel_width]
                );
            }
        }
    }
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Proves borrowed OpenCV views become independent package-owned matrices.
#[test]
fn matrix_view_operations_copy_checked_storage() {
    let _guard = matrix_test_guard();
    assert_eq!(ffi::live_mat_count(), 0);
    let bytes = (0_u8..12).collect::<Vec<_>>();
    let source = ffi::mat_from_u8(&bytes, 3, 4, 1);
    let source_ref = source.as_ref().expect("matrix view source");

    let rows = source_ref.row_range_copy_owned(1, 3);
    let columns = source_ref.col_range_copy_owned(1, 3);
    let diagonal = source_ref.diagonal_copy_owned(0);
    let upper_diagonal = source_ref.diagonal_copy_owned(1);
    let reshaped = source_ref.reshape_copy_owned(2, 2);

    let rows_ref = rows.as_ref().expect("copied row range");
    assert_eq!((rows_ref.rows(), rows_ref.cols()), (2, 4));
    assert_eq!(mat_bytes(rows_ref), [4, 5, 6, 7, 8, 9, 10, 11]);
    assert!(!rows_ref.is_submatrix());

    let columns_ref = columns.as_ref().expect("copied column range");
    assert_eq!((columns_ref.rows(), columns_ref.cols()), (3, 2));
    assert_eq!(mat_bytes(columns_ref), [1, 2, 5, 6, 9, 10]);
    assert!(!columns_ref.is_submatrix());

    let diagonal_ref = diagonal.as_ref().expect("copied diagonal");
    assert_eq!((diagonal_ref.rows(), diagonal_ref.cols()), (3, 1));
    assert_eq!(mat_bytes(diagonal_ref), [0, 5, 10]);
    assert!(!diagonal_ref.is_submatrix());

    let upper_diagonal_ref = upper_diagonal.as_ref().expect("copied upper diagonal");
    assert_eq!(
        (upper_diagonal_ref.rows(), upper_diagonal_ref.cols()),
        (3, 1)
    );
    assert_eq!(mat_bytes(upper_diagonal_ref), [1, 6, 11]);

    let reshaped_ref = reshaped.as_ref().expect("copied reshape");
    assert_eq!((reshaped_ref.rows(), reshaped_ref.cols()), (2, 3));
    assert_eq!(reshaped_ref.channels(), 2);
    assert_eq!(mat_bytes(reshaped_ref), bytes);
    assert!(!reshaped_ref.is_submatrix());

    macro_rules! assert_invalid {
        ($operation:expr) => {{
            let invalid = $operation;
            assert!(invalid.is_null());
            assert_eq!(ffi::take_last_failure(), 1);
        }};
    }
    assert_invalid!(source_ref.row_range_copy_owned(-1, 1));
    assert_invalid!(source_ref.row_range_copy_owned(1, 1));
    assert_invalid!(source_ref.row_range_copy_owned(0, 4));
    assert_invalid!(source_ref.col_range_copy_owned(-1, 1));
    assert_invalid!(source_ref.col_range_copy_owned(1, 1));
    assert_invalid!(source_ref.col_range_copy_owned(0, 5));
    assert_invalid!(source_ref.diagonal_copy_owned(-3));
    assert_invalid!(source_ref.diagonal_copy_owned(4));
    assert_invalid!(source_ref.reshape_copy_owned(0, 1));
    assert_invalid!(source_ref.reshape_copy_owned(2, 0));
    assert_invalid!(source_ref.reshape_copy_owned(2, 513));
    assert_invalid!(source_ref.reshape_copy_owned(5, 1));

    assert_eq!(mat_bytes(source_ref), bytes);
    drop(reshaped);
    drop(upper_diagonal);
    drop(diagonal);
    drop(columns);
    drop(rows);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one matrix's contiguous storage for concise assertions.
fn mat_bytes(matrix: &ffi::OpenCvMat) -> Vec<u8> {
    matrix
        .to_bytes()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
