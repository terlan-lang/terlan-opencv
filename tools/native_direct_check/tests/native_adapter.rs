//! Direct ownership and operation tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves copied buffers, owned results, image operations, and deterministic drops.
#[test]
fn owned_matrix_lifecycle_and_first_surface_operations_work() {
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_component_analysis_count(), 0);
    assert_eq!(ffi::live_contour_analysis_count(), 0);

    let mut source = vec![0_u8, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0];
    let original = source.clone();
    let mat = ffi::mat_from_u8(&source, 2, 2, 3);
    assert!(!mat.is_null());
    source.fill(17);

    let mat_ref = mat.as_ref().expect("non-null matrix");
    assert_eq!(mat_ref.rows(), 2);
    assert_eq!(mat_ref.cols(), 2);
    assert_eq!(mat_ref.channels(), 3);
    assert!(mat_ref.is_contiguous());
    assert_eq!(
        mat_ref
            .to_u8()
            .as_ref()
            .expect("copied matrix bytes")
            .as_slice(),
        original
    );

    let cloned = mat_ref.clone_owned();
    assert!(!cloned.is_null());
    assert_eq!(ffi::live_mat_count(), 2);
    let difference_input =
        ffi::mat_from_u8(&[10, 8, 25, 20, 60, 40, 65, 90, 70, 100, 80, 120], 2, 2, 3);
    let difference_input_ref = difference_input.as_ref().expect("difference input");
    let difference = mat_ref.abs_diff_owned(difference_input_ref);
    assert_eq!(
        difference
            .as_ref()
            .expect("absolute difference")
            .to_u8()
            .as_ref()
            .expect("absolute difference bytes")
            .as_slice(),
        [10, 8, 25, 235, 195, 215, 65, 90, 185, 100, 175, 120]
    );
    let aliased_difference = mat_ref.abs_diff_owned(mat_ref);
    assert_eq!(
        aliased_difference
            .as_ref()
            .expect("aliased difference")
            .to_u8()
            .as_ref()
            .expect("aliased difference bytes")
            .as_slice(),
        [0; 12]
    );
    let weighted_first = ffi::mat_from_u8(
        &[0, 40, 80, 120, 160, 200, 20, 60, 100, 140, 180, 220],
        2,
        2,
        3,
    );
    let weighted_second = ffi::mat_from_u8(
        &[120, 80, 40, 0, 40, 80, 220, 180, 140, 100, 60, 20],
        2,
        2,
        3,
    );
    let weighted_first_ref = weighted_first.as_ref().expect("first weighted input");
    let weighted_second_ref = weighted_second.as_ref().expect("second weighted input");
    let weighted = weighted_first_ref.add_weighted_owned(weighted_second_ref, 0.25, 0.75, 10.0);
    assert_eq!(
        weighted
            .as_ref()
            .expect("weighted result")
            .to_u8()
            .as_ref()
            .expect("weighted bytes")
            .as_slice(),
        [100, 80, 60, 40, 80, 120, 180, 160, 140, 120, 100, 80]
    );
    let aliased_weighted = weighted_first_ref.add_weighted_owned(weighted_first_ref, 0.5, 0.5, 0.0);
    assert_eq!(
        aliased_weighted
            .as_ref()
            .expect("aliased weighted result")
            .to_u8()
            .as_ref()
            .expect("aliased weighted bytes")
            .as_slice(),
        [0, 40, 80, 120, 160, 200, 20, 60, 100, 140, 180, 220]
    );
    let saturated_weighted =
        weighted_first_ref.add_weighted_owned(weighted_second_ref, 1.0, 1.0, 100.0);
    assert_eq!(
        saturated_weighted
            .as_ref()
            .expect("saturated weighted result")
            .to_u8()
            .as_ref()
            .expect("saturated weighted bytes")
            .as_slice(),
        [220, 220, 220, 220, 255, 255, 255, 255, 255, 255, 255, 255]
    );
    for (first_weight, second_weight, offset) in [
        (16.01, 0.0, 0.0),
        (0.0, -16.01, 0.0),
        (f64::NAN, 0.0, 0.0),
        (0.0, f64::INFINITY, 0.0),
        (0.0, 0.0, 255.01),
        (0.0, 0.0, f64::NEG_INFINITY),
    ] {
        let invalid = weighted_first_ref.add_weighted_owned(
            weighted_second_ref,
            first_weight,
            second_weight,
            offset,
        );
        assert!(invalid.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let wrong_difference_shape = ffi::mat_zeros(1, 2, 3);
    let invalid_difference = mat_ref.abs_diff_owned(
        wrong_difference_shape
            .as_ref()
            .expect("wrong difference shape"),
    );
    assert!(invalid_difference.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let invalid_weighted = mat_ref.add_weighted_owned(
        wrong_difference_shape
            .as_ref()
            .expect("wrong weighted shape"),
        0.5,
        0.5,
        0.0,
    );
    assert!(invalid_weighted.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let wrong_difference_channels = ffi::mat_zeros(2, 2, 1);
    let invalid_difference = mat_ref.abs_diff_owned(
        wrong_difference_channels
            .as_ref()
            .expect("wrong difference channels"),
    );
    assert!(invalid_difference.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    drop(wrong_difference_channels);
    drop(wrong_difference_shape);
    drop(saturated_weighted);
    drop(aliased_weighted);
    drop(weighted);
    drop(weighted_second);
    drop(weighted_first);
    drop(aliased_difference);
    drop(difference);
    drop(difference_input);

    let resized = mat_ref.resize_owned(1, 1, 0);
    let resized_ref = resized.as_ref().expect("resized matrix");
    assert_eq!((resized_ref.rows(), resized_ref.cols()), (1, 1));
    assert_eq!(resized_ref.channels(), 3);

    let gray = mat_ref.cvt_color_owned(0);
    let gray_ref = gray.as_ref().expect("grayscale matrix");
    assert_eq!(gray_ref.channels(), 1);
    assert_eq!(
        gray_ref
            .to_u8()
            .as_ref()
            .expect("grayscale bytes")
            .as_slice(),
        [0, 255, 76, 150]
    );
    let equalized = gray_ref.equalize_hist_owned();
    let equalized_ref = equalized.as_ref().expect("equalized grayscale matrix");
    assert_eq!(
        equalized_ref
            .to_u8()
            .as_ref()
            .expect("equalized grayscale bytes")
            .as_slice(),
        [0, 255, 85, 170]
    );
    let histogram = gray_ref.histogram_u8_owned();
    let histogram_ref = histogram.as_ref().expect("copied grayscale histogram");
    assert_eq!(histogram_ref.len(), 256);
    assert_eq!(histogram_ref.get(0), Some(&1));
    assert_eq!(histogram_ref.get(76), Some(&1));
    assert_eq!(histogram_ref.get(150), Some(&1));
    assert_eq!(histogram_ref.get(255), Some(&1));
    assert_eq!(histogram_ref.iter().sum::<i64>(), 4);
    let component_mask = ffi::mat_from_u8(&[0, 255, 0, 0, 255, 0, 0, 0, 255], 3, 3, 1);
    let component_mask_ref = component_mask.as_ref().expect("component mask");
    let components_four = component_mask_ref.connected_components_owned(4);
    let components_four_ref = components_four.as_ref().expect("four-neighbor components");
    assert_eq!(components_four_ref.foreground_count(), 2);
    assert_eq!(
        components_four_ref
            .labels_owned()
            .as_ref()
            .expect("four labels")
            .as_slice(),
        [0, 1, 0, 0, 1, 0, 0, 0, 2]
    );
    assert_eq!(
        components_four_ref
            .areas_owned()
            .as_ref()
            .expect("four areas")
            .as_slice(),
        [2, 1]
    );
    assert_eq!(
        components_four_ref
            .bounds_owned()
            .as_ref()
            .expect("four bounds")
            .as_slice(),
        [1, 0, 1, 2, 2, 2, 1, 1]
    );
    let components_eight = component_mask_ref.connected_components_owned(8);
    let components_eight_ref = components_eight
        .as_ref()
        .expect("eight-neighbor components");
    assert_eq!(components_eight_ref.foreground_count(), 1);
    assert_eq!(
        components_eight_ref
            .labels_owned()
            .as_ref()
            .expect("eight labels")
            .as_slice(),
        [0, 1, 0, 0, 1, 0, 0, 0, 1]
    );
    assert_eq!(
        components_eight_ref
            .areas_owned()
            .as_ref()
            .expect("eight areas")
            .as_slice(),
        [3]
    );
    assert_eq!(
        components_eight_ref
            .bounds_owned()
            .as_ref()
            .expect("eight bounds")
            .as_slice(),
        [1, 0, 2, 3]
    );
    assert_eq!(ffi::live_component_analysis_count(), 2);
    let contour_mask = ffi::mat_from_u8(
        &[
            0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0,
            0, 0,
        ],
        5,
        5,
        1,
    );
    let contour_mask_ref = contour_mask.as_ref().expect("contour mask");
    let contours = contour_mask_ref.find_external_contours_owned();
    let contours_ref = contours.as_ref().expect("external contours");
    assert_eq!(ffi::live_contour_analysis_count(), 1);
    assert_eq!(contours_ref.count(), 1);
    assert_eq!(
        contours_ref
            .points_owned()
            .as_ref()
            .expect("contour points")
            .as_slice(),
        [1, 1, 1, 3, 3, 3, 3, 1]
    );
    assert_eq!(
        contours_ref
            .offsets_owned()
            .as_ref()
            .expect("contour offsets")
            .as_slice(),
        [0, 4]
    );
    assert_eq!(
        contours_ref
            .bounds_owned()
            .as_ref()
            .expect("contour bounds")
            .as_slice(),
        [1, 1, 3, 3]
    );
    assert_eq!(
        contours_ref
            .areas_owned()
            .as_ref()
            .expect("contour areas")
            .as_slice(),
        [4.0]
    );
    assert_eq!(
        contours_ref
            .perimeters_owned()
            .as_ref()
            .expect("contour perimeters")
            .as_slice(),
        [8.0]
    );
    assert_eq!(
        contours_ref
            .convex_areas_owned()
            .as_ref()
            .expect("contour convex areas")
            .as_slice(),
        [4.0]
    );
    assert_eq!(
        contours_ref
            .solidities_owned()
            .as_ref()
            .expect("contour solidities")
            .as_slice(),
        [1.0]
    );
    let contour_circularities = contours_ref.circularities_owned();
    let contour_circularities = contour_circularities
        .as_ref()
        .expect("contour circularities")
        .as_slice();
    assert_eq!(contour_circularities.len(), 1);
    assert!((contour_circularities[0] - std::f64::consts::FRAC_PI_4).abs() < 1.0e-12);
    let contour_extents = contours_ref.extents_owned();
    let contour_extents = contour_extents
        .as_ref()
        .expect("contour extents")
        .as_slice();
    assert_eq!(contour_extents.len(), 1);
    assert!((contour_extents[0] - 4.0 / 9.0).abs() < 1.0e-12);
    assert_eq!(
        contours_ref
            .centroids_owned()
            .as_ref()
            .expect("contour centroids")
            .as_slice(),
        [2.0, 2.0]
    );
    let contour_hu_moments = contours_ref.hu_moments_owned();
    let contour_hu_moments = contour_hu_moments
        .as_ref()
        .expect("contour Hu moments")
        .as_slice();
    assert_eq!(contour_hu_moments.len(), 7);
    assert!(contour_hu_moments.iter().all(|value| value.is_finite()));
    assert!(contour_hu_moments[0] > 0.0);
    let contour_minimum_area_boxes = contours_ref.minimum_area_boxes_owned();
    let contour_minimum_area_boxes = contour_minimum_area_boxes
        .as_ref()
        .unwrap_or_else(|| {
            panic!(
                "contour minimum-area boxes failure code {}",
                ffi::take_last_failure()
            )
        })
        .as_slice();
    assert_eq!(contour_minimum_area_boxes.len(), 8);
    assert!(contour_minimum_area_boxes
        .iter()
        .zip([1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0])
        .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-4));
    let contour_minimum_enclosing_circles = contours_ref.minimum_enclosing_circles_owned();
    let contour_minimum_enclosing_circles = contour_minimum_enclosing_circles
        .as_ref()
        .expect("contour minimum enclosing circles")
        .as_slice();
    assert_eq!(contour_minimum_enclosing_circles.len(), 3);
    assert!((contour_minimum_enclosing_circles[0] - 2.0).abs() <= 1.0e-4);
    assert!((contour_minimum_enclosing_circles[1] - 2.0).abs() <= 1.0e-4);
    assert!((1.414..1.415).contains(&contour_minimum_enclosing_circles[2]));
    let contour_minimum_enclosing_triangles = contours_ref.minimum_enclosing_triangles_owned();
    let contour_minimum_enclosing_triangles = contour_minimum_enclosing_triangles
        .as_ref()
        .expect("contour minimum enclosing triangles")
        .as_slice();
    assert_eq!(contour_minimum_enclosing_triangles.len(), 6);
    assert!(contour_minimum_enclosing_triangles
        .iter()
        .all(|value| value.is_finite()));
    let contour_fitted_lines = contours_ref.fitted_lines_owned();
    let contour_fitted_lines = contour_fitted_lines
        .as_ref()
        .expect("contour fitted lines")
        .as_slice();
    assert_eq!(contour_fitted_lines.len(), 4);
    assert!(contour_fitted_lines.iter().all(|value| value.is_finite()));
    assert!((contour_fitted_lines[0].hypot(contour_fitted_lines[1]) - 1.0).abs() <= 1.0e-3);
    assert_eq!(
        contours_ref
            .fitted_ellipses_owned()
            .as_ref()
            .expect("contour fitted ellipses")
            .as_slice(),
        [2.0, 2.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        contours_ref
            .shape_match_scores_owned(0)
            .as_ref()
            .expect("contour self-match scores")
            .as_slice(),
        [0.0, 0.0, 0.0]
    );
    for reference_index in [-1, 1] {
        assert!(contours_ref
            .shape_match_scores_owned(reference_index)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    drop(contours);
    drop(contour_mask);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
    let feature_bytes = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255,
        0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let feature_mask = ffi::mat_from_u8(&feature_bytes, 7, 7, 1);
    let feature_mask_ref = feature_mask.as_ref().expect("feature mask");
    let coordinates = feature_mask_ref.good_features_to_track_owned(8, 0.01, 1.0);
    let coordinates_ref = coordinates.as_ref().expect("feature coordinates");
    assert!((2..=8).contains(&coordinates_ref.len()));
    assert_eq!(coordinates_ref.len() % 2, 0);
    for point in coordinates_ref.as_slice().chunks_exact(2) {
        assert!(point[0].is_finite() && point[1].is_finite());
        assert!((0.0..7.0).contains(&point[0]));
        assert!((0.0..7.0).contains(&point[1]));
    }
    assert_eq!(
        feature_mask_ref
            .to_u8()
            .as_ref()
            .expect("unchanged feature bytes")
            .as_slice(),
        feature_bytes
    );
    let hu_moments = feature_mask_ref.hu_moments_owned();
    let hu_moments_ref = hu_moments.as_ref().expect("Hu moments");
    assert_eq!(hu_moments_ref.len(), 7);
    assert!(hu_moments_ref.iter().all(|value| value.is_finite()));
    let first_hu_moment = *hu_moments_ref.get(0).expect("first Hu moment");
    assert!(first_hu_moment > 0.0 && first_hu_moment < 1.0);
    let limited_coordinates = feature_mask_ref.good_features_to_track_owned(1, 0.01, 0.0);
    assert_eq!(
        limited_coordinates
            .as_ref()
            .expect("one feature coordinate")
            .len(),
        2
    );
    drop(limited_coordinates);
    drop(hu_moments);
    drop(coordinates);
    drop(feature_mask);
    let empty_feature_mask = ffi::mat_zeros(7, 7, 1);
    let empty_hu_moments = empty_feature_mask
        .as_ref()
        .expect("empty feature mask")
        .hu_moments_owned();
    assert_eq!(
        empty_hu_moments
            .as_ref()
            .expect("empty Hu moments")
            .as_slice(),
        [0.0; 7]
    );
    drop(empty_hu_moments);
    drop(empty_feature_mask);
    let thresholded = gray_ref.threshold_owned(100.0, 200.0, 0);
    let thresholded_ref = thresholded.as_ref().expect("thresholded matrix");
    assert_eq!(
        thresholded_ref
            .to_u8()
            .as_ref()
            .expect("thresholded bytes")
            .as_slice(),
        [0, 200, 0, 200]
    );
    let normalized = thresholded_ref.normalize_min_max_owned(0.0, 255.0);
    let normalized_ref = normalized.as_ref().expect("normalized matrix");
    assert_eq!(
        normalized_ref
            .to_u8()
            .as_ref()
            .expect("normalized bytes")
            .as_slice(),
        [0, 255, 0, 255]
    );
    let eroded = thresholded_ref.erode_owned(3, 3, 1);
    let eroded_ref = eroded.as_ref().expect("eroded matrix");
    assert_eq!(
        eroded_ref
            .to_u8()
            .as_ref()
            .expect("eroded bytes")
            .as_slice(),
        [0, 0, 0, 0]
    );
    let dilated = thresholded_ref.dilate_owned(3, 3, 1);
    let dilated_ref = dilated.as_ref().expect("dilated matrix");
    assert_eq!(
        dilated_ref
            .to_u8()
            .as_ref()
            .expect("dilated bytes")
            .as_slice(),
        [200, 200, 200, 200]
    );
    let edge_source = ffi::mat_zeros(3, 3, 1);
    let edge_source_ref = edge_source.as_ref().expect("edge source matrix");
    let edges = edge_source_ref.canny_owned(10.0, 20.0, 3, false);
    let edges_ref = edges.as_ref().expect("Canny edge matrix");
    assert_eq!(
        edges_ref
            .to_u8()
            .as_ref()
            .expect("Canny edge bytes")
            .as_slice(),
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    let edges_l2 = edge_source_ref.canny_owned(10.0, 20.0, 3, true);
    let edges_l2_ref = edges_l2.as_ref().expect("L2 Canny edge matrix");
    assert_eq!(
        edges_l2_ref
            .to_u8()
            .as_ref()
            .expect("L2 Canny edge bytes")
            .as_slice(),
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    );

    let blurred = mat_ref.gaussian_blur_owned(3, 3, 0.0);
    let blurred_ref = blurred.as_ref().expect("blurred matrix");
    assert_eq!((blurred_ref.rows(), blurred_ref.cols()), (2, 2));
    assert_eq!(blurred_ref.channels(), 3);

    let cropped = mat_ref.crop_owned(1, 0, 1, 2);
    let cropped_ref = cropped.as_ref().expect("cropped matrix");
    assert_eq!((cropped_ref.rows(), cropped_ref.cols()), (2, 1));
    assert_eq!(cropped_ref.channels(), 3);
    assert_eq!(
        cropped_ref
            .to_u8()
            .as_ref()
            .expect("cropped bytes")
            .as_slice(),
        [255, 255, 255, 0, 255, 0]
    );
    let padded = cropped_ref.pad_owned(1, 0, 1, 0, 0);
    let padded_ref = padded.as_ref().expect("padded matrix");
    assert_eq!((padded_ref.rows(), padded_ref.cols()), (3, 2));
    assert_eq!(padded_ref.channels(), 3);

    let flipped = mat_ref.flip_owned(1);
    let flipped_ref = flipped.as_ref().expect("horizontally flipped matrix");
    assert_eq!(
        flipped_ref
            .to_u8()
            .as_ref()
            .expect("flipped bytes")
            .as_slice(),
        [255, 255, 255, 0, 0, 0, 0, 255, 0, 0, 0, 255]
    );
    let rotated = cropped_ref.rotate_owned(0);
    let rotated_ref = rotated.as_ref().expect("clockwise rotated matrix");
    assert_eq!((rotated_ref.rows(), rotated_ref.cols()), (1, 2));
    assert_eq!(
        rotated_ref
            .to_u8()
            .as_ref()
            .expect("rotated bytes")
            .as_slice(),
        [0, 255, 0, 255, 255, 255]
    );
    let affine = mat_ref.warp_affine_owned(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2, 2, 0, 0);
    let affine_ref = affine.as_ref().expect("identity affine matrix");
    assert_eq!(
        affine_ref
            .to_u8()
            .as_ref()
            .expect("identity affine bytes")
            .as_slice(),
        original
    );
    let perspective =
        mat_ref.warp_perspective_owned(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2, 2, 0, 0);
    let perspective_ref = perspective.as_ref().expect("identity perspective matrix");
    assert_eq!(
        perspective_ref
            .to_u8()
            .as_ref()
            .expect("identity perspective bytes")
            .as_slice(),
        original
    );

    let encoded = gray_ref.encode_owned(0);
    let encoded_ref = encoded.as_ref().expect("BMP bytes");
    assert_eq!(&encoded_ref.as_slice()[..2], b"BM");
    let decoded = ffi::decode_image(encoded_ref.as_slice(), 0);
    let decoded_ref = decoded.as_ref().expect("decoded BMP matrix");
    assert_eq!((decoded_ref.rows(), decoded_ref.cols()), (2, 2));
    assert_eq!(decoded_ref.channels(), 1);
    assert_eq!(
        decoded_ref
            .to_u8()
            .as_ref()
            .expect("decoded bytes")
            .as_slice(),
        [0, 255, 76, 150]
    );

    let png = gray_ref.encode_owned(1);
    let png_ref = png.as_ref().expect("PNG bytes");
    assert_eq!(&png_ref.as_slice()[..8], b"\x89PNG\r\n\x1a\n");
    let decoded_png = ffi::decode_image(png_ref.as_slice(), 0);
    let decoded_png_ref = decoded_png.as_ref().expect("decoded PNG matrix");
    assert_eq!((decoded_png_ref.rows(), decoded_png_ref.cols()), (2, 2));
    assert_eq!(decoded_png_ref.channels(), 1);
    assert_eq!(
        decoded_png_ref
            .to_u8()
            .as_ref()
            .expect("decoded PNG bytes")
            .as_slice(),
        [0, 255, 76, 150]
    );

    let jpeg = gray_ref.encode_owned(2);
    let jpeg_ref = jpeg.as_ref().expect("JPEG bytes");
    assert_eq!(&jpeg_ref.as_slice()[..3], b"\xff\xd8\xff");
    let repeated_jpeg = gray_ref.encode_owned(2);
    assert_eq!(
        jpeg_ref.as_slice(),
        repeated_jpeg
            .as_ref()
            .expect("repeated JPEG bytes")
            .as_slice()
    );
    let decoded_jpeg = ffi::decode_image(jpeg_ref.as_slice(), 0);
    let decoded_jpeg_ref = decoded_jpeg.as_ref().expect("decoded JPEG matrix");
    assert_eq!((decoded_jpeg_ref.rows(), decoded_jpeg_ref.cols()), (2, 2));
    assert_eq!(decoded_jpeg_ref.channels(), 1);
    assert_eq!(
        decoded_jpeg_ref
            .to_u8()
            .as_ref()
            .expect("decoded JPEG bytes")
            .len(),
        4
    );

    let webp = gray_ref.encode_owned(3);
    let webp_ref = webp.as_ref().expect("WebP bytes");
    assert_eq!(&webp_ref.as_slice()[..4], b"RIFF");
    assert_eq!(&webp_ref.as_slice()[8..12], b"WEBP");
    let repeated_webp = gray_ref.encode_owned(3);
    assert_eq!(
        webp_ref.as_slice(),
        repeated_webp
            .as_ref()
            .expect("repeated WebP bytes")
            .as_slice()
    );
    let decoded_webp = ffi::decode_image(webp_ref.as_slice(), 0);
    let decoded_webp_ref = decoded_webp.as_ref().expect("decoded WebP matrix");
    assert_eq!((decoded_webp_ref.rows(), decoded_webp_ref.cols()), (2, 2));
    assert_eq!(decoded_webp_ref.channels(), 1);

    let tiff = gray_ref.encode_owned(4);
    let tiff_ref = tiff.as_ref().expect("TIFF bytes");
    assert!(matches!(
        &tiff_ref.as_slice()[..4],
        b"II\x2a\x00" | b"MM\x00\x2a"
    ));
    let repeated_tiff = gray_ref.encode_owned(4);
    assert_eq!(
        tiff_ref.as_slice(),
        repeated_tiff
            .as_ref()
            .expect("repeated TIFF bytes")
            .as_slice()
    );
    let decoded_tiff = ffi::decode_image(tiff_ref.as_slice(), 0);
    let decoded_tiff_ref = decoded_tiff.as_ref().expect("decoded TIFF matrix");
    assert_eq!((decoded_tiff_ref.rows(), decoded_tiff_ref.cols()), (2, 2));
    assert_eq!(decoded_tiff_ref.channels(), 1);
    assert_eq!(
        decoded_tiff_ref
            .to_u8()
            .as_ref()
            .expect("decoded TIFF bytes")
            .as_slice(),
        [0, 255, 76, 150]
    );

    let invalid_encoded = ffi::decode_image(b"not-bmp", 0);
    assert!(invalid_encoded.is_null());
    assert_eq!(ffi::take_last_failure(), 6);
    let invalid_decode_mode = ffi::decode_image(b"BM", 99);
    assert!(invalid_decode_mode.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let invalid_format = mat_ref.encode_owned(99);
    assert!(invalid_format.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    for (x, y, width, height) in [(2, 0, 1, 1), (0, 0, i64::MAX, 1)] {
        let invalid_crop = mat_ref.crop_owned(x, y, width, height);
        assert!(invalid_crop.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (top, bottom, left, right, border_mode) in
        [(-1, 0, 0, 0, 0), (i64::MAX, 0, 0, 0, 0), (0, 0, 0, 0, 99)]
    {
        let invalid_pad = mat_ref.pad_owned(top, bottom, left, right, border_mode);
        assert!(invalid_pad.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let invalid_flip = mat_ref.flip_owned(99);
    assert!(invalid_flip.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let invalid_rotate = mat_ref.rotate_owned(99);
    assert!(invalid_rotate.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    for (threshold_value, maximum_value, mode) in [
        (100.0, 200.0, 99),
        (-0.1, 200.0, 0),
        (255.1, 200.0, 0),
        (f64::NAN, 200.0, 0),
        (f64::INFINITY, 200.0, 0),
        (100.0, -0.1, 0),
        (100.0, 255.1, 0),
        (100.0, f64::NAN, 0),
        (100.0, f64::INFINITY, 0),
    ] {
        let invalid_threshold = gray_ref.threshold_owned(threshold_value, maximum_value, mode);
        assert!(invalid_threshold.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (kernel_width, kernel_height, iterations) in [
        (0, 3, 1),
        (2, 3, 1),
        (3, 0, 1),
        (3, 2, 1),
        (257, 3, 1),
        (3, 257, 1),
        (3, 3, 0),
        (3, 3, 65),
    ] {
        let invalid_erode = gray_ref.erode_owned(kernel_width, kernel_height, iterations);
        assert!(invalid_erode.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        let invalid_dilate = gray_ref.dilate_owned(kernel_width, kernel_height, iterations);
        assert!(invalid_dilate.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let invalid_color_canny = mat_ref.canny_owned(10.0, 20.0, 3, false);
    assert!(invalid_color_canny.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    for (low_threshold, high_threshold, aperture_size) in [
        (-0.1, 20.0, 3),
        (10.0, 65536.0, 3),
        (f64::NAN, 20.0, 3),
        (10.0, f64::NAN, 3),
        (f64::INFINITY, 20.0, 3),
        (10.0, f64::INFINITY, 3),
        (20.0, 10.0, 3),
        (10.0, 20.0, 1),
        (10.0, 20.0, 4),
        (10.0, 20.0, 9),
    ] {
        let invalid_canny =
            gray_ref.canny_owned(low_threshold, high_threshold, aperture_size, false);
        assert!(invalid_canny.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (minimum, maximum) in [
        (f64::NAN, 255.0),
        (0.0, f64::NAN),
        (f64::INFINITY, 255.0),
        (0.0, f64::INFINITY),
        (20.0, 20.0),
        (20.0, 10.0),
    ] {
        let invalid_normalization = gray_ref.normalize_min_max_owned(minimum, maximum);
        assert!(invalid_normalization.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (m00, width, interpolation, border_mode) in [
        (f64::NAN, 2, 0, 0),
        (f64::INFINITY, 2, 0, 0),
        (1_000_001.0, 2, 0, 0),
        (1.0, 0, 0, 0),
        (1.0, 2, 99, 0),
        (1.0, 2, 0, 99),
    ] {
        let invalid_affine = mat_ref.warp_affine_owned(
            m00,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            width,
            2,
            interpolation,
            border_mode,
        );
        assert!(invalid_affine.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (m20, width, interpolation, border_mode) in [
        (f64::NAN, 2, 0, 0),
        (f64::INFINITY, 2, 0, 0),
        (1_000_001.0, 2, 0, 0),
        (0.0, 0, 0, 0),
        (0.0, 2, 99, 0),
        (0.0, 2, 0, 99),
    ] {
        let invalid_perspective = mat_ref.warp_perspective_owned(
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            m20,
            0.0,
            1.0,
            width,
            2,
            interpolation,
            border_mode,
        );
        assert!(invalid_perspective.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let invalid_color_equalization = mat_ref.equalize_hist_owned();
    assert!(invalid_color_equalization.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let invalid_color_histogram = mat_ref.histogram_u8_owned();
    assert!(invalid_color_histogram.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let invalid_color_components = mat_ref.connected_components_owned(4);
    assert!(invalid_color_components.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let invalid_color_contours = mat_ref.find_external_contours_owned();
    assert!(invalid_color_contours.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let invalid_color_features = mat_ref.good_features_to_track_owned(1, 0.01, 1.0);
    assert!(invalid_color_features.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let invalid_color_moments = mat_ref.hu_moments_owned();
    assert!(invalid_color_moments.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    for (max_corners, quality_level, minimum_distance) in [
        (0, 0.01, 1.0),
        (4097, 0.01, 1.0),
        (1, 0.0, 1.0),
        (1, 1.01, 1.0),
        (1, f64::NAN, 1.0),
        (1, f64::INFINITY, 1.0),
        (1, 0.01, -0.1),
        (1, 0.01, 1_000_001.0),
        (1, 0.01, f64::NAN),
        (1, 0.01, f64::INFINITY),
    ] {
        let invalid_features =
            gray_ref.good_features_to_track_owned(max_corners, quality_level, minimum_distance);
        assert!(invalid_features.is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let invalid_connectivity = component_mask_ref.connected_components_owned(6);
    assert!(invalid_connectivity.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let oversized_component_source = ffi::mat_zeros(1025, 1024, 1);
    let oversized_component_source_ref = oversized_component_source
        .as_ref()
        .expect("valid matrix above component limit");
    let invalid_oversized_components = oversized_component_source_ref.connected_components_owned(8);
    assert!(invalid_oversized_components.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let invalid_oversized_contours = oversized_component_source_ref.find_external_contours_owned();
    assert!(invalid_oversized_contours.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let invalid_oversized_features =
        oversized_component_source_ref.good_features_to_track_owned(1, 0.01, 1.0);
    assert!(invalid_oversized_features.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let invalid_oversized_moments = oversized_component_source_ref.hu_moments_owned();
    assert!(invalid_oversized_moments.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    drop(oversized_component_source);
    let oversized_histogram_source = ffi::mat_zeros(4097, 4096, 1);
    let oversized_histogram_source_ref = oversized_histogram_source
        .as_ref()
        .expect("valid matrix above exact histogram limit");
    let invalid_oversized_histogram = oversized_histogram_source_ref.histogram_u8_owned();
    assert!(invalid_oversized_histogram.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    drop(oversized_histogram_source);

    let invalid_bytes = ffi::mat_from_u8(&[1, 2], 2, 2, 3);
    let invalid_long_bytes = ffi::mat_from_u8(&[0; 13], 2, 2, 3);
    let invalid_negative_rows = ffi::mat_from_u8(&[], -1, 1, 3);
    let invalid_overflowing_rows = ffi::mat_from_u8(&[], i64::MAX, 1, 3);
    let invalid_layout = ffi::mat_zeros(2, 2, 99);
    let invalid_allocation = ffi::mat_zeros(16384, 16384, 3);
    let invalid_resize = mat_ref.resize_owned(0, 1, 0);
    let invalid_large_resize = mat_ref.resize_owned(i64::MAX, 1, 0);
    let invalid_color = mat_ref.cvt_color_owned(99);
    let invalid_color_pair = mat_ref.cvt_color_owned(1);
    let invalid_gray_pair = gray_ref.cvt_color_owned(0);
    let invalid_blur = mat_ref.gaussian_blur_owned(2, 3, 0.0);
    let invalid_large_blur = mat_ref.gaussian_blur_owned(257, 3, 0.0);
    let invalid_negative_sigma = mat_ref.gaussian_blur_owned(3, 3, -0.1);
    let invalid_nan_sigma = mat_ref.gaussian_blur_owned(3, 3, f64::NAN);
    let invalid_infinite_sigma = mat_ref.gaussian_blur_owned(3, 3, f64::INFINITY);
    assert!(invalid_bytes.is_null());
    assert!(invalid_long_bytes.is_null());
    assert!(invalid_negative_rows.is_null());
    assert!(invalid_overflowing_rows.is_null());
    assert!(invalid_layout.is_null());
    assert!(invalid_allocation.is_null());
    assert!(invalid_resize.is_null());
    assert!(invalid_large_resize.is_null());
    assert!(invalid_color.is_null());
    assert!(invalid_color_pair.is_null());
    assert!(invalid_gray_pair.is_null());
    assert!(invalid_blur.is_null());
    assert!(invalid_large_blur.is_null());
    assert!(invalid_negative_sigma.is_null());
    assert!(invalid_nan_sigma.is_null());
    assert!(invalid_infinite_sigma.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert_eq!(ffi::take_last_failure(), 0);
    assert_eq!(ffi::live_mat_count(), 25);
    assert_eq!(ffi::live_component_analysis_count(), 2);

    drop(components_eight);
    drop(components_four);
    assert_eq!(ffi::live_component_analysis_count(), 0);
    drop(component_mask);
    drop(histogram);
    drop(equalized);
    drop(perspective);
    drop(affine);
    drop(edges_l2);
    drop(edges);
    drop(edge_source);
    drop(normalized);
    drop(dilated);
    drop(eroded);
    drop(thresholded);
    drop(rotated);
    drop(flipped);
    drop(padded);
    drop(cropped);
    drop(decoded_tiff);
    drop(repeated_tiff);
    drop(tiff);
    drop(decoded_webp);
    drop(repeated_webp);
    drop(webp);
    drop(decoded_jpeg);
    drop(repeated_jpeg);
    drop(jpeg);
    drop(decoded_png);
    drop(png);
    drop(decoded);
    drop(encoded);
    drop(blurred);
    drop(gray);
    drop(resized);
    drop(cloned);
    drop(mat);
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_component_analysis_count(), 0);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
}
