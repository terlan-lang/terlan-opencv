//! Owned maximally stable extremal-region contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_ints, expect_typed_handle, expect_unit,
    int_arg, matrix, NativeHandle, NativeHelper,
};

const MSER_TYPE: &str = "opencv.Features.MserRegions";

/// Complete copied MSER count, points, offsets, and bounds projection.
type MserProjection = (i64, Vec<i64>, Vec<i64>, Vec<i64>);

/// Runs extraction, canonical projection, ownership, and rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.features.live_mser_count", &[])?, 0)?;
    let fixture = nested_regions_fixture();
    let image = matrix(helper, &fixture, 160, 160, "gray")?;
    let first = detect(helper, &image, 64)?;
    let second = detect(helper, &image, 64)?;
    expect_int(helper.call("opencv.features.live_mser_count", &[])?, 2)?;
    let first_projection = projection(helper, &first)?;
    validate_projection(&first_projection, 64)?;
    let repeated = projection(helper, &second)?;
    if repeated != first_projection {
        return Err("repeated MSER extraction was not deterministic".to_owned());
    }
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, &fixture)?;

    let truncated = detect(helper, &image, 1)?;
    let truncated_projection = projection(helper, &truncated)?;
    if truncated_projection.0 != 1 {
        return Err(format!(
            "expected one truncated MSER region, received {}",
            truncated_projection.0
        ));
    }
    validate_projection(&truncated_projection, 1)?;

    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)?;
    if projection(helper, &first)? != first_projection {
        return Err("MSER owner retained a source-image borrow".to_owned());
    }

    let empty_image = matrix(helper, &[0; 160 * 160], 160, 160, "gray")?;
    let empty = detect(helper, &empty_image, 8)?;
    let empty_projection = projection(helper, &empty)?;
    if empty_projection != (0, Vec::new(), vec![0], Vec::new()) {
        return Err("uniform image produced MSER regions".to_owned());
    }

    for arguments in [
        [0, 16, 4096, 8],
        [256, 16, 4096, 8],
        [5, 0, 4096, 8],
        [5, 17, 16, 8],
        [5, 16, 25601, 8],
        [5, 16, 4096, 0],
        [5, 16, 4096, 4097],
        [5, 16, 4096, 257],
    ] {
        expect_error_code(
            helper.call(
                "opencv.features.detect_mser",
                &[
                    empty_image.arg(),
                    int_arg(arguments[0]),
                    int_arg(arguments[1]),
                    int_arg(arguments[2]),
                    int_arg(arguments[3]),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.features.detect_mser",
            &[
                empty_image.arg_with_owner("foreign-worker"),
                int_arg(5),
                int_arg(16),
                int_arg(4096),
                int_arg(8),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call("opencv.features.mser_count", &[empty_image.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.mat.rows", &[empty.arg()])?,
        "handle_type_mismatch",
    )?;

    let color = matrix(helper, &[0; 3 * 3 * 3], 3, 3, "bgr")?;
    let too_small = matrix(helper, &[0; 2 * 2], 2, 2, "gray")?;
    let oversized = matrix(helper, &[0; 1025 * 1024], 1025, 1024, "gray")?;
    for source in [&color, &too_small, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.features.detect_mser",
                &[source.arg(), int_arg(5), int_arg(1), int_arg(4), int_arg(1)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    for regions in [&first, &second, &truncated, &empty] {
        expect_unit(helper.call("opencv.features.dispose_mser", &[regions.arg()])?)?;
        expect_error_code(
            helper.call("opencv.features.mser_count", &[regions.arg()])?,
            "stale_handle",
        )?;
    }
    for source in [&oversized, &too_small, &color, &empty_image] {
        expect_unit(helper.call("opencv.mat.dispose", &[source.arg()])?)?;
    }
    expect_int(helper.call("opencv.features.live_mser_count", &[])?, 0)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Creates nested stable grayscale regions with multiple intensity plateaus.
fn nested_regions_fixture() -> Vec<u8> {
    let mut pixels = vec![0; 160 * 160];
    for (start, end, value) in [(20, 140, 64), (40, 120, 128), (60, 100, 255)] {
        for y in start..end {
            for x in start..end {
                pixels[y * 160 + x] = value;
            }
        }
    }
    pixels
}

/// Detects one fixed-policy MSER resource.
fn detect(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.features.detect_mser",
            &[
                image.arg(),
                int_arg(5),
                int_arg(16),
                int_arg(4096),
                int_arg(maximum),
            ],
        )?,
        MSER_TYPE,
    )
}

/// Reads one complete copied MSER projection.
fn projection(helper: &mut NativeHelper, regions: &NativeHandle) -> Result<MserProjection, String> {
    let count = match helper.call("opencv.features.mser_count", &[regions.arg()])? {
        crate::protocol::NativeReply::Int(value) => value,
        other => return Err(format!("expected MSER count, received {other:?}")),
    };
    Ok((
        count,
        expect_ints(helper.call("opencv.features.mser_points", &[regions.arg()])?)?,
        expect_ints(helper.call("opencv.features.mser_offsets", &[regions.arg()])?)?,
        expect_ints(helper.call("opencv.features.mser_bounds", &[regions.arg()])?)?,
    ))
}

/// Validates aligned canonical MSER points, offsets, and exact bounds.
fn validate_projection(projection: &MserProjection, maximum: i64) -> Result<(), String> {
    let (count, points, offsets, bounds) = projection;
    if !(1..=maximum).contains(count)
        || offsets.len() != *count as usize + 1
        || offsets.first() != Some(&0)
        || bounds.len() != *count as usize * 4
        || points.len() != *offsets.last().unwrap_or(&-1) as usize * 2
    {
        return Err("MSER projections are not row aligned".to_owned());
    }
    let mut previous_region: Option<(i64, i64, i64, i64, i64)> = None;
    for item in 0..*count as usize {
        let start = offsets[item] as usize;
        let end = offsets[item + 1] as usize;
        let bound = &bounds[item * 4..item * 4 + 4];
        let size = (end - start) as i64;
        let order = (bound[1], bound[0], bound[3], bound[2], size);
        if !(16..=4096).contains(&size)
            || bound[0] < 0
            || bound[1] < 0
            || bound[2] <= 0
            || bound[3] <= 0
            || previous_region.is_some_and(|previous| order < previous)
        {
            return Err(format!("invalid MSER region {item}"));
        }
        let mut minimum_x = 160;
        let mut minimum_y = 160;
        let mut maximum_x = -1;
        let mut maximum_y = -1;
        let mut previous_point: Option<(i64, i64)> = None;
        for point in start..end {
            let x = points[point * 2];
            let y = points[point * 2 + 1];
            if !(0..160).contains(&x)
                || !(0..160).contains(&y)
                || previous_point.is_some_and(|previous| (y, x) <= previous)
            {
                return Err(format!("invalid MSER point in region {item}"));
            }
            minimum_x = minimum_x.min(x);
            minimum_y = minimum_y.min(y);
            maximum_x = maximum_x.max(x);
            maximum_y = maximum_y.max(y);
            previous_point = Some((y, x));
        }
        if [
            minimum_x,
            minimum_y,
            maximum_x - minimum_x + 1,
            maximum_y - minimum_y + 1,
        ] != bound
        {
            return Err(format!("MSER bounds do not cover region {item}"));
        }
        previous_region = Some(order);
    }
    Ok(())
}
