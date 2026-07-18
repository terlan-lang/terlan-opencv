//! Shared native-helper argument and reply contract support.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use super::{NativeHandle, NativeHelper, MAT_TYPE};
use crate::protocol::{NativeRecordValue, NativeReply};

/// Deterministic seven-by-seven grayscale image containing four strong corners.
#[rustfmt::skip]
pub(super) const CORNER_IMAGE: &[u8] = &[
    0, 0,   0,   0,   0, 0, 0,
    0, 0,   0,   0,   0, 0, 0,
    0, 0, 255, 255, 255, 0, 0,
    0, 0, 255, 255, 255, 0, 0,
    0, 0, 255, 255, 255, 0, 0,
    0, 0,   0,   0,   0, 0, 0,
    0, 0,   0,   0,   0, 0, 0,
];

/// Builds a high-contrast checkerboard with interior multi-scale corners.
pub(super) fn checkerboard_fixture() -> Vec<u8> {
    let mut image = vec![0; 160 * 160];
    for row in 0..160 {
        for col in 0..160 {
            if (row / 16 + col / 16) % 2 == 0 {
                image[row * 160 + col] = 255;
            }
        }
    }
    image
}

/// Encodes one signed integer argument.
pub(super) fn int_arg(value: i64) -> String {
    format!("i:{value}")
}

/// Encodes one floating-point argument.
pub(super) fn float_arg(value: f64) -> String {
    format!("f:{value}")
}

/// Encodes one copied floating-point list, preserving the protocol's empty list form.
pub(super) fn float_list_arg(values: &[f64]) -> String {
    if values.is_empty() {
        return "ls:".to_owned();
    }
    format!(
        "lf:{}",
        values
            .iter()
            .map(f64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Encodes one copied signed-integer list, preserving the empty list form.
pub(super) fn int_list_arg(values: &[i64]) -> String {
    if values.is_empty() {
        return "ls:".to_owned();
    }
    format!(
        "li:{}",
        values
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Creates one copied matrix through the generated helper protocol.
pub(super) fn matrix(
    helper: &mut NativeHelper,
    bytes: &[u8],
    rows: i64,
    cols: i64,
    layout: &str,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[
                bytes_arg(bytes),
                int_arg(rows),
                int_arg(cols),
                atom_arg(layout),
            ],
        )?,
        MAT_TYPE,
    )
}

/// Encodes one boolean argument.
pub(super) fn bool_arg(value: bool) -> String {
    format!("b:{value}")
}

/// Encodes one finite symbolic argument.
pub(super) fn atom_arg(value: &str) -> String {
    format!("a:{}", STANDARD.encode(value))
}

/// Encodes one copied byte argument.
pub(super) fn bytes_arg(value: &[u8]) -> String {
    format!("x:{}", STANDARD.encode(value))
}

/// Encodes one copied row-major affine transform argument.
pub(super) fn affine_arg(values: [f64; 6]) -> String {
    record_arg(
        "AffineTransform",
        &["m00", "m01", "m02", "m10", "m11", "m12"],
        &values,
    )
}

/// Encodes one copied row-major perspective transform argument.
pub(super) fn perspective_arg(values: [f64; 9]) -> String {
    record_arg(
        "PerspectiveTransform",
        &[
            "m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22",
        ],
        &values,
    )
}

/// Encodes copied pinhole-camera intrinsic parameters.
pub(super) fn camera_intrinsics_arg(values: [f64; 4]) -> String {
    record_arg("CameraIntrinsics", &["fx", "fy", "cx", "cy"], &values)
}

/// Encodes copied five-coefficient lens distortion parameters.
pub(super) fn distortion_arg(values: [f64; 5]) -> String {
    record_arg(
        "DistortionCoefficients",
        &["k1", "k2", "p1", "p2", "k3"],
        &values,
    )
}

/// Encodes one copied camera pose.
pub(super) fn camera_pose_arg(values: [f64; 6]) -> String {
    record_arg("CameraPose", &["rx", "ry", "rz", "tx", "ty", "tz"], &values)
}

/// Encodes one copied three-dimensional vector.
pub(super) fn vector3_arg(values: [f64; 3]) -> String {
    record_arg("Vector3", &["x", "y", "z"], &values)
}

/// Encodes one copied floating-point value record.
fn record_arg<const N: usize>(type_name: &str, names: &[&str; N], values: &[f64; N]) -> String {
    let fields = names
        .iter()
        .zip(values)
        .map(|(name, value)| format!("{}:f:{value}", STANDARD.encode(name)))
        .collect::<Vec<_>>()
        .join(",");
    format!("r:{}:{fields}", STANDARD.encode(type_name))
}

/// Requires an integer reply with the expected value.
pub(super) fn expect_int(reply: NativeReply, expected: i64) -> Result<(), String> {
    match reply {
        NativeReply::Int(actual) if actual == expected => Ok(()),
        other => Err(format!("expected integer {expected}, received {other:?}")),
    }
}

/// Requires a Boolean reply with the expected value.
pub(super) fn expect_bool(reply: NativeReply, expected: bool) -> Result<(), String> {
    match reply {
        NativeReply::Bool(actual) if actual == expected => Ok(()),
        other => Err(format!("expected Boolean {expected}, received {other:?}")),
    }
}

/// Requires an exact copied byte reply.
pub(super) fn expect_bytes(reply: NativeReply, expected: &[u8]) -> Result<(), String> {
    match reply {
        NativeReply::Bytes(actual) if actual == expected => Ok(()),
        other => Err(format!("expected bytes {expected:?}, received {other:?}")),
    }
}

/// Extracts one copied byte reply for a subsequent native operation.
pub(super) fn expect_bytes_value(reply: NativeReply) -> Result<Vec<u8>, String> {
    match reply {
        NativeReply::Bytes(actual) => Ok(actual),
        other => Err(format!("expected copied bytes, received {other:?}")),
    }
}

/// Extracts one copied integer-list reply for deterministic inspection.
pub(super) fn expect_ints(reply: NativeReply) -> Result<Vec<i64>, String> {
    match reply {
        NativeReply::Ints(actual) => Ok(actual),
        other => Err(format!("expected copied integers, received {other:?}")),
    }
}

/// Extracts one copied floating-point list for deterministic inspection.
pub(super) fn expect_floats(reply: NativeReply) -> Result<Vec<f64>, String> {
    match reply {
        NativeReply::Floats(actual) => Ok(actual),
        other => Err(format!("expected copied floats, received {other:?}")),
    }
}

/// Extracts one copied all-float record in the requested field order.
pub(super) fn expect_float_record(
    reply: NativeReply,
    expected_type: &str,
    field_names: &[&str],
) -> Result<Vec<f64>, String> {
    let NativeReply::Record { type_name, fields } = reply else {
        return Err(format!("expected copied {expected_type} record"));
    };
    if type_name != expected_type || fields.len() != field_names.len() {
        return Err(format!(
            "expected {expected_type} fields {field_names:?}, received {type_name} {fields:?}"
        ));
    }
    field_names
        .iter()
        .map(|name| match fields.get(*name) {
            Some(NativeRecordValue::Float(value)) if value.is_finite() => Ok(*value),
            other => Err(format!("invalid {expected_type}.{name} field: {other:?}")),
        })
        .collect()
}

/// Validates finite bounded feature coordinates in canonical row-major order.
pub(super) fn validate_feature_points(
    family: &str,
    points: &[f64],
    count: usize,
    width: f64,
    height: f64,
) -> Result<(), String> {
    if points.len() != count * 2 {
        return Err(format!("unexpected {family} coordinates {points:?}"));
    }
    let mut previous = None;
    for point in points.chunks_exact(2) {
        let current = (point[1], point[0]);
        if !point[0].is_finite()
            || !point[1].is_finite()
            || !(0.0..width).contains(&point[0])
            || !(0.0..height).contains(&point[1])
            || previous.is_some_and(|value| value > current)
        {
            return Err(format!("invalid {family} coordinates {points:?}"));
        }
        previous = Some(current);
    }
    Ok(())
}

/// Builds isolated high-contrast points that satisfy FAST-family arc tests.
pub(super) fn detector_fixture() -> Vec<u8> {
    let mut image = vec![0; 32 * 32];
    for (row, column) in [(8, 8), (8, 23), (23, 8), (23, 23)] {
        image[row * 32 + column] = 255;
    }
    image
}

/// Validates bounded unique detector coordinates in canonical row-major order.
pub(super) fn validate_detector_points(
    family: &str,
    points: &[f64],
    minimum: usize,
    maximum: usize,
    width: f64,
    height: f64,
) -> Result<(), String> {
    if points.len() & 1 != 0 || points.len() < minimum * 2 || points.len() > maximum * 2 {
        return Err(format!("unexpected {family} point count {points:?}"));
    }
    let mut previous = None;
    for point in points.chunks_exact(2) {
        let current = (point[1], point[0]);
        if !point[0].is_finite()
            || !point[1].is_finite()
            || !(0.0..width).contains(&point[0])
            || !(0.0..height).contains(&point[1])
            || previous.is_some_and(|value| value >= current)
        {
            return Err(format!("invalid {family} coordinates {points:?}"));
        }
        previous = Some(current);
    }
    Ok(())
}

/// Extracts and validates one matrix handle reply.
pub(super) fn expect_handle(reply: NativeReply) -> Result<NativeHandle, String> {
    expect_typed_handle(reply, MAT_TYPE)
}

/// Extracts and validates one typed native resource handle reply.
pub(super) fn expect_typed_handle(
    reply: NativeReply,
    expected_type: &str,
) -> Result<NativeHandle, String> {
    match reply {
        NativeReply::Handle {
            owner,
            id,
            generation,
            type_name,
        } if type_name == expected_type => Ok(NativeHandle {
            owner,
            id,
            generation,
            type_name,
        }),
        other => Err(format!(
            "expected {expected_type} handle, received {other:?}"
        )),
    }
}

/// Requires a successful unit reply.
pub(super) fn expect_unit(reply: NativeReply) -> Result<(), String> {
    match reply {
        NativeReply::Unit => Ok(()),
        other => Err(format!("expected unit, received {other:?}")),
    }
}

/// Requires one stable helper error code.
pub(super) fn expect_error_code(reply: NativeReply, expected: &str) -> Result<(), String> {
    match reply {
        NativeReply::Error { code, .. } if code == expected => Ok(()),
        other => Err(format!("expected error {expected}, received {other:?}")),
    }
}
