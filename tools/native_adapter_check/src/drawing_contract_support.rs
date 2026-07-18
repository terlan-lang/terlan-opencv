//! Shared native-helper support for immutable drawing contracts.

use super::{expect_typed_handle, int_arg, NativeHandle, NativeHelper};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Invokes one valid drawing operation and returns its fresh matrix owner.
pub(super) fn draw<const N: usize>(
    helper: &mut NativeHelper,
    operation: &str,
    source: &NativeHandle,
    parameters: [i64; N],
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(operation, &draw_arguments(source.arg(), parameters))?,
        MAT_TYPE,
    )
}

/// Builds one native-helper drawing argument vector.
pub(super) fn draw_arguments<const N: usize>(source: String, parameters: [i64; N]) -> Vec<String> {
    let mut arguments = vec![source];
    arguments.extend(parameters.into_iter().map(int_arg));
    arguments
}
