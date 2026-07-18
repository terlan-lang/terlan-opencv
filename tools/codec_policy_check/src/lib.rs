//! Offline validation for the package-owned OpenCV codec policy.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const POLICY_SCHEMA: &str = "terlan.opencv.codec-policy.v1";
const MAX_ENCODED_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DECODED_BYTES: u64 = 256 * 1024 * 1024;

/// Validates the committed codec policy and every package integration point.
pub fn validate_package(root: &Path) -> Result<(), String> {
    let policy = read_toml(&root.join("bindings/opencv.codecs.toml"))?;
    let lock = read_toml(&root.join("upstream/opencv.lock.toml"))?;
    let package = read_toml(&root.join("terlan.toml"))?;
    let manifest = read_json(&root.join("bindings/opencv.native-binding.json"))?;
    let cmake = read_text(&root.join("cmake/opencv-cpu.cmake"))?;
    let probe = read_text(&root.join("tests/cpp/opencv_link_probe.cc"))?;
    validate_policy(&policy)?;
    validate_module_scope(&lock, &package, &manifest, &cmake)?;
    validate_public_surface(&manifest)?;
    validate_link_probe(&probe)
}

/// Reads one UTF-8 package input with a path-qualified error.
fn read_text(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))
}

/// Reads one TOML package input through a structured parser.
fn read_toml(path: &Path) -> Result<Value, String> {
    let text = read_text(path)?;
    basic_toml::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

/// Reads one JSON package input through a structured parser.
fn read_json(path: &Path) -> Result<Value, String> {
    let text = read_text(path)?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

/// Returns success when a policy condition holds.
fn require(condition: bool, message: impl Into<String>) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

/// Obtains one required object member.
fn object<'a>(value: &'a Value, key: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{key} must be an object"))
}

/// Obtains one required array member.
fn array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array"))
}

/// Obtains one required non-empty string member.
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

/// Obtains one required unsigned integer member.
fn unsigned(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{key} must be an unsigned integer"))
}

/// Converts one string array into an ordered set.
fn string_set(value: &Value, key: &str) -> Result<BTreeSet<String>, String> {
    array(value, key)?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| format!("{key} entries must be non-empty strings"))
        })
        .collect()
}

/// Parses the package's quoted comma-separated CMake `BUILD_LIST` assignment.
fn cmake_build_modules(cmake: &str) -> Result<BTreeSet<String>, String> {
    let assignment = cmake
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("set(BUILD_LIST"))
        .ok_or_else(|| "CMake BUILD_LIST assignment is missing".to_string())?;
    let (_, quoted) = assignment
        .split_once('"')
        .ok_or_else(|| "CMake BUILD_LIST must use a quoted module list".to_string())?;
    let (modules, _) = quoted
        .split_once('"')
        .ok_or_else(|| "CMake BUILD_LIST has an unterminated module list".to_string())?;
    let parsed = modules
        .split(',')
        .map(str::trim)
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    require(
        !parsed.contains("") && !parsed.is_empty(),
        "CMake BUILD_LIST contains an empty module",
    )?;
    Ok(parsed)
}

/// Validates byte ownership, finite formats, finite modes, and stable errors.
fn validate_policy(policy: &Value) -> Result<(), String> {
    require(
        string(policy, "schema")? == POLICY_SCHEMA,
        "invalid codec policy schema",
    )?;
    require(
        string(policy, "module")? == "imgcodecs",
        "codec module must be imgcodecs",
    )?;
    require(
        string(policy, "public_module")? == "opencv.Codecs",
        "codec public module must be opencv.Codecs",
    )?;
    for key in ["input_boundary", "output_boundary"] {
        require(
            string(policy, key)? == "copied_bytes",
            format!("{key} must remain copied_bytes"),
        )?;
    }
    require(
        string(policy, "filesystem_policy")? == "forbidden",
        "ambient filesystem access must remain forbidden",
    )?;
    require(
        string(policy, "availability_source")? == "pinned_build_and_link_probe",
        "codec availability must come from the pinned build and link probe",
    )?;
    for key in ["unavailable_error", "invalid_data_error"] {
        require(
            string(policy, key)?.starts_with("opencv."),
            format!("{key} must use the opencv.* error family"),
        )?;
    }
    require(
        unsigned(policy, "max_encoded_bytes")? == MAX_ENCODED_BYTES,
        "encoded byte limit drift",
    )?;
    require(
        unsigned(policy, "max_decoded_bytes")? == MAX_DECODED_BYTES,
        "decoded byte limit drift",
    )?;
    require(
        string_set(policy, "public_functions")?
            == BTreeSet::from(["decode".into(), "encode".into()]),
        "codec API must contain only decode and encode",
    )?;
    require(
        string_set(policy, "forbidden_functions")?
            == BTreeSet::from(["imread".into(), "imwrite".into()]),
        "filesystem rejection inventory must contain imread and imwrite",
    )?;
    validate_formats(array(policy, "formats")?)?;
    validate_decode_modes(array(policy, "decode_modes")?)
}

/// Requires the first codec surface to be one guaranteed BMP byte codec.
fn validate_formats(formats: &[Value]) -> Result<(), String> {
    require(
        formats.len() == 1,
        "first codec surface must contain exactly BMP",
    )?;
    let format = &formats[0];
    require(string(format, "name")? == "bmp", "first codec must be BMP")?;
    require(
        string(format, "extension")? == ".bmp",
        "BMP extension must be .bmp",
    )?;
    require(
        format.get("decode") == Some(&Value::Bool(true)),
        "BMP decoding must be enabled",
    )?;
    require(
        format.get("encode") == Some(&Value::Bool(true)),
        "BMP encoding must be enabled",
    )?;
    require(
        string(format, "availability")? == "required",
        "BMP availability must be a build requirement",
    )
}

/// Requires a finite, duplicate-free set of reviewed OpenCV decode modes.
fn validate_decode_modes(modes: &[Value]) -> Result<(), String> {
    require(
        modes.len() == 3,
        "decode modes must contain exactly unchanged, grayscale, and color",
    )?;
    let observed = modes
        .iter()
        .map(|mode| {
            Ok((
                string(mode, "name")?.to_owned(),
                mode.get("value").and_then(Value::as_i64),
            ))
        })
        .collect::<Result<BTreeSet<_>, String>>()?;
    let expected = BTreeSet::from([
        ("color".to_string(), Some(1)),
        ("grayscale".to_string(), Some(0)),
        ("unchanged".to_string(), Some(-1)),
    ]);
    require(
        observed == expected,
        "decode modes must be unchanged, grayscale, and color",
    )
}

/// Validates that every package build description carries `imgcodecs`.
fn validate_module_scope(
    lock: &Value,
    package: &Value,
    manifest: &Value,
    cmake: &str,
) -> Result<(), String> {
    let lock_build = Value::Object(object(lock, "build")?.clone());
    require(
        string_set(&lock_build, "modules")?.contains("imgcodecs"),
        "upstream lock omits imgcodecs",
    )?;
    let native = Value::Object(object(package, "native")?.clone());
    let cpp = Value::Object(object(&native, "cpp")?.clone());
    require(
        string_set(&cpp, "modules")?.contains("imgcodecs"),
        "package metadata omits imgcodecs",
    )?;
    let build = Value::Object(object(manifest, "build")?.clone());
    let linked = array(&build, "linked_libraries")?
        .iter()
        .filter_map(|library| library.get("name").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    require(
        linked.contains("opencv_imgcodecs"),
        "native build plan omits opencv_imgcodecs",
    )?;
    require(
        cmake_build_modules(cmake)?.contains("imgcodecs"),
        "CMake build scope omits imgcodecs",
    )
}

/// Rejects filesystem call names and path-shaped public codec arguments.
fn validate_public_surface(manifest: &Value) -> Result<(), String> {
    let mut codec_module = None;
    for module in array(manifest, "modules")? {
        if string(module, "module")? == "opencv.Codecs" {
            require(codec_module.is_none(), "duplicate opencv.Codecs module")?;
            codec_module = Some(module);
        }
        for function in module
            .get("functions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let name = string(function, "name")?;
            require(
                !matches!(name, "imread" | "imwrite"),
                "filesystem codec function is forbidden",
            )?;
            for argument in function
                .get("args")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let argument_name = string(argument, "name")?;
                require(
                    !matches!(argument_name, "path" | "filename" | "file"),
                    "path-shaped codec argument is forbidden",
                )?;
            }
        }
    }
    validate_codec_module(
        codec_module.ok_or_else(|| "opencv.Codecs module is missing".to_string())?,
    )
}

/// Requires the exact copied-byte decode and encode public signatures.
fn validate_codec_module(module: &Value) -> Result<(), String> {
    let functions = array(module, "functions")?;
    require(
        functions.len() == 2,
        "opencv.Codecs must contain exactly two functions",
    )?;
    let decode = functions
        .iter()
        .find(|function| function.get("name").and_then(Value::as_str) == Some("decode"))
        .ok_or_else(|| "opencv.Codecs.decode is missing".to_string())?;
    let encode = functions
        .iter()
        .find(|function| function.get("name").and_then(Value::as_str) == Some("encode"))
        .ok_or_else(|| "opencv.Codecs.encode is missing".to_string())?;
    validate_signature(
        decode,
        "opencv.codecs.decode",
        &[
            ("bytes", "std.vm.Bytes.Bytes"),
            ("mode", "opencv.Types.DecodeMode"),
        ],
        "opencv.Mat.Mat",
    )?;
    validate_signature(
        encode,
        "opencv.codecs.encode",
        &[
            ("mat", "opencv.Mat.Mat"),
            ("format", "opencv.Types.ImageFormat"),
        ],
        "std.vm.Bytes.Bytes",
    )
}

/// Validates one generated operation identity and ordered public type list.
fn validate_signature(
    function: &Value,
    operation: &str,
    expected_args: &[(&str, &str)],
    expected_return: &str,
) -> Result<(), String> {
    require(
        string(function, "operation")? == operation,
        format!("{operation} operation identity drift"),
    )?;
    let args = array(function, "args")?;
    require(
        args.len() == expected_args.len(),
        format!("{operation} argument count drift"),
    )?;
    for (argument, (name, ty)) in args.iter().zip(expected_args) {
        require(
            string(argument, "name")? == *name && string(argument, "ty")? == *ty,
            format!("{operation} argument contract drift"),
        )?;
    }
    require(
        string(function, "returns")? == expected_return,
        format!("{operation} result contract drift"),
    )?;
    require(
        string(function, "blocking")? == "blocking",
        format!("{operation} must remain a blocking native operation"),
    )
}

/// Requires the native probe to link both byte codec directions without files.
fn validate_link_probe(probe: &str) -> Result<(), String> {
    for required in [
        "opencv2/imgcodecs.hpp",
        "cv::imdecode",
        "cv::imencode",
        "\".bmp\"",
    ] {
        require(
            probe.contains(required),
            format!("link probe omits {required}"),
        )?;
    }
    for forbidden in ["cv::imread", "cv::imwrite"] {
        require(
            !probe.contains(forbidden),
            format!("link probe exposes {forbidden}"),
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "lib_test.rs"]
mod tests;
