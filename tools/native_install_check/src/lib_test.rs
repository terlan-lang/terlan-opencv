//! Tests for native OpenCV installation validation.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use super::*;

/// Creates an isolated complete fake installation layout and probe source.
fn fixture(name: &str) -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "terlan_opencv_install_{name}_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("include/opencv4/opencv2")).expect("create include fixture");
    fs::create_dir_all(root.join("lib")).expect("create library fixture");
    for header in [
        "core.hpp",
        "imgproc.hpp",
        "imgcodecs.hpp",
        "features2d.hpp",
        "calib3d.hpp",
        "ml.hpp",
        "dnn.hpp",
    ] {
        fs::write(root.join("include/opencv4/opencv2").join(header), "")
            .expect("write header fixture");
    }
    for library in [
        "libopencv_core.so",
        "libopencv_imgproc.so.4.13",
        "libopencv_imgcodecs.so",
        "libopencv_features2d.so",
        "libopencv_calib3d.so",
        "libopencv_ml.so",
        "libopencv_dnn.so",
    ] {
        fs::write(root.join("lib").join(library), "").expect("write library fixture");
    }
    let probe = root.join("probe.cc");
    fs::write(&probe, "int main() { return 0; }\n").expect("write probe fixture");
    (root, probe)
}

/// Requires one deterministic checker diagnostic.
fn expect_error(result: Result<InstallLayout, String>, expected: &str) {
    let error = result.expect_err("invalid install must fail");
    assert!(error.contains(expected), "unexpected error: {error}");
}

/// Accepts complete shared-library layouts including versioned Linux names.
#[test]
fn complete_layout_is_accepted() {
    let (root, _) = fixture("complete");
    let layout = validate_layout(&root).expect("complete layout");
    assert_eq!(layout.include_dir, root.join("include/opencv4"));
    assert_eq!(layout.library_dir, root.join("lib"));
    assert_eq!(layout.core_library, root.join("lib/libopencv_core.so"));
    assert_eq!(
        layout.imgproc_library,
        root.join("lib/libopencv_imgproc.so.4.13")
    );
    assert_eq!(
        layout.imgcodecs_library,
        root.join("lib/libopencv_imgcodecs.so")
    );
    assert_eq!(
        layout.features2d_library,
        root.join("lib/libopencv_features2d.so")
    );
    assert_eq!(
        layout.calib3d_library,
        root.join("lib/libopencv_calib3d.so")
    );
    assert_eq!(layout.ml_library, root.join("lib/libopencv_ml.so"));
    assert_eq!(layout.dnn_library, root.join("lib/libopencv_dnn.so"));
    fs::remove_dir_all(root).expect("remove fixture");
}

/// Rejects each missing required header before compiler execution.
#[test]
fn missing_headers_are_rejected() {
    for header in [
        "core.hpp",
        "imgproc.hpp",
        "imgcodecs.hpp",
        "features2d.hpp",
        "calib3d.hpp",
        "ml.hpp",
        "dnn.hpp",
    ] {
        let (root, _) = fixture(header);
        fs::remove_file(root.join("include/opencv4/opencv2").join(header)).expect("remove header");
        expect_error(validate_layout(&root), "opencv.header_missing");
        fs::remove_dir_all(root).expect("remove fixture");
    }
}

/// Rejects absent library directories and each required component.
#[test]
fn missing_libraries_are_rejected() {
    let (root, _) = fixture("library_directory");
    fs::remove_dir_all(root.join("lib")).expect("remove library directory");
    expect_error(validate_layout(&root), "opencv.library_missing");
    fs::remove_dir_all(root).expect("remove fixture");

    for library in [
        "libopencv_core.so",
        "libopencv_imgproc.so.4.13",
        "libopencv_imgcodecs.so",
        "libopencv_features2d.so",
        "libopencv_calib3d.so",
        "libopencv_ml.so",
        "libopencv_dnn.so",
    ] {
        let (root, _) = fixture(library);
        fs::remove_file(root.join("lib").join(library)).expect("remove library");
        expect_error(validate_layout(&root), "opencv.library_missing");
        fs::remove_dir_all(root).expect("remove fixture");
    }
}

/// Rejects missing probes before invoking the configured compiler.
#[test]
fn missing_probe_is_rejected() {
    let (root, probe) = fixture("missing_probe");
    fs::remove_file(&probe).expect("remove probe");
    expect_error(
        validate_install_with(&root, &probe, Path::new("c++"), |_, _, _| {
            panic!("compiler must not run")
        }),
        "opencv.probe_missing",
    );
    fs::remove_dir_all(root).expect("remove fixture");
}

/// Classifies unavailable compilers and compile incompatibility independently.
#[test]
fn compiler_and_header_failures_are_distinct() {
    for (name, result, expected) in [
        (
            "compiler_unavailable",
            CompilerResult::Unavailable,
            "opencv.compiler_unavailable",
        ),
        (
            "header_incompatible",
            CompilerResult::Failed,
            "opencv.header_incompatible",
        ),
    ] {
        let (root, probe) = fixture(name);
        expect_error(
            validate_install_with(&root, &probe, Path::new("c++"), |phase, _, _| {
                assert_eq!(phase, ProbePhase::Compile);
                result
            }),
            expected,
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }
}

/// Classifies missing link symbols after a successful compile phase.
#[test]
fn missing_symbols_are_rejected_after_compile() {
    let (root, probe) = fixture("missing_symbols");
    let mut phases = Vec::new();
    expect_error(
        validate_install_with(&root, &probe, Path::new("c++"), |phase, _, _| {
            phases.push(phase);
            match phase {
                ProbePhase::Compile => CompilerResult::Success,
                ProbePhase::Link => CompilerResult::Failed,
            }
        }),
        "opencv.symbol_missing",
    );
    assert_eq!(phases, [ProbePhase::Compile, ProbePhase::Link]);
    fs::remove_dir_all(root).expect("remove fixture");
}

/// Emits reviewed C++14 compile and component link arguments in stable order.
#[test]
fn successful_probe_uses_reviewed_compile_and_link_arguments() {
    let (root, probe) = fixture("arguments");
    let mut calls = Vec::<(ProbePhase, Vec<OsString>)>::new();
    let layout = validate_install_with(&root, &probe, Path::new("c++"), |phase, _, args| {
        calls.push((phase, args.to_vec()));
        CompilerResult::Success
    })
    .expect("successful probe");
    assert_eq!(layout.library_dir, root.join("lib"));
    assert_eq!(calls.len(), 2);
    assert!(calls[0].1.contains(&OsString::from("-std=c++14")));
    assert!(calls[0].1.contains(&OsString::from("-c")));
    assert!(calls[1]
        .1
        .contains(&root.join("lib/libopencv_calib3d.so").into_os_string()));
    assert!(calls[1]
        .1
        .contains(&root.join("lib/libopencv_ml.so").into_os_string()));
    assert!(calls[1]
        .1
        .contains(&root.join("lib/libopencv_dnn.so").into_os_string()));
    assert!(calls[1]
        .1
        .contains(&root.join("lib/libopencv_features2d.so").into_os_string()));
    assert!(calls[1]
        .1
        .contains(&root.join("lib/libopencv_imgcodecs.so").into_os_string()));
    assert!(calls[1]
        .1
        .contains(&root.join("lib/libopencv_imgproc.so.4.13").into_os_string()));
    assert!(calls[1]
        .1
        .contains(&root.join("lib/libopencv_core.so").into_os_string()));
    fs::remove_dir_all(root).expect("remove fixture");
}

/// Rejects unrelated filenames that merely contain a component name.
#[test]
fn unrelated_library_names_are_not_accepted() {
    assert!(library_rank("libopencv_core_backup.so", "opencv_core").is_none());
    assert!(library_rank("other_opencv_core.lib", "opencv_core").is_none());
    assert!(library_rank("libopencv_core.so.backup", "opencv_core").is_none());
    assert!(library_rank("libopencv_core.so.4..13", "opencv_core").is_none());
    assert!(library_rank("libopencv_core.a", "opencv_core").is_some());
    assert!(library_rank("libopencv_core.so.4.13.0", "opencv_core").is_some());
}

/// Prefers a shared library when static and dynamic artifacts coexist.
#[test]
fn shared_libraries_are_selected_before_static_archives() {
    let (root, _) = fixture("shared_preference");
    fs::write(root.join("lib/libopencv_core.a"), "").expect("write static library");
    let layout = validate_layout(&root).expect("mixed layout");
    assert_eq!(layout.core_library, root.join("lib/libopencv_core.so"));
    fs::remove_dir_all(root).expect("remove fixture");
}
