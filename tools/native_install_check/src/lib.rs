//! Validation for a prepared native OpenCV installation.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Prepared include and library roots required by the package adapter.
#[derive(Debug, Eq, PartialEq)]
pub struct InstallLayout {
    /// Directory containing the `opencv2` include tree.
    pub include_dir: PathBuf,
    /// Directory containing the selected OpenCV libraries.
    pub library_dir: PathBuf,
    /// Exact OpenCV core library selected for the link probe.
    pub core_library: PathBuf,
    /// Exact OpenCV image-processing library selected for the link probe.
    pub imgproc_library: PathBuf,
    /// Exact OpenCV image-codec library selected for the link probe.
    pub imgcodecs_library: PathBuf,
    /// Exact OpenCV feature-detection library selected for the link probe.
    pub features2d_library: PathBuf,
    /// Exact OpenCV camera-calibration library selected for the link probe.
    pub calib3d_library: PathBuf,
    /// Exact OpenCV machine-learning library selected for the link probe.
    pub ml_library: PathBuf,
    /// Exact OpenCV deep-neural-network library selected for the link probe.
    pub dnn_library: PathBuf,
}

/// Validates required headers and libraries without invoking a compiler.
pub fn validate_layout(install_root: &Path) -> Result<InstallLayout, String> {
    let include_dir = install_root.join("include/opencv4");
    for header in [
        "opencv2/core.hpp",
        "opencv2/imgproc.hpp",
        "opencv2/imgcodecs.hpp",
        "opencv2/features2d.hpp",
        "opencv2/calib3d.hpp",
        "opencv2/ml.hpp",
        "opencv2/dnn.hpp",
    ] {
        let path = include_dir.join(header);
        if !path.is_file() {
            return Err(format!(
                "error[opencv.header_missing]: required OpenCV header `{header}` is missing"
            ));
        }
    }

    let library_dir = install_root.join("lib");
    if !library_dir.is_dir() {
        return Err(
            "error[opencv.library_missing]: OpenCV library directory is missing".to_string(),
        );
    }
    let core_library = required_library(&library_dir, "opencv_core")?;
    let imgproc_library = required_library(&library_dir, "opencv_imgproc")?;
    let imgcodecs_library = required_library(&library_dir, "opencv_imgcodecs")?;
    let features2d_library = required_library(&library_dir, "opencv_features2d")?;
    let calib3d_library = required_library(&library_dir, "opencv_calib3d")?;
    let ml_library = required_library(&library_dir, "opencv_ml")?;
    let dnn_library = required_library(&library_dir, "opencv_dnn")?;
    Ok(InstallLayout {
        include_dir,
        library_dir,
        core_library,
        imgproc_library,
        imgcodecs_library,
        features2d_library,
        calib3d_library,
        ml_library,
        dnn_library,
    })
}

/// Compiles and links the package probe against one prepared installation.
pub fn validate_install(
    install_root: &Path,
    probe_source: &Path,
    compiler: &Path,
) -> Result<InstallLayout, String> {
    validate_install_with(install_root, probe_source, compiler, run_compiler)
}

/// One compiler phase used to classify deterministic diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbePhase {
    Compile,
    Link,
}

/// Result of invoking the configured compiler process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompilerResult {
    Success,
    Unavailable,
    Failed,
}

/// Validates an installation through an injected compiler runner.
fn validate_install_with<Runner>(
    install_root: &Path,
    probe_source: &Path,
    compiler: &Path,
    mut runner: Runner,
) -> Result<InstallLayout, String>
where
    Runner: FnMut(ProbePhase, &Path, &[OsString]) -> CompilerResult,
{
    let layout = validate_layout(install_root)?;
    if !probe_source.is_file() {
        return Err("error[opencv.probe_missing]: native OpenCV link probe is missing".to_string());
    }
    let scratch = scratch_dir();
    fs::create_dir_all(&scratch)
        .map_err(|_| "error[opencv.probe_io]: cannot create probe workspace".to_string())?;
    let object = scratch.join("opencv_link_probe.o");
    let executable = scratch.join("opencv_link_probe");
    let compile_args = vec![
        OsString::from("-std=c++14"),
        OsString::from("-Werror"),
        OsString::from("-I"),
        layout.include_dir.as_os_str().to_owned(),
        OsString::from("-c"),
        probe_source.as_os_str().to_owned(),
        OsString::from("-o"),
        object.as_os_str().to_owned(),
    ];
    match runner(ProbePhase::Compile, compiler, &compile_args) {
        CompilerResult::Success => {}
        CompilerResult::Unavailable => {
            remove_scratch(&scratch);
            return Err(
                "error[opencv.compiler_unavailable]: configured C++ compiler is unavailable"
                    .to_string(),
            );
        }
        CompilerResult::Failed => {
            remove_scratch(&scratch);
            return Err(
                "error[opencv.header_incompatible]: OpenCV headers do not compile against the package probe"
                    .to_string(),
            );
        }
    }

    let link_args = vec![
        object.as_os_str().to_owned(),
        layout.dnn_library.as_os_str().to_owned(),
        layout.ml_library.as_os_str().to_owned(),
        layout.calib3d_library.as_os_str().to_owned(),
        layout.features2d_library.as_os_str().to_owned(),
        layout.imgcodecs_library.as_os_str().to_owned(),
        layout.imgproc_library.as_os_str().to_owned(),
        layout.core_library.as_os_str().to_owned(),
        OsString::from("-o"),
        executable.as_os_str().to_owned(),
    ];
    let result = match runner(ProbePhase::Link, compiler, &link_args) {
        CompilerResult::Success => Ok(layout),
        CompilerResult::Unavailable => Err(
            "error[opencv.compiler_unavailable]: configured C++ compiler is unavailable".to_string(),
        ),
        CompilerResult::Failed => Err(
            "error[opencv.symbol_missing]: OpenCV libraries do not provide the required package symbols"
                .to_string(),
        ),
    };
    remove_scratch(&scratch);
    result
}

/// Invokes one compiler phase without exposing compiler output as package errors.
fn run_compiler(_phase: ProbePhase, compiler: &Path, args: &[OsString]) -> CompilerResult {
    match Command::new(compiler).args(args).output() {
        Ok(output) if output.status.success() => CompilerResult::Success,
        Ok(_) => CompilerResult::Failed,
        Err(_) => CompilerResult::Unavailable,
    }
}

/// Requires one exact native component library from the prepared install.
fn required_library(directory: &Path, component: &str) -> Result<PathBuf, String> {
    find_library(directory, component)?.ok_or_else(|| {
        format!("error[opencv.library_missing]: required OpenCV library `{component}` is missing")
    })
}

/// Locates one static, shared, or import library by reviewed filename patterns.
fn find_library(directory: &Path, component: &str) -> Result<Option<PathBuf>, String> {
    let entries = fs::read_dir(directory).map_err(|_| {
        "error[opencv.library_missing]: cannot inspect OpenCV libraries".to_string()
    })?;
    let mut matches = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter_map(|path| {
            let rank = path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| library_rank(name, component))?;
            Some((rank, path))
        })
        .collect::<Vec<_>>();
    matches.sort();
    Ok(matches.into_iter().next().map(|(_, path)| path))
}

/// Ranks supported names with shared libraries before static archives.
fn library_rank(name: &str, component: &str) -> Option<u8> {
    if name == format!("lib{component}.so") || name == format!("lib{component}.dylib") {
        return Some(0);
    }
    if let Some(version) = name.strip_prefix(&format!("lib{component}.so.")) {
        if !version.is_empty()
            && version
                .chars()
                .all(|character| character.is_ascii_digit() || character == '.')
            && version.split('.').all(|segment| !segment.is_empty())
        {
            return Some(1);
        }
    }
    if name == format!("lib{component}.a") || name == format!("{component}.lib") {
        return Some(2);
    }
    None
}

/// Creates a collision-resistant process-local probe directory path.
fn scratch_dir() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!(
        "terlan-opencv-install-check-{}-{nonce}",
        std::process::id()
    ))
}

/// Removes a completed probe workspace without masking its validation result.
fn remove_scratch(path: &Path) {
    let _ = fs::remove_dir_all(path);
}

#[cfg(test)]
#[path = "lib_test.rs"]
mod tests;
