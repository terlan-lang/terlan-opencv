//! OpenCV adapter metadata reproduction command.

#![forbid(unsafe_code)]

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use terlan_opencv_metadata_check::{package_root, validate_live, validate_offline};

/// Runs offline validation or live LibTooling reproduction.
fn main() -> ExitCode {
    let root = package_root();
    let result = match env::args_os().nth(1) {
        Some(extractor) => validate_live(&root, PathBuf::from(extractor).as_path()),
        None => validate_offline(&root),
    };
    match result {
        Ok(()) => {
            println!("opencv-metadata-check: adapter metadata contract verified");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("opencv-metadata-check: {error}");
            ExitCode::FAILURE
        }
    }
}
