//! OpenCV adapter metadata reproduction command.

#![forbid(unsafe_code)]

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use terlan_opencv_metadata_check::{
    package_root, refresh_metadata, validate_generated_aot_adapter, validate_live, validate_offline,
};

/// Runs offline validation or live LibTooling reproduction.
fn main() -> ExitCode {
    let root = package_root();
    let arguments = env::args_os().skip(1).collect::<Vec<_>>();
    let result = match arguments.as_slice() {
        [argument] if argument == "--generated-aot" => validate_generated_aot_adapter(&root),
        [flag, extractor] if flag == "--refresh" => {
            refresh_metadata(&root, PathBuf::from(extractor).as_path())
        }
        [extractor] => validate_live(&root, PathBuf::from(extractor).as_path()),
        [] => validate_offline(&root),
        _ => Err(
            "usage: terlan-opencv-metadata-check [--generated-aot | --refresh EXTRACTOR | EXTRACTOR]"
                .to_string(),
        ),
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
