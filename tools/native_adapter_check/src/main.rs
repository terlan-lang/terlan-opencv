//! Executable OCV0.2 native adapter lifecycle gate.

#![forbid(unsafe_code)]

use std::env;
use std::process::ExitCode;

use terlan_opencv_native_adapter_check::run_native_contract;

/// Runs the native contract against the generated helper path.
fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1) else {
        eprintln!("usage: terlan-opencv-native-adapter-check <native-helper>");
        return ExitCode::FAILURE;
    };
    match run_native_contract(path) {
        Ok(()) => {
            println!("OpenCV native adapter contract passed");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("OpenCV native adapter contract failed: {error}");
            ExitCode::FAILURE
        }
    }
}
