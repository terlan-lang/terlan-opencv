//! Command-line gate for the package-owned OpenCV codec policy.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::path::Path;

use terlan_opencv_codec_policy_check::validate_package;

/// Executes the policy gate from the package root argument.
fn run() -> Result<(), String> {
    let root = std::env::args_os().nth(1).unwrap_or_else(|| ".".into());
    validate_package(Path::new(&root))
}

/// Prints a stable gate result and returns a failing exit code on drift.
fn main() {
    if let Err(error) = run() {
        eprintln!("opencv-codec-policy-check: {error}");
        std::process::exit(1);
    }
    println!("opencv-codec-policy-check: copied-byte BMP policy is complete and filesystem-free");
}
