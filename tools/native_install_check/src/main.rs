//! Command-line entry point for prepared OpenCV installation validation.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::path::Path;

use terlan_opencv_native_install_check::validate_install;

/// Validates positional arguments and executes the native install gate.
fn run() -> Result<(), String> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if !(2..=3).contains(&args.len()) {
        return Err(
            "usage: terlan-opencv-native-install-check <install-root> <probe.cc> [c++-compiler]"
                .to_string(),
        );
    }
    let compiler = args
        .get(2)
        .map(Path::new)
        .unwrap_or_else(|| Path::new("c++"));
    validate_install(Path::new(&args[0]), Path::new(&args[1]), compiler)?;
    Ok(())
}

/// Prints only stable package diagnostics on failure.
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
    println!("opencv-install-check: required headers, libraries, and symbols are available");
}
