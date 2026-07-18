//! Executable entry point for the package-owned OpenCV runtime guard.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

/// Runs the runtime guard and reports only guard-owned fatal I/O failures.
fn main() {
    if let Err(error) = terlan_opencv_runtime_guard::run_from_environment() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
