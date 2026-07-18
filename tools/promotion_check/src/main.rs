//! Command-line entrypoint for OpenCV promotion evidence validation.

#![forbid(unsafe_code)]

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use terlan_opencv_promotion_check::{collect_promotion_evidence, write_promotion_evidence};

/// Parsed promotion checker command-line arguments.
struct Arguments {
    /// Compiler repository to reproduce.
    compiler: PathBuf,
    /// OpenCV package repository to reproduce.
    package: PathBuf,
    /// Pinned upstream OpenCV repository.
    opencv: PathBuf,
    /// Required immutable OpenCV revision.
    expected_opencv_commit: String,
    /// Destination for structured promotion evidence.
    report: PathBuf,
    /// Whether fresh-workspace execution already succeeded.
    verified: bool,
}

/// Parses required named command-line arguments.
fn parse_arguments() -> Result<Arguments, String> {
    let mut compiler = None;
    let mut package = None;
    let mut opencv = None;
    let mut expected_opencv_commit = None;
    let mut report = None;
    let mut verified = false;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = match argument.as_str() {
            "--verified" => {
                verified = true;
                continue;
            }
            "--compiler" | "--package" | "--opencv" | "--expected-opencv-commit" | "--report" => {
                arguments.next().ok_or_else(|| {
                    format!("error[opencv_promotion_args]: `{argument}` requires a value")
                })?
            }
            _ => {
                return Err(format!(
                    "error[opencv_promotion_args]: unsupported argument `{argument}`"
                ))
            }
        };
        match argument.as_str() {
            "--compiler" => compiler = Some(PathBuf::from(value)),
            "--package" => package = Some(PathBuf::from(value)),
            "--opencv" => opencv = Some(PathBuf::from(value)),
            "--expected-opencv-commit" => expected_opencv_commit = Some(value),
            "--report" => report = Some(PathBuf::from(value)),
            _ => unreachable!("named argument was matched above"),
        }
    }
    Ok(Arguments {
        compiler: compiler.ok_or("error[opencv_promotion_args]: missing `--compiler`")?,
        package: package.ok_or("error[opencv_promotion_args]: missing `--package`")?,
        opencv: opencv.ok_or("error[opencv_promotion_args]: missing `--opencv`")?,
        expected_opencv_commit: expected_opencv_commit
            .ok_or("error[opencv_promotion_args]: missing `--expected-opencv-commit`")?,
        report: report.ok_or("error[opencv_promotion_args]: missing `--report`")?,
        verified,
    })
}

/// Validates repositories and writes the requested evidence report.
fn run() -> Result<(), String> {
    let arguments = parse_arguments()?;
    let evidence = collect_promotion_evidence(
        &arguments.compiler,
        &arguments.package,
        &arguments.opencv,
        &arguments.expected_opencv_commit,
        arguments.verified,
    )?;
    write_promotion_evidence(&evidence, &arguments.report)?;
    println!(
        "opencv-promotion-check: recorded compiler {}, package {}, and OpenCV {}",
        evidence.compiler.commit, evidence.package.commit, evidence.opencv.commit
    );
    Ok(())
}

/// Process entrypoint with stable diagnostic behavior.
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("opencv-promotion-check: {error}");
            ExitCode::FAILURE
        }
    }
}
