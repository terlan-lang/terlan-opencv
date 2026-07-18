//! Package-owned availability boundary for the generated OpenCV worker.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::ffi::OsStr;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

/// Environment variable selecting the OpenCV-linked generated worker.
pub const WORKER_PATH_ENV: &str = "TERLAN_OPENCV_NATIVE_WORKER_PATH";

const UNAVAILABLE_CODE: &str = "opencv.unavailable";
const UNAVAILABLE_MESSAGE: &str = "OpenCV native runtime is unavailable";

/// Runs the runtime guard over standard input and output.
///
/// The guard remains alive when its OpenCV-linked worker is unavailable and
/// emits one correlated `opencv.unavailable` reply for every incoming call.
pub fn run_from_environment() -> Result<(), String> {
    let worker_path = std::env::var_os(WORKER_PATH_ENV);
    let mut guard = RuntimeGuard::new(worker_path.as_deref());
    let input = io::stdin();
    let output = io::stdout();
    guard
        .run(input.lock(), output.lock())
        .map_err(|error| format!("OpenCV runtime guard I/O failed: {error}"))
}

/// Stateful proxy owning at most one generated worker.
struct RuntimeGuard {
    worker: Option<NativeWorker>,
}

impl RuntimeGuard {
    /// Creates a guard and contains all worker startup failures.
    fn new(worker_path: Option<&OsStr>) -> Self {
        let worker = worker_path.and_then(|path| NativeWorker::spawn(path).ok());
        Self { worker }
    }

    /// Proxies input lines or returns stable unavailable replies.
    fn run<R: BufRead, W: Write>(&mut self, mut input: R, mut output: W) -> io::Result<()> {
        let mut request = String::new();
        loop {
            request.clear();
            if input.read_line(&mut request)? == 0 {
                return Ok(());
            }
            let request = request.trim_end_matches(['\r', '\n']);
            if request.is_empty() {
                continue;
            }
            let reply = self.exchange(request);
            writeln!(output, "{reply}")?;
            output.flush()?;
        }
    }

    /// Exchanges one request and permanently contains a failed worker.
    fn exchange(&mut self, request: &str) -> String {
        let result = self.worker.as_mut().map(|worker| worker.exchange(request));
        match result {
            Some(Ok(reply)) => reply,
            Some(Err(_)) => {
                self.worker = None;
                unavailable_reply(request_id(request))
            }
            None => unavailable_reply(request_id(request)),
        }
    }
}

/// OpenCV-linked child process with line-oriented protocol streams.
struct NativeWorker {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl NativeWorker {
    /// Starts the selected worker without exposing loader diagnostics.
    fn spawn(path: &OsStr) -> io::Result<Self> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("OpenCV worker stdin is unavailable"))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("OpenCV worker stdout is unavailable"))?;
        Ok(Self {
            child,
            input,
            output: BufReader::new(output),
        })
    }

    /// Sends one request and requires one complete worker reply.
    fn exchange(&mut self, request: &str) -> io::Result<String> {
        writeln!(self.input, "{request}")?;
        self.input.flush()?;
        let mut reply = String::new();
        if self.output.read_line(&mut reply)? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "OpenCV worker exited before replying",
            ));
        }
        Ok(reply.trim_end_matches(['\r', '\n']).to_string())
    }
}

impl Drop for NativeWorker {
    /// Stops the linked worker when the VM closes the guard transport.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Extracts a request identity without trusting the remaining payload.
fn request_id(request: &str) -> u64 {
    let mut fields = request.split_whitespace();
    if fields.next() != Some("call") {
        return 0;
    }
    fields
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
}

/// Encodes one correlated package-unavailable protocol reply.
fn unavailable_reply(request_id: u64) -> String {
    format!(
        "reply {request_id} 1 err {} {}",
        STANDARD.encode(UNAVAILABLE_CODE),
        STANDARD.encode(UNAVAILABLE_MESSAGE)
    )
}

#[cfg(test)]
#[path = "lib_test.rs"]
mod tests;
