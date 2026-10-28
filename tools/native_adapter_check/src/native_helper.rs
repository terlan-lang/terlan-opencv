//! Process client and transport-level checks for the generated adapter helper.

use std::ffi::OsStr;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::protocol::{decode_reply, NativeReply};

const MAX_ADAPTER_FRAME_BYTES: usize = 1_048_576;
const MAX_ADAPTER_TRANSFER_BYTES: usize = 16_777_216;
const MAX_ADAPTER_MESSAGE_BYTES: usize =
    MAX_ADAPTER_TRANSFER_BYTES * 4 / 3 + MAX_ADAPTER_FRAME_BYTES;

/// One live helper process with correlated request sequencing.
pub(crate) struct NativeHelper {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    next_request_id: u64,
}

impl NativeHelper {
    /// Starts the generated helper with isolated standard streams.
    pub(crate) fn spawn(path: impl AsRef<OsStr>) -> Result<Self, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("cannot start generated helper: {error}"))?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| "generated helper stdin is unavailable".to_string())?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| "generated helper stdout is unavailable".to_string())?;
        Ok(Self {
            child,
            input,
            output: BufReader::new(output),
            next_request_id: 0,
        })
    }

    /// Proves bounded frames and one-shot monotonic request admission.
    pub(crate) fn verify_public_adapter_transport(path: impl AsRef<OsStr>) -> Result<(), String> {
        let path = path.as_ref();
        let mut sequenced = Self::spawn(path)?;
        match sequenced.call("opencv.mat.live_count", &[])? {
            NativeReply::Int(0) => {}
            reply => return Err(format!("unexpected adapter readiness reply: {reply:?}")),
        }
        match sequenced.request(sequenced.next_request_id, "opencv.mat.live_count", &[])? {
            NativeReply::Error { code, .. } if code == "request_not_monotonic" => {}
            reply => return Err(format!("adapter accepted replayed request id: {reply:?}")),
        }
        drop(sequenced);

        let mut bounded = Self::spawn(path)?;
        let frame = vec![b'x'; MAX_ADAPTER_FRAME_BYTES + 1];
        let write_result = bounded
            .input
            .write_all(&frame)
            .and_then(|()| bounded.input.write_all(b"\n"))
            .and_then(|()| bounded.input.flush());
        if let Err(error) = write_result {
            // A conforming helper may close its input as soon as it has read the
            // over-limit prefix; the protocol reply remains authoritative.
            if error.kind() != std::io::ErrorKind::BrokenPipe {
                return Err(format!("cannot write oversized helper frame: {error}"));
            }
        }
        let mut line = String::new();
        let read = bounded
            .output
            .by_ref()
            .read_line(&mut line)
            .map_err(|error| format!("cannot read oversized-frame reply: {error}"))?;
        if read == 0 {
            return Err("generated helper exited without rejecting oversized frame".to_string());
        }
        let expected = format!("err {}", STANDARD.encode("frame_too_large"));
        if !line.starts_with(&expected) {
            return Err(format!(
                "generated helper returned unexpected oversized-frame reply: {}",
                line.trim_end()
            ));
        }
        Ok(())
    }

    /// Executes one protocol call and decodes its correlated reply.
    pub(crate) fn call(&mut self, operation: &str, args: &[String]) -> Result<NativeReply, String> {
        self.next_request_id += 1;
        self.request(self.next_request_id, operation, args)
    }

    /// Sends one exact request identity, including adversarial replays.
    fn request(
        &mut self,
        request_id: u64,
        operation: &str,
        args: &[String],
    ) -> Result<NativeReply, String> {
        let mut fields = vec![
            "call".to_string(),
            request_id.to_string(),
            STANDARD.encode(operation),
        ];
        fields.extend(args.iter().cloned());
        let request = fields.join(" ");
        self.write_request(request_id, &request).map_err(|error| {
            format!(
                "cannot write helper request for {operation} with {} arguments ({} bytes): {error}",
                args.len(),
                request.len() + 1
            )
        })?;
        let reply_line = self.read_response(request_id)?;
        let reply = decode_reply(&reply_line, request_id)
            .map_err(|error| format!("operation {operation} with arguments {args:?}: {error}"))?;
        if std::env::var_os("TERLAN_OPENCV_TRACE_HELPER_ERRORS").is_some()
            && matches!(reply, NativeReply::Error { .. })
        {
            eprintln!("helper error from {operation} {args:?}: {reply:?}");
        }
        Ok(reply)
    }

    fn write_request(&mut self, request_id: u64, request: &str) -> std::io::Result<()> {
        if request.len() < MAX_ADAPTER_FRAME_BYTES {
            writeln!(self.input, "{request}")?;
            return self.input.flush();
        }
        if request.len() > MAX_ADAPTER_MESSAGE_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "request exceeds the bounded adapter transfer allowance",
            ));
        }
        let chunk_bytes = (MAX_ADAPTER_FRAME_BYTES - 128) * 3 / 4;
        let chunks = request.as_bytes().chunks(chunk_bytes);
        let chunk_count = chunks.len();
        for (index, chunk) in chunks.enumerate() {
            let final_chunk = usize::from(index + 1 == chunk_count);
            writeln!(
                self.input,
                "chunk {request_id} {index} {final_chunk} {}",
                STANDARD.encode(chunk)
            )?;
        }
        self.input.flush()
    }

    fn read_response(&mut self, request_id: u64) -> Result<String, String> {
        let first = self.read_frame()?;
        if !first.starts_with("reply_chunk ") {
            return Ok(first);
        }
        let mut assembled = Vec::new();
        let mut next_index = 0_u64;
        let mut frame = first;
        loop {
            let mut fields = frame.split_whitespace();
            let _reply_chunk = fields.next();
            let actual_request_id = parse_u64(fields.next(), "reply chunk request id")?;
            let index = parse_u64(fields.next(), "reply chunk index")?;
            let final_chunk = match fields.next() {
                Some("0") => false,
                Some("1") => true,
                value => return Err(format!("invalid reply final-chunk marker: {value:?}")),
            };
            let encoded = fields
                .next()
                .ok_or_else(|| "reply chunk is missing its payload".to_string())?;
            if fields.next().is_some() || actual_request_id != request_id || index != next_index {
                return Err("reply chunk identity or sequence mismatch".to_string());
            }
            let chunk = STANDARD
                .decode(encoded)
                .map_err(|error| format!("invalid encoded reply chunk: {error}"))?;
            if assembled.len().saturating_add(chunk.len()) > MAX_ADAPTER_MESSAGE_BYTES {
                return Err("chunked reply exceeds the adapter transfer allowance".to_string());
            }
            assembled.extend_from_slice(&chunk);
            next_index += 1;
            if final_chunk {
                return String::from_utf8(assembled)
                    .map_err(|error| format!("chunked reply is not UTF-8: {error}"));
            }
            frame = self.read_frame()?;
        }
    }

    fn read_frame(&mut self) -> Result<String, String> {
        let mut frame = Vec::new();
        let read = self
            .output
            .by_ref()
            .take((MAX_ADAPTER_FRAME_BYTES + 1) as u64)
            .read_until(b'\n', &mut frame)
            .map_err(|error| format!("cannot read helper reply: {error}"))?;
        if read == 0 {
            return Err("generated helper exited without replying".to_string());
        }
        if frame.len() > MAX_ADAPTER_FRAME_BYTES {
            return Err("generated helper emitted an oversized reply frame".to_string());
        }
        String::from_utf8(frame)
            .map(|line| line.trim_end_matches(['\r', '\n']).to_string())
            .map_err(|error| format!("helper reply is not UTF-8: {error}"))
    }
}

fn parse_u64(value: Option<&str>, field: &str) -> Result<u64, String> {
    value
        .ok_or_else(|| format!("missing {field}"))?
        .parse::<u64>()
        .map_err(|error| format!("invalid {field}: {error}"))
}

impl Drop for NativeHelper {
    /// Stops the helper even when a contract assertion fails.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
