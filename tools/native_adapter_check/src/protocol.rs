//! Generated helper reply decoding.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use std::collections::BTreeMap;

/// One primitive field copied from a generated native value record.
#[derive(Debug, PartialEq)]
pub(crate) enum NativeRecordValue {
    /// Signed integer field.
    Int(i64),
    /// Floating-point field.
    Float(f64),
    /// Boolean field.
    Bool(bool),
}

/// One decoded generated-helper reply payload.
#[derive(Debug, PartialEq)]
pub(crate) enum NativeReply {
    /// Successful unit value.
    Unit,
    /// Successful signed integer.
    Int(i64),
    /// Successful Boolean.
    Bool(bool),
    /// Successful copied bytes.
    Bytes(Vec<u8>),
    /// Successful copied signed integers.
    Ints(Vec<i64>),
    /// Successful copied floating-point values.
    Floats(Vec<f64>),
    /// Successful copied value record.
    Record {
        /// Generated Terlan record name.
        type_name: String,
        /// Decoded primitive fields by generated field name.
        fields: BTreeMap<String, NativeRecordValue>,
    },
    /// Successful newly owned resource.
    Handle {
        /// Cryptographically random worker identity preventing cross-owner use.
        owner: String,
        /// Monotonic process-local handle identity.
        id: u64,
        /// Generation protecting stale aliases.
        generation: u64,
        /// Canonical generated Terlan resource type.
        type_name: String,
    },
    /// Stable protocol or native adapter failure.
    Error {
        /// Finite machine-readable error code.
        code: String,
        /// Public diagnostic text.
        message: String,
    },
}

/// Decodes one reply and enforces request identity and one-credit flow control.
pub(crate) fn decode_reply(line: &str, expected_request_id: u64) -> Result<NativeReply, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    let ["reply", request_id, "1", payload @ ..] = fields.as_slice() else {
        return Err(format!("malformed correlated helper reply: {line}"));
    };
    let request_id = request_id
        .parse::<u64>()
        .map_err(|error| format!("invalid helper request id: {error}"))?;
    if request_id != expected_request_id {
        return Err(format!(
            "helper replied to request {request_id}, expected {expected_request_id}"
        ));
    }
    decode_payload(payload)
}

/// Decodes one helper value payload.
fn decode_payload(fields: &[&str]) -> Result<NativeReply, String> {
    match fields {
        ["ok_unit"] => Ok(NativeReply::Unit),
        ["ok_int", value] => value
            .parse::<i64>()
            .map(NativeReply::Int)
            .map_err(|error| format!("invalid helper integer: {error}")),
        ["ok_bool", value] => value
            .parse::<bool>()
            .map(NativeReply::Bool)
            .map_err(|error| format!("invalid helper Boolean: {error}")),
        ["ok_bytes"] => Ok(NativeReply::Bytes(Vec::new())),
        ["ok_bytes", value] => STANDARD
            .decode(value)
            .map(NativeReply::Bytes)
            .map_err(|error| format!("invalid helper bytes: {error}")),
        ["ok_ints"] => Ok(NativeReply::Ints(Vec::new())),
        ["ok_ints", values] => values
            .split(',')
            .map(|value| {
                value
                    .parse::<i64>()
                    .map_err(|error| format!("invalid helper integer list: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(NativeReply::Ints),
        ["ok_floats"] => Ok(NativeReply::Floats(Vec::new())),
        ["ok_floats", values] => values
            .split(',')
            .map(|value| {
                value
                    .parse::<f64>()
                    .map_err(|error| format!("invalid helper float list: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(NativeReply::Floats),
        ["ok_record", type_name, fields] => Ok(NativeReply::Record {
            type_name: decode_text(type_name)?,
            fields: parse_record_fields(fields)?,
        }),
        ["ok_handle", owner, id, generation, type_name] => Ok(NativeReply::Handle {
            owner: decode_text(owner)?,
            id: parse_u64(id, "handle id")?,
            generation: parse_u64(generation, "handle generation")?,
            type_name: decode_text(type_name)?,
        }),
        ["err", code, message] => Ok(NativeReply::Error {
            code: decode_text(code)?,
            message: decode_text(message)?,
        }),
        _ => Err(format!("unsupported helper payload: {}", fields.join(" "))),
    }
}

/// Decodes comma-delimited primitive fields from one copied value record.
fn parse_record_fields(value: &str) -> Result<BTreeMap<String, NativeRecordValue>, String> {
    if value.is_empty() {
        return Ok(BTreeMap::new());
    }
    value
        .split(',')
        .try_fold(BTreeMap::new(), |mut fields, field| {
            let mut parts = field.splitn(3, ':');
            let name = parts
                .next()
                .ok_or_else(|| "record field has no name".to_owned())?;
            let kind = parts
                .next()
                .ok_or_else(|| "record field has no kind".to_owned())?;
            let value = parts
                .next()
                .ok_or_else(|| "record field has no value".to_owned())?;
            let value = match kind {
                "i" => value
                    .parse::<i64>()
                    .map(NativeRecordValue::Int)
                    .map_err(|error| format!("invalid record integer: {error}"))?,
                "f" => value
                    .parse::<f64>()
                    .map(NativeRecordValue::Float)
                    .map_err(|error| format!("invalid record float: {error}"))?,
                "b" => value
                    .parse::<bool>()
                    .map(NativeRecordValue::Bool)
                    .map_err(|error| format!("invalid record Boolean: {error}"))?,
                _ => return Err(format!("unsupported record field kind `{kind}`")),
            };
            let name = decode_text(name)?;
            if fields.insert(name.clone(), value).is_some() {
                return Err(format!("duplicate record field `{name}`"));
            }
            Ok(fields)
        })
}

/// Parses one unsigned protocol integer.
fn parse_u64(value: &str, kind: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|error| format!("invalid {kind}: {error}"))
}

/// Decodes one base64 UTF-8 protocol field.
fn decode_text(value: &str) -> Result<String, String> {
    let bytes = STANDARD
        .decode(value)
        .map_err(|error| format!("invalid base64 text: {error}"))?;
    String::from_utf8(bytes).map_err(|error| format!("invalid UTF-8 text: {error}"))
}

#[cfg(test)]
#[path = "protocol_test.rs"]
mod tests;
