//! Tests for generated helper reply decoding.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use std::collections::BTreeMap;

use super::{decode_reply, NativeRecordValue, NativeReply};

/// Decodes every reply shape consumed by the lifecycle checker.
#[test]
fn decodes_supported_replies() {
    assert_eq!(decode_reply("reply 1 1 ok_unit", 1), Ok(NativeReply::Unit));
    assert_eq!(
        decode_reply("reply 2 1 ok_int -4", 2),
        Ok(NativeReply::Int(-4))
    );
    assert_eq!(
        decode_reply("reply 3 1 ok_bool true", 3),
        Ok(NativeReply::Bool(true))
    );
    assert_eq!(
        decode_reply(
            &format!("reply 4 1 ok_bytes {}", STANDARD.encode([1, 2])),
            4
        ),
        Ok(NativeReply::Bytes(vec![1, 2]))
    );
    assert_eq!(
        decode_reply("reply 4 1 ok_bytes", 4),
        Ok(NativeReply::Bytes(Vec::new()))
    );
    assert_eq!(
        decode_reply("reply 5 1 ok_ints", 5),
        Ok(NativeReply::Ints(Vec::new()))
    );
    assert_eq!(
        decode_reply("reply 6 1 ok_ints 0,-4,9", 6),
        Ok(NativeReply::Ints(vec![0, -4, 9]))
    );
    assert_eq!(
        decode_reply("reply 7 1 ok_floats 4,8.5", 7),
        Ok(NativeReply::Floats(vec![4.0, 8.5]))
    );
    assert_eq!(
        decode_reply(
            &format!(
                "reply 7 1 ok_handle {} 7 2 {}",
                STANDARD.encode("worker-owner"),
                STANDARD.encode("opencv.Mat.Mat")
            ),
            7
        ),
        Ok(NativeReply::Handle {
            owner: "worker-owner".to_string(),
            id: 7,
            generation: 2,
            type_name: "opencv.Mat.Mat".to_string()
        })
    );
    assert_eq!(
        decode_reply(
            &format!(
                "reply 8 1 ok_record {} {}:f:1.5,{}:i:-2,{}:b:true",
                STANDARD.encode("Snapshot"),
                STANDARD.encode("ratio"),
                STANDARD.encode("count"),
                STANDARD.encode("ready")
            ),
            8
        ),
        Ok(NativeReply::Record {
            type_name: "Snapshot".to_owned(),
            fields: BTreeMap::from([
                ("count".to_owned(), NativeRecordValue::Int(-2)),
                ("ratio".to_owned(), NativeRecordValue::Float(1.5)),
                ("ready".to_owned(), NativeRecordValue::Bool(true)),
            ])
        })
    );
}

/// Rejects malformed correlation, flow-control, and value payloads.
#[test]
fn rejects_malformed_replies() {
    for line in [
        "reply 8 1 ok_unit",
        "reply 7 0 ok_unit",
        "reply 7 1 ok_bool maybe",
        "reply 7 1 ok_bytes *",
        "reply 7 1 ok_ints 1,nope,2",
        "reply 7 1 ok_ints 1,,2",
        "reply 7 1 ok_floats 1,nope",
        "reply 7 1 ok_handle * 1 1 dHlwZQ==",
        "reply 7 1 ok_record U25hcHNob3Q= bmFtZQ==:s:value",
        "reply 7 1 ok_record U25hcHNob3Q= bmFtZQ==:f:nope",
        "reply 7 1 ok_record U25hcHNob3Q= *:f:1",
        "reply 7 1 ok_record U25hcHNob3Q= bmFtZQ==:f:1,bmFtZQ==:f:2",
        "reply 7 1 invented",
    ] {
        assert!(decode_reply(line, 7).is_err(), "accepted {line}");
    }
}
