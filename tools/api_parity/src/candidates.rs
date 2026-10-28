//! Deterministic review candidates derived from maintained wrapper call metadata.

use super::{read_json, read_toml, require, Inventory, MappingDocument, UPSTREAM_COMMIT};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// Schema used by the generated public-operation reconciliation ledger.
const RECONCILIATION_SCHEMA: &str = "terlan.opencv.api-reconciliation.v1";

/// Native binding fields needed to associate wrapper symbols with public operations.
#[derive(Deserialize)]
struct NativeBinding {
    /// Public generated Terlan modules.
    modules: Vec<NativeModule>,
    /// Maintained Clang metadata for adapter implementation symbols.
    cpp_metadata: WrapperMetadata,
}

/// One generated Terlan module.
#[derive(Deserialize)]
struct NativeModule {
    /// Public functions declared by the module.
    #[serde(default)]
    functions: Vec<NativeFunction>,
}

/// Public operation and the C++ wrapper symbol implementing it.
#[derive(Deserialize)]
struct NativeFunction {
    /// Stable public Terlan operation identity.
    operation: String,
    /// Stable package-owned C++ wrapper identity.
    #[serde(default)]
    cpp_symbol: Option<String>,
}

/// Maintained metadata for package-owned adapter symbols.
#[derive(Deserialize)]
struct WrapperMetadata {
    /// Extracted adapter declarations.
    symbols: Vec<WrapperSymbol>,
}

/// One adapter declaration and its directly invoked declarations.
#[derive(Deserialize)]
struct WrapperSymbol {
    /// Stable package-owned declaration identity.
    id: String,
    /// Direct calls observed in the wrapper definition.
    #[serde(default)]
    direct_calls: Vec<DirectCall>,
}

/// One direct call emitted by the maintained Clang extractor.
#[derive(Deserialize)]
struct DirectCall {
    /// Canonical identity of the invoked declaration.
    id: String,
}

/// Reviewable upstream identities reached by one public Terlan operation.
#[derive(Debug, Eq, PartialEq)]
struct Candidate {
    /// Public Terlan operation.
    operation: String,
    /// Package wrapper declarations contributing exact calls.
    wrappers: BTreeSet<String>,
    /// Exact source-stable OpenCV identities not already in the mapping ledger.
    upstream: BTreeSet<String>,
}

/// Complete deterministic candidate analysis.
#[derive(Debug, Default, Eq, PartialEq)]
struct CandidateReport {
    /// Public operations with at least one pending exact OpenCV call.
    candidates: Vec<Candidate>,
    /// Wrappers that call exact OpenCV identities but own no public operation.
    unowned_wrappers: BTreeSet<String>,
}

/// Deterministic classification of every public package operation.
#[derive(Deserialize, Eq, PartialEq, Serialize)]
struct ReconciliationDocument {
    /// Reconciliation document schema.
    schema: String,
    /// Exact pinned OpenCV source revision.
    upstream_commit: String,
    /// Number of unique public operations.
    total: usize,
    /// Public operations ordered by stable operation identity.
    operations: Vec<ReconciledOperation>,
}

/// Mapping or gap state for one public package operation.
#[derive(Deserialize, Eq, PartialEq, Serialize)]
struct ReconciledOperation {
    /// Stable public Terlan operation identity.
    operation: String,
    /// Either `mapped` or `gap`.
    status: String,
    /// Stable gap code, omitted for mapped operations.
    #[serde(skip_serializing_if = "Option::is_none")]
    gap: Option<String>,
    /// Package-owned wrappers implementing the operation.
    wrappers: Vec<String>,
    /// Exact inventory identities called directly by those wrappers.
    upstream: Vec<String>,
}

/// Writes the deterministic public-operation reconciliation ledger.
pub(super) fn refresh_reconciliation(root: &Path) -> Result<(), String> {
    let inventory: Inventory = read_json(&root.join("bindings/opencv.api-inventory.json"))?;
    let mappings: MappingDocument = read_toml(&root.join("bindings/opencv.api-parity.toml"))?;
    let binding: NativeBinding = read_json(&root.join("bindings/opencv.native-binding.json"))?;
    let document = reconciliation_document(&inventory, &mappings, &binding)?;
    let encoded = serde_json::to_string_pretty(&document)
        .map_err(|error| format!("cannot encode API reconciliation: {error}"))?;
    fs::write(
        root.join("bindings/opencv.api-reconciliation.json"),
        format!("{encoded}\n"),
    )
    .map_err(|error| format!("cannot write API reconciliation: {error}"))?;
    println!(
        "OpenCV API reconciliation refreshed: {} public operations",
        document.total
    );
    Ok(())
}

/// Requires the committed public-operation reconciliation to match current inputs.
pub(super) fn validate_reconciliation(
    root: &Path,
    inventory: &Inventory,
    mappings: &MappingDocument,
) -> Result<(), String> {
    let binding: NativeBinding = read_json(&root.join("bindings/opencv.native-binding.json"))?;
    let actual = reconciliation_document(inventory, mappings, &binding)?;
    let expected: ReconciliationDocument =
        read_json(&root.join("bindings/opencv.api-reconciliation.json"))?;
    require(
        actual == expected,
        "committed OpenCV API reconciliation is stale; run `make opencv-api-parity-reconcile`",
    )
}

/// Classifies every public operation from binding, mapping, and direct-call metadata.
fn reconciliation_document(
    inventory: &Inventory,
    mappings: &MappingDocument,
    binding: &NativeBinding,
) -> Result<ReconciliationDocument, String> {
    let upstream = inventory
        .identities
        .iter()
        .map(|identity| (identity.upstream_id.as_str(), identity.id.as_str()))
        .collect::<BTreeMap<_, _>>();
    require(
        upstream.len() == inventory.identities.len(),
        "inventory upstream identities are not unique",
    )?;
    let wrapper_calls = binding
        .cpp_metadata
        .symbols
        .iter()
        .map(|symbol| (symbol.id.as_str(), symbol.direct_calls.as_slice()))
        .collect::<BTreeMap<_, _>>();
    require(
        wrapper_calls.len() == binding.cpp_metadata.symbols.len(),
        "wrapper metadata identities are not unique",
    )?;
    let mapped_operations = mappings
        .mappings
        .iter()
        .flat_map(|mapping| {
            std::iter::once(mapping.terlan.as_str())
                .chain(mapping.additional_terlan.iter().map(String::as_str))
        })
        .map(str::to_ascii_lowercase)
        .collect::<BTreeSet<_>>();
    let mut functions = BTreeMap::<String, BTreeSet<String>>::new();
    for function in binding
        .modules
        .iter()
        .flat_map(|module| module.functions.iter())
    {
        let wrappers = functions.entry(function.operation.clone()).or_default();
        if let Some(wrapper) = &function.cpp_symbol {
            wrappers.insert(wrapper.clone());
        }
    }

    let mut operations = Vec::with_capacity(functions.len());
    for (operation, wrappers) in functions {
        let mut exact = BTreeSet::new();
        for wrapper in &wrappers {
            if let Some(calls) = wrapper_calls.get(wrapper.as_str()) {
                exact.extend(
                    calls
                        .iter()
                        .filter_map(|call| upstream.get(call.id.as_str()).copied())
                        .map(str::to_string),
                );
            }
        }
        let mapped = mapped_operations.contains(&operation.to_ascii_lowercase());
        let gap = if mapped {
            None
        } else if wrappers.is_empty()
            || wrappers
                .iter()
                .all(|wrapper| !wrapper_calls.contains_key(wrapper.as_str()))
        {
            Some("no_wrapper".to_string())
        } else if exact.is_empty() {
            Some("no_exact_call".to_string())
        } else if exact.len() == 1 {
            Some("direct_behavior_review".to_string())
        } else {
            Some("composed_behavior_review".to_string())
        };
        operations.push(ReconciledOperation {
            operation,
            status: if mapped { "mapped" } else { "gap" }.to_string(),
            gap,
            wrappers: wrappers.into_iter().collect(),
            upstream: exact.into_iter().collect(),
        });
    }

    Ok(ReconciliationDocument {
        schema: RECONCILIATION_SCHEMA.to_string(),
        upstream_commit: UPSTREAM_COMMIT.to_string(),
        total: operations.len(),
        operations,
    })
}

/// Prints the deterministic parity reconciliation queue.
pub(super) fn report(root: &Path) -> Result<(), String> {
    let inventory: Inventory = read_json(&root.join("bindings/opencv.api-inventory.json"))?;
    let mappings: MappingDocument = read_toml(&root.join("bindings/opencv.api-parity.toml"))?;
    let binding: NativeBinding = read_json(&root.join("bindings/opencv.native-binding.json"))?;
    let report = candidate_report(&inventory, &mappings, &binding)?;
    let direct = report
        .candidates
        .iter()
        .filter(|candidate| candidate.upstream.len() == 1)
        .count();
    let composed = report.candidates.len() - direct;

    println!("OpenCV API parity reconciliation candidates");
    println!("direct: {direct}");
    println!("composed: {composed}");
    println!("unowned wrappers: {}", report.unowned_wrappers.len());
    for candidate in report.candidates {
        let classification = if candidate.upstream.len() == 1 {
            "direct"
        } else {
            "composed"
        };
        println!(
            "\n[{classification}] {} via {}",
            candidate.operation,
            candidate
                .wrappers
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
        for upstream in candidate.upstream {
            println!("  {upstream}");
        }
    }
    if !report.unowned_wrappers.is_empty() {
        println!("\n[unowned]");
        for wrapper in report.unowned_wrappers {
            println!("  {wrapper}");
        }
    }
    Ok(())
}

/// Builds candidates without treating a direct implementation call as parity credit.
fn candidate_report(
    inventory: &Inventory,
    mappings: &MappingDocument,
    binding: &NativeBinding,
) -> Result<CandidateReport, String> {
    let upstream = inventory
        .identities
        .iter()
        .map(|identity| (identity.upstream_id.as_str(), identity.id.as_str()))
        .collect::<BTreeMap<_, _>>();
    require(
        upstream.len() == inventory.identities.len(),
        "inventory upstream identities are not unique",
    )?;
    let mapped = mappings
        .mappings
        .iter()
        .flat_map(|mapping| mapping.upstream.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    let owners = operation_owners(binding)?;
    let mut candidates = BTreeMap::<String, Candidate>::new();
    let mut unowned_wrappers = BTreeSet::new();

    for wrapper in &binding.cpp_metadata.symbols {
        let pending = wrapper
            .direct_calls
            .iter()
            .filter_map(|call| upstream.get(call.id.as_str()).copied())
            .filter(|identity| !mapped.contains(identity))
            .collect::<BTreeSet<_>>();
        if pending.is_empty() {
            continue;
        }
        let Some(operation) = owners.get(wrapper.id.as_str()) else {
            unowned_wrappers.insert(wrapper.id.clone());
            continue;
        };
        let candidate = candidates
            .entry(operation.to_string())
            .or_insert_with(|| Candidate {
                operation: operation.to_string(),
                wrappers: BTreeSet::new(),
                upstream: BTreeSet::new(),
            });
        candidate.wrappers.insert(wrapper.id.clone());
        candidate
            .upstream
            .extend(pending.into_iter().map(str::to_string));
    }

    Ok(CandidateReport {
        candidates: candidates.into_values().collect(),
        unowned_wrappers,
    })
}

/// Builds one-to-one wrapper ownership and rejects conflicting public declarations.
fn operation_owners(binding: &NativeBinding) -> Result<BTreeMap<&str, &str>, String> {
    let mut owners = BTreeMap::new();
    for function in binding
        .modules
        .iter()
        .flat_map(|module| module.functions.iter())
    {
        let Some(symbol) = function.cpp_symbol.as_deref() else {
            continue;
        };
        if let Some(existing) = owners.insert(symbol, function.operation.as_str()) {
            require(
                existing == function.operation,
                format!(
                    "wrapper `{symbol}` has conflicting operations `{existing}` and `{}`",
                    function.operation
                ),
            )?;
        }
    }
    Ok(owners)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Identity, Mapping, Source};

    /// Creates one normalized identity for candidate tests.
    fn identity(id: &str, upstream_id: &str) -> Identity {
        Identity {
            id: id.to_string(),
            upstream_id: upstream_id.to_string(),
            kind: "function".to_string(),
            module: "core".to_string(),
            cpp_name: "call".to_string(),
            overload_set: "cv::call".to_string(),
            source: Source {
                path: "core.hpp".to_string(),
                line: 1,
                column: 1,
            },
        }
    }

    /// Creates a minimal binding with one public operation and optional unowned wrapper.
    fn binding(calls: &[&str], include_unowned: bool) -> NativeBinding {
        let mut symbols = vec![WrapperSymbol {
            id: "wrapper.call".to_string(),
            direct_calls: calls
                .iter()
                .map(|id| DirectCall { id: id.to_string() })
                .collect(),
        }];
        if include_unowned {
            symbols.push(WrapperSymbol {
                id: "wrapper.internal".to_string(),
                direct_calls: vec![DirectCall {
                    id: "cv::second".to_string(),
                }],
            });
        }
        NativeBinding {
            modules: vec![NativeModule {
                functions: vec![NativeFunction {
                    operation: "opencv.call".to_string(),
                    cpp_symbol: Some("wrapper.call".to_string()),
                }],
            }],
            cpp_metadata: WrapperMetadata { symbols },
        }
    }

    /// Creates a mapping document containing the selected exact identity.
    fn mappings(upstream: &[&str]) -> MappingDocument {
        MappingDocument {
            schema: "test".to_string(),
            upstream_commit: "test".to_string(),
            mappings: upstream
                .iter()
                .map(|id| Mapping {
                    upstream: vec![id.to_string()],
                    terlan: "opencv.call".to_string(),
                    additional_terlan: Vec::new(),
                    disposition: "direct".to_string(),
                    rationale: "test".to_string(),
                    evidence: vec!["test".to_string()],
                })
                .collect(),
        }
    }

    #[test]
    fn candidates_are_exact_deduplicated_and_exclude_mapped_identities() {
        let inventory = Inventory {
            schema: "test".to_string(),
            upstream_commit: "test".to_string(),
            total: 2,
            modules: BTreeMap::new(),
            identities: vec![
                identity("exact.first", "cv::first"),
                identity("exact.second", "cv::second"),
            ],
        };
        let report = candidate_report(
            &inventory,
            &mappings(&["exact.first"]),
            &binding(&["cv::first", "cv::second", "cv::second", "helper"], true),
        )
        .expect("candidate report");

        assert_eq!(
            report,
            CandidateReport {
                candidates: vec![Candidate {
                    operation: "opencv.call".to_string(),
                    wrappers: BTreeSet::from(["wrapper.call".to_string()]),
                    upstream: BTreeSet::from(["exact.second".to_string()]),
                }],
                unowned_wrappers: BTreeSet::from(["wrapper.internal".to_string()]),
            }
        );
    }

    #[test]
    fn conflicting_wrapper_ownership_is_rejected() {
        let binding = NativeBinding {
            modules: vec![
                NativeModule {
                    functions: vec![NativeFunction {
                        operation: "opencv.first".to_string(),
                        cpp_symbol: Some("wrapper.call".to_string()),
                    }],
                },
                NativeModule {
                    functions: vec![NativeFunction {
                        operation: "opencv.second".to_string(),
                        cpp_symbol: Some("wrapper.call".to_string()),
                    }],
                },
            ],
            cpp_metadata: WrapperMetadata {
                symbols: Vec::new(),
            },
        };

        assert_eq!(
            operation_owners(&binding).unwrap_err(),
            "wrapper `wrapper.call` has conflicting operations `opencv.first` and `opencv.second`"
        );
    }

    #[test]
    fn reconciliation_classifies_mapped_direct_and_wrapperless_operations() {
        let inventory = Inventory {
            schema: "test".to_string(),
            upstream_commit: "test".to_string(),
            total: 2,
            modules: BTreeMap::new(),
            identities: vec![
                identity("exact.first", "cv::first"),
                identity("exact.second", "cv::second"),
            ],
        };
        let mut binding = binding(&["cv::first"], false);
        binding.modules[0].functions.push(NativeFunction {
            operation: "opencv.dispose".to_string(),
            cpp_symbol: None,
        });
        binding.modules[0].functions.push(NativeFunction {
            operation: "opencv.helper".to_string(),
            cpp_symbol: Some("wrapper.helper".to_string()),
        });
        binding.cpp_metadata.symbols.push(WrapperSymbol {
            id: "wrapper.helper".to_string(),
            direct_calls: Vec::new(),
        });

        let document = reconciliation_document(&inventory, &mappings(&["exact.first"]), &binding)
            .expect("reconciliation");

        assert_eq!(document.total, 3);
        assert_eq!(document.operations[0].operation, "opencv.call");
        assert_eq!(document.operations[0].status, "mapped");
        assert_eq!(document.operations[0].gap, None);
        assert_eq!(document.operations[1].operation, "opencv.dispose");
        assert_eq!(document.operations[1].gap.as_deref(), Some("no_wrapper"));
        assert_eq!(document.operations[2].operation, "opencv.helper");
        assert_eq!(document.operations[2].gap.as_deref(), Some("no_exact_call"));
    }
}
