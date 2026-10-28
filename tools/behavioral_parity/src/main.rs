//! Behavioral completion accounting for the pinned OpenCV package.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Source contract schema.
const LEDGER_SCHEMA: &str = "terlan.opencv.behavioral-parity.v1";
/// Generated report schema.
const REPORT_SCHEMA: &str = "terlan.opencv.behavioral-parity-report.v1";
/// Exact supported upstream source revision.
const UPSTREAM_COMMIT: &str = "fe38fc608f6acb8b68953438a62305d8318f4fcd";
/// Expected declaration inventory schema.
const INVENTORY_SCHEMA: &str = "terlan.opencv.api-inventory.v1";
/// Expected operation reconciliation schema.
const RECONCILIATION_SCHEMA: &str = "terlan.opencv.api-reconciliation.v1";
/// Expected exact-identity mapping schema.
const MAPPING_SCHEMA: &str = "terlan.opencv.api-parity.v1";

/// Versioned behavioral completion source contract.
#[derive(Clone, Deserialize)]
struct Ledger {
    /// Contract schema.
    schema: String,
    /// Pinned OpenCV source revision.
    upstream_commit: String,
    /// Required declaration inventory schema.
    inventory_schema: String,
    /// Required operation reconciliation schema.
    reconciliation_schema: String,
    /// Stable module behaviors.
    behaviors: Vec<Behavior>,
}

/// One stable OpenCV behavior family.
#[derive(Clone, Deserialize)]
struct Behavior {
    /// Stable behavior identity.
    id: String,
    /// Owning upstream OpenCV module.
    module: String,
    /// User-facing behavior title.
    title: String,
    /// `implemented`, `incomplete`, or `optional_backend`.
    status: String,
    /// Admitted input and output domain.
    domain: String,
    /// Ownership contract.
    ownership: String,
    /// Failure contract.
    errors: String,
    /// Prefixes selecting current Terlan operations.
    #[serde(default)]
    operation_prefixes: Vec<String>,
    /// Exact current Terlan operations owned by this behavior.
    #[serde(default)]
    operations: Vec<String>,
    /// Upstream documentation or header groups represented by the behavior.
    upstream_sections: Vec<String>,
    /// Permanent package evidence.
    evidence: Vec<String>,
    /// Whether unmatched identities in this module belong to this behavior.
    #[serde(default)]
    fallback: bool,
}

/// Normalized pinned OpenCV declaration inventory.
#[derive(Deserialize)]
struct Inventory {
    /// Inventory schema.
    schema: String,
    /// Pinned OpenCV revision.
    upstream_commit: String,
    /// Declared total identity count.
    total: usize,
    /// Identity counts by module.
    modules: BTreeMap<String, usize>,
    /// Exact declaration identities.
    identities: Vec<Identity>,
}

/// One exact source-stable OpenCV declaration.
#[derive(Deserialize)]
struct Identity {
    /// Exact identity including source location.
    id: String,
    /// Declaration kind.
    kind: String,
    /// Owning upstream module.
    module: String,
    /// Qualified overload-set identity.
    overload_set: String,
}

/// Current generated Terlan operation reconciliation.
#[derive(Deserialize)]
struct Reconciliation {
    /// Reconciliation schema.
    schema: String,
    /// Pinned OpenCV revision.
    upstream_commit: String,
    /// Declared operation count.
    total: usize,
    /// Current public package operations.
    operations: Vec<ReconciledOperation>,
}

/// One current public Terlan operation.
#[derive(Deserialize)]
struct ReconciledOperation {
    /// Fully qualified operation identity.
    operation: String,
}

/// Reviewed exact declaration-to-operation mappings.
#[derive(Deserialize)]
struct MappingLedger {
    /// Mapping schema.
    schema: String,
    /// Pinned OpenCV revision.
    upstream_commit: String,
    /// Reviewed mappings.
    #[serde(default)]
    mappings: Vec<Mapping>,
}

/// One reviewed exact identity mapping.
#[derive(Deserialize)]
struct Mapping {
    /// Exact upstream identities.
    upstream: Vec<String>,
    /// Primary public Terlan operation.
    terlan: String,
    /// Additional cooperating operations.
    #[serde(default)]
    additional_terlan: Vec<String>,
}

/// Deterministic generated behavioral report.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct Report {
    /// Report schema.
    schema: String,
    /// Pinned OpenCV revision.
    upstream_commit: String,
    /// Aggregate completion counts.
    summary: Summary,
    /// Module completion reports.
    modules: Vec<ModuleReport>,
    /// Expanded behavior reports.
    behaviors: Vec<BehaviorReport>,
    /// Exact declaration classifications.
    identity_classifications: Vec<IdentityClassification>,
}

/// Aggregate report totals.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct Summary {
    /// Number of selected OpenCV modules.
    modules: usize,
    /// Number of stable behaviors.
    behaviors: usize,
    /// Implemented behaviors.
    implemented_behaviors: usize,
    /// Incomplete admitted CPU behaviors.
    incomplete_behaviors: usize,
    /// Optional host or platform behaviors.
    optional_behaviors: usize,
    /// Current Terlan operations covered by exactly one behavior.
    operations: usize,
    /// Exact upstream identities classified.
    identities: usize,
    /// Classification totals by category.
    classifications: BTreeMap<String, usize>,
}

/// One module-level completion summary.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ModuleReport {
    /// Upstream module name.
    module: String,
    /// `complete`, `incomplete`, or `optional_backend`.
    status: String,
    /// Behavior IDs owned by the module.
    behaviors: Vec<String>,
    /// Current operations assigned to module behaviors.
    operations: usize,
    /// Exact identities in the pinned inventory.
    identities: usize,
    /// Identity classification totals.
    classifications: BTreeMap<String, usize>,
}

/// One expanded stable behavior.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct BehaviorReport {
    /// Stable behavior identity.
    id: String,
    /// Upstream module.
    module: String,
    /// Completion status.
    status: String,
    /// Exact selected Terlan operations.
    operations: Vec<String>,
    /// Exact identities attributed to the behavior.
    identities: usize,
    /// Permanent evidence paths.
    evidence: Vec<String>,
}

/// One exact declaration classification.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct IdentityClassification {
    /// Exact source-stable identity.
    identity: String,
    /// Upstream module.
    module: String,
    /// `behavior`, `overload`, `cpp_shape`, or `optional_backend`.
    category: String,
    /// Stable behavior receiving completion responsibility.
    behavior: String,
}

/// Validated behavior expansion shared by reporting and classification.
struct Expansion {
    /// Exact operation-to-behavior ownership.
    operation_behaviors: BTreeMap<String, String>,
    /// Expanded operations by behavior.
    behavior_operations: BTreeMap<String, Vec<String>>,
    /// Fallback behavior by module.
    fallbacks: BTreeMap<String, String>,
    /// Behavior definitions by ID.
    behaviors: BTreeMap<String, Behavior>,
}

/// Runs the behavioral parity command.
fn main() {
    if let Err(message) = run(env::args().skip(1).collect()) {
        eprintln!("{message}");
        std::process::exit(1);
    }
}

/// Validates, refreshes, or strictly checks the behavioral contract.
fn run(arguments: Vec<String>) -> Result<(), String> {
    let root = package_root()?;
    let report = report_from_files(&root)?;
    match arguments.as_slice() {
        [] => {
            require_current_report(&root, &report)?;
            print_report(&report);
            Ok(())
        }
        [flag] if flag == "--refresh" => {
            write_report(&root, &report)?;
            print_report(&report);
            Ok(())
        }
        [flag] if flag == "--strict" => {
            require_current_report(&root, &report)?;
            print_report(&report);
            require_strict(&report)
        }
        _ => Err("usage: terlan-opencv-behavioral-parity [--refresh | --strict]".to_string()),
    }
}

/// Resolves the package root from this tool's location.
fn package_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .ok_or_else(|| "behavioral parity tool must remain under tools".to_string())
}

/// Reads all committed contracts and computes the exact report.
fn report_from_files(root: &Path) -> Result<Report, String> {
    let ledger: Ledger = read_toml(&root.join("bindings/opencv.behavioral-parity.toml"))?;
    let inventory: Inventory = read_json(&root.join("bindings/opencv.api-inventory.json"))?;
    let reconciliation: Reconciliation =
        read_json(&root.join("bindings/opencv.api-reconciliation.json"))?;
    let mappings: MappingLedger = read_toml(&root.join("bindings/opencv.api-parity.toml"))?;
    build_report(root, &ledger, &inventory, &reconciliation, &mappings)
}

/// Computes one validated deterministic behavioral report.
fn build_report(
    root: &Path,
    ledger: &Ledger,
    inventory: &Inventory,
    reconciliation: &Reconciliation,
    mappings: &MappingLedger,
) -> Result<Report, String> {
    validate_document_headers(ledger, inventory, reconciliation, mappings)?;
    let expansion = expand_behaviors(root, ledger, inventory, reconciliation)?;
    let classifications = classify_identities(inventory, mappings, &expansion)?;
    assemble_report(inventory, ledger, &expansion, classifications)
}

/// Validates schemas, revisions, and declared totals.
fn validate_document_headers(
    ledger: &Ledger,
    inventory: &Inventory,
    reconciliation: &Reconciliation,
    mappings: &MappingLedger,
) -> Result<(), String> {
    require(
        ledger.schema == LEDGER_SCHEMA,
        "invalid behavioral ledger schema",
    )?;
    require(
        ledger.upstream_commit == UPSTREAM_COMMIT,
        "behavioral ledger upstream commit drift",
    )?;
    require(
        ledger.inventory_schema == INVENTORY_SCHEMA,
        "behavioral ledger inventory schema drift",
    )?;
    require(
        ledger.reconciliation_schema == RECONCILIATION_SCHEMA,
        "behavioral ledger reconciliation schema drift",
    )?;
    require(
        inventory.schema == INVENTORY_SCHEMA,
        "invalid inventory schema",
    )?;
    require(
        inventory.upstream_commit == UPSTREAM_COMMIT,
        "inventory upstream commit drift",
    )?;
    require(
        inventory.total == inventory.identities.len(),
        "inventory total does not match identities",
    )?;
    require(
        reconciliation.schema == RECONCILIATION_SCHEMA,
        "invalid reconciliation schema",
    )?;
    require(
        reconciliation.upstream_commit == UPSTREAM_COMMIT,
        "reconciliation upstream commit drift",
    )?;
    require(
        reconciliation.total == reconciliation.operations.len(),
        "reconciliation total does not match operations",
    )?;
    require(mappings.schema == MAPPING_SCHEMA, "invalid mapping schema")?;
    require(
        mappings.upstream_commit == UPSTREAM_COMMIT,
        "mapping upstream commit drift",
    )
}

/// Validates behaviors and expands operation prefix selectors.
fn expand_behaviors(
    root: &Path,
    ledger: &Ledger,
    inventory: &Inventory,
    reconciliation: &Reconciliation,
) -> Result<Expansion, String> {
    let modules = inventory.modules.keys().cloned().collect::<BTreeSet<_>>();
    let operations = reconciliation
        .operations
        .iter()
        .map(|item| normalize_operation(&item.operation))
        .collect::<BTreeSet<_>>();
    require(
        operations.len() == reconciliation.operations.len(),
        "reconciliation contains duplicate operations",
    )?;
    let mut behaviors = BTreeMap::new();
    let mut fallbacks = BTreeMap::new();
    for behavior in &ledger.behaviors {
        validate_behavior(root, behavior, &modules)?;
        require(
            behaviors
                .insert(behavior.id.clone(), behavior.clone())
                .is_none(),
            format!("duplicate behavior ID `{}`", behavior.id),
        )?;
        if behavior.fallback {
            require(
                fallbacks
                    .insert(behavior.module.clone(), behavior.id.clone())
                    .is_none(),
                format!(
                    "module `{}` has multiple fallback behaviors",
                    behavior.module
                ),
            )?;
        }
    }
    require(
        fallbacks.keys().cloned().collect::<BTreeSet<_>>() == modules,
        "every selected module must define exactly one fallback behavior",
    )?;
    let mut operation_behaviors = BTreeMap::new();
    let mut behavior_operations = BTreeMap::new();
    for operation in operations {
        let matches = ledger
            .behaviors
            .iter()
            .filter(|behavior| {
                behavior.operations.contains(&operation)
                    || behavior
                        .operation_prefixes
                        .iter()
                        .any(|prefix| operation.starts_with(prefix))
            })
            .collect::<Vec<_>>();
        require(
            matches.len() == 1,
            format!(
                "operation `{operation}` must belong to exactly one behavior; matched {}",
                matches.len()
            ),
        )?;
        let behavior = matches[0];
        operation_behaviors.insert(operation.clone(), behavior.id.clone());
        behavior_operations
            .entry(behavior.id.clone())
            .or_insert_with(Vec::new)
            .push(operation);
    }
    for behavior in &ledger.behaviors {
        for prefix in &behavior.operation_prefixes {
            require(
                operation_behaviors
                    .keys()
                    .any(|item| item.starts_with(prefix)),
                format!(
                    "{}: operation prefix `{prefix}` selects no current operation",
                    behavior.id
                ),
            )?;
        }
        for selected in &behavior.operations {
            require(
                operation_behaviors.contains_key(selected),
                format!(
                    "{}: exact operation `{selected}` is not a current operation",
                    behavior.id
                ),
            )?;
        }
        if behavior.status == "implemented" {
            require(
                behavior_operations
                    .get(&behavior.id)
                    .is_some_and(|items| !items.is_empty()),
                format!("{}: implemented behavior has no operations", behavior.id),
            )?;
        }
    }
    Ok(Expansion {
        operation_behaviors,
        behavior_operations,
        fallbacks,
        behaviors,
    })
}

/// Validates one behavior definition and all evidence paths.
fn validate_behavior(
    root: &Path,
    behavior: &Behavior,
    modules: &BTreeSet<String>,
) -> Result<(), String> {
    require(
        modules.contains(&behavior.module),
        format!("{}: unknown module `{}`", behavior.id, behavior.module),
    )?;
    require(
        behavior.id.starts_with(&format!("{}.", behavior.module)),
        format!("{}: behavior ID must start with its module", behavior.id),
    )?;
    require(
        matches!(
            behavior.status.as_str(),
            "implemented" | "incomplete" | "optional_backend"
        ),
        format!("{}: invalid status `{}`", behavior.id, behavior.status),
    )?;
    for (field, value) in [
        ("title", behavior.title.as_str()),
        ("domain", behavior.domain.as_str()),
        ("ownership", behavior.ownership.as_str()),
        ("errors", behavior.errors.as_str()),
    ] {
        require(
            !value.trim().is_empty(),
            format!("{}: {field} must not be empty", behavior.id),
        )?;
    }
    require(
        !behavior.upstream_sections.is_empty()
            && behavior
                .upstream_sections
                .iter()
                .all(|item| !item.trim().is_empty()),
        format!("{}: upstream sections are required", behavior.id),
    )?;
    require(
        !behavior.evidence.is_empty(),
        format!("{}: evidence is required", behavior.id),
    )?;
    for evidence in &behavior.evidence {
        require(
            root.join(evidence).is_file(),
            format!("{}: evidence path does not exist: {evidence}", behavior.id),
        )?;
    }
    require(
        behavior
            .operation_prefixes
            .iter()
            .all(|prefix| prefix.starts_with("opencv.") && prefix.ends_with('.')),
        format!(
            "{}: operation prefixes must be qualified prefixes",
            behavior.id
        ),
    )?;
    require(
        behavior
            .operations
            .iter()
            .all(|operation| operation.starts_with("opencv.")),
        format!("{}: exact operations must be qualified", behavior.id),
    )?;
    if behavior.fallback {
        require(
            behavior.operation_prefixes.is_empty() && behavior.operations.is_empty(),
            format!(
                "{}: fallback behavior cannot select operations",
                behavior.id
            ),
        )?;
        require(
            behavior.status != "implemented",
            format!("{}: fallback behavior cannot be implemented", behavior.id),
        )?;
    }
    if behavior.status == "implemented" {
        require(
            behavior
                .evidence
                .iter()
                .any(|path| path.starts_with("tests/") || path.starts_with("samples/")),
            format!(
                "{}: implemented behavior requires executable evidence",
                behavior.id
            ),
        )?;
    }
    Ok(())
}

/// Classifies every exact identity into one behavior or C++ audit shape.
fn classify_identities(
    inventory: &Inventory,
    mappings: &MappingLedger,
    expansion: &Expansion,
) -> Result<Vec<IdentityClassification>, String> {
    let known = inventory
        .identities
        .iter()
        .map(|identity| identity.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut mapped_operations = BTreeMap::<String, Vec<String>>::new();
    for mapping in &mappings.mappings {
        let operations = std::iter::once(mapping.terlan.as_str())
            .chain(mapping.additional_terlan.iter().map(String::as_str))
            .map(normalize_operation)
            .collect::<Vec<_>>();
        for upstream in &mapping.upstream {
            require(
                known.contains(upstream.as_str()),
                format!("mapping references unknown identity `{upstream}`"),
            )?;
            require(
                mapped_operations
                    .insert(upstream.clone(), operations.clone())
                    .is_none(),
                format!("duplicate identity mapping `{upstream}`"),
            )?;
        }
    }
    let mut overload_groups = BTreeMap::<(String, String), Vec<&Identity>>::new();
    for identity in &inventory.identities {
        overload_groups
            .entry((identity.module.clone(), identity.overload_set.clone()))
            .or_default()
            .push(identity);
    }
    for identities in overload_groups.values_mut() {
        identities.sort_by(|left, right| left.id.cmp(&right.id));
    }
    let mut result = Vec::with_capacity(inventory.identities.len());
    for identity in &inventory.identities {
        let fallback = expansion
            .fallbacks
            .get(&identity.module)
            .ok_or_else(|| format!("{}: module has no fallback", identity.id))?;
        let fallback_status = &expansion
            .behaviors
            .get(fallback)
            .expect("validated fallback")
            .status;
        let same_module_mapping = mapped_operations.get(&identity.id).and_then(|operations| {
            behavior_for_operations(operations, &identity.module, expansion)
        });
        let group = overload_groups
            .get(&(identity.module.clone(), identity.overload_set.clone()))
            .expect("identity belongs to an overload group");
        let group_mapping = group.iter().find_map(|candidate| {
            mapped_operations.get(&candidate.id).and_then(|operations| {
                behavior_for_operations(operations, &identity.module, expansion)
            })
        });
        let (category, behavior) = if fallback_status == "optional_backend" {
            ("optional_backend", fallback.clone())
        } else if matches!(identity.kind.as_str(), "record" | "enum") {
            ("cpp_shape", fallback.clone())
        } else if let Some(behavior) = same_module_mapping {
            ("behavior", behavior)
        } else if let Some(behavior) = group_mapping {
            ("overload", behavior)
        } else if group.first().is_some_and(|first| first.id == identity.id) {
            ("behavior", fallback.clone())
        } else {
            ("overload", fallback.clone())
        };
        result.push(IdentityClassification {
            identity: identity.id.clone(),
            module: identity.module.clone(),
            category: category.to_string(),
            behavior,
        });
    }
    result.sort_by(|left, right| left.identity.cmp(&right.identity));
    require(
        result.len() == inventory.total,
        "not every upstream identity was classified",
    )?;
    Ok(result)
}

/// Selects one same-module behavior for a reviewed mapping operation set.
fn behavior_for_operations(
    operations: &[String],
    module: &str,
    expansion: &Expansion,
) -> Option<String> {
    let behaviors = operations
        .iter()
        .filter_map(|operation| expansion.operation_behaviors.get(operation))
        .filter(|behavior| {
            expansion
                .behaviors
                .get(*behavior)
                .is_some_and(|item| item.module == module)
        })
        .cloned()
        .collect::<BTreeSet<_>>();
    (behaviors.len() == 1).then(|| behaviors.into_iter().next().expect("one behavior"))
}

/// Assembles module, behavior, and aggregate reports.
fn assemble_report(
    inventory: &Inventory,
    ledger: &Ledger,
    expansion: &Expansion,
    classifications: Vec<IdentityClassification>,
) -> Result<Report, String> {
    let mut category_totals = BTreeMap::new();
    let mut behavior_identity_totals = BTreeMap::new();
    for item in &classifications {
        *category_totals.entry(item.category.clone()).or_insert(0) += 1;
        *behavior_identity_totals
            .entry(item.behavior.clone())
            .or_insert(0) += 1;
    }
    let mut behaviors = ledger
        .behaviors
        .iter()
        .map(|behavior| BehaviorReport {
            id: behavior.id.clone(),
            module: behavior.module.clone(),
            status: behavior.status.clone(),
            operations: expansion
                .behavior_operations
                .get(&behavior.id)
                .cloned()
                .unwrap_or_default(),
            identities: behavior_identity_totals
                .get(&behavior.id)
                .copied()
                .unwrap_or_default(),
            evidence: behavior.evidence.clone(),
        })
        .collect::<Vec<_>>();
    behaviors.sort_by(|left, right| left.id.cmp(&right.id));
    let mut modules = Vec::new();
    for (module, identities) in &inventory.modules {
        let owned = behaviors
            .iter()
            .filter(|behavior| behavior.module == *module)
            .collect::<Vec<_>>();
        require(
            !owned.is_empty(),
            format!("module `{module}` has no behaviors"),
        )?;
        let status = if owned.iter().all(|item| item.status == "optional_backend") {
            "optional_backend"
        } else if owned.iter().any(|item| item.status == "incomplete") {
            "incomplete"
        } else {
            "complete"
        };
        let mut module_categories = BTreeMap::new();
        for item in classifications.iter().filter(|item| item.module == *module) {
            *module_categories.entry(item.category.clone()).or_insert(0) += 1;
        }
        modules.push(ModuleReport {
            module: module.clone(),
            status: status.to_string(),
            behaviors: owned.iter().map(|item| item.id.clone()).collect(),
            operations: owned.iter().map(|item| item.operations.len()).sum(),
            identities: *identities,
            classifications: module_categories,
        });
    }
    let summary = Summary {
        modules: modules.len(),
        behaviors: behaviors.len(),
        implemented_behaviors: behaviors
            .iter()
            .filter(|item| item.status == "implemented")
            .count(),
        incomplete_behaviors: behaviors
            .iter()
            .filter(|item| item.status == "incomplete")
            .count(),
        optional_behaviors: behaviors
            .iter()
            .filter(|item| item.status == "optional_backend")
            .count(),
        operations: expansion.operation_behaviors.len(),
        identities: classifications.len(),
        classifications: category_totals,
    };
    Ok(Report {
        schema: REPORT_SCHEMA.to_string(),
        upstream_commit: UPSTREAM_COMMIT.to_string(),
        summary,
        modules,
        behaviors,
        identity_classifications: classifications,
    })
}

/// Normalizes package operation module spelling.
fn normalize_operation(operation: &str) -> String {
    operation.to_ascii_lowercase()
}

/// Writes the deterministic generated report.
fn write_report(root: &Path, report: &Report) -> Result<(), String> {
    let encoded = serde_json::to_string_pretty(report)
        .map_err(|error| format!("cannot encode behavioral report: {error}"))?;
    fs::write(
        root.join("bindings/opencv.behavioral-parity-report.json"),
        format!("{encoded}\n"),
    )
    .map_err(|error| format!("cannot write behavioral report: {error}"))
}

/// Requires the committed generated report to equal a fresh computation.
fn require_current_report(root: &Path, actual: &Report) -> Result<(), String> {
    let expected: Report = read_json(&root.join("bindings/opencv.behavioral-parity-report.json"))?;
    require(
        &expected == actual,
        "committed behavioral parity report is stale; run make opencv-behavioral-parity-refresh",
    )
}

/// Fails while admitted CPU behaviors remain incomplete.
fn require_strict(report: &Report) -> Result<(), String> {
    require(
        report.summary.incomplete_behaviors == 0,
        format!(
            "OpenCV behavioral parity is incomplete: {} admitted behaviors remain",
            report.summary.incomplete_behaviors
        ),
    )
}

/// Prints concise module completion progress.
fn print_report(report: &Report) {
    println!("OpenCV 4.13.0 behavioral parity");
    for module in &report.modules {
        println!(
            "{}: {}, {} behaviors, {} operations, {} identities",
            module.module,
            module.status,
            module.behaviors.len(),
            module.operations,
            module.identities
        );
    }
    println!(
        "total: {} implemented, {} incomplete, {} optional; {} operations and {} identities classified",
        report.summary.implemented_behaviors,
        report.summary.incomplete_behaviors,
        report.summary.optional_behaviors,
        report.summary.operations,
        report.summary.identities
    );
}

/// Reads one JSON contract.
fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_str(&source).map_err(|error| format!("{}: {error}", path.display()))
}

/// Reads one TOML contract.
fn read_toml<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    toml::from_str(&source).map_err(|error| format!("{}: {error}", path.display()))
}

/// Returns success when a contract invariant holds.
fn require(condition: bool, message: impl Into<String>) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loads all real package inputs for mutation-based contract tests.
    fn inputs() -> (PathBuf, Ledger, Inventory, Reconciliation, MappingLedger) {
        let root = package_root().expect("package root");
        let ledger = read_toml(&root.join("bindings/opencv.behavioral-parity.toml"))
            .expect("behavior ledger");
        let inventory =
            read_json(&root.join("bindings/opencv.api-inventory.json")).expect("inventory");
        let reconciliation = read_json(&root.join("bindings/opencv.api-reconciliation.json"))
            .expect("reconciliation");
        let mappings =
            read_toml(&root.join("bindings/opencv.api-parity.toml")).expect("mapping ledger");
        (root, ledger, inventory, reconciliation, mappings)
    }

    /// The source ledger covers every current operation and exact identity.
    #[test]
    fn real_contract_covers_every_operation_and_identity() {
        let (root, ledger, inventory, reconciliation, mappings) = inputs();
        let report = build_report(&root, &ledger, &inventory, &reconciliation, &mappings).unwrap();
        assert_eq!(report.summary.operations, 429);
        assert_eq!(report.summary.identities, 7_419);
        assert_eq!(
            report.summary.classifications.values().sum::<usize>(),
            7_419
        );
    }

    /// Duplicate behavior ownership is rejected before reporting.
    #[test]
    fn overlapping_operation_selectors_are_rejected() {
        let (root, mut ledger, inventory, reconciliation, mappings) = inputs();
        let fallback = ledger
            .behaviors
            .iter_mut()
            .find(|behavior| behavior.id == "imgproc.remaining")
            .unwrap();
        fallback.operation_prefixes = vec!["opencv.imgproc.".to_string()];
        assert!(
            build_report(&root, &ledger, &inventory, &reconciliation, &mappings)
                .unwrap_err()
                .contains("fallback behavior cannot select operations")
        );
    }

    /// Unknown modules cannot enter the source ledger.
    #[test]
    fn unknown_behavior_modules_are_rejected() {
        let (root, mut ledger, inventory, reconciliation, mappings) = inputs();
        ledger.behaviors[0].module = "unknown".to_string();
        assert!(
            build_report(&root, &ledger, &inventory, &reconciliation, &mappings)
                .unwrap_err()
                .contains("unknown module")
        );
    }

    /// Every behavior must retain resolvable package evidence.
    #[test]
    fn missing_behavior_evidence_is_rejected() {
        let (root, mut ledger, inventory, reconciliation, mappings) = inputs();
        ledger.behaviors[0].evidence = vec!["missing-evidence.terl".to_string()];
        assert!(
            build_report(&root, &ledger, &inventory, &reconciliation, &mappings)
                .unwrap_err()
                .contains("evidence path does not exist")
        );
    }

    /// Strict completion remains red while admitted behaviors are incomplete.
    #[test]
    fn strict_gate_rejects_incomplete_admitted_behaviors() {
        let (root, ledger, inventory, reconciliation, mappings) = inputs();
        let report = build_report(&root, &ledger, &inventory, &reconciliation, &mappings).unwrap();
        assert!(require_strict(&report)
            .unwrap_err()
            .contains("admitted behaviors remain"));
    }
}
