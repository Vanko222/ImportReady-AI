"""P0-1 integration tests for the JSON compliance repository and CaseState.

Tests use the real approved JSON files under ``data/`` for baseline validation
and temporary copies for failure-behavior tests (the real files are never
modified).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src import config
from src.models import CalculatorUse, ComplianceCost, ProductAttribute
from src.repositories import ComplianceRepository
from src.repositories.compliance_repository import (
    DataDirectoryError,
    DatasetLoadError,
    DatasetValidationError,
    DuplicateIDError,
    JsonComplianceRepository,
    ReferenceIntegrityError,
)
from src.state import AnalysisStatus, CaseState, FactOrigin, KnowledgeSnapshot, ProductFact

DATA_DIR = config.get_data_dir()
DATA_FILES = list(config.DATA_FILE_NAMES)


@pytest.fixture(scope="module")
def repo() -> JsonComplianceRepository:
    return JsonComplianceRepository()


@pytest.fixture()
def temp_data(tmp_path: Path) -> Path:
    """Fresh copy of the real data dir inside the pytest tmp dir."""
    target = tmp_path / "data"
    shutil.copytree(DATA_DIR, target)
    return target


def _write_dataset(dir_path: Path, file_name: str, payload) -> None:
    (dir_path / file_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_dataset(dir_path: Path, file_name: str):
    return json.loads((dir_path / file_name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# A. Startup + B. Baseline counts
# --------------------------------------------------------------------------- #
def test_all_six_files_load(repo: JsonComplianceRepository) -> None:
    assert repo.counts["attributes"] == 86
    assert repo.counts["rules"] == 41
    assert repo.counts["sources"] == 70
    assert repo.counts["costs"] == 32
    assert repo.counts["known_gaps"] == 16
    assert repo.counts["pending_policy_updates"] == 0


def test_schema_version_and_metadata(repo: JsonComplianceRepository) -> None:
    assert repo.schema_version == "1.0"
    assert repo.generated_at == "2026-09-09T11:13:35Z"
    snapshot = repo.knowledge_snapshot()
    assert isinstance(snapshot, KnowledgeSnapshot)
    assert snapshot.schema_version == "1.0"
    assert snapshot.generated_at == repo.generated_at


# --------------------------------------------------------------------------- #
# C. Canonical lookup
# --------------------------------------------------------------------------- #
def test_known_ids_resolve(repo: JsonComplianceRepository) -> None:
    attribute = repo.get_attribute("A-CMN-001")
    assert isinstance(attribute, ProductAttribute)
    assert attribute.attribute_name == "product_category"

    rule = repo.get_rule("R-CMN-001")
    assert rule is not None and rule.rule_id == "R-CMN-001"

    source = repo.get_source("S001")
    assert source is not None and source.source_id == "S001"

    cost = repo.get_cost("C-E-001")
    assert isinstance(cost, ComplianceCost)
    assert cost.cost_id == "C-E-001"


def test_unknown_ids_return_none(repo: JsonComplianceRepository) -> None:
    # Documented lookup behavior: unknown canonical ID -> None.
    assert repo.get_attribute("A-NOT-REAL") is None
    assert repo.get_rule("R-NOT-REAL") is None
    assert repo.get_source("S-NOT-REAL") is None
    assert repo.get_cost("C-NOT-REAL") is None


# --------------------------------------------------------------------------- #
# D. Cross-reference validation on the approved baseline
# --------------------------------------------------------------------------- #
def test_baseline_cross_references_resolve(repo: JsonComplianceRepository) -> None:
    # Startup already validated; calling again must not raise.
    repo.validate_references()


# --------------------------------------------------------------------------- #
# E. Category queries
# --------------------------------------------------------------------------- #
def test_rules_for_category(repo: JsonComplianceRepository) -> None:
    toy_rules = repo.get_rules_for_category("childrens_toys")
    assert toy_rules
    assert all(r.category == "childrens_toys" for r in toy_rules)

    elec_rules = repo.get_rules_for_category("small_consumer_electronics")
    assert elec_rules
    assert all(r.category == "small_consumer_electronics" for r in elec_rules)


def test_costs_for_category(repo: JsonComplianceRepository) -> None:
    toy_costs = repo.get_costs_for_category("childrens_toys")
    assert toy_costs
    assert all(c.category == "childrens_toys" for c in toy_costs)

    elec_costs = repo.get_costs_for_category("small_consumer_electronics")
    assert elec_costs
    assert all(c.category == "small_consumer_electronics" for c in elec_costs)


def test_unknown_category_returns_empty(repo: JsonComplianceRepository) -> None:
    assert repo.get_rules_for_category("not_a_real_category") == []
    assert repo.get_costs_for_category("not_a_real_category") == []


# --------------------------------------------------------------------------- #
# F. Cost representation semantics
# --------------------------------------------------------------------------- #
def test_all_costs_expose_additional_source_ids(repo: JsonComplianceRepository) -> None:
    costs = repo.costs
    assert len(costs) == 32
    for cost in costs:
        assert isinstance(cost.additional_source_ids, list)


def test_non_empty_additional_sources_preserved(repo: JsonComplianceRepository) -> None:
    cost = repo.get_cost("C-P-001")
    assert cost is not None
    assert cost.additional_source_ids == ["S054", "S069"]


def test_quote_required_never_becomes_zero(repo: JsonComplianceRepository) -> None:
    quote_required = [c for c in repo.costs if c.calculator_use == CalculatorUse.QUOTE_REQUIRED]
    assert quote_required  # baseline has such records
    for cost in quote_required:
        assert cost.exact_amount is None
        assert cost.low_amount is None
        assert cost.high_amount is None
        assert cost.price_status == "QUOTE_REQUIRED"
        # Guards against any accidental normalization to numeric zero.
        assert cost.exact_amount != 0
        assert cost.low_amount != 0
        assert cost.high_amount != 0


# --------------------------------------------------------------------------- #
# G. CaseState
# --------------------------------------------------------------------------- #
def test_case_ids_unique_and_defaults_isolated() -> None:
    first = CaseState()
    second = CaseState()
    assert first.case_id != second.case_id

    first.product_facts.append(ProductFact(attribute_id="A-CMN-001", value="childrens_toys"))
    assert second.product_facts == []  # mutable defaults are per-instance


def test_case_initial_status_and_created_at() -> None:
    case = CaseState()
    assert case.analysis_status == AnalysisStatus.IN_PROGRESS
    assert case.created_at is not None
    assert case.case_id.startswith("case_")


def test_case_stores_raw_input_and_facts() -> None:
    case = CaseState(raw_product_input="wooden blocks for children")
    case.missing_attribute_ids = ["A-CMN-001", "A-CMN-002"]
    fact = ProductFact(attribute_id="A-CMN-001", value="childrens_toys", origin=FactOrigin.CLASSIFIER)
    case.add_fact(fact)
    assert len(case.product_facts) == 1
    assert case.product_facts[0].origin == FactOrigin.CLASSIFIER
    assert "A-CMN-001" not in case.missing_attribute_ids
    assert "A-CMN-002" in case.missing_attribute_ids


def test_case_json_round_trip() -> None:
    case = CaseState(
        raw_product_input="bluetooth earphones",
        knowledge_snapshot=KnowledgeSnapshot(schema_version="1.0", generated_at="2026-09-09T11:13:35Z"),
    )
    case.add_fact(ProductFact(attribute_id="A-ELEC-001", value="yes"))
    dumped = json.dumps(case.to_json_dict())
    restored = CaseState.model_validate(json.loads(dumped))
    assert restored.case_id == case.case_id
    assert restored.raw_product_input == case.raw_product_input
    assert restored.created_at == case.created_at
    assert restored.product_facts == case.product_facts
    assert restored.knowledge_snapshot == case.knowledge_snapshot
    assert restored.analysis_status == case.analysis_status


# --------------------------------------------------------------------------- #
# H. Failure behavior on temporary (never-real) data
# --------------------------------------------------------------------------- #
def test_missing_data_directory(tmp_path: Path) -> None:
    with pytest.raises(DataDirectoryError):
        JsonComplianceRepository(data_dir=tmp_path / "does_not_exist")


def test_missing_data_file(temp_data: Path) -> None:
    (temp_data / "policy_sources.json").unlink()
    with pytest.raises(DatasetLoadError):
        JsonComplianceRepository(data_dir=temp_data)


def test_invalid_json(temp_data: Path) -> None:
    (temp_data / "compliance_costs.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(DatasetLoadError):
        JsonComplianceRepository(data_dir=temp_data)


def test_wrong_schema_version(temp_data: Path) -> None:
    doc = _read_dataset(temp_data, "product_taxonomy.json")
    doc["schema_version"] = "2.0"
    _write_dataset(temp_data, "product_taxonomy.json", doc)
    with pytest.raises(DatasetValidationError):
        JsonComplianceRepository(data_dir=temp_data)


def test_missing_top_level_key(temp_data: Path) -> None:
    doc = _read_dataset(temp_data, "known_gaps.json")
    del doc["generated_at"]
    _write_dataset(temp_data, "known_gaps.json", doc)
    with pytest.raises(DatasetValidationError):
        JsonComplianceRepository(data_dir=temp_data)


def test_invalid_record_structure(temp_data: Path) -> None:
    doc = _read_dataset(temp_data, "compliance_costs.json")
    doc["records"][0]["calculator_use"] = "NOT_A_REAL_VALUE"
    _write_dataset(temp_data, "compliance_costs.json", doc)
    with pytest.raises(DatasetValidationError):
        JsonComplianceRepository(data_dir=temp_data)


def test_duplicate_ids_fail(temp_data: Path) -> None:
    doc = _read_dataset(temp_data, "product_taxonomy.json")
    doc["records"].append(dict(doc["records"][0]))  # duplicate attribute_id
    _write_dataset(temp_data, "product_taxonomy.json", doc)
    with pytest.raises(DuplicateIDError):
        JsonComplianceRepository(data_dir=temp_data)


def test_broken_rule_to_attribute_reference_fails(temp_data: Path) -> None:
    doc = _read_dataset(temp_data, "compliance_rules.json")
    doc["records"][0]["required_attribute_ids"] = ["A-DOES-NOT-EXIST"]
    _write_dataset(temp_data, "compliance_rules.json", doc)
    with pytest.raises(ReferenceIntegrityError):
        JsonComplianceRepository(data_dir=temp_data)


def test_broken_rule_to_source_reference_fails(temp_data: Path) -> None:
    doc = _read_dataset(temp_data, "compliance_rules.json")
    doc["records"][0]["source_ids"] = ["S-DOES-NOT-EXIST"]
    _write_dataset(temp_data, "compliance_rules.json", doc)
    with pytest.raises(ReferenceIntegrityError):
        JsonComplianceRepository(data_dir=temp_data)


def test_broken_attribute_to_rule_reference_fails(temp_data: Path) -> None:
    doc = _read_dataset(temp_data, "product_taxonomy.json")
    doc["records"][0]["triggered_rule_ids"] = ["R-DOES-NOT-EXIST"]
    _write_dataset(temp_data, "product_taxonomy.json", doc)
    with pytest.raises(ReferenceIntegrityError):
        JsonComplianceRepository(data_dir=temp_data)


def test_broken_cost_source_reference_fails(temp_data: Path) -> None:
    doc = _read_dataset(temp_data, "compliance_costs.json")
    doc["records"][0]["source_id"] = "S-DOES-NOT-EXIST"
    _write_dataset(temp_data, "compliance_costs.json", doc)
    with pytest.raises(ReferenceIntegrityError):
        JsonComplianceRepository(data_dir=temp_data)


def test_broken_cost_additional_source_reference_fails(temp_data: Path) -> None:
    doc = _read_dataset(temp_data, "compliance_costs.json")
    doc["records"][0]["additional_source_ids"] = ["S-DOES-NOT-EXIST"]
    _write_dataset(temp_data, "compliance_costs.json", doc)
    with pytest.raises(ReferenceIntegrityError):
        JsonComplianceRepository(data_dir=temp_data)


def test_errors_carry_context(tmp_path: Path) -> None:
    # Missing-file error names the file.
    first_dir = tmp_path / "first"
    shutil.copytree(DATA_DIR, first_dir)
    (first_dir / "policy_sources.json").unlink()
    with pytest.raises(DatasetLoadError, match="policy_sources.json"):
        JsonComplianceRepository(data_dir=first_dir)

    # Broken-reference error names the reference kind.
    second_dir = tmp_path / "second"
    shutil.copytree(DATA_DIR, second_dir)
    doc = _read_dataset(second_dir, "compliance_rules.json")
    doc["records"][0]["required_attribute_ids"] = ["A-DOES-NOT-EXIST"]
    _write_dataset(second_dir, "compliance_rules.json", doc)
    with pytest.raises(ReferenceIntegrityError, match="required_attribute_id"):
        JsonComplianceRepository(data_dir=second_dir)


# --------------------------------------------------------------------------- #
# I. Hardening: interface conformance
# --------------------------------------------------------------------------- #
def test_repository_satisfies_interface(repo: JsonComplianceRepository) -> None:
    # Structural conformance: the JSON repository is usable as the abstract
    # ComplianceRepository interface (runtime-checkable Protocol).
    assert isinstance(repo, ComplianceRepository)

    interface: ComplianceRepository = repo
    assert interface.get_attribute("A-CMN-001") is not None
    assert interface.get_rule("R-CMN-001") is not None
    assert interface.get_source("S001") is not None
    assert interface.get_cost("C-E-001") is not None
    assert interface.get_known_gaps()
    assert interface.get_pending_policy_updates() == []
    assert interface.knowledge_snapshot() is not None
    interface.validate_references()  # must not raise on the approved baseline


def test_rules_for_category_is_exact_and_does_not_inject_common(
    repo: JsonComplianceRepository,
) -> None:
    # get_rules_for_category must remain an exact-category query: requesting a
    # classified category must NOT silently include `common` rules.
    common_ids = {r.rule_id for r in repo.rules if r.category == "common"}
    assert common_ids  # the approved baseline does contain common rules

    for category in ("childrens_toys", "small_consumer_electronics"):
        returned = repo.get_rules_for_category(category)
        returned_ids = {r.rule_id for r in returned}
        assert returned_ids.isdisjoint(common_ids)
        assert all(r.category == category for r in returned)

    # Querying "common" returns only common rules.
    common_returned = repo.get_rules_for_category("common")
    assert common_returned
    assert {r.rule_id for r in common_returned} == common_ids


# --------------------------------------------------------------------------- #
# J. Hardening: invalid UTF-8 handling
# --------------------------------------------------------------------------- #
def test_invalid_utf8_raises_dataset_load_error(temp_data: Path) -> None:
    # Overwrite one dataset with bytes that are not valid UTF-8. The repository
    # must fail with DatasetLoadError naming the file, chained from the
    # underlying UnicodeDecodeError - never leak the raw decode exception.
    target = temp_data / "product_taxonomy.json"
    target.write_bytes(b'\xff\xfe\x00{"schema_version": "1.0"')

    with pytest.raises(DatasetLoadError, match="product_taxonomy.json") as excinfo:
        JsonComplianceRepository(data_dir=temp_data)

    cause = excinfo.value.__cause__
    assert isinstance(cause, UnicodeDecodeError)
