"""契约测试：标准 Draft 2020-12 与 OpenAPI 3.1 校验，并检查示例覆盖与状态码精度。

对应 T30（离线/在线证据身份）与 T33（全定义样例、错误复用、health 状态码）。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from openapi_spec_validator import validate as validate_openapi
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "docs" / "contracts"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@pytest.fixture(scope="module")
def core():
    return load(CONTRACTS / "核心对象.schema.json")


@pytest.fixture(scope="module")
def api():
    return load(CONTRACTS / "openapi.json")


@pytest.fixture(scope="module")
def examples():
    return load(CONTRACTS / "契约样例.json")


def test_core_schema_is_valid_draft_2020_12(core):
    Draft202012Validator.check_schema(core)


def test_all_examples_validate_with_format_checker(core, examples):
    resource = Resource.from_contents(core)
    registry = Registry().with_resources([(core["$id"], resource)])
    for example in examples:
        schema = {"$ref": core["$id"] + "#/$defs/" + example["schema"]}
        Draft202012Validator(schema, registry=registry, format_checker=FormatChecker()).validate(example["value"])


def test_every_definition_has_direct_example(core, examples):
    covered = {example["schema"] for example in examples}
    assert set(core["$defs"]) == covered


def test_openapi_bundled_is_valid(api, core):
    def bind(value):
        if isinstance(value, dict):
            result = {key: bind(item) for key, item in value.items()}
            ref = result.get("$ref", "")
            for prefix in ("./核心对象.schema.json#/$defs/", "#/$defs/"):
                if ref.startswith(prefix):
                    result["$ref"] = "#/components/schemas/Core_" + ref[len(prefix):]
                    break
            return result
        if isinstance(value, list):
            return [bind(item) for item in value]
        return value

    bundled = bind(copy.deepcopy(api))
    bundled["components"]["schemas"].update({f"Core_{name}": bind(schema) for name, schema in core["$defs"].items()})
    validate_openapi(bundled)


def test_health_status_codes_are_minimal(api):
    assert set(api["paths"]["/health/live"]["get"]["responses"]) == {"200"}
    assert set(api["paths"]["/health/ready"]["get"]["responses"]) == {"200", "503"}


def test_upload_only_returns_413_and_never_on_health(api):
    assert "413" in api["paths"]["/api/documents"]["post"]["responses"]
    for path in ("/health/live", "/health/ready"):
        assert "413" not in api["paths"][path]["get"]["responses"]
        assert "429" not in api["paths"][path]["get"]["responses"]


def test_common_errors_are_reused_from_components(api):
    responses = api["components"]["responses"]
    for name in ("BadRequest", "NotFound", "Conflict", "Gone", "PayloadTooLarge", "ValidationError", "LimitExceeded", "Unavailable"):
        assert name in responses
        assert responses[name]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorEnvelope")


def test_document_endpoints_return_document_list_item(api):
    detail = api["paths"]["/api/documents/{document_id}"]["get"]["responses"]["200"]
    assert detail["content"]["application/json"]["schema"]["$ref"].endswith("DocumentListItem")
    listing = api["paths"]["/api/documents"]["get"]["responses"]["200"]
    items = listing["content"]["application/json"]["schema"]["properties"]["items"]
    assert items["items"]["$ref"].endswith("DocumentListItem")


def test_graph_relations_include_nodes_and_support_chunks(api):
    schema = api["paths"]["/api/graph/relations"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert {"paths", "warnings", "nodes", "support_chunks"} <= set(schema["required"])


def test_events_endpoint_declares_event_types(api):
    operation = api["paths"]["/api/queries/{query_id}/events"]["get"]
    assert operation["x-event-types"] == [
        "started",
        "stage",
        "evidence",
        "delta",
        "warning",
        "done",
        "failed",
        "cancelled",
    ]
    assert "410" in operation["responses"]


def test_no_experiment_or_report_endpoints(api):
    paths = " ".join(api["paths"])
    assert "experiment" not in paths
    assert "report" not in paths
