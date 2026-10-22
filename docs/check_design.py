"""离线检查本设计的链接、引用、默认值与契约样例，无网络和付费调用。

样例校验覆盖当前契约所用关键字；本脚本不是通用 JSON Schema 认证器。
应用开发仍应使用标准校验库与 OpenAPI 工具检查实际 API。
"""

import datetime as dt
import copy
import json
import math
import re
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "docs" / "contracts"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve(ref, base):
    name, _, fragment = ref.partition("#")
    target = (base.parent / unquote(name)).resolve() if name else base
    if not target.is_relative_to(CONTRACTS.resolve()):
        raise AssertionError(f"契约引用越界：{ref}")
    value = load(target)
    for token in fragment.strip("/").split("/") if fragment.strip("/") else []:
        value = value[token.replace("~1", "/").replace("~0", "~")]
    return value, target


def walk_refs(value, base):
    if isinstance(value, dict):
        if "$ref" in value:
            resolve(value["$ref"], base)
        for item in value.values():
            walk_refs(item, base)
    elif isinstance(value, list):
        for item in value:
            walk_refs(item, base)


def matches_type(value, kind):
    return {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }[kind]


def validate(value, schema, base, where="样例"):
    if "$ref" in schema:
        target, target_path = resolve(schema["$ref"], base)
        return validate(value, target, target_path, where)
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                validate(value, branch, base, where)
                return
            except (AssertionError, ValueError):
                pass
        raise AssertionError(f"{where} 不符合 anyOf")
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        assert any(matches_type(value, item) for item in types), f"{where} 类型错误"
    if "enum" in schema:
        assert value in schema["enum"], f"{where} 枚举错误"
    if "const" in schema:
        assert value == schema["const"], f"{where} 常量错误"
    if isinstance(value, dict):
        assert all(key in value for key in schema.get("required", [])), f"{where} 缺少字段"
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], base, f"{where}.{key}")
            else:
                additional = schema.get("additionalProperties", True)
                assert additional is not False, f"{where} 未定义字段 {key}"
                if isinstance(additional, dict):
                    validate(item, additional, base, f"{where}.{key}")
    if isinstance(value, list):
        assert len(value) >= schema.get("minItems", 0), f"{where} 元素过少"
        assert len(value) <= schema.get("maxItems", float("inf")), f"{where} 元素过多"
        if schema.get("uniqueItems"):
            assert len({json.dumps(item, sort_keys=True) for item in value}) == len(value)
        for index, item in enumerate(value):
            validate(item, schema.get("items", {}), base, f"{where}[{index}]")
    if isinstance(value, str):
        assert len(value) >= schema.get("minLength", 0)
        assert len(value) <= schema.get("maxLength", float("inf"))
        if "pattern" in schema:
            assert re.search(schema["pattern"], value), f"{where} 格式错误"
        if schema.get("format") == "uuid":
            uuid.UUID(value)
        if schema.get("format") == "date":
            dt.date.fromisoformat(value)
        if schema.get("format") == "date-time":
            assert dt.datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
        if schema.get("format") == "uri":
            assert urlparse(value).scheme, f"{where} URL 缺少协议"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        assert math.isfinite(value), f"{where} 非有限数值"
        assert value >= schema.get("minimum", -float("inf"))
        assert value <= schema.get("maximum", float("inf"))


def check_contract_closure(api, core, examples, core_path):
    """检查本次修订暴露的跨对象约束，防止仅靠合法 JSON 掩盖语义冲突。"""
    definitions = core["$defs"]
    covered = {item["schema"] for item in examples}
    assert covered == set(definitions), f"样例覆盖不完整：{set(definitions) - covered}"
    assert all(len(item.get("required", [])) == len(set(item.get("required", []))) for item in definitions.values())
    assert "query_id" not in definitions["SampleEvidence"]["properties"]
    assert "query_id" in definitions["Evidence"]["required"]
    assert definitions["DatasetSample"]["properties"]["evidence"]["items"]["$ref"].endswith("/SampleEvidence")
    assert {"pipeline_revision", "published_at"} <= set(definitions["Version"]["required"])
    assert {"facts", "reasoning_code", "path_status"} <= set(definitions["QueryStatus"]["required"])
    assert "facts" in definitions["Message"]["required"]
    for example in examples:
        name, value = example["schema"], example["value"]
        if name == "DocumentListItem":
            document, version = value["document"], value["version"]
            assert (version["version_id"] if version else None) == document["current_version_id"]
            assert value["availability"] == ("unavailable" if document["tombstoned"] or not version else version["availability"])
            if version:
                assert version["document_id"] == document["document_id"]
        if name in {"DatasetSample", "QueryStatus"}:
            evidence = value["evidence"]
            assert len({item["evidence_id"] for item in evidence}) == len(evidence)
            if name == "QueryStatus":
                assert all(item["query_id"] == value["query_id"] for item in evidence)
                if value["status"] not in {"queued", "running"}:
                    assert not {"pending", "running"}.intersection(value["path_status"].values())
                if value["behavior"] == "clarify":
                    assert value["reasoning_code"] == "FACTS_MISSING"
        if name == "Path":
            assert len(value["relation_types"]) == len(value["node_ids"]) - 1

    paths = api["paths"]
    listing = paths["/api/documents"]["get"]
    item_ref = listing["responses"]["200"]["content"]["application/json"]["schema"]["properties"]["items"]["items"]["$ref"]
    assert item_ref.endswith("/DocumentListItem")
    params = {item["name"]: item for item in listing["parameters"]}
    assert {"q", "doc_type", "availability"} <= set(params)
    assert params["availability"]["schema"]["enum"] == definitions["Version"]["properties"]["availability"]["enum"]
    assert set(paths["/health/live"]["get"]["responses"]) == {"200"}
    assert set(paths["/health/ready"]["get"]["responses"]) == {"200", "503"}
    assert {"400", "410", "422", "503"} <= set(paths["/api/queries/{query_id}/events"]["get"]["responses"])
    assert "/api/graph/entities" in paths
    graph = paths["/api/graph/relations"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert {"paths", "nodes", "support_chunks", "warnings"} <= set(graph["required"])
    for path, methods in paths.items():
        for method, operation in methods.items():
            if "413" in operation["responses"]:
                assert method == "post" and path in {"/api/documents", "/api/sessions/{session_id}/attachments"}
            for code, response in operation["responses"].items():
                if code.startswith(("4", "5")):
                    assert response["$ref"].startswith("#/components/responses/")

    # 这些反例曾被旧契约遗漏，必须验证会被拒绝。
    by = {item["schema"]: item["value"] for item in examples}
    bad_sample = copy.deepcopy(by["SampleEvidence"])
    bad_sample["query_id"] = by["Evidence"]["query_id"]
    bad_evidence = copy.deepcopy(by["Evidence"])
    del bad_evidence["query_id"]
    bad_version = copy.deepcopy(by["Version"])
    del bad_version["pipeline_revision"]
    bad_query = copy.deepcopy(by["QueryStatus"])
    bad_query["path_status"]["graph"] = "假装成功"
    for name, value in [("SampleEvidence", bad_sample), ("Evidence", bad_evidence), ("Version", bad_version), ("QueryStatus", bad_query)]:
        try:
            validate(value, definitions[name], core_path, f"反例.{name}")
        except (AssertionError, ValueError):
            continue
        raise AssertionError(f"反例未被拒绝：{name}")
    return len(covered), 4


def main():
    json_paths = list(CONTRACTS.glob("*.json"))
    for path in json_paths:
        walk_refs(load(path), path)
    api = load(CONTRACTS / "openapi.json")
    operation_ids = set()
    count = 0
    for path, methods in api["paths"].items():
        for method, operation in methods.items():
            count += 1
            assert operation["operationId"] not in operation_ids
            operation_ids.add(operation["operationId"])
            defined = {p["name"] for p in operation.get("parameters", []) if p["in"] == "path"}
            assert defined == set(re.findall(r"\{([^}]+)\}", path)), (method, path)
    core_path = CONTRACTS / "核心对象.schema.json"
    core = load(core_path)
    examples = load(CONTRACTS / "契约样例.json")
    for example in examples:
        validate(example["value"], core["$defs"][example["schema"]], core_path, example["schema"])
    coverage, rejected = check_contract_closure(api, core, examples, core_path)
    config = load(CONTRACTS / "默认配置.json")
    budget = config["budget"]
    assert budget["total_limit"] == 300
    assert budget["api_limit"] + budget["gpu_limit"] + budget["reserve_limit"] == 300
    assert config["project"]["human_review"] is False
    assert config["dataset"]["human_annotation"] is False
    assert config["project"]["web_default"] is False
    assert config["training"]["program_integration_required"] is True
    assert config["serving"]["frontend_management"] is False
    assert config["serving"]["automatic_cloud_fallback"] is False
    assert config["retrieval"]["graph_max_hops"] == 2
    assert config["models"]["embedding_dimension"] == 1024
    assert not any(path.startswith(("/api/reports", "/api/model-experiments")) for path in api["paths"])
    paths = [ROOT / "README.md", ROOT / "掌柜智库_概要设计总纲.md"]
    paths += list((ROOT / "docs").glob("*.md")) + [ROOT / "designs" / "界面设计说明.md"]
    links = 0
    for path in paths:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8-sig")):
            if target.startswith(("http://", "https://", "#")):
                continue
            assert (path.parent / unquote(target.split("#", 1)[0])).exists(), (path, target)
            links += 1
    pen = load(ROOT / "designs" / "法律咨询工作台_低保真.pen")
    assert len(pen["children"]) == 5
    assert all(n.get("width") == 1440 and n.get("height") == 960 and not n.get("placeholder") for n in pen["children"])
    print(json.dumps({"json_files": len(json_paths), "core_definitions": len(core["$defs"]), "api_operations": count, "validated_examples": len(examples), "covered_definitions": coverage, "rejected_regressions": rejected, "local_links": links, "pen_screens": 5, "result": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
