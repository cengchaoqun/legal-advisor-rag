"""使用标准库校验设计契约；需额外安装 jsonschema、openapi-spec-validator。

不联网、不启动应用、不产生 API/GPU 费用。外部文件引用先由 check_design
核验，再绑定到内存 registry；OpenAPI 的核心对象引用转换为等价内部引用，
避免标准工具通过核心 Schema 的声明性 $id 发起网络请求。
"""

import copy
import json

from jsonschema import Draft202012Validator, FormatChecker
from openapi_spec_validator import validate as validate_openapi
from referencing import Registry, Resource

from check_design import CONTRACTS, check_contract_closure, load, walk_refs


def main():
    core_path = CONTRACTS / "核心对象.schema.json"
    api_path = CONTRACTS / "openapi.json"
    core = load(core_path)
    api = load(api_path)
    examples = load(CONTRACTS / "契约样例.json")
    for path in CONTRACTS.glob("*.json"):
        walk_refs(load(path), path)
    Draft202012Validator.check_schema(core)
    resource = Resource.from_contents(core)
    registry = Registry().with_resources([(core["$id"], resource), (core_path.as_uri(), resource)])
    for example in examples:
        schema = {"$ref": core["$id"] + "#/$defs/" + example["schema"]}
        Draft202012Validator(schema, registry=registry, format_checker=FormatChecker()).validate(example["value"])
    check_contract_closure(api, core, examples, core_path)

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
    bundled["components"]["schemas"].update({"Core_" + name: bind(schema) for name, schema in core["$defs"].items()})
    validate_openapi(bundled)
    print(json.dumps({"json_schema": "draft2020-12", "format_checker": True, "validated_examples": len(examples), "openapi": api["openapi"], "api_operations": sum(len(methods) for methods in api["paths"].values()), "result": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
