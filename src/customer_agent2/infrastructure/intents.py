"""Strict loader for the packaged M4-C intent tree."""

import json
from hashlib import sha256
from importlib.resources import files
from typing import cast

from customer_agent2.domain.models import IntentDefinition, IntentRoute, IntentTree


def load_default_intent_tree() -> IntentTree:
    """Load and validate the versioned tree included in the installed package."""
    resource = files("customer_agent2.config").joinpath("intent_tree.json")
    return load_intent_tree_json(resource.read_text(encoding="utf-8"))


def load_intent_tree_json(content: str) -> IntentTree:
    """Parse an exact JSON object without passing untyped mappings across layers."""
    raw: object = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("Intent Tree 必须是 JSON 对象")
    tree = cast(dict[str, object], raw)
    if set(tree) != {"version", "routes"}:
        raise ValueError("Intent Tree 顶层字段无效")
    version = tree["version"]
    raw_routes = tree["routes"]
    if not isinstance(version, str) or not isinstance(raw_routes, list):
        raise TypeError("Intent Tree 顶层类型无效")

    definitions: list[IntentDefinition] = []
    for raw_route in cast(list[object], raw_routes):
        if not isinstance(raw_route, dict):
            raise TypeError("Intent Tree 路由必须是对象")
        route = cast(dict[str, object], raw_route)
        if not {"name", "description"} <= set(route) or set(route) - {
            "name",
            "description",
            "knowledge_base_slugs",
        }:
            raise ValueError("Intent Tree 路由字段无效")
        name = route["name"]
        description = route["description"]
        raw_slugs = route.get("knowledge_base_slugs", [])
        if not isinstance(name, str) or not isinstance(description, str):
            raise TypeError("Intent Tree 路由类型无效")
        if not isinstance(raw_slugs, list) or any(
            not isinstance(slug, str) for slug in cast(list[object], raw_slugs)
        ):
            raise TypeError("Intent Tree knowledge_base_slugs 类型无效")
        try:
            intent_route = IntentRoute(name)
        except ValueError:
            raise ValueError("Intent Tree 包含未知路由") from None
        definitions.append(
            IntentDefinition(
                intent_route,
                description,
                tuple(cast(list[str], raw_slugs)),
            )
        )
    return IntentTree(version, tuple(definitions))


def intent_tree_fingerprint(intent_tree: IntentTree) -> str:
    """Hash normalized tree semantics so checkpoints reject edited candidates."""
    normalized = {
        "version": intent_tree.version,
        "routes": [
            {
                "name": definition.route.value,
                "description": definition.description,
                "knowledge_base_slugs": list(definition.knowledge_base_slugs),
            }
            for definition in intent_tree.definitions
        ],
    }
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()
