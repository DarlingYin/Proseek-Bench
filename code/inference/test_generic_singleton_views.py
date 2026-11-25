"""Generic interface coverage for Apps with multiple singleton views."""

import unittest

from generic_tool_adapter import GenericToolAdapter


def catalogs(*, extended: bool):
    source = {
        "source_id": "demo",
        "label": "Demo",
        "description": "Demo account state.",
    }
    internal = [
        {
            "tool_id": "demo.get_user_state",
            "operation": "singleton",
            "entity_types": ["demo.user_state"],
            "parameters": {},
            "result_fields": ["record_id", "profile_name"],
            "cost_units": 1,
        }
    ]
    public = {
        "demo.get_my_state": {
            "tool_id": "demo.get_my_state",
            "description": "Read account state.",
            "source": source,
            "role": "singleton",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
                "additionalProperties": False,
            },
            "cost_units": 1,
            "runtime_binding": {
                "source_id": "demo",
                "entity_type": "demo.user_state",
                "internal_tool_id": "demo.get_user_state",
                "internal_id_field": "record_id",
            },
        }
    }
    if extended:
        internal.append({
            "tool_id": "demo.get_console",
            "operation": "singleton",
            "entity_types": ["demo.console"],
            "parameters": {},
            "result_fields": ["semantic_ref", "status"],
            "cost_units": 1,
        })
        public["demo.get_my_console"] = {
            "tool_id": "demo.get_my_console",
            "description": "Read console state.",
            "source": source,
            "role": "singleton",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
                "additionalProperties": False,
            },
            "cost_units": 1,
            "runtime_binding": {
                "source_id": "demo",
                "entity_type": "demo.console",
                "internal_tool_id": "demo.get_console",
                "internal_id_field": "semantic_ref",
            },
        }
    semantic = {"sources": {"demo": {"tools": internal}}}
    capability = {"tools": public}
    return semantic, capability


class GenericSingletonViewTests(unittest.TestCase):
    def test_singleton_only_catalog_keeps_original_contract(self):
        adapter = GenericToolAdapter(*catalogs(extended=False))
        schema = adapter.public_catalog()["tools"]["read_user_state"]["parameters"]
        self.assertEqual(set(schema["properties"]), {"app"})
        self.assertEqual(
            adapter.resolve("read_user_state", {"app": "demo"}, {}),
            ("demo.get_my_state", {}),
        )

    def test_extended_singletons_add_an_explicit_view_without_changing_default(self):
        adapter = GenericToolAdapter(*catalogs(extended=True))
        catalog = adapter.public_catalog()
        schema = catalog["tools"]["read_user_state"]["parameters"]
        self.assertEqual(set(schema["properties"]), {"app", "view"})
        self.assertIn("console", catalog["user_states"]["demo"]["views"])
        self.assertEqual(
            adapter.resolve("read_user_state", {"app": "demo"}, {}),
            ("demo.get_my_state", {}),
        )
        self.assertEqual(
            adapter.resolve(
                "read_user_state", {"app": "demo", "view": "console"}, {}
            ),
            ("demo.get_my_console", {}),
        )
        self.assertEqual(
            adapter.from_legacy("demo.get_my_console", {}),
            ("read_user_state", {"app": "demo", "view": "console"}),
        )


if __name__ == "__main__":
    unittest.main()
