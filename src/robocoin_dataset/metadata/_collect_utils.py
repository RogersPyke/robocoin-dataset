"""
Metadata field resolution and context building utilities.

Purpose:
    Core logic for resolving all schema fields from distributed dataset sources
    (local_dataset_info.yaml, meta/info.json, annotations/, etc.).

    This module is the public entry point for the Collect stage and is used
    exclusively by metadata/collect.py.  The actual logic lives in the private
    sub-modules below; this file re-exports the single public symbol needed by
    callers so that the import surface stays stable.

Sub-modules:
    _yaml_io           — YAML / JSON / JSONL file loading primitives
    _feature_extractors — feature-dict parsing (sensors, cameras, depth, units)
    _auto_fields       — auto-field computation and template-compat transforms
    _schema_resolver   — master field-resolution loop (public API)
"""

from robocoin_dataset.metadata._schema_resolver import resolve_context_from_schema

__all__ = ["resolve_context_from_schema"]
