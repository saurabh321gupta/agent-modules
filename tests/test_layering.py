"""Enforces the layering rule stated in `agent_modules/__init__.py`.

A module may import only from a lower layer. That is the property which makes each module testable
alone: nothing above it can leak into its tests. Without a test, the rule decays within a week.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "agent_modules"

#: Every module and its layer. Adding a module without adding it here fails the suite, which is the
#: point: the layer is a design decision, not an accident.
LAYERS: dict[str, int] = {
    # L0 - leaves
    "models": 0,
    "config": 0,
    "journey": 0,
    "timing": 0,
    # L1 - pure logic over L0
    "files": 1,
    "policy": 1,
    "validator": 1,
    "verifier": 1,
    "extractor": 1,
    # L2 - prompt builders and browser-bound modules
    "prompts_llm": 2,
    "prompts_jev": 2,
    "reader": 2,
    "overlays": 2,
    "executor": 2,
    # L3 - transport
    "llm_client": 3,
    "jev_client": 3,
    # L4 - planners
    "normalizer": 4,
    "planner_llm": 4,
    "planner_jev": 4,
    "planner_staged": 4,
    # L5 - orchestration
    "orchestrator": 5,
    # L6 - entry points
    "cli": 6,
    "__main__": 6,
    "profile_builder": 6,
}


def module_paths() -> list[Path]:
    return sorted(p for p in PACKAGE.glob("*.py") if p.name != "__init__.py")


def internal_imports(path: Path) -> set[str]:
    """The sibling modules this file imports, via relative or absolute form."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                if node.level:
                    found.add(node.module.split(".")[0])
                elif node.module.startswith("agent_modules."):
                    found.add(node.module.split(".", 1)[1].split(".")[0])
            else:
                found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("agent_modules."):
                    found.add(alias.name.split(".", 1)[1].split(".")[0])
    return found


def test_every_module_declares_a_layer():
    undeclared = sorted(p.stem for p in module_paths() if p.stem not in LAYERS)
    assert not undeclared, (
        f"these modules have no declared layer: {undeclared}. "
        "Add them to LAYERS in tests/test_layering.py."
    )


def test_no_module_imports_upward():
    violations: list[str] = []
    for path in module_paths():
        own = LAYERS.get(path.stem)
        if own is None:
            continue
        for imported in sorted(internal_imports(path)):
            other = LAYERS.get(imported)
            if other is None or other <= own:
                continue
            violations.append(f"{path.name} (L{own}) imports {imported} (L{other})")
    assert not violations, "upward imports:\n" + "\n".join(violations)


def test_layers_are_not_skipped_by_the_map_itself():
    """A layer with no modules is dead weight and a sign the map has drifted from the code."""
    numbers = sorted(set(LAYERS.values()))
    assert numbers == list(range(len(numbers))), f"layer numbering has a gap: {numbers}"


def test_no_module_shadows_a_standard_library_module():
    """A module named `types` broke `functools`, and the error pointed at the stdlib, not at us.

    Running a file inside a package by path puts that directory first on `sys.path`, at which point a
    module sharing a standard-library name is imported instead of the real one. Nothing about the
    failure names the culprit, so the name is what has to change.
    """
    import sys

    clashes = sorted(
        path.stem for path in module_paths() if path.stem in sys.stdlib_module_names
    )
    assert not clashes, f"these module names shadow the standard library: {clashes}"


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
