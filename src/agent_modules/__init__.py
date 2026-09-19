"""Small, independently testable modules for driving a job application in a real browser.

A module in this package may import only from a lower layer:

    L6 cli              -> L5
    L5 orchestrator     -> L4, L3, L2, L1, L0
    L4 planners         -> L3, L2, L1, L0
    L3 clients          -> L0
    L2 builders/browser -> L1, L0
    L1 pure logic       -> L0
    L0 leaves           -> nothing

`tests/test_layering.py` enforces this.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
