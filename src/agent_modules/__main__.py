"""Allows `python -m agent_modules`, which is the shortest correct way to run this package.

Running a module file by path (`python src/agent_modules/cli.py`) cannot work: the file becomes
`__main__` with no parent package, so its relative imports fail, and the directory it lives in is
placed on `sys.path` where it can shadow standard-library modules. Both problems disappear when the
package is imported properly, which is what this file makes convenient.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
