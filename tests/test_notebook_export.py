from prepare_notebook_exports import normalize_future_imports


def test_later_cell_future_import_moves_without_changing_embedded_source():
    source = '''"""Notebook title."""
setting = 1
from __future__ import annotations
embedded = "from __future__ import annotations\\nprint('child script')"
def identity(value: MissingRuntimeType) -> MissingRuntimeType:
    return value
'''
    fixed = normalize_future_imports(source)
    namespace = {}
    exec(compile(fixed, "export.py", "exec"), namespace)
    assert namespace["__doc__"] == "Notebook title."
    assert namespace["setting"] == 1
    assert namespace["embedded"] == "from __future__ import annotations\nprint('child script')"
    assert namespace["identity"](7) == 7
    assert normalize_future_imports(fixed) == fixed
