# Root conftest.py: prevent pytest from collecting the package __init__.py
# (which uses relative imports and cannot be imported as a top-level module).
collect_ignore = ["__init__.py"]
