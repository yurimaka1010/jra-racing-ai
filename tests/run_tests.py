"""Run synthetic tests with all Python socket connections blocked."""
import socket
import importlib.abc
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class BlockRealDatabaseDriver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "psycopg" or fullname.startswith("psycopg."):
            raise AssertionError("real_database_driver_forbidden_in_tests")
        return None


def main():
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    guard = BlockRealDatabaseDriver()
    sys.meta_path.insert(0, guard)
    try:
        with patch.object(socket, "socket", side_effect=AssertionError("network_forbidden_in_tests")), \
                patch.object(socket, "create_connection", side_effect=AssertionError("network_forbidden_in_tests")):
            suite = unittest.defaultTestLoader.discover(str(project / "tests"), pattern="test_*.py")
            result = unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        sys.meta_path.remove(guard)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
