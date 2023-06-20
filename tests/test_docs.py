"""Run doctests.

ref. https://docs.python.org/3/library/doctest.html#unittest-api
"""

import doctest
import unittest

import happypose


def load_tests(loader, tests, ignore):
    """Load docstrings and find tests there."""
    tests.addTests(doctest.DocTestSuite(happypose.pose_estimators.cosypose.placeholder))
    return tests


if __name__ == "__main__":
    unittest.main()
