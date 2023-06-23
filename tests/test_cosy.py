"""Unittests for Cosypose."""

import unittest

from happypose.pose_estimators.cosypose.placeholder import cosy_placeholder


class TestCosy(unittest.TestCase):
    """Test cosypose."""

    def test_mock(self):
        """Test the placeholder function."""
        self.assertEqual(cosy_placeholder(), 42)


if __name__ == "__main__":
    unittest.main()
