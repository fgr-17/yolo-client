import pytest
import unittest
from mock import patch

import package


class TestPackage(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_pass(self):
        self.assertEqual(0, 0)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestPackage('test_input_account_data'))
    return suite


if __name__ == '__main__':
    # unittest.main()
    runner = unittest.TextTestRunner()
    runner.run(suite())
