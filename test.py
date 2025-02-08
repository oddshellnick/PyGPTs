import unittest
from unit_tests import main_test_suite


if __name__ == "__main__":
	runner = unittest.TextTestRunner()
	runner.run(main_test_suite())
