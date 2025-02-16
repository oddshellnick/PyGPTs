from unittest import TextTestRunner
from unit_tests import main_test_suite


if __name__ == "__main__":
	runner = TextTestRunner()
	runner.run(main_test_suite())
