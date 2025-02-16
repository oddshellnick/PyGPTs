from unit_tests.Gemini import gemini_test_suite
from unittest import (
	TestSuite,
	TextTestRunner
)


def main_test_suite() -> TestSuite:
	suite = TestSuite()
	
	suite.addTest(gemini_test_suite())
	
	return suite


if __name__ == "__main__":
	runner = TextTestRunner()
	runner.run(main_test_suite())
