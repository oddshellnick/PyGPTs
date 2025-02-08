import unittest
from unit_tests.Gemini import gemini_test_suite


def main_test_suite():
	suite = unittest.TestSuite()
	
	suite.addTest(gemini_test_suite())
	
	return suite
