import unittest
from unit_tests.Gemini.data import data_test_suite
from unit_tests.Gemini.main import main_test_suite
from unit_tests.Gemini.functions import functions_test_suite


def gemini_test_suite():
	suite = unittest.TestSuite()
	
	suite.addTest(main_test_suite())
	suite.addTest(data_test_suite())
	suite.addTest(functions_test_suite())
	
	return suite
