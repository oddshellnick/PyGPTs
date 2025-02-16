from unit_tests.Gemini.data import data_test_suite
from unit_tests.Gemini.chat import chat_test_suite
from unittest import (
	TestSuite,
	TextTestRunner
)
from unit_tests.Gemini.model import model_test_suite
from unit_tests.Gemini.client import client_test_suite
from unit_tests.Gemini.limiter import limiter_test_suite
from unit_tests.Gemini.functions import functions_test_suite
from unit_tests.Gemini.clients_manager import clients_manager_test_suite


def gemini_test_suite() -> TestSuite:
	suite = TestSuite()
	
	suite.addTest(data_test_suite())
	suite.addTest(functions_test_suite())
	suite.addTest(limiter_test_suite())
	suite.addTest(model_test_suite())
	suite.addTest(chat_test_suite())
	suite.addTest(client_test_suite())
	suite.addTest(clients_manager_test_suite())
	
	return suite


if __name__ == "__main__":
	runner = TextTestRunner()
	runner.run(gemini_test_suite())
