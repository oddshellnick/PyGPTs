import unittest
from parameterized import parameterized
from PyGPTs.Gemini.functions import find_base_model


class TestFindBaseModel(unittest.TestCase):
	@parameterized.expand(
			[
				("gemini-1.0-pro", "gemini-1.0-pro"),
				("gemini-1.5-pro-001", "gemini-1.5-pro"),
				("gemini-2.0-flash-latest", "gemini-2.0-flash"),
				("gemini-2.0-pro-exp", "gemini-2.0-pro"),
				("gemini-2.0-flash-lite", "gemini-2.0-flash-lite"),
				("gemini-2.0-flash-thinking", "gemini-2.0-flash-thinking"),
				("gemini-1.5-flash-8b", "gemini-1.5-flash-8b"),
				("gemini-1.5-flash-8b-001", "gemini-1.5-flash-8b"),
				("gemini-1.5-flash-8b-it-001", "gemini-1.5-flash-8b-it"),
				("some-invalid-model-name", None),
				("invalid-1.0", None),
				("", None),
				("gemini-1.5-pro-extra-hyphens", "gemini-1.5-pro"),
				("gemini-2.0-flash-lite-02-05", "gemini-2.0-flash-lite"),
			]
	)
	def test_base_model_name(self, model_version, expected_base_model):
		"""Test find_base_model with various model name formats."""
		actual_base_model = find_base_model(model_version)
		self.assertEqual(actual_base_model, expected_base_model)


def functions_test_suite():
	suite = unittest.TestSuite()
	test_loader = unittest.TestLoader()
	
	suite.addTest(test_loader.loadTestsFromTestCase(TestFindBaseModel))
	
	return suite
