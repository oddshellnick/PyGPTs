import unittest
from PyGPTs.Gemini.functions import find_base_model
from PyGPTs.Gemini.data import (
	GeminiLimits,
	GeminiModels
)
from PyVarTools.python_instances_tools import get_class_attributes


class TestGeminiLimitsIntegration(unittest.TestCase):
	def test_gemini_limits_completeness(self):
		"""
		Test that all base Gemini models defined in GeminiModels are also present in GeminiLimits.
		This ensures that rate limits and context limits are defined for every model.
		"""
		limit_attributes = [
			name for name in get_class_attributes(GeminiLimits, start_exclude="__", end_exclude="__")
		]
		
		all_models = []
		for model_group_field in get_class_attributes(GeminiModels, start_exclude="__", end_exclude="__"):
			model_group = getattr(GeminiModels, model_group_field)
		
			for model_field in get_class_attributes(model_group, start_exclude="__", end_exclude="__"):
				model_name = getattr(model_group, model_field)
				all_models.append(model_name)
		
		for model_name in all_models:
			base_model_name = find_base_model(model_name)
		
			if base_model_name:
				for limit_attr_name in limit_attributes:
					limit_dict = getattr(GeminiLimits, limit_attr_name)
		
					self.assertIn(
							base_model_name,
							limit_dict,
							f"Model '{base_model_name}' (derived from '{model_name}') is missing in GeminiLimits.{limit_attr_name}"
					)
			else:
				self.fail(f"Could not extract base model name from '{model_name}' using regex.")


def data_test_suite():
	suite = unittest.TestSuite()
	test_loader = unittest.TestLoader()
	
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiLimitsIntegration))
	
	return suite
