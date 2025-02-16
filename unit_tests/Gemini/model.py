from PyGPTs.Gemini.model import (
	GeminiModel,
	GeminiModelSettings
)
from PyGPTs.Gemini.limiter import (
	GeminiLimiter,
	GeminiLimiterSettings
)
from unittest import (
	TestCase,
	TestLoader,
	TestSuite,
	TextTestRunner
)
from PyGPTs.Gemini.data import (
	GeminiLimits,
	GeminiMimeTypes,
	GeminiModels
)
from google.genai.types import (
	CountTokensConfigDict,
	GenerateContentConfigDict,
	GenerationConfigDict,
	HarmBlockThreshold,
	HarmCategory
)


class TestGeminiModel(TestCase):
	def setUp(self):
		self.model_settings = GeminiModelSettings(
				model_name=GeminiModels.Gemini_1_5_pro.latest_stable,
				limiter_settings=GeminiLimiterSettings(request_per_day_limit=20, request_per_minute_limit=5)
		)
		self.gemini_model = GeminiModel(self.model_settings)
	
	def test_init(self):
		self.assertEqual(self.gemini_model.model_name, self.model_settings.model_name)
		self.assertEqual(
				self.gemini_model.generation_config,
				self.model_settings.generation_config
		)
		self.assertEqual(
				self.gemini_model.count_tokens_config,
				self.model_settings.count_tokens_config
		)
		self.assertIsInstance(self.gemini_model, GeminiLimiter)
		self.assertEqual(
				self.gemini_model.request_per_day_limit,
				self.model_settings.request_per_day_limit
		)
	
	def test_model_settings_getter(self):
		retrieved_settings = self.gemini_model.model_settings
		
		self.assertIsInstance(retrieved_settings, GeminiModelSettings)
		self.assertEqual(retrieved_settings.model_name, self.gemini_model.model_name)
		self.assertEqual(
				retrieved_settings.generation_config,
				self.gemini_model.generation_config
		)
		self.assertEqual(
				retrieved_settings.count_tokens_config,
				self.gemini_model.count_tokens_config
		)
		self.assertEqual(
				retrieved_settings.request_per_day_used,
				self.gemini_model.request_per_day_used
		)
		self.assertEqual(
				retrieved_settings.request_per_day_limit,
				self.gemini_model.request_per_day_limit
		)
		self.assertEqual(
				retrieved_settings.request_per_minute_limit,
				self.gemini_model.request_per_minute_limit
		)
		self.assertEqual(
				retrieved_settings.tokens_per_minute_limit,
				self.gemini_model.tokens_per_minute_limit
		)
		self.assertEqual(retrieved_settings.context_used, self.gemini_model.context_used)
		self.assertEqual(retrieved_settings.context_limit, self.gemini_model.context_limit)
		self.assertEqual(
				retrieved_settings.raise_error_on_minute_limit,
				self.gemini_model.raise_error_on_minute_limit
		)
	
	def test_model_settings_setter(self):
		new_settings = GeminiModelSettings(
				model_name=GeminiModels.Gemini_2_0_flash.latest_stable,
				generation_config=GenerateContentConfigDict(temperature=0.9),
				limiter_settings=GeminiLimiterSettings(request_per_day_limit=30)
		)
		self.gemini_model.model_settings = new_settings
		
		self.assertEqual(self.gemini_model.model_name, new_settings.model_name)
		self.assertEqual(
				self.gemini_model.request_per_day_limit,
				new_settings.request_per_day_limit
		)
		self.assertEqual(self.gemini_model.generation_config, new_settings.generation_config)
		self.assertEqual(
				self.gemini_model.count_tokens_config,
				new_settings.count_tokens_config
		)


class TestGeminiModelSettings(TestCase):
	def test_default_generation_config_values(self):
		settings = GeminiModelSettings()
		default_gen_config = settings.generation_config
		
		self.assertEqual(default_gen_config['temperature'], 0.7)
		self.assertEqual(default_gen_config['top_p'], 0.5)
		self.assertEqual(default_gen_config['top_k'], 40)
		self.assertEqual(default_gen_config['candidate_count'], 1)
		self.assertEqual(default_gen_config['response_mime_type'], GeminiMimeTypes.text_plain)
		
		safety_settings = {s["category"]: s["threshold"] for s in default_gen_config["safety_settings"]}
		self.assertEqual(
				safety_settings[HarmCategory.HARM_CATEGORY_HATE_SPEECH],
				HarmBlockThreshold.OFF
		)
		self.assertEqual(
				safety_settings[HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT],
				HarmBlockThreshold.OFF
		)
		self.assertEqual(
				safety_settings[HarmCategory.HARM_CATEGORY_HARASSMENT],
				HarmBlockThreshold.OFF
		)
		self.assertEqual(
				safety_settings[HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT],
				HarmBlockThreshold.OFF
		)
		self.assertEqual(
				safety_settings[HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY],
				HarmBlockThreshold.OFF
		)
	
	def test_init_custom_count_tokens_config(self):
		custom_count_tokens_config = CountTokensConfigDict(generation_config=GenerationConfigDict(max_output_tokens=100))
		settings = GeminiModelSettings(count_tokens_config=custom_count_tokens_config)
		
		self.assertEqual(settings.count_tokens_config, custom_count_tokens_config)
	
	def test_init_custom_generation_config(self):
		custom_gen_config = GenerateContentConfigDict(temperature=0.9, top_k=10)
		settings = GeminiModelSettings(generation_config=custom_gen_config)
		
		self.assertEqual(settings.generation_config, custom_gen_config)
		self.assertIsInstance(settings.count_tokens_config, dict)
	
	def test_init_custom_limiter_settings(self):
		custom_limiter_settings = GeminiLimiterSettings(request_per_day_limit=50)
		settings = GeminiModelSettings(limiter_settings=custom_limiter_settings)
		
		self.assertEqual(settings.limiter_settings, custom_limiter_settings)
		self.assertEqual(settings.request_per_day_limit, 50)
	
	def test_init_custom_model_name(self):
		model = GeminiModels.Gemini_1_5_pro.latest_stable
		settings = GeminiModelSettings(model_name=model)
		
		self.assertEqual(settings.model_name, model)
		self.assertEqual(settings.request_per_day_limit, GeminiLimits.request_per_day[model])
		self.assertEqual(
				settings.request_per_minute_limit,
				GeminiLimits.request_per_minute[model]
		)
		self.assertEqual(settings.tokens_per_minute_limit, GeminiLimits.tokens_per_minute[model])
		self.assertEqual(settings.context_limit, GeminiLimits.context_limit[model])
	
	def test_init_default(self):
		settings = GeminiModelSettings()
		default_model = GeminiModels.Gemini_2_0_flash.latest_stable
		
		self.assertEqual(settings.model_name, default_model)
		self.assertIsInstance(settings.generation_config, dict)
		self.assertIsInstance(settings.count_tokens_config, dict)
		self.assertIsInstance(settings.limiter_settings, GeminiLimiterSettings)
		self.assertEqual(
				settings.request_per_day_limit,
				GeminiLimits.request_per_day[default_model]
		)
		self.assertEqual(
				settings.request_per_minute_limit,
				GeminiLimits.request_per_minute[default_model]
		)
		self.assertEqual(
				settings.tokens_per_minute_limit,
				GeminiLimits.tokens_per_minute[default_model]
		)
		self.assertEqual(settings.context_limit, GeminiLimits.context_limit[default_model])
	
	def test_init_unknown_model_no_error_with_limits(self):
		settings = GeminiModelSettings(
				model_name="unknown-model",
				limiter_settings=GeminiLimiterSettings(
						request_per_day_limit=100,
						request_per_minute_limit=10,
						tokens_per_minute_limit=1000,
						context_limit=2000
				)
		)
		
		self.assertEqual(settings.model_name, "unknown-model")
		self.assertEqual(settings.request_per_day_limit, 100)
		self.assertEqual(settings.request_per_minute_limit, 10)
		self.assertEqual(settings.tokens_per_minute_limit, 1000)
		self.assertEqual(settings.context_limit, 2000)
	
	def test_init_unknown_model_raises_error_no_limits_specified(self):
		with self.assertRaises(ValueError) as context:
			GeminiModelSettings(model_name="unknown-model")
		self.assertEqual(
				str(context.exception),
				"unknown-model is not a default model name. Specify 'request_per_day_limit'."
		)
	
	def test_to_dict(self):
		model = GeminiModels.Gemini_1_5_flash_8b.latest_stable
		settings = GeminiModelSettings(model_name=model)
		settings_dict = settings.to_dict()
		
		self.assertEqual(settings_dict["model_name"], model)
		self.assertIn("generation_config", settings_dict)
		self.assertIn("count_tokens_config", settings_dict)
		self.assertIn("limiter_settings", settings_dict)


def model_test_suite() -> TestSuite:
	suite = TestSuite()
	test_loader = TestLoader()
	
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiModelSettings))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiModel))
	
	return suite


if __name__ == "__main__":
	runner = TextTestRunner()
	runner.run(model_test_suite())
