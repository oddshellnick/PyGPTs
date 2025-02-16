from typing import Any, Optional
from PyGPTs.Gemini.functions import find_base_model
from PyGPTs.Gemini.limiter import (
	GeminiLimiter,
	GeminiLimiterSettings
)
from PyGPTs.Gemini.data import (
	GeminiLimits,
	GeminiMimeTypes,
	GeminiModels
)
from google.genai.types import (
	CountTokensConfigDict,
	CountTokensConfigOrDict,
	GenerateContentConfigDict,
	GenerateContentConfigOrDict,
	GenerationConfigDict,
	HarmBlockThreshold,
	HarmCategory,
	SafetySettingDict
)


class GeminiModelSettings(GeminiLimiterSettings):
	"""
	A class for configuring settings for a specific Gemini model.  It extends `GeminiLimiterSettings` to incorporate model-specific configurations.

	Attributes:
		model_name (str): The name of the Gemini model to use (e.g., "gemini-pro", "gemini-ultra"). Defaults to the latest stable Gemini 2.0 Flash model.
		generation_config (Optional[GenerateContentConfigOrDict]): Configuration for text generation, controlling aspects like temperature, top_p, and top_k. Defaults to a pre-defined, conservative configuration if not specified.
		count_tokens_config (Optional[CountTokensConfigOrDict]): Configuration for token counting. If not provided, it's derived from `generation_config`.
		limiter_settings (Optional[GeminiLimiterSettings]): Settings for the rate limiter. If not provided, default `GeminiLimiterSettings` will be used.
	"""
	
	def __init__(
			self,
			model_name: str = GeminiModels.Gemini_2_0_flash.latest_stable,
			generation_config: Optional[GenerateContentConfigOrDict] = None,
			count_tokens_config: Optional[CountTokensConfigOrDict] = None,
			limiter_settings: Optional[GeminiLimiterSettings] = None
	):
		"""
		Initializes an instance of the GeminiSettings class.

		Args:
			model_name (str): The name of the Gemini model to use. Defaults to "gemini_2_0_flash".
			generation_config (Optional[GenerateContentConfigOrDict]): Configuration for text generation. Defaults to a conservative configuration.
			count_tokens_config (Optional[CountTokensConfigOrDict]): Configuration for token counting. If None, it will be derived from `generation_config`.
			limiter_settings (Optional[GeminiLimiterSettings]): Settings for rate limiting. Defaults to default `GeminiLimiterSettings`.
		"""
		if generation_config is None:
			generation_config = GenerateContentConfigDict(
					temperature=0.7,
					top_p=0.5,
					top_k=40,
					candidate_count=1,
					response_mime_type=GeminiMimeTypes.text_plain,
					safety_settings=[
						SafetySettingDict(
								category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
								threshold=HarmBlockThreshold.OFF
						),
						SafetySettingDict(
								category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
								threshold=HarmBlockThreshold.OFF
						),
						SafetySettingDict(
								category=HarmCategory.HARM_CATEGORY_HARASSMENT,
								threshold=HarmBlockThreshold.OFF
						),
						SafetySettingDict(
								category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
								threshold=HarmBlockThreshold.OFF
						),
						SafetySettingDict(
								category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
								threshold=HarmBlockThreshold.OFF
						)
					]
			)
		
		if count_tokens_config is None:
			count_tokens_config = CountTokensConfigDict()
		
		if limiter_settings is None:
			limiter_settings = GeminiLimiterSettings()
		
		self.model_name = model_name
		self.generation_config = generation_config
		self.count_tokens_config = count_tokens_config
		self.limiter_settings = limiter_settings
		
		super().__init__(**limiter_settings.to_dict())
		
		base_model_name = find_base_model(model_name)
		
		if self.request_per_day_limit is None:
			if base_model_name not in GeminiLimits.request_per_day:
				raise ValueError(
						f"{model_name} is not a default model name. Specify 'request_per_day_limit'."
				)
			
			self.request_per_day_limit = GeminiLimits.request_per_day[base_model_name]
			self.limiter_settings.request_per_day_limit = GeminiLimits.request_per_day[base_model_name]
		
		if self.request_per_minute_limit is None:
			if base_model_name not in GeminiLimits.request_per_minute:
				raise ValueError(
						f"{model_name} is not a default model name. Specify 'request_per_minute_limit'."
				)
			
			self.request_per_minute_limit = GeminiLimits.request_per_minute[base_model_name]
			self.limiter_settings.request_per_minute_limit = GeminiLimits.request_per_minute[base_model_name]
		
		if self.tokens_per_minute_limit is None:
			if base_model_name not in GeminiLimits.tokens_per_minute:
				raise ValueError(
						f"{model_name} is not a default model name. Specify 'tokens_per_minute_limit'."
				)
			
			self.tokens_per_minute_limit = GeminiLimits.tokens_per_minute[base_model_name]
			self.limiter_settings.tokens_per_minute_limit = GeminiLimits.tokens_per_minute[base_model_name]
		
		if self.context_limit is None:
			if base_model_name not in GeminiLimits.context_limit:
				raise ValueError(f"{model_name} is not a default model name. Specify 'context_limit'.")
			
			self.context_limit = GeminiLimits.context_limit[base_model_name]
			self.limiter_settings.context_limit = GeminiLimits.context_limit[base_model_name]
	
	def to_dict(self) -> dict[str, Any]:
		"""
		Converts the GeminiModelSettings object to a dictionary, including nested settings.

		Returns:
			dict[str, Any]: A dictionary representation of the GeminiModelSettings object.
		"""
		return {
			"model_name": self.model_name,
			"generation_config": self.generation_config,
			"count_tokens_config": self.count_tokens_config,
			"limiter_settings": self.limiter_settings
		}


class GeminiModel(GeminiLimiter):
	"""
	Represents a Gemini model with associated rate limiting and generation configurations.

	This class combines the functionalities of `GeminiLimiter` for rate limit management with model-specific settings
	such as the model name and generation configuration.
	It inherits rate limiting capabilities from `GeminiLimiter` and adds attributes to store and manage the model's configuration.

	Attributes:
		model_name (str): The name of the Gemini model.
		generation_config (GenerateContentConfigOrDict): Configuration settings for content generation with this model.
	"""
	
	def __init__(self, gemini_model_settings: GeminiModelSettings):
		"""
		Initializes a GeminiModel instance.

		Args:
			gemini_model_settings (GeminiModelSettings): A GeminiModelSettings object containing the model configuration.
		"""
		super().__init__(gemini_model_settings.limiter_settings)
		
		self.model_name = gemini_model_settings.model_name
		self.generation_config = gemini_model_settings.generation_config
		self.count_tokens_config = gemini_model_settings.count_tokens_config
	
	@property
	def model_settings(self) -> GeminiModelSettings:
		"""
		Returns the current settings of the Gemini model, including updated usage statistics from the limiter.

		Returns:
			GeminiModelSettings: A `GeminiModelSettings` object representing the current settings of the model in this chat session.
		"""
		return GeminiModelSettings(
				model_name=self.model_name,
				generation_config=self.generation_config,
				count_tokens_config=self.count_tokens_config,
				limiter_settings=self.limiter_settings
		)
	
	@model_settings.setter
	def model_settings(self, gemini_model_settings: GeminiModelSettings):
		"""
		Sets the model settings for the Gemini model.

		Args:
			gemini_model_settings (GeminiModelSettings): The new GeminiModelSettings to apply.
		"""
		self.model_name = gemini_model_settings.model_name
		self.generation_config = gemini_model_settings.generation_config
		self.count_tokens_config = gemini_model_settings.count_tokens_config
		self.limiter_settings = gemini_model_settings.limiter_settings
