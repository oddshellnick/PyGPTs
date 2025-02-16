import typing
from PyGPTs.Gemini.functions import extract_text_from_gemini_response
from PyGPTs.Gemini.client import (
	GeminiClient,
	GeminiClientSettings
)


class GeminiClientsManager:
	def __init__(self, gemini_clients_settings: list[GeminiClientSettings]):
		"""
		Initializes a new GeminiManager instance.

		Args:
			gemini_clients_settings (List[GeminiSettings]): A list of GeminiSettings objects.

		Raises:
			GeminiNoUsefulModelsException: If none of the provided models have available quota.
		"""
		self.clients = [
			GeminiClient(client_settings)
			for client_settings in gemini_clients_settings
		]
		
		self.current_model_index = self.lowest_useful_client_index
	
	def get_client_index(self, model_api_key: str) -> typing.Optional[int]:
		"""
		Retrieves the index of a model based on its API KEY.

		Args:
			model_api_key (str): The API key of the model to search for.

		Returns:
		   typing.Optional[int]: The index of the model if found, None otherwise.
		"""
		for i in range(len(self.clients)):
			if self.clients[i].api_key == model_api_key:
				return i
		
		return None
	
	def client(
			self,
			model_index: typing.Optional[int] = None,
			model_api_key: typing.Optional[str] = None
	) -> typing.Optional[GeminiClient]:
		"""
		Switches to a specific Gemini model by index or API key.

		Args:
			model_index (typing.Optional[int]): The index of the model to use.
			model_api_key (typing.Optional[str]): The API key of the model to use.

		Returns:
			typing.Optional[GeminiClient]: The selected Gemini instance, None if model not found.

		Raises:
			ValueError: If both `model_index` and `model_api_key` are provided.
		"""
		if model_index is not None and model_api_key is not None:
			raise ValueError("You can't use both 'model_index' and 'model_api_key'")
		
		if model_api_key is not None:
			self.current_model_index = self.get_client_index(model_api_key)
		elif model_index is not None:
			self.current_model_index = model_index if model_index < len(self.clients) else None
		
		return self.clients[self.current_model_index] if self.current_model_index is not None else None
	
	@property
	def has_useful_model(self) -> bool:
		"""
		Checks if any of the managed models have available quota.

		Returns:
			bool: True if any model has available quota, False otherwise.
		"""
		return any(client_settings.has_day_limits for client_settings in self.clients)
	
	@property
	def next_client(self) -> GeminiClient:
		"""
		Switches to the next available Gemini model.

		Returns:
			GeminiClient: The next available Gemini instance.
		"""
		self.current_model_index = (self.current_model_index + 1) % len(self.clients) if self.current_model_index is not None else 0
		
		return self.client()
	
	@property
	def lowest_useful_client_index(self) -> typing.Optional[int]:
		"""
		Finds the index of the first model with available quota.

		Returns:
			typing.Optional[int]: The index of the first available model, None if no models have available quota.
		"""
		for i in range(len(self.clients)):
			if self.clients[i].has_day_limits:
				return i
		
		return None
	
	def reset_clients(self, gemini_clients_settings: list[GeminiClientSettings]):
		"""
		Resets the managed models.

		Args:
			gemini_clients_settings (List[GeminiSettings]): A new list of GeminiSettings objects.

		Raises:
			GeminiNoUsefulModelsException: if there are no models with available quota
		"""
		self.clients = [
			GeminiClient(client_settings)
			for client_settings in gemini_clients_settings
		]
		self.current_model_index = self.lowest_useful_client_index
