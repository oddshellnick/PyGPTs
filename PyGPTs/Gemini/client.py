from google.genai import Client
from PyGPTs.Gemini import errors, types
import google.genai.types as genai_types
from google.ai.generativelanguage_v1 import GenerateContentResponse
from PyGPTs.Gemini.model import (
	GeminiModel,
	GeminiModelSettings
)
from typing import (
	Any,
	AsyncGenerator,
	Generator,
	Optional,
	Union
)
from PyGPTs.Gemini.chat import (
	GeminiAsyncChat,
	GeminiAsyncChatSettings,
	GeminiChat,
	GeminiChatSettings
)


class GeminiClientSettings(GeminiModelSettings):
	"""
	A class for configuring settings for a Gemini client.

	Attributes:
		api_key (str): Your Gemini API key.
		model_settings (GeminiModelSettings): Settings for the Gemini model. If `None`, uses default `GeminiModelSettings`.
		chats (list[Union[GeminiChat, GeminiAsyncChat]]): list of chats.
	"""
	
	def __init__(
			self,
			api_key: str,
			chats: Optional[list[Union[GeminiChat, GeminiAsyncChat]]] = None,
			model_settings: Optional[GeminiModelSettings] = None
	):
		"""
		Initializes an instance of the GeminiClientSettings class.

		Args:
			api_key (str): Your Gemini API key.
			chats (Optional[list[Union[GeminiChat, GeminiAsyncChat]]]): list of chats.
			model_settings (Optional[GeminiModelSettings]): Settings for the Gemini model. If `None`, uses default `GeminiModelSettings`.
		"""
		self.api_key = api_key
		
		self.chats = chats if chats is not None else []
		
		if model_settings is None:
			model_settings = GeminiModelSettings()
		
		self.model_settings = model_settings
		
		super().__init__(**self.model_settings.to_dict())
	
	def to_dict(self) -> dict[str, Any]:
		return {
			"api_key": self.api_key,
			"chats": self.chats,
			"model_settings": self.model_settings
		}


class GeminiClient(GeminiModel):
	"""
	A wrapper class for interacting with Google Gemini models using the `genai` library.

	Attributes:
		api_key (str): The API key used for authentication.
		client (Client): The underlying Google AI client instance.
		chats (List[Union[GeminiChat, GeminiAsyncChat]]): A list of active chat sessions, which can be either synchronous or asynchronous.
	"""
	
	def __init__(self, client_settings: GeminiClientSettings):
		"""
		Initializes a new Gemini instance.

		Args:
			client_settings (GeminiClientSettings): An instance of GeminiClientSettings containing configuration parameters.
		"""
		super().__init__(client_settings.model_settings)
		
		self.api_key = client_settings.api_key
		self.client = Client(api_key=self.api_key)
		self.chats: list[Union[GeminiChat, GeminiAsyncChat]] = client_settings.chats
	
	async def async_generate_content(
			self,
			message: types.gemini_generate_input,
			count_tokens_config: Optional[genai_types.CountTokensConfigOrDict] = None,
			generate_config: Optional[genai_types.GenerateContentConfigOrDict] = None
	) -> GenerateContentResponse:
		"""
		Asynchronously generates content based on the provided message. This is the asynchronous version of `generate_content`.

		Args:
			message (types.gemini_generate_input): The input message.
			count_tokens_config (Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.
			generate_config (Optional[genai_types.GenerateContentConfigOrDict]): Overrides the default generation config.

		Returns:
			Coroutine[Any, Any, GenerateContentResponse]: A coroutine that resolves to the generated content response.
		"""
		await self.async_add_data(
				self.client.models.count_tokens(
						model=self.model_name,
						contents=message,
						config=count_tokens_config
						if count_tokens_config is not None
						else self.count_tokens_config
				).total_tokens
		)
		
		response = await self.client.aio.models.generate_content(
				model=self.model_name,
				contents=message,
				config=generate_config
				if generate_config is not None
				else self.generation_config
		)
		
		return response
	
	async def async_generate_content_stream(
			self,
			message: types.gemini_generate_input,
			count_tokens_config: Optional[genai_types.CountTokensConfigOrDict] = None,
			generate_config: Optional[genai_types.GenerateContentConfigOrDict] = None
	) -> AsyncGenerator[GenerateContentResponse, Any]:
		"""
		Asynchronously generates content as a stream. This is the asynchronous version of `generate_content_stream`.

		This allows you to process parts of the response as they become available,
		rather than waiting for the entire response to be generated.

		Args:
			message (types.gemini_generate_input): The input message.
			count_tokens_config (Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.
			generate_config (Optional[genai_types.GenerateContentConfigOrDict]): Overrides the default generation config.

		Returns:
			AsyncGenerator[GenerateContentResponse, Any]: An async iterator that yields `GenerateContentResponse` objects as they become available.
		"""
		await self.async_add_data(
				self.client.models.count_tokens(
						model=self.model_name,
						contents=message,
						config=count_tokens_config
						if count_tokens_config is not None
						else self.count_tokens_config
				).total_tokens
		)
		
		async for response in await self.client.aio.models.generate_content_stream(
				model=self.model_name,
				contents=message,
				config=generate_config
				if generate_config is not None
				else self.generation_config
		):
			yield response
	
	def chat(self, chat_index: int = -1) -> Union[GeminiChat, GeminiAsyncChat]:
		"""
		Returns a specific chat session.

		Args:
			chat_index (int): The index of the chat session to retrieve. Defaults to -1, which returns the last created chat session.

		Returns:
			Union[GeminiChat, GeminiAsyncChat]: The `GeminiChat` or 'GeminiAsyncChat' object at the specified index.
		"""
		return self.chats[chat_index]
	
	async def async_send_message(self, message: types.gemini_message_input, chat_index: int = -1) -> GenerateContentResponse:
		"""
		Sends a message to an asynchronous chat session.

		Args:
			message (types.gemini_message_input): The message to send to the async chat session.
			chat_index (int): The index of the asynchronous chat session to send the message to. Defaults to -1, which targets the last created asynchronous chat session.

		Returns:
			GenerateContentResponse: The response from the Gemini model for the sent message.

		Raises:
			GeminiChatTypeException: If the chat session at the given index is not an asynchronous chat (`GeminiAsyncChat`).
		"""
		chat = self.chat(chat_index)
		
		if not chat.is_async:
			raise errors.GeminiChatTypeException(chat_index, "asynchronous")
		
		return await chat.send_message(message=message)
	
	async def async_send_message_stream(self, message: types.gemini_message_input, chat_index: int = -1) -> AsyncGenerator[GenerateContentResponse, Any]:
		"""
		Sends a message to an asynchronous chat session and returns an asynchronous stream of responses.

		Args:
			message (types.gemini_message_input): The message to send to the async chat session.
			chat_index (int): The index of the asynchronous chat session to send the message to. Defaults to -1, which targets the last created asynchronous chat session.

		Returns:
			AsyncGenerator[GenerateContentResponse, Any]: An async generator that yields responses from the Gemini model as they become available in a stream.

		Raises:
			GeminiChatTypeException: If the chat session at the given index is not an asynchronous chat (`GeminiAsyncChat`).
		"""
		chat = self.chat(chat_index)
		
		if not chat.is_async:
			raise errors.GeminiChatTypeException(chat_index, "asynchronous")
		
		return chat.send_message_stream(message=message)
	
	@property
	def client_settings(self) -> GeminiClientSettings:
		"""
		Returns a `GeminiClientSettings` object representing the current configuration of the Gemini client.

		This property provides a snapshot of the current settings being used by the `GeminiClient`,
		including the API key, the list of active chat sessions, and the model settings.
		It's useful for inspecting or serializing the client's configuration.

		Returns:
			GeminiClientSettings: A `GeminiClientSettings` object containing the current client's API key, chat sessions, and model settings.
		"""
		return GeminiClientSettings(
				api_key=self.api_key,
				chats=self.chats,
				model_settings=self.model_settings
		)
	
	@client_settings.setter
	def client_settings(self, client_settings: GeminiClientSettings):
		"""
		Sets the client settings for the Gemini client, allowing for a complete reconfiguration.

		This setter updates the `GeminiClient` instance with new settings provided in a `GeminiClientSettings` object.
		It allows changing the model settings, API key, and replacing the entire list of active chat sessions.
		This is useful for dynamically switching configurations or resetting the client with new settings.

		Args:
			client_settings (GeminiClientSettings): A `GeminiClientSettings` object containing the new settings to apply to the client.
		"""
		self.model_settings = client_settings.model_settings
		self.client = Client(api_key=client_settings.api_key)
		self.api_key = client_settings.api_key
		self.chats: list[Union[GeminiChat, GeminiAsyncChat]] = client_settings.chats
	
	def close_chat(self, chat_index: int = -1):
		"""
		Closes a chat session.

		Args:
			chat_index (int): The index of the chat session to close. Defaults to -1 (the last chat session).
		"""
		self.chats.pop(chat_index)
	
	def generate_content(
			self,
			message: types.gemini_generate_input,
			count_tokens_config: Optional[genai_types.CountTokensConfigOrDict] = None,
			generate_config: Optional[genai_types.GenerationConfigOrDict] = None
	) -> GenerateContentResponse:
		"""
		Generates content based on the provided message using the configured Gemini model.

		Args:
			message (types.gemini_generate_input): The input message or content to generate from. Can be a string, a list of strings, or a more complex structure defined in `types.gemini_generate_input`.
			count_tokens_config (Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.
			generate_config (Optional[genai_types.GenerateContentConfigOrDict]): Overrides the default `generation_config` for this specific call.
		"""
		self.add_data(
				self.client.models.count_tokens(
						model=self.model_name,
						contents=message,
						config=count_tokens_config
						if count_tokens_config is not None
						else self.count_tokens_config
				).total_tokens
		)
		
		response = self.client.models.generate_content(
				model=self.model_name,
				contents=message,
				config=generate_config
				if generate_config is not None
				else self.generation_config
		)
		
		return response
	
	def generate_content_stream(
			self,
			message: types.gemini_generate_input,
			count_tokens_config: Optional[genai_types.CountTokensConfigOrDict] = None,
			generate_config: Optional[genai_types.GenerateContentConfigOrDict] = None
	) -> Generator[GenerateContentResponse, Any, None]:
		"""
		Generates content as a stream (synchronous version).

		Args:
			message (types.gemini_generate_input): The input message.
			count_tokens_config (Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.
			generate_config (Optional[genai_types.GenerateContentConfigOrDict]): Overrides the default generation config.

		Returns:
			Generator[GenerateContentResponse, Any, None]: An iterator that yields `GenerateContentResponse` objects.
		"""
		self.add_data(
				self.client.models.count_tokens(
						model=self.model_name,
						contents=message,
						config=count_tokens_config
						if count_tokens_config is not None
						else self.count_tokens_config
				).total_tokens
		)
		
		for response in self.client.models.generate_content_stream(
				model=self.model_name,
				contents=message,
				config=generate_config
				if generate_config is not None
				else self.generation_config
		):
			yield response
	
	def get_chats(self) -> list[Union[GeminiChat, GeminiAsyncChat]]:
		"""
		Returns the list of chat sessions managed by this `GeminiClient`.

		Returns:
			list[Union[GeminiChat, GeminiAsyncChat]]: A list of `GeminiChat` and `GeminiAsyncChat` objects, representing the active chat sessions.
		"""
		return self.chats
	
	def send_message(self, message: types.gemini_message_input, chat_index: int = -1) -> GenerateContentResponse:
		"""
		Sends a message to a synchronous chat session.

		Args:
			message (types.gemini_message_input): The message to send to the chat session.
			chat_index (int): The index of the synchronous chat session to send the message to. Defaults to -1, which targets the last created synchronous chat session.

		Returns:
			GenerateContentResponse: The response from the Gemini model for the sent message.

		Raises:
			GeminiChatTypeException: If the chat session at the given index is not a synchronous chat (`GeminiChat`).
		"""
		chat = self.chat(chat_index)
		
		if chat.is_async:
			raise errors.GeminiChatTypeException(chat_index, "synchronous")
		
		return chat.send_message(message=message)
	
	def send_message_stream(self, message: types.gemini_message_input, chat_index: int = -1) -> Generator[GenerateContentResponse, Any, None]:
		"""
		Sends a message to a synchronous chat session and returns a stream of responses.

		Args:
			message (types.gemini_message_input): The message to send to the chat session.
			chat_index (int): The index of the synchronous chat session to send the message to. Defaults to -1, which targets the last created synchronous chat session.

		Returns:
			Generator[GenerateContentResponse, Any, None]: A generator that yields responses from the Gemini model as they become available in a stream.

		Raises:
			GeminiChatTypeException: If the chat session at the given index is not a synchronous chat (`GeminiChat`).
		"""
		chat = self.chat(chat_index)
		
		if chat.is_async:
			raise errors.GeminiChatTypeException(chat_index, "synchronous")
		
		return chat.send_message_stream(message=message)
	
	def start_async_chat(
			self,
			model_settings: Optional[GeminiModelSettings] = None,
			history: Optional[list[types.gemini_history]] = None
	):
		"""
		Starts new async chat and appends to chats list

		Args:
			model_settings (Optional[genai_types.GenerateContentConfigOrDict]): Overrides the default `model_settings` for this specific call.
			history (Optional[list[types.gemini_history]]): you can specify the history for this chat.
		"""
		self.chats.append(
				GeminiAsyncChat(
						chat_settings=GeminiAsyncChatSettings(
								client=self.client,
								model_settings=model_settings
								if model_settings is not None
								else self.model_settings,
								history=history
						)
				)
		)
	
	def start_chat(
			self,
			model_settings: Optional[GeminiModelSettings] = None,
			history: Optional[list[types.gemini_history]] = None
	):
		"""
		Starts new chat and appends to chats list

		Args:
			model_settings (Optional[GeminiModelSettings]): Overrides the default `model_settings` for this specific call.
			history (Optional[list[types.gemini_history]]): you can specify the history for this chat
		"""
		self.chats.append(
				GeminiChat(
						chat_settings=GeminiChatSettings(
								client=self.client,
								model_settings=model_settings
								if model_settings is not None
								else self.model_settings,
								history=history
						)
				)
		)
