from google.genai import Client
from google.genai.types import Content
from google.genai.chats import AsyncChat, Chat
from google.ai.generativelanguage_v1 import GenerateContentResponse
from PyGPTs.Gemini.model import (
	GeminiModel,
	GeminiModelSettings
)
from PyGPTs.Gemini.types import (
	gemini_history,
	gemini_message_input
)
from typing import (
	Any,
	AsyncGenerator,
	Generator,
	Optional,
	Union
)
from PyGPTs.Gemini.functions import extract_token_count_from_gemini_response


class GeminiBaseChatSettings(GeminiModelSettings):
	"""
	Configuration class for creating Gemini chat sessions.

	This class bundles together all settings required to initialize a `GeminiChat` or `GeminiAsyncChat` instance.

	Attributes:
		client (Client): The Gemini API client instance (`genai.Client`).
		is_async (Optional[bool]):  A flag indicating whether the chat session should be asynchronous. Defaults to `None`.
		history (Optional[list[gemini_history]]):  The initial chat history. Defaults to `None` (empty history).
		model_settings (Optional[GeminiModelSettings]): The settings for the Gemini model. If `None`, default `GeminiModelSettings` will be used.
	"""
	
	def __init__(
			self,
			client: Client,
			is_async: Optional[bool] = None,
			history: Optional[list[gemini_history]] = None,
			model_settings: Optional[GeminiModelSettings] = None
	):
		"""
		Initializes a GeminiBaseChatSettings instance.

		Args:
			client (Client): The Gemini API client instance (`genai.Client`).
			is_async (Optional[bool]):  A flag indicating whether the chat session should be asynchronous. Defaults to `None`.
			history (Optional[list[gemini_history]]):  The initial chat history. Defaults to `None` (empty history).
			model_settings (Optional[GeminiModelSettings]): The settings for the Gemini model. If `None`, default `GeminiModelSettings` will be used.
		"""
		if model_settings is None:
			model_settings = GeminiModelSettings()
		
		self.client = client
		self.is_async = is_async
		self.history = history
		self.model_settings = model_settings
		
		super().__init__(**self.model_settings.to_dict())
		
		self.limiter_settings.context_used = sum(
				self.client.models.count_tokens(model=model_settings.model_name, contents=message).total_tokens
				for message in history
		) if history is not None else 0
	
	def to_dict(self) -> dict[str, Any]:
		"""
		Converts the GeminiBaseChatSettings object to a dictionary, including nested settings.

		Returns:
			dict[str, Any]: A dictionary representation of the GeminiBaseChatSettings object.
		"""
		return {
			"client": self.client,
			"is_async": self.is_async,
			"history": self.history,
			"model_settings": self.model_settings
		}


class GeminiChatSettings(GeminiBaseChatSettings):
	"""
	Configuration class for creating Gemini chat sessions.

	This class bundles together all settings required to initialize a `GeminiChat` instance.

	Attributes:
		client (Client): The Gemini API client instance (`genai.Client`).
		is_async (Optional[bool]):  A flag indicating whether the chat session should be asynchronous. Defaults to `None`.
		history (Optional[list[gemini_history]]):  The initial chat history. Defaults to `None` (empty history).
		model_settings (Optional[GeminiModelSettings]): The settings for the Gemini model. If `None`, default `GeminiModelSettings` will be used.
	"""
	
	def __init__(
			self,
			client: Client,
			history: Optional[list[gemini_history]] = None,
			model_settings: Optional[GeminiModelSettings] = None
	):
		"""
		Initializes a GeminiChatSettings instance.

		Args:
			client (Client): The Gemini API client instance (`genai.Client`).
			history (Optional[list[gemini_history]]):  The initial chat history. Defaults to `None` (empty history).
			model_settings (Optional[GeminiModelSettings]): The settings for the Gemini model. If `None`, default `GeminiModelSettings` will be used.
		"""
		super().__init__(
				client=client,
				is_async=False,
				history=history,
				model_settings=model_settings
		)


class BaseGeminiChat(GeminiModel):
	"""
	A base class representing a chat session with a Gemini model, providing common functionalities for both synchronous and asynchronous chats.

	Attributes:
		client (genai.Client): The `genai.Client` instance used for interacting with the Gemini API.
		chat (Union[Chat, AsyncChat]): The underlying `Chat` object, which can be either synchronous (`Chat`) or asynchronous (`AsyncChat`).
		is_async (Optional[bool]): An optional boolean flag indicating whether the chat session is intended to be asynchronous. `None` in the base class.
	"""
	
	def __init__(self, chat_settings: GeminiBaseChatSettings):
		"""
		Initializes a `BaseGeminiChat` instance.

		Args:
			chat_settings (GeminiBaseChatSettings): An instance of `GeminiChatSettings` containing the configuration for the chat session.
		"""
		super().__init__(chat_settings.model_settings)
		
		self.client = chat_settings.client
		self.is_async = chat_settings.is_async
		
		self.chat = self.create_chat(model_settings=self.model_settings, history=chat_settings.history)
	
	@property
	def history(self) -> Optional[list[Content]]:
		"""
		Returns the history of the chat session.

		Returns:
			Optional[list[Content]]: The history of the chat.
		"""
		return self.chat._curated_history if self.chat is not None else None
	
	def create_chat(
			self,
			model_settings: GeminiModelSettings,
			history: Optional[list[gemini_history]] = None
	) -> Optional[Union[Chat, AsyncChat]]:
		self.model_settings = model_settings
		self.context_used = self.client.models.count_tokens(
				model=self.model_settings.model_name,
				contents=history,
				config=self.count_tokens_config
		).total_tokens
		
		return None
	
	@property
	def chat_settings(self) -> GeminiBaseChatSettings:
		"""
		Returns a `GeminiBaseChatSettings` object that encapsulates the complete configuration of the current chat session.

		This property provides access to a settings object that contains all the parameters used to initialize and configure this `BaseGeminiChat` instance.
		This includes the Gemini API client, asynchronicity flag, chat history, and the model settings.

		Returns:
			GeminiBaseChatSettings: A `GeminiBaseChatSettings` object representing the current configuration of this chat session.
		"""
		return GeminiBaseChatSettings(
				client=self.client,
				is_async=self.is_async,
				history=self.history,
				model_settings=self.model_settings
		)
	
	@chat_settings.setter
	def chat_settings(self, model_settings: GeminiModelSettings):
		"""
		Sets new model settings for the chat session, effectively changing the underlying Gemini model while preserving the chat history.

		This setter updates the chat session to use a new Gemini model configuration specified by the provided `GeminiModelSettings` object.
		It recreates the chat session using the `create_chat` method, applying the new `model_settings` and retaining the existing chat history.
		This allows for dynamic switching of models during a conversation without losing the context.

		Args:
			model_settings (GeminiModelSettings): The new `GeminiModelSettings` object containing the configuration for the desired Gemini model.
		"""
		self.chat = self.create_chat(model_settings=model_settings, history=self.history)
	
	def clear_chat_history(self):
		"""
		Clears the history of the current chat session and resets the context usage to 0.

		This effectively starts a new conversation within the same `BaseGeminiChat` object, but with a clean slate.
		"""
		self.chat = self.create_chat(model_settings=self.model_settings, history=[])
		self.clear_context()
	
	def reset_history(self, history: list[gemini_history]):
		"""
		Resets the chat history with a new history and updates the context usage accordingly.

		Args:
			history (list[gemini_history]): The new chat history to set.
		"""
		self.chat = self.create_chat(model_settings=self.model_settings, history=history)
		self.context_used = self.client.models.count_tokens(
				model=self.model_name,
				contents=history,
				config=self.count_tokens_config
		).total_tokens
	
	def slice_history(self, start: Optional[int] = None, end: Optional[int] = None):
		"""
		Slices the current chat history, keeping only a portion of it, and resets the chat history to this sliced part.

		This is useful for managing context window size by discarding older messages.

		Args:
			start (Optional[int]): The starting index for the slice. If None, defaults to the beginning of the history.
			end (Optional[int]): The ending index for the slice (exclusive). If None, defaults to the end of the history.
		"""
		self.reset_history(self.history[slice(start, end)])


class GeminiChat(BaseGeminiChat):
	"""
	A class representing a chat session with a Gemini model. This class encapsulates the `Chat` object and manages its own rate limiting.
	"""
	
	def __init__(self, chat_settings: GeminiChatSettings):
		"""
		Initializes a `GeminiChat` instance.
		"""
		super().__init__(chat_settings=chat_settings)
	
	def create_chat(
			self,
			model_settings: GeminiModelSettings,
			history: Optional[list[gemini_history]] = None
	) -> Chat:
		self.model_settings = model_settings
		self.context_used = self.client.models.count_tokens(
				model=self.model_settings.model_name,
				contents=history,
				config=self.count_tokens_config
		).total_tokens
		
		return self.client.chats.create(
				model=model_settings.model_name,
				config=model_settings.generation_config,
				history=history
		)
	
	def send_message(self, message: gemini_message_input) -> GenerateContentResponse:
		"""
		Sends a message to a chat session.

		Args:
			message (str): The message to send

		Returns:
			GenerateContentResponse: The response from the Gemini model.
		"""
		self.add_data(
				self.client.models.count_tokens(
						model=self.model_name,
						contents=message,
						config=self.count_tokens_config
				).total_tokens
		)
		
		response = self.chat.send_message(message=message)
		self.add_context(extract_token_count_from_gemini_response(response))
		
		return response
	
	def send_message_stream(self, message: gemini_message_input) -> Generator[GenerateContentResponse, Any, None]:
		"""
		Sends a message to a chat session and returns stream.

		Args:
			message (str): The message to send.

		Returns:
			Generator[GenerateContentResponse, Any, None]: The response from the Gemini model.
		"""
		self.add_data(
				self.client.models.count_tokens(
						model=self.model_name,
						contents=message,
						config=self.count_tokens_config
				).total_tokens
		)
		
		for response in self.chat.send_message_stream(message=message):
			self.add_context(extract_token_count_from_gemini_response(response))
			yield response


class GeminiAsyncChatSettings(GeminiBaseChatSettings):
	"""
	Configuration class for creating Gemini asynchronous chat sessions.

	This class bundles together all settings required to initialize a `GeminiAsyncChat` instance.

	Attributes:
		client (Client): The Gemini API client instance (`genai.Client`).
		is_async (Optional[bool]):  A flag indicating whether the chat session should be asynchronous. Defaults to `None`.
		history (Optional[list[gemini_history]]):  The initial chat history. Defaults to `None` (empty history).
		model_settings (Optional[GeminiModelSettings]): The settings for the Gemini model. If `None`, default `GeminiModelSettings` will be used.
	"""
	
	def __init__(
			self,
			client: Client,
			history: Optional[list[gemini_history]] = None,
			model_settings: Optional[GeminiModelSettings] = None
	):
		"""
		Initializes a GeminiAsyncChatSettings instance.

		Args:
			client (Client): The Gemini API client instance (`genai.Client`).
			history (Optional[list[gemini_history]]):  The initial chat history. Defaults to `None` (empty history).
			model_settings (Optional[GeminiModelSettings]): The settings for the Gemini model. If `None`, default `GeminiModelSettings` will be used.
		"""
		super().__init__(
				client=client,
				is_async=True,
				history=history,
				model_settings=model_settings
		)


class GeminiAsyncChat(BaseGeminiChat):
	"""
	A class representing a chat session with a Gemini model. This class encapsulates the `AsyncChat` object and manages its own rate limiting.
	"""
	
	def __init__(self, chat_settings: GeminiAsyncChatSettings):
		"""
		Initializes a `GeminiAsyncChat` instance.
		"""
		super().__init__(chat_settings=chat_settings)
	
	def create_chat(
			self,
			model_settings: GeminiModelSettings,
			history: Optional[list[gemini_history]] = None
	) -> AsyncChat:
		self.model_settings = model_settings
		self.context_used = self.client.models.count_tokens(
				model=self.model_settings.model_name,
				contents=history,
				config=self.count_tokens_config
		).total_tokens
		
		return self.client.aio.chats.create(
				model=model_settings.model_name,
				config=model_settings.generation_config,
				history=history
		)
	
	async def send_message(self, message: gemini_message_input) -> GenerateContentResponse:
		"""
		Sends a message to a chat session.

		Args:
			message (str): The message to send.

		Returns:
			GenerateContentResponse: The response from the Gemini model.
		"""
		await self.async_add_data(
				self.client.models.count_tokens(
						model=self.model_name,
						contents=message,
						config=self.count_tokens_config
				).total_tokens
		)
		
		response = await self.chat.send_message(message=message)
		self.add_context(extract_token_count_from_gemini_response(response))
		
		return response
	
	async def send_message_stream(self, message: gemini_message_input) -> AsyncGenerator[GenerateContentResponse, Any]:
		"""
		Sends a message to a chat session and returns stream.

		Args:
			message (str): The message to send.

		Returns:
			AsyncGenerator[GenerateContentResponse, Any]: The response from the Gemini model.
		"""
		await self.async_add_data(
				self.client.models.count_tokens(
						model=self.model_name,
						contents=message,
						config=self.count_tokens_config
				).total_tokens
		)
		
		async for response in await self.chat.send_message_stream(message=message):
			self.add_context(extract_token_count_from_gemini_response(response))
			yield response
