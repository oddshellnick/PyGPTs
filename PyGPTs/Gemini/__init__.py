import time
import pytz
import typing
import asyncio
import datetime
from google import genai
import PyGPTs.Gemini.data as data
import PyGPTs.Gemini.types as types
import PyGPTs.Gemini.errors as errors
import google.genai.types as genai_types
from google.genai.chats import AsyncChat, Chat
from google.ai.generativelanguage_v1 import GenerateContentResponse
from PyGPTs.Gemini.functions import (
	extract_token_count_from_gemini_response,
	find_base_model
)


class GeminiLimiter:
	"""
	Manages rate limiting for Gemini API requests.

	Attributes:
		limit_day (datetime.datetime): The limit day for tracking daily usage limits.
		request_per_day_used (int): The number of requests used so far today.
		request_per_day_limit (int): The maximum number of requests allowed per day.
		request_per_minute_limit (int): The maximum number of requests allowed per minute.
		tokens_per_minute_limit (int): The maximum number of tokens allowed per minute.
		raise_error_on_minute_limit (bool): Whether to raise an error when a rate limit is exceeded. Defaults to True.
		request_per_minute_used (int): The number of requests used so far this minute.
		tokens_per_minute_used (int): The number of tokens used so far this minute.
		context_used (int): Represents the current amount of context used. This is distinct from tokens and requests.
		context_limit (int): The maximum allowed context usage.
		start_time (float): The timestamp of the start of the current minute.
	"""
	
	def __init__(
			self,
			limit_day: datetime.datetime,
			request_per_day_used: int,
			request_per_day_limit: int,
			request_per_minute_limit: int,
			tokens_per_minute_limit: int,
			context_used: int,
			context_limit: int,
			raise_error_on_minute_limit: bool = True,
	):
		"""
		Initializes an instance of the GeminiLimiter class.

		Args:
			limit_day (datetime.datetime): The limit day for tracking daily usage.
			request_per_day_used (int): Initial count of requests used per day.
			request_per_day_limit (int): Maximum requests allowed per day.
			request_per_minute_limit (int): Maximum requests allowed per minute.
			tokens_per_minute_limit (int): Maximum tokens allowed per minute.
			context_used (int): The initial amount of context used.
			context_limit (int): The maximum amount of context allowed.
			raise_error_on_minute_limit (bool): Whether to raise exceptions when hitting minute limits.
		"""
		self.limit_day = limit_day
		self.request_per_day_limit = request_per_day_limit
		self.request_per_minute_limit = request_per_minute_limit
		self.tokens_per_minute_limit = tokens_per_minute_limit
		self.raise_error_on_minute_limit = raise_error_on_minute_limit
		self.request_per_day_used = request_per_day_used
		self.request_per_minute_used = 0
		self.tokens_per_minute_used = 0
		self.context_used = context_used
		self.context_limit = context_limit
		self.start_time = time.time()
	
	def check_limits(self, last_tokens: int):
		"""
		Checks if any rate limits have been exceeded. Resets minute counters if a minute has passed.
		Pauses execution or raises an error if a limit is exceeded, depending on raise_error_on_limit.

		Args:
			last_tokens (int): The number of tokens used in the last request.

		Raises:
			GeminiDayLimitException: If the daily request limit has been exceeded.
			GeminiMinuteLimitException: If the per-minute request or token limit has been exceeded and raise_error_on_limit is True.
			GeminiContextLimitException: If the context limit has been exceeded.
		"""
		elapsed_time = time.time() - self.start_time
		current_date = datetime.datetime.now(tz=pytz.timezone("America/New_York"))
		
		if current_date.date() == self.limit_day.date() and self.request_per_day_used > self.request_per_day_limit:
			raise errors.GeminiDayLimitException()
		
		if self.context_used > self.context_limit:
			raise errors.GeminiContextLimitException()
		
		if elapsed_time < 60:
			if self.request_per_day_used > self.request_per_day_limit:
				self.request_per_day_used = 1
				self.limit_day = datetime.datetime(
						year=current_date.year,
						month=current_date.month,
						day=current_date.day,
						tzinfo=current_date.tzinfo,
				)
			elif (
					self.request_per_minute_used > self.request_per_minute_limit
					or self.tokens_per_minute_used > self.tokens_per_minute_limit
			):
				if self.raise_error_on_minute_limit:
					raise errors.GeminiMinuteLimitException()
		
				time.sleep(60 - elapsed_time)
		
				self.request_per_minute_used = 1
				self.tokens_per_minute_used = last_tokens
		
				self.start_time = time.time()
		else:
			self.request_per_minute_used = 1
			self.tokens_per_minute_used = last_tokens
		
			self.start_time = time.time()
	
	def add_context(self, tokens: int):
		"""
		Adds to the context usage counter and checks if the context limit has been exceeded.

		Args:
			tokens (int): The number of tokens to add to the context usage.

		Raises:
			GeminiContextLimitException: If adding the tokens causes the context usage to exceed the limit.
		"""
		if self.context_used + tokens > self.context_limit:
			raise errors.GeminiContextLimitException()
		
		self.context_used += tokens
	
	def add_data(self, tokens: int):
		"""
		Increments the usage counters for requests, tokens and context.

		Args:
			tokens (int): The number of tokens used in the last request.
		"""
		self.request_per_day_used += 1
		self.request_per_minute_used += 1
		self.tokens_per_minute_used += tokens
		
		self.add_context(tokens)
		self.check_limits(tokens)
	
	def check_day_limits(self):
		"""
		Checks if the current day's request limit has been reached or if the date has changed.

		Returns:
			bool: True if requests can still be made within the daily limit, False otherwise.
		"""
		return self.request_per_day_used < self.request_per_day_limit or datetime.datetime.now(tz=pytz.timezone("America/New_York")).date() != self.limit_day.date()
	
	async def async_check_limits(self, last_tokens: int):
		"""
		Checks if any rate limits have been exceeded. Resets minute counters if a minute has passed.
		Pauses execution or raises an error if a limit is exceeded, depending on `raise_error_on_limit`.
		This is the asynchronous version of `check_limits`.

		Args:
			last_tokens (int): The number of tokens used in the last request.

		Raises:
			GeminiDayLimitException: If the daily request limit has been exceeded.
			GeminiMinuteLimitException: If the per-minute request or token limit has been exceeded and `raise_error_on_limit` is True.
			GeminiContextLimitException: If the context limit has been exceeded.
		"""
		elapsed_time = time.time() - self.start_time
		current_date = datetime.datetime.now(tz=pytz.timezone("America/New_York"))
		
		if not self.check_day_limits():
			raise errors.GeminiDayLimitException()
		
		if self.context_used > self.context_limit:
			raise errors.GeminiContextLimitException()
		
		if elapsed_time < 60:
			if self.request_per_day_used > self.request_per_day_limit:
				self.request_per_day_used = 1
				self.limit_day = datetime.datetime(
						year=current_date.year,
						month=current_date.month,
						day=current_date.day,
						tzinfo=current_date.tzinfo,
				)
			elif (
					self.request_per_minute_used > self.request_per_minute_limit
					or self.tokens_per_minute_used > self.tokens_per_minute_limit
			):
				if self.raise_error_on_minute_limit:
					raise errors.GeminiMinuteLimitException()
		
				await asyncio.sleep(60 - elapsed_time)
		
				self.request_per_minute_used = 1
				self.tokens_per_minute_used = last_tokens
		
				self.start_time = time.time()
		else:
			self.request_per_minute_used = 1
			self.tokens_per_minute_used = last_tokens
		
			self.start_time = time.time()
	
	async def async_add_data(self, tokens: int):
		"""
		Increments the usage counters for requests, tokens and context. This is the asynchronous version of `add_data`.

		Args:
			tokens (int): The number of tokens used in the last request.
		"""
		self.request_per_day_used += 1
		self.request_per_minute_used += 1
		self.tokens_per_minute_used += tokens
		
		self.add_context(tokens)
		await self.async_check_limits(tokens)
	
	def clear_context(self):
		"""
		Resets the current context usage count to 0.
		"""
		self.context_used = 0
	
	def close_day_limit(self):
		"""
		Sets the per-day usage counter to its limit, effectively blocking further requests for the current day.
		"""
		self.request_per_day_used = self.request_per_day_limit
	
	def close_minute_limit(self):
		"""
		Sets the per-minute usage counters to their limits, effectively blocking further requests for the current minute.
		"""
		self.request_per_minute_used = self.request_per_minute_limit
		self.tokens_per_minute_used = self.tokens_per_minute_limit
	
	def decrease_context(self, tokens: int):
		"""
		Decreases the current context usage count.

		Args:
			tokens (int): The number of tokens to decrease from the context usage.

		Raises:
			ValueError: If the decrease would result in a negative context usage count.
		"""
		if self.context_used - tokens < 0:
			raise ValueError("Cannot decrease context below 0")
		
		self.context_used -= tokens


class GeminiModelSettings:
	"""
	A class for configuring settings for a specific Gemini model.

	Attributes:
		model_name (str): The name of the Gemini model (e.g., "gemini-pro", "gemini-ultra").
		generation_config (genai_types.GenerateContentConfigOrDict, optional): Configuration for text generation, controlling aspects like temperature, top_p, and top_k. Defaults to a pre-defined, conservative configuration.
		limit_day (datetime.datetime, optional): The starting date for tracking daily request limits. Defaults to the current date in the "America/New_York" timezone.
		request_per_day_used (int): The number of requests made today. Defaults to 0.
		request_per_day_limit (int, optional): The maximum number of requests allowed per day. If not provided, it defaults to the value specified in `data.GeminiLimits` for the given `model_name`.
		request_per_minute_limit (int, optional): The maximum number of requests allowed per minute. Defaults to the model's limit in `data.GeminiLimits`.
		tokens_per_minute_limit (int, optional): The maximum number of tokens allowed per minute. Defaults to the model's limit in `data.GeminiLimits`.
		context_used (int): The initial amount of context used. Defaults to 0.
		context_limit (int, optional): The maximum amount of context allowed. Defaults to model's limit in `data.GeminiLimits`.
		raise_error_on_minute_limit (bool): If True, raises a `GeminiMinuteLimitException` when the per-minute rate limit is exceeded. If False, it pauses execution until the rate limit resets. Defaults to True.
	"""
	
	def __init__(
			self,
			model_name: str = data.GeminiModels.Gemini_2_0_flash.latest_stable,
			generation_config: typing.Optional[genai_types.GenerateContentConfigOrDict] = None,
			limit_day: typing.Optional[datetime.datetime] = None,
			request_per_day_used: int = 0,
			request_per_day_limit: typing.Optional[int] = None,
			request_per_minute_limit: typing.Optional[int] = None,
			tokens_per_minute_limit: typing.Optional[int] = None,
			context_used: int = 0,
			context_limit: typing.Optional[int] = None,
			raise_error_on_minute_limit: bool = True,
	):
		"""
		Initializes an instance of the GeminiSettings class.

		Args:
			model_name (str): The name of the Gemini model to use. Defaults to "gemini_2_0_flash".
			generation_config (genai.GenerationConfig): Configuration for text generation. Defaults to a conservative configuration.
			limit_day (datetime.datetime): The limit day for tracking usage limits. Defaults to the current day in the America/New_York time zone.
			request_per_day_used (int): The number of requests used so far today. Defaults to 0.
			request_per_day_limit (typing.Optional[int]): The maximum number of requests allowed per day. Defaults to the limit specified in "data.GeminiLimits" for the chosen model.
			request_per_minute_limit (typing.Optional[int]): The maximum number of requests allowed per minute. Defaults to the limit specified in "data.GeminiLimits" for the chosen model.
			tokens_per_minute_limit (typing.Optional[int]): The maximum number of tokens allowed per minute. Defaults to the limit specified in "data.GeminiLimits" for the chosen model.
			context_used (int): The initial amount of context used. Defaults to 0.
			context_limit (int, optional): The maximum amount of context allowed. Defaults to model's limit in data.GeminiLimits
			raise_error_on_minute_limit (bool): Whether to raise an error when a minute rate limit is exceeded. Defaults to True.
		"""
		if generation_config is None:
			generation_config = genai_types.GenerateContentConfigDict(
					temperature=0.7,
					top_p=0.5,
					top_k=40,
					candidate_count=1,
					response_mime_type=data.GeminiMimeTypes.text_plain,
					safety_settings=[
						genai_types.SafetySettingDict(
								category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
								threshold=genai_types.HarmBlockThreshold.OFF
						),
						genai_types.SafetySettingDict(
								category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
								threshold=genai_types.HarmBlockThreshold.OFF
						),
						genai_types.SafetySettingDict(
								category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
								threshold=genai_types.HarmBlockThreshold.OFF
						),
						genai_types.SafetySettingDict(
								category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
								threshold=genai_types.HarmBlockThreshold.OFF
						),
						genai_types.SafetySettingDict(
								category=genai_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
								threshold=genai_types.HarmBlockThreshold.OFF
						)
					]
			)
		
		if limit_day is None:
			limit_day = datetime.datetime.now(tz=pytz.timezone("America/New_York"))
		else:
			limit_day = limit_day.astimezone(pytz.timezone("America/New_York"))
		
		self.model_name = model_name
		self.generation_config = generation_config
		self.raise_error_on_minute_limit = raise_error_on_minute_limit
		self.request_per_day_used = request_per_day_used
		self.context_used = context_used
		
		self.limit_day = datetime.datetime(
				year=limit_day.year,
				month=limit_day.month,
				day=limit_day.day,
				tzinfo=limit_day.tzinfo,
		)
		
		base_model_name = find_base_model(model_name)
		
		if request_per_day_limit is None:
			if base_model_name not in data.GeminiLimits.request_per_day:
				raise ValueError(
						f"{model_name} is not a default model name. Specify 'request_per_day_limit'."
				)
			
			self.request_per_day_limit = data.GeminiLimits.request_per_day[base_model_name]
		else:
			self.request_per_day_limit = request_per_day_limit
		
		if request_per_minute_limit is None:
			if base_model_name not in data.GeminiLimits.request_per_minute:
				raise ValueError(
						f"{model_name} is not a default model name. Specify 'request_per_minute_limit'."
				)
			
			self.request_per_minute_limit = data.GeminiLimits.request_per_minute[base_model_name]
		else:
			self.request_per_minute_limit = request_per_minute_limit
		
		if tokens_per_minute_limit is None:
			if base_model_name not in data.GeminiLimits.tokens_per_minute:
				raise ValueError(
						f"{model_name} is not a default model name. Specify 'tokens_per_minute_limit'."
				)
			
			self.tokens_per_minute_limit = data.GeminiLimits.tokens_per_minute[base_model_name]
		else:
			self.tokens_per_minute_limit = tokens_per_minute_limit
		
		if context_limit is None:
			if base_model_name not in data.GeminiLimits.context_limit:
				raise ValueError(f"{model_name} is not a default model name. Specify 'context_limit'.")
			
			self.context_limit = data.GeminiLimits.context_limit[base_model_name]
		else:
			self.context_limit = context_limit


class GeminiModel:
	"""
	A class representing a specific Gemini model and its associated limiter.

	Attributes:
		model_name (str): The name of the Gemini model.
		generation_config (genai_types.GenerateContentConfigOrDict): Configuration for text generation.
		limiter (GeminiLimiter): A `GeminiLimiter` instance to manage rate limits for this model.
	"""
	
	def __init__(self, gemini_model_settings: GeminiModelSettings):
		"""
		Initializes a GeminiModel instance.

		Args:
			gemini_model_settings (GeminiModelSettings): A GeminiModelSettings object containing the model configuration.
		"""
		self.model_name = gemini_model_settings.model_name
		self.generation_config = gemini_model_settings.generation_config
		
		self.limiter = GeminiLimiter(
				limit_day=gemini_model_settings.limit_day,
				request_per_day_used=gemini_model_settings.request_per_day_used,
				request_per_day_limit=gemini_model_settings.request_per_day_limit,
				request_per_minute_limit=gemini_model_settings.request_per_minute_limit,
				tokens_per_minute_limit=gemini_model_settings.tokens_per_minute_limit,
				context_used=gemini_model_settings.context_used,
				context_limit=gemini_model_settings.context_limit,
				raise_error_on_minute_limit=gemini_model_settings.raise_error_on_minute_limit
		)
		
		self.settings = gemini_model_settings


class BaseGeminiChat:
	"""
	A class representing a chat session with a Gemini model.

	Attributes:
		client (genai.Client): The `genai.Client` instance used to interact with the Gemini API.
		chat (typing.Union[Chat, AsyncChat]): The underlying `Chat` object.
		limiter (GeminiLimiter): A `GeminiLimiter` instance to manage rate limits for this chat session.
		settings (GeminiModelSettings): The settings for the Gemini chat.
	"""
	
	def __init__(
			self,
			client: genai.Client,
			model_settings: GeminiModelSettings,
			is_async: typing.Optional[bool] = None,
			history: typing.Optional[list[types.gemini_history]] = None
	):
		"""
		Initializes a `GeminiChat` instance.

		Args:
			client (genai.Client): The `genai.Client` instance.
			model_settings (GeminiModelSettings): The settings for the Gemini model.
			history (typing.Optional[list[types.gemini_history]]): The initial chat history. Defaults to None.
		"""
		self.client = client
		
		self.chat, self.model_name = self.create_chat(model_settings=model_settings, history=history)
		
		self.limiter = GeminiLimiter(
				limit_day=model_settings.limit_day,
				request_per_day_used=model_settings.request_per_day_used,
				request_per_day_limit=model_settings.request_per_day_limit,
				request_per_minute_limit=model_settings.request_per_minute_limit,
				tokens_per_minute_limit=model_settings.tokens_per_minute_limit,
				context_used=sum(
						self.client.models.count_tokens(model=model_settings.model_name, contents=message).total_tokens
						for message in history
				) if history is not None else 0,
				context_limit=model_settings.context_limit,
				raise_error_on_minute_limit=model_settings.raise_error_on_minute_limit
		)
		
		self.settings = model_settings
		self.is_async = is_async
	
	def create_chat(
			self,
			model_settings: GeminiModelSettings,
			history: typing.Optional[list[types.gemini_history]] = None
	) -> tuple[typing.Union[Chat, AsyncChat], str]:
		pass
	
	@property
	def history(self) -> list[genai_types.Content]:
		"""
		Returns the history of the chat session.

		Returns:
			list[genai_types.Content]: The history of the chat.
		"""
		return self.chat._curated_history
	
	def change_settings(self, model_settings: GeminiModelSettings):
		"""
		Changes the settings used in the chat session, preserving the chat history. The new model's limiter is also initialized, and its context usage is set based on the existing history.

		Args:
			model_settings (GeminiModelSettings): The settings for the new model to use.
		"""
		history = self.history
		
		self.chat, self.model_name = self.create_chat(model_settings=model_settings, history=history)
		self.limiter = GeminiLimiter(
				limit_day=model_settings.limit_day,
				request_per_day_used=model_settings.request_per_day_used,
				request_per_day_limit=model_settings.request_per_day_limit,
				request_per_minute_limit=model_settings.request_per_minute_limit,
				tokens_per_minute_limit=model_settings.tokens_per_minute_limit,
				context_used=self.client.models.count_tokens(model=model_settings.model_name, contents=history).total_tokens,
				context_limit=model_settings.context_limit,
				raise_error_on_minute_limit=model_settings.raise_error_on_minute_limit
		)
		self.settings = model_settings
	
	def clear_chat_history(self):
		"""
		Clears the history of the current chat session and resets the context usage to 0.

		This effectively starts a new conversation within the same `BaseGeminiChat` object, but with a clean slate.
		"""
		self.chat, self.model_name = self.create_chat(model_settings=self.settings, history=[])
		self.limiter.clear_context()
	
	@property
	def context_usage(self) -> dict[str, int]:
		"""
		Returns the current context usage and limit.

		Returns:
			dict[str, int]: A dictionary containing `context_used` and `context_limit`.
		"""
		return {
			"context_used": self.limiter.context_used,
			"context_limit": self.limiter.context_limit
		}
	
	@property
	def current_limit_day(self) -> datetime.datetime:
		"""
		Returns the current day being used for tracking daily limits.

		Returns:
			datetime.datetime: Current day being used for tracking daily limits.
		"""
		return self.limiter.limit_day
	
	@property
	def day_usage(self) -> dict[str, typing.Union[int, datetime.datetime]]:
		"""
		Returns the current per-day usage and limits.

		Returns:
			dict[str, typing.Union[int, datetime.datetime]]: A dictionary containing `used_requests`, `requests_limit`, and `date`.
		"""
		return {
			"used_requests": self.limiter.request_per_day_used,
			"requests_limit": self.limiter.request_per_day_limit,
			"date": self.limiter.limit_day,
		}
	
	@property
	def minute_usage(self) -> dict[str, int]:
		"""
		Returns the current per-minute usage and limits.

		Returns:
			dict[str, int]: A dictionary containing `used_requests`, `requests_limit`, `used_tokens`, and `tokens_limit`.
		"""
		return {
			"used_requests": self.limiter.request_per_minute_used,
			"requests_limit": self.limiter.request_per_minute_limit,
			"used_tokens": self.limiter.tokens_per_minute_used,
			"tokens_limit": self.limiter.tokens_per_minute_limit,
		}
	
	def reset_history(
			self,
			history: list[types.gemini_history],
			count_tokens_config: typing.Optional[genai_types.CountTokensConfigOrDict] = None
	):
		"""
		Resets the chat history with a new history and updates the context usage accordingly.

		Args:
			history (list[types.gemini_history]): The new chat history to set.
			count_tokens_config (typing.Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens when updating context usage. If None, default configuration will be used.
		"""
		self.chat, self.model_name = self.create_chat(model_settings=self.settings, history=history)
		self.limiter.context_used = self.client.models.count_tokens(model=self.model_name, contents=history, config=count_tokens_config).total_tokens
	
	def slice_history(
			self,
			start: typing.Optional[int] = None,
			end: typing.Optional[int] = None,
			count_tokens_config: typing.Optional[genai_types.CountTokensConfigOrDict] = None
	):
		"""
		Slices the current chat history, keeping only a portion of it, and resets the chat history to this sliced part.

		This is useful for managing context window size by discarding older messages.

		Args:
			start (typing.Optional[int]): The starting index for the slice. If None, defaults to the beginning of the history.
			end (typing.Optional[int]): The ending index for the slice (exclusive). If None, defaults to the end of the history.
			count_tokens_config (typing.Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens when updating context usage. If None, default configuration will be used.
		"""
		self.reset_history(self.history[slice(start, end)], count_tokens_config)


class GeminiAsyncChat(BaseGeminiChat):
	"""
	A class representing a chat session with a Gemini model. This class encapsulates the `AsyncChat` object and manages its own rate limiting.

	Attributes:
		client (genai.Client): The `genai.Client` instance used to interact with the Gemini API.
		chat (AsyncChat): The underlying `AsyncChat` object.
		limiter (GeminiLimiter): A `GeminiLimiter` instance to manage rate limits for this chat session.
	"""
	
	def __init__(
			self,
			client: genai.Client,
			model_settings: GeminiModelSettings,
			history: typing.Optional[list[types.gemini_history]] = None
	):
		super().__init__(
				client=client,
				model_settings=model_settings,
				is_async=True,
				history=history
		)
	
	def create_chat(
			self,
			model_settings: GeminiModelSettings,
			history: typing.Optional[list[types.gemini_history]] = None
	) -> tuple[AsyncChat, str]:
		return (
				self.client.aio.chats.create(
						model=model_settings.model_name,
						config=model_settings.generation_config,
						history=history
				),
				model_settings.model_name
		)
	
	async def send_message(
			self,
			message: types.gemini_message_input,
			count_tokens_config: typing.Optional[genai_types.CountTokensConfigOrDict] = None
	) -> GenerateContentResponse:
		"""
		Sends a message to a chat session.

		Args:
			message (str): The message to send.
			count_tokens_config (typing.Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.

		Returns:
			GenerateContentResponse: The response from the Gemini model.
		"""
		await self.limiter.async_add_data(
				self.client.models.count_tokens(model=self.model_name, contents=message, config=count_tokens_config).total_tokens
		)
		
		response = await self.chat.send_message(message=message)
		self.limiter.add_context(extract_token_count_from_gemini_response(response))
		
		return response
	
	async def send_message_stream(
			self,
			message: types.gemini_message_input,
			count_tokens_config: typing.Optional[genai_types.CountTokensConfigOrDict] = None
	) -> typing.AsyncGenerator[GenerateContentResponse, typing.Any]:
		"""
		Sends a message to a chat session and returns stream.

		Args:
			message (str): The message to send.
			count_tokens_config (typing.Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.

		Returns:
			typing.AsyncGenerator[GenerateContentResponse, typing.Any]: The response from the Gemini model.
		"""
		await self.limiter.async_add_data(
				self.client.models.count_tokens(model=self.model_name, contents=message, config=count_tokens_config).total_tokens
		)
		
		async for response in await self.chat.send_message_stream(message=message):
			self.limiter.add_context(extract_token_count_from_gemini_response(response))
			yield response


class GeminiChat(BaseGeminiChat):
	"""
	A class representing a chat session with a Gemini model. This class encapsulates the `Chat` object and manages its own rate limiting.

	Attributes:
		client (genai.Client): The `genai.Client` instance used to interact with the Gemini API.
		chat (Chat): The underlying `Chat` object.
		limiter (GeminiLimiter): A `GeminiLimiter` instance to manage rate limits for this chat session.
	"""
	
	def __init__(
			self,
			client: genai.Client,
			model_settings: GeminiModelSettings,
			history: typing.Optional[list[types.gemini_history]] = None
	):
		"""
		Initializes a `GeminiChat` instance.

		Args:
			client (genai.Client): The `genai.Client` instance.
			model_settings (GeminiModelSettings): The settings for the Gemini model.
			history (typing.Optional[list[types.gemini_history]]): The initial chat history. Defaults to None.
		"""
		super().__init__(
				client=client,
				model_settings=model_settings,
				is_async=False,
				history=history
		)
	
	def create_chat(
			self,
			model_settings: GeminiModelSettings,
			history: typing.Optional[list[types.gemini_history]] = None
	) -> tuple[Chat, str]:
		return (
				self.client.chats.create(
						model=model_settings.model_name,
						config=model_settings.generation_config,
						history=history
				),
				model_settings.model_name
		)
	
	def send_message(
			self,
			message: types.gemini_message_input,
			count_tokens_config: typing.Optional[genai_types.CountTokensConfigOrDict] = None
	) -> GenerateContentResponse:
		"""
		Sends a message to a chat session.

		Args:
			message (str): The message to send.
			count_tokens_config (typing.Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.

		Returns:
			GenerateContentResponse: The response from the Gemini model.
		"""
		self.limiter.add_data(
				self.client.models.count_tokens(model=self.model_name, contents=message, config=count_tokens_config).total_tokens
		)
		
		response = self.chat.send_message(message=message)
		self.limiter.add_context(extract_token_count_from_gemini_response(response))
		
		return response
	
	def send_message_stream(
			self,
			message: types.gemini_message_input,
			count_tokens_config: typing.Optional[genai_types.CountTokensConfigOrDict] = None
	) -> typing.Generator[GenerateContentResponse, typing.Any, None]:
		"""
		Sends a message to a chat session and returns stream.

		Args:
			message (str): The message to send.
			count_tokens_config (typing.Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.

		Returns:
			typing.Generator[GenerateContentResponse, typing.Any, None]: The response from the Gemini model.
		"""
		self.limiter.add_data(
				self.client.models.count_tokens(model=self.model_name, contents=message, config=count_tokens_config).total_tokens
		)
		
		for response in self.chat.send_message_stream(message=message):
			self.limiter.add_context(extract_token_count_from_gemini_response(response))
			yield response


class GeminiClientSettings:
	"""
	A class for configuring settings for a Gemini client.

	Attributes:
		api_key (str): Your Gemini API key.
		model_settings (GeminiModelSettings, optional): Settings for the Gemini model. If `None`, uses default `GeminiModelSettings`.
		chats (list[typing.Union[GeminiChat, GeminiAsyncChat]], optional): list of chats.
	"""
	
	def __init__(
			self,
			api_key: str,
			model_settings: typing.Optional[GeminiModelSettings] = None,
			chats: typing.Optional[list[typing.Union[GeminiChat, GeminiAsyncChat]]] = None
	):
		"""
		Initializes an instance of the GeminiClientSettings class.

		Args:
			api_key (str): Your Gemini API key.
			model_settings (GeminiModelSettings, optional): Settings for the Gemini model. If `None`, uses default `GeminiModelSettings`.
			chats (list[typing.Union[GeminiChat, GeminiAsyncChat]], optional): list of chats.
		"""
		self.api_key = api_key
		
		self.model_settings = model_settings if model_settings is not None else GeminiModelSettings()
		
		self.chats = chats if chats is not None else []


class GeminiClient:
	"""
	A wrapper class for interacting with Google Gemini models using the `genai` library.

	Attributes:
		api_key (str): The API key used for authentication.
		client (genai.Client): The underlying Google AI client instance.
		model (GeminiModel): The Gemini model configuration and limiter.
		chats (List[typing.Union[GeminiChat, GeminiAsyncChat]]): A list of active chat sessions, which can be either synchronous or asynchronous.
	"""
	
	def __init__(self, gemini_client_settings: GeminiClientSettings):
		"""
		Initializes a new Gemini instance.

		Args:
			gemini_client_settings (GeminiClientSettings): An instance of GeminiClientSettings containing configuration parameters.
		"""
		self.client = genai.Client(api_key=gemini_client_settings.api_key)
		
		self.api_key = gemini_client_settings.api_key
		self.model = GeminiModel(gemini_client_settings.model_settings)
		self.chats: list[typing.Union[GeminiChat, GeminiAsyncChat]] = gemini_client_settings.chats
	
	@property
	def generation_config(self) -> genai_types.GenerateContentConfigOrDict:
		"""
		Returns the generation config of current model.

		Returns:
			genai_types.GenerateContentConfigOrDict: The generation config using in current model.
		"""
		return self.model.generation_config
	
	@property
	def model_name(self) -> str:
		"""
		Returns name of current model.

		Returns:
			str: The name of current model.
		"""
		return self.model.model_name
	
	async def async_generate_content(
			self,
			message: types.gemini_generate_input,
			count_tokens_config: typing.Optional[genai_types.CountTokensConfigOrDict] = None,
			generate_config: typing.Optional[genai_types.GenerateContentConfigOrDict] = None
	) -> GenerateContentResponse:
		"""
		Asynchronously generates content based on the provided message. This is the asynchronous version of `generate_content`.

		Args:
			message (types.gemini_generate_input): The input message.
			count_tokens_config (typing.Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.
			generate_config (typing.Optional[genai_types.GenerateContentConfigOrDict]): Overrides the default generation config.

		Returns:
			typing.Coroutine[typing.Any, typing.Any, GenerateContentResponse]: A coroutine that resolves to the generated content response.
		"""
		await self.limiter.async_add_data(
				self.client.models.count_tokens(model=self.model_name, contents=message, config=count_tokens_config).total_tokens
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
			count_tokens_config: typing.Optional[genai_types.CountTokensConfigOrDict] = None,
			generate_config: typing.Optional[genai_types.GenerateContentConfigOrDict] = None
	) -> typing.AsyncGenerator[GenerateContentResponse, typing.Any]:
		"""
		Asynchronously generates content as a stream. This is the asynchronous version of `generate_content_stream`.

		This allows you to process parts of the response as they become available,
		rather than waiting for the entire response to be generated.

		Args:
			message (types.gemini_generate_input): The input message.
			count_tokens_config (typing.Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.
			generate_config (typing.Optional[genai_types.GenerateContentConfigOrDict]): Overrides the default generation config.

		Returns:
			typing.AsyncGenerator[GenerateContentResponse, typing.Any]: An async iterator that yields `GenerateContentResponse` objects as they become available.
		"""
		await self.limiter.async_add_data(
				self.client.models.count_tokens(model=self.model_name, contents=message, config=count_tokens_config).total_tokens
		)
		
		async for response in await self.client.aio.models.generate_content_stream(
				model=self.model_name,
				contents=message,
				config=generate_config
				if generate_config is not None
				else self.generation_config
		):
			yield response
	
	def chat(self, chat_index: int = -1) -> typing.Union[GeminiChat, GeminiAsyncChat]:
		"""
		Returns a specific chat session.

		Args:
			chat_index (int): The index of the chat session to retrieve. Defaults to -1, which returns the last created chat session.

		Returns:
			typing.Union[GeminiChat, GeminiAsyncChat]: The `GeminiChat` or 'GeminiAsyncChat' object at the specified index.
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
	
	async def async_send_message_stream(self, message: types.gemini_message_input, chat_index: int = -1) -> typing.AsyncGenerator[GenerateContentResponse, typing.Any]:
		"""
		Sends a message to an asynchronous chat session and returns an asynchronous stream of responses.

		Args:
			message (types.gemini_message_input): The message to send to the async chat session.
			chat_index (int): The index of the asynchronous chat session to send the message to. Defaults to -1, which targets the last created asynchronous chat session.

		Returns:
			typing.AsyncGenerator[GenerateContentResponse, typing.Any]: An async generator that yields responses from the Gemini model as they become available in a stream.

		Raises:
			GeminiChatTypeException: If the chat session at the given index is not an asynchronous chat (`GeminiAsyncChat`).
		"""
		chat = self.chat(chat_index)
		
		if not chat.is_async:
			raise errors.GeminiChatTypeException(chat_index, "asynchronous")
		
		return chat.send_message_stream(message=message)
	
	def check_day_limits(self) -> bool:
		"""
		Checks if request can be provided in current day limits

		Returns:
			bool: True if request can be provided, False if not
		"""
		return self.limiter.check_day_limits()
	
	def close_chat(self, chat_index: int = -1):
		"""
		Closes a chat session.

		Args:
			chat_index (int): The index of the chat session to close. Defaults to -1 (the last chat session).
		"""
		self.chats.pop(chat_index)
	
	def close_day_limit(self):
		"""Manually closes the day limit."""
		self.limiter.close_day_limit()
	
	def close_minute_limit(self):
		"""Manually closes the minute limit."""
		self.limiter.close_minute_limit()
	
	@property
	def context_usage(self) -> dict[str, int]:
		"""
		Returns the current context usage and limit.

		Returns:
			dict[str, int]: A dictionary containing `context_used` and `context_limit`.
		"""
		return {
			"context_used": self.limiter.context_used,
			"context_limit": self.limiter.context_limit
		}
	
	@property
	def current_limit_day(self) -> datetime.datetime:
		"""
		Returns the current day being used for tracking daily limits.

		Returns:
			datetime.datetime: Current day being used for tracking daily limits.
		"""
		return self.limiter.limit_day
	
	@property
	def day_usage(self) -> dict[str, typing.Union[int, datetime.datetime]]:
		"""
		Returns the current per-day usage and limits.

		Returns:
			dict[str, typing.Union[int, datetime.datetime]]: A dictionary containing `used_requests`, `requests_limit`, and `date`.
		"""
		return {
			"used_requests": self.limiter.request_per_day_used,
			"requests_limit": self.limiter.request_per_day_limit,
			"date": self.limiter.limit_day
		}
	
	def generate_content(
			self,
			message: types.gemini_generate_input,
			count_tokens_config: typing.Optional[genai_types.CountTokensConfigOrDict] = None,
			generate_config: typing.Optional[genai_types.GenerationConfigOrDict] = None
	) -> GenerateContentResponse:
		"""
		Generates content based on the provided message using the configured Gemini model.

		Args:
			message (types.gemini_generate_input): The input message or content to generate from. Can be a string, a list of strings, or a more complex structure defined in `types.gemini_generate_input`.
			count_tokens_config (typing.Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.
			generate_config (typing.Optional[genai_types.GenerateContentConfigOrDict]): Overrides the default `generation_config` for this specific call.
		"""
		self.limiter.add_data(
				self.client.models.count_tokens(model=self.model_name, contents=message, config=count_tokens_config).total_tokens
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
			count_tokens_config: typing.Optional[genai_types.CountTokensConfigOrDict] = None,
			generate_config: typing.Optional[genai_types.GenerateContentConfigOrDict] = None
	) -> typing.Generator[GenerateContentResponse, typing.Any, None]:
		"""
		Generates content as a stream (synchronous version).

		Args:
			message (types.gemini_generate_input): The input message.
			count_tokens_config (typing.Optional[genai_types.CountTokensConfigOrDict]): Configuration for counting tokens.
			generate_config (typing.Optional[genai_types.GenerateContentConfigOrDict]): Overrides the default generation config.

		Returns:
			typing.Generator[GenerateContentResponse, typing.Any, None]: An iterator that yields `GenerateContentResponse` objects.
		"""
		self.limiter.add_data(
				self.client.models.count_tokens(model=self.model_name, contents=message, config=count_tokens_config).total_tokens
		)
		
		for response in self.client.models.generate_content_stream(
				model=self.model_name,
				contents=message,
				config=generate_config
				if generate_config is not None
				else self.generation_config
		):
			yield response
	
	def get_chats(self) -> list[typing.Union[GeminiChat, GeminiAsyncChat]]:
		"""
		Returns the list of chat sessions managed by this `GeminiClient`.

		Returns:
			list[typing.Union[GeminiChat, GeminiAsyncChat]]: A list of `GeminiChat` and `GeminiAsyncChat` objects, representing the active chat sessions.
		"""
		return self.chats
	
	@property
	def limiter(self) -> GeminiLimiter:
		"""
		Returns the limiter of current model.

		Returns:
			GeminiLimiter: The limiter using in current model.
		"""
		return self.model.limiter
	
	@property
	def minute_usage(self) -> dict[str, int]:
		"""
		Returns the current per-minute usage and limits.

		Returns:
			dict[str, int]: A dictionary containing `used_requests`, `requests_limit`, `used_tokens`, and `tokens_limit`.
		"""
		return {
			"used_requests": self.limiter.request_per_minute_used,
			"requests_limit": self.limiter.request_per_minute_limit,
			"used_tokens": self.limiter.tokens_per_minute_used,
			"tokens_limit": self.limiter.tokens_per_minute_limit,
		}
	
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
	
	def send_message_stream(self, message: types.gemini_message_input, chat_index: int = -1) -> typing.Generator[GenerateContentResponse, typing.Any, None]:
		"""
		Sends a message to a synchronous chat session and returns a stream of responses.

		Args:
			message (types.gemini_message_input): The message to send to the chat session.
			chat_index (int): The index of the synchronous chat session to send the message to. Defaults to -1, which targets the last created synchronous chat session.

		Returns:
			typing.Generator[GenerateContentResponse, typing.Any, None]: A generator that yields responses from the Gemini model as they become available in a stream.

		Raises:
			GeminiChatTypeException: If the chat session at the given index is not a synchronous chat (`GeminiChat`).
		"""
		chat = self.chat(chat_index)
		
		if chat.is_async:
			raise errors.GeminiChatTypeException(chat_index, "synchronous")
		
		return chat.send_message_stream(message=message)
	
	@property
	def model_settings(self) -> GeminiModelSettings:
		"""
		Returns the model settings of current model.

		Returns:
			GeminiModelSettings: The model settings using in current model.
		"""
		return self.model.settings
	
	def start_async_chat(
			self,
			model_settings: typing.Optional[GeminiModelSettings] = None,
			history: typing.Optional[list[types.gemini_message_input]] = None
	):
		"""
		Starts new async chat and appends to chats list

		Args:
			model_settings (typing.Optional[genai_types.GenerateContentConfigOrDict]): Overrides the default `model_settings` for this specific call.
			history (typing.Optional[list[types.gemini_message_input]], optional): you can specify the history for this chat.
		"""
		self.chats.append(
				GeminiAsyncChat(
						client=self.client,
						model_settings=model_settings
						if model_settings is not None
						else self.model_settings,
						history=history if history is not None else []
				)
		)
	
	def start_chat(
			self,
			model_settings: typing.Optional[GeminiModelSettings] = None,
			history: typing.Optional[list[types.gemini_message_input]] = None
	):
		"""
		Starts new chat and appends to chats list

		Args:
			model_settings (typing.Optional[GeminiModelSettings], optional): Overrides the default `model_settings` for this specific call.
			history (typing.Optional[list[types.gemini_message_input]], optional): you can specify the history for this chat
		"""
		self.chats.append(
				GeminiChat(
						client=self.client,
						model_settings=model_settings
						if model_settings is not None
						else self.model_settings,
						history=history if history is not None else []
				)
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
	def next_client(self) -> GeminiClient:
		"""
		Switches to the next available Gemini model.

		Returns:
			GeminiClient: The next available Gemini instance.
		"""
		self.current_model_index = (self.current_model_index + 1) % len(self.clients) if self.current_model_index is not None else 0
		
		return self.client()
	
	@property
	def has_useful_model(self) -> bool:
		"""
		Checks if any of the managed models have available quota.

		Returns:
			bool: True if any model has available quota, False otherwise.
		"""
		return any(client_settings.check_day_limits() for client_settings in self.clients)
	
	@property
	def lowest_useful_client_index(self) -> typing.Optional[int]:
		"""
		Finds the index of the first model with available quota.

		Returns:
			typing.Optional[int]: The index of the first available model, None if no models have available quota.
		"""
		if self.has_useful_model:
			for i in range(len(self.clients)):
				if self.clients[i].check_day_limits():
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
