import time
import pytz
import asyncio
from datetime import datetime
from typing import Any, Optional, Union
from PyVarTools.python_instances_tools import get_function_parameters
from PyGPTs.Gemini.errors import (
	GeminiContextLimitException,
	GeminiDayLimitException,
	GeminiMinuteLimitException
)


class GeminiLimiterSettings:
	"""
	A class for configuring settings for a Gemini limiter.

	Attributes:
		limit_day (datetime): The starting date for tracking daily request limits. Defaults to the current date in the "America/New_York" timezone.
		request_per_day_used (int): The number of requests made today. Defaults to 0.
		request_per_day_limit (int, optional): The maximum number of requests allowed per day. If not provided, it needs to be set externally or will default to `None`.
		request_per_minute_limit (int, optional): The maximum number of requests allowed per minute. If not provided, it needs to be set externally or will default to `None`.
		tokens_per_minute_limit (int, optional): The maximum number of tokens allowed per minute. If not provided, it needs to be set externally or will default to `None`.
		context_used (int): The initial amount of context used. Defaults to 0.
		context_limit (int, optional): The maximum amount of context allowed. If not provided, it needs to be set externally or will default to `None`.
		raise_error_on_minute_limit (bool): If True, raises a `GeminiMinuteLimitException` when the per-minute rate limit is exceeded. If False, it pauses execution until the rate limit resets. Defaults to True.
	"""
	
	def __init__(
			self,
			limit_day: Optional[datetime] = None,
			request_per_day_used: int = 0,
			request_per_day_limit: Optional[int] = None,
			request_per_minute_limit: Optional[int] = None,
			tokens_per_minute_limit: Optional[int] = None,
			context_used: int = 0,
			context_limit: Optional[int] = None,
			raise_error_on_minute_limit: bool = True
	):
		"""
		Initializes an instance of the GeminiLimiter class.

		Args:
			limit_day (Optional[datetime]): The limit day for tracking daily usage.
			request_per_day_used (int): Initial count of requests used per day.
			request_per_day_limit (Optional[int]): Maximum requests allowed per day.
			request_per_minute_limit (Optional[int]): Maximum requests allowed per minute.
			tokens_per_minute_limit (Optional[int]): Maximum tokens allowed per minute.
			context_used (int): The initial amount of context used.
			context_limit (Optional[int]): The maximum amount of context allowed.
			raise_error_on_minute_limit (bool): Whether to raise exceptions when hitting minute limits.
		"""
		if limit_day is None:
			limit_day = datetime.now().astimezone(pytz.timezone("America/New_York"))
		else:
			limit_day = limit_day.astimezone(pytz.timezone("America/New_York"))
		
		self.limit_day = datetime(
				year=limit_day.year,
				month=limit_day.month,
				day=limit_day.day,
				tzinfo=limit_day.tzinfo
		)
		
		self.request_per_day_used = request_per_day_used
		self.request_per_day_limit = request_per_day_limit
		self.request_per_minute_limit = request_per_minute_limit
		self.tokens_per_minute_limit = tokens_per_minute_limit
		self.context_used = context_used
		self.context_limit = context_limit
		self.raise_error_on_minute_limit = raise_error_on_minute_limit
	
	def to_dict(self) -> dict[str, Any]:
		"""
		Converts the GeminiLimiterSettings object to a dictionary.

		Returns:
			dict[str, Any]: A dictionary representation of the GeminiLimiterSettings object.
		"""
		return {
			"limit_day": self.limit_day,
			"request_per_day_used": self.request_per_day_used,
			"request_per_day_limit": self.request_per_day_limit,
			"request_per_minute_limit": self.request_per_minute_limit,
			"tokens_per_minute_limit": self.tokens_per_minute_limit,
			"context_used": self.context_used,
			"context_limit": self.context_limit,
			"raise_error_on_minute_limit": self.raise_error_on_minute_limit
		}


class GeminiLimiter:
	"""
	Manages rate limiting for Gemini API requests.

	Attributes:
		limit_day (datetime): The start day for tracking daily usage limits.
		request_per_day_used (int): The number of requests used so far today.
		request_per_day_limit (int): The maximum number of requests allowed per day.
		request_per_minute_limit (int): The maximum number of requests allowed per minute.
		tokens_per_minute_limit (int): The maximum number of tokens allowed per minute.
		context_used (int): Represents the current amount of context used.
		context_limit (int): The maximum allowed context usage.
		raise_error_on_minute_limit (bool): Whether to raise an error when a rate limit is exceeded. Defaults to True.
		request_per_minute_used (int): The number of requests used so far this minute.
		tokens_per_minute_used (int): The number of tokens used so far this minute.
		start_time (float): The timestamp of the start of the current minute.
	"""
	
	def __init__(self, limiter_settings: GeminiLimiterSettings):
		"""
		Initializes an instance of the GeminiLimiter class.

		Args:
			limiter_settings (GeminiLimiterSettings): Settings object containing all limiter configurations.
		"""
		self.limit_day = limiter_settings.limit_day
		self.request_per_day_used = limiter_settings.request_per_day_used
		self.request_per_day_limit = limiter_settings.request_per_day_limit
		self.request_per_minute_limit = limiter_settings.request_per_minute_limit
		self.tokens_per_minute_limit = limiter_settings.tokens_per_minute_limit
		self.context_used = limiter_settings.context_used
		self.context_limit = limiter_settings.context_limit
		self.raise_error_on_minute_limit = limiter_settings.raise_error_on_minute_limit
		self.request_per_minute_used = 0
		self.tokens_per_minute_used = 0
		self.start_time = time.time()
	
	@property
	def has_minute_limits(self) -> bool:
		"""
		Checks if both the per-minute request and token usage are within their respective limits.

		Returns:
			bool: True if both `request_per_minute_used` is less than `request_per_minute_limit` and `tokens_per_minute_used` is less than `tokens_per_minute_limit`, False otherwise.
		"""
		return (
				self.request_per_minute_used < self.request_per_minute_limit
				and self.tokens_per_minute_used < self.tokens_per_minute_limit
		)
	
	def restart_minute_counters(self, last_tokens: int):
		"""
		Restarts the per-minute usage counters and resets the `start_time`.

		This method sets `request_per_minute_used` to 1, `tokens_per_minute_used` to the provided `last_tokens` value, and updates `start_time` to the current time.
		This is typically called when a new minute begins or after a minute limit pause.

		Args:
			last_tokens (int): The token count of the last processed request, which is used to initialize `tokens_per_minute_used`.
		"""
		self.request_per_minute_used = 1
		self.tokens_per_minute_used = last_tokens
		
		self.start_time = time.time()
	
	@property
	def minute_exceeded(self) -> bool:
		"""
		Checks if a minute has passed since the `start_time`.

		Returns:
			bool: True if 60 seconds or more have elapsed since `start_time`, False otherwise.
		"""
		return time.time() - self.start_time >= 60
	
	def restart_day_counters(self):
		"""
		Restarts the per-day usage counters and updates the `limit_day` to the current date.

		This method sets `request_per_day_used` to 1 and updates `limit_day` to the current date in "America/New_York" timezone.
		This is typically called when a new day begins for resetting daily limits.
		"""
		self.request_per_day_used = 1
		current_date = datetime.now(tz=pytz.timezone("America/New_York"))
		
		self.limit_day = datetime(
				year=current_date.year,
				month=current_date.month,
				day=current_date.day,
				tzinfo=current_date.tzinfo,
		)
	
	@property
	def has_context(self) -> bool:
		"""
		Checks if the current context usage is below the context limit.

		Returns:
			bool: True if `context_used` is less than `context_limit`, False otherwise.
		"""
		return self.context_used < self.context_limit
	
	@property
	def has_day_limits(self) -> bool:
		"""
		Checks if the current day's request limit has been reached or if the date has changed.

		Returns:
			bool: True if requests can still be made within the daily limit, False otherwise.
		"""
		return self.request_per_day_used < self.request_per_day_limit
	
	@property
	def limit_day_exceeded(self) -> bool:
		"""
		Checks if the current date is different from the limiter's `limit_day`, indicating if a new day has started for daily limits.

		Returns:
			bool: True if the current day is different from `limit_day`, False otherwise.
		"""
		return datetime.now(tz=pytz.timezone("America/New_York")).date() != self.limit_day.date()
	
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
		if not self.limit_day_exceeded and not self.has_day_limits:
			raise GeminiDayLimitException()
		
		if not self.has_context:
			raise GeminiContextLimitException()
		
		if self.limit_day_exceeded:
			self.restart_day_counters()
		
		if self.minute_exceeded:
			self.restart_minute_counters(last_tokens)
			return
		
		if not self.has_minute_limits:
			if self.raise_error_on_minute_limit:
				raise GeminiMinuteLimitException()
		
			time.sleep(60 - (time.time() - self.start_time))
		
			self.restart_minute_counters(last_tokens)
	
	def add_context(self, tokens: int):
		"""
		Adds to the context usage counter and checks if the context limit has been exceeded.

		Args:
			tokens (int): The number of tokens to add to the context usage.

		Raises:
			GeminiContextLimitException: If adding the tokens causes the context usage to exceed the limit.
		"""
		if self.context_used + tokens > self.context_limit:
			raise GeminiContextLimitException()
		
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
		if not self.limit_day_exceeded and not self.has_day_limits:
			raise GeminiDayLimitException()
		
		if not self.has_context:
			raise GeminiContextLimitException()
		
		if self.limit_day_exceeded:
			self.restart_day_counters()
		
		if self.minute_exceeded:
			self.restart_minute_counters(last_tokens)
			return
		
		if not self.has_minute_limits:
			if self.raise_error_on_minute_limit:
				raise GeminiMinuteLimitException()
		
			await asyncio.sleep(60 - (time.time() - self.start_time))
		
			self.restart_minute_counters(last_tokens)
	
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
	
	@property
	def context_usage(self) -> dict[str, int]:
		"""
		Returns the current context usage and limit.

		Returns:
			dict[str, int]: A dictionary containing `context_used` and `context_limit`.
		"""
		return {"context_used": self.context_used, "context_limit": self.context_limit}
	
	@property
	def day_usage(self) -> dict[str, Union[int, datetime]]:
		"""
		Returns the current per-day usage and limits.

		Returns:
			dict[str, Union[int, datetime]]: A dictionary containing `used_requests`, `requests_limit`, and `date`.
		"""
		return {
			"used_requests": self.request_per_day_used,
			"requests_limit": self.request_per_day_limit,
			"date": self.limit_day
		}
	
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
	
	@property
	def limiter_settings(self) -> GeminiLimiterSettings:
		"""
		Returns a `GeminiLimiterSettings` object reflecting the current state of the limiter properties.

		This property creates and returns a new `GeminiLimiterSettings` instance initialized with the current values of all individual limiter settings.
		It provides a way to get a snapshot of the current limiter configuration as a single object.

		Returns:
			GeminiLimiterSettings: A new `GeminiLimiterSettings` object populated with the current limiter settings.
		"""
		return GeminiLimiterSettings(
				**{
					key: getattr(self, key)
					for key in get_function_parameters(GeminiLimiterSettings).keys()
				}
		)
	
	@limiter_settings.setter
	def limiter_settings(self, limiter_settings: GeminiLimiterSettings):
		"""
		Sets multiple limiter settings at once using a `GeminiLimiterSettings` object.

		This setter allows for updating all limiter parameters in a single operation by providing a `GeminiLimiterSettings` object.
		It updates limiter settings to the values specified in the provided `GeminiLimiterSettings` object.
		It also resets the per-minute usage counters and restarts the minute timer.

		Args:
			limiter_settings (GeminiLimiterSettings): A `GeminiLimiterSettings` object containing the new settings to apply to the limiter.
		"""
		self.limit_day = limiter_settings.limit_day
		self.request_per_day_used = limiter_settings.request_per_day_used
		self.request_per_day_limit = limiter_settings.request_per_day_limit
		self.request_per_minute_limit = limiter_settings.request_per_minute_limit
		self.tokens_per_minute_limit = limiter_settings.tokens_per_minute_limit
		self.context_used = limiter_settings.context_used
		self.context_limit = limiter_settings.context_limit
		self.raise_error_on_minute_limit = limiter_settings.raise_error_on_minute_limit
		self.request_per_minute_used = 0
		self.tokens_per_minute_used = 0
		self.start_time = time.time()
	
	@property
	def minute_usage(self) -> dict[str, int]:
		"""
		Returns the current per-minute usage and limits.

		Returns:
			dict[str, int]: A dictionary containing `used_requests`, `requests_limit`, `used_tokens`, and `tokens_limit`.
		"""
		return {
			"used_requests": self.request_per_minute_used,
			"requests_limit": self.request_per_minute_limit,
			"used_tokens": self.tokens_per_minute_used,
			"tokens_limit": self.tokens_per_minute_limit
		}
