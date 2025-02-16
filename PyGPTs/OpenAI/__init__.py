import time
import pytz
import asyncio
import datetime
from openai import OpenAI
from PyGPTs.OpenAI import errors
from PyGPTs.OpenAI.data import (
	OpenAIGenerationConfig,
	OpenAILimits,
	OpenAIModels
)


class OpenAILimiter:
	"""
	Manages rate limiting for OpenAI API requests.

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
		Initializes an instance of the OpenAILimiter class.

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
			OpenAIDayLimitException: If the daily request limit has been exceeded.
			OpenAIMinuteLimitException: If the per-minute request or token limit has been exceeded and raise_error_on_limit is True.
			OpenAIContextLimitException: If the context limit has been exceeded.
		"""
		elapsed_time = time.time() - self.start_time
		current_date = datetime.datetime.now(tz=pytz.timezone("America/New_York"))
		
		if current_date.date() == self.limit_day.date() and self.request_per_day_used > self.request_per_day_limit:
			raise errors.OpenAIDayLimitException()
		
		if self.context_used > self.context_limit:
			raise errors.OpenAIContextLimitException()
		
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
					raise errors.OpenAIMinuteLimitException()
		
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
			OpenAIContextLimitException: If adding the tokens causes the context usage to exceed the limit.
		"""
		if self.context_used + tokens > self.context_limit:
			raise errors.OpenAIContextLimitException()
		
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
			OpenAIDayLimitException: If the daily request limit has been exceeded.
			OpenAIMinuteLimitException: If the per-minute request or token limit has been exceeded and `raise_error_on_limit` is True.
			OpenAIContextLimitException: If the context limit has been exceeded.
		"""
		elapsed_time = time.time() - self.start_time
		current_date = datetime.datetime.now(tz=pytz.timezone("America/New_York"))
		
		if not self.check_day_limits():
			raise errors.OpenAIDayLimitException()
		
		if self.context_used > self.context_limit:
			raise errors.OpenAIContextLimitException()
		
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
					raise errors.OpenAIMinuteLimitException()
		
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
