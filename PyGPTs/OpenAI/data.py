import httpx
from dataclasses import dataclass
from typing import Optional, Union
from openai import NOT_GIVEN, NotGiven
from openai._types import Body, Headers, Query
from openai.types.chat import ChatCompletionStreamOptionsParam


@dataclass(frozen=True)
class OpenAIModels:
	"""
	Provides a structured way to access different OpenAI model names.

	This class uses nested dataclasses to organize and easily retrieve model names, including different versions and variations (e.g., latest, stable, specific versions).
	"""
	
	@dataclass(frozen=True)
	class DALLE:
		"""
		Names for DALL-E models.

		Attributes:
			e_2 (str): The name of DALL-E 2 model.
			e_3 (str): The name of DALL-E 3 model.
		"""
		e_2 = "dall-e-2"
		e_3 = "dall-e-3"
	
	@dataclass(frozen=True)
	class TTS:
		"""
		Names for TTS (Text-to-Speech) models.

		Attributes:
			_1 (str): The name of TTS model version 1.
			_1_hd (str): The name of TTS model version 1 HD.
		"""
		_1 = "tts-1"
		_1_hd = "tts-1-hd"
	
	@dataclass(frozen=True)
	class Whisper:
		"""
		Names for Whisper models (Speech-to-Text).

		Attributes:
			_1 (str): The name of Whisper model version 1.
		"""
		_1 = "whisper-1"
	
	@dataclass(frozen=True)
	class gpt_3_5_turbo:
		"""
		Names for GPT-3.5 Turbo models.

		Attributes:
			latest_stable (str): The name of the latest stable GPT-3.5 Turbo model.
			_1106 (str): The name of GPT-3.5 Turbo model version 1106.
			_0125 (str): The name of GPT-3.5 Turbo model version 0125.
			instruct (str): The name of GPT-3.5 Turbo Instruct model.
		"""
		latest_stable = "gpt-3.5-turbo"
		_1106 = "gpt-3.5-turbo-1106"
		_0125 = "gpt-3.5-turbo-0125"
		instruct = "gpt-3.5-turbo-instruct"
	
	@dataclass(frozen=True)
	class gpt_4:
		"""
		Names for GPT-4 models.

		Attributes:
			latest_stable (str): The name of the latest stable GPT-4 model.
			_0314 (str): The name of GPT-4 model version 0314.
			_0613 (str): The name of GPT-4 model version 0613.
		"""
		latest_stable = "gpt-4"
		_0314 = "gpt-4-0314"
		_0613 = "gpt-4-0613"
	
	@dataclass(frozen=True)
	class gpt_4_turbo:
		"""
		Names for GPT-4 Turbo models.

		Attributes:
			latest_stable (str): The name of the latest stable GPT-4 Turbo model.
			_2024_04_09 (str): The name of GPT-4 Turbo model version from 2024-04-09.
			preview_latest_stable (str): The name of the latest stable GPT-4 Turbo Preview model.
		"""
		latest_stable = "gpt-4-turbo"
		_2024_04_09 = "gpt-4-turbo-2024-04-09"
		preview_latest_stable = "gpt-4-turbo-preview"
	
	@dataclass(frozen=True)
	class gpt_4o:
		"""
		Names for GPT-4o models.

		Attributes:
			latest (str): The name of the latest GPT-4o model.
			latest_stable (str): The name of the latest stable GPT-4o model.
			_2024_05_13 (str): The name of GPT-4o model version from 2024-05-13.
			_2024_08_06 (str): The name of GPT-4o model version from 2024-08-06.
			_2024_11_20 (str): The name of GPT-4o model version from 2024-11-20.
		"""
		latest = "chatgpt-4o-latest"
		latest_stable = "gpt-4o"
		_2024_05_13 = "gpt-4o-2024-05-13"
		_2024_08_06 = "gpt-4o-2024-08-06"
		_2024_11_20 = "gpt-4o-2024-11-20"
	
	@dataclass(frozen=True)
	class gpt_4o_audio:
		"""
		Names for GPT-4o Audio models.

		Attributes:
			preview_latest_stable (str): The name of the latest stable GPT-4o Audio Preview model.
			preview_2024_10_01 (str): The name of GPT-4o Audio Preview model version from 2024-10-01.
			preview_2024_12_17 (str): The name of GPT-4o Audio Preview model version from 2024-12-17.
		"""
		preview_latest_stable = "gpt-4o-audio-preview"
		preview_2024_10_01 = "gpt-4o-audio-preview-2024-10-01"
		preview_2024_12_17 = "gpt-4o-audio-preview-2024-12-17"
	
	@dataclass(frozen=True)
	class gpt_4o_mini:
		"""
		Names for GPT-4o Mini models.

		Attributes:
			latest_stable (str): The name of the latest stable GPT-4o Mini model.
			_2024_07_18 (str): The name of GPT-4o Mini model version from 2024-07-18.
		"""
		latest_stable = "gpt-4o-mini"
		_2024_07_18 = "gpt-4o-mini-2024-07-18"
	
	@dataclass(frozen=True)
	class gpt_4o_mini_audio:
		"""
		Names for GPT-4o Mini Audio models.

		Attributes:
			preview_latest (str): The name of the latest GPT-4o Mini Audio Preview model.
			preview_latest_2024_12_17 (str): The name of the latest GPT-4o Mini Audio Preview model version from 2024-12-17.
		"""
		preview_latest = "gpt-4o-mini-audio-preview"
		preview_latest_2024_12_17 = "gpt-4o-mini-audio-preview-2024-12-17"
	
	@dataclass(frozen=True)
	class gpt_4o_mini_realtime:
		"""
		Names for GPT-4o Mini Realtime models.

		Attributes:
			preview_latest (str): The name of the latest GPT-4o Mini Realtime Preview model.
			preview_latest_2024_12_17 (str): The name of the latest GPT-4o Mini Realtime Preview model version from 2024-12-17.
		"""
		preview_latest = "gpt-4o-mini-realtime-preview"
		preview_latest_2024_12_17 = "gpt-4o-mini-realtime-preview-2024-12-17"
	
	@dataclass(frozen=True)
	class gpt_4o_realtime:
		"""
		Names for GPT-4o Realtime models.

		Attributes:
			preview_latest_stable (str): The name of the latest stable GPT-4o Realtime Preview model.
			preview_2024_10_01 (str): The name of GPT-4o Realtime Preview model version from 2024-10-01.
			preview_2024_12_17 (str): The name of GPT-4o Realtime Preview model version from 2024-12-17.
		"""
		preview_latest_stable = "gpt-4o-realtime-preview"
		preview_2024_10_01 = "gpt-4o-realtime-preview-2024-10-01"
		preview_2024_12_17 = "gpt-4o-realtime-preview-2024-12-17"
	
	@dataclass(frozen=True)
	class o1:
		"""
		Names for o1 models.

		Attributes:
			latest_stable (str): The name of the latest stable o1 model.
			_2024_12_17 (str): The name of o1 model version from 2024-12-17.
			preview_latest_stable (str): The name of the latest stable o1 Preview model.
			preview_2024_09_12 (str): The name of o1 Preview model version from 2024-09-12.
		"""
		latest_stable = "o1"
		_2024_12_17 = "o1-2024-12-17"
		preview_latest_stable = "o1-preview"
		preview_2024_09_12 = "o1-preview-2024-09-12"
	
	@dataclass(frozen=True)
	class o1_mini:
		"""
		Names for o1 Mini models.

		Attributes:
			latest_stable (str): The name of the latest stable o1 Mini model.
			_2024_09_12 (str): The name of o1 Mini model version from 2024-09-12.
		"""
		latest_stable = "o1-mini"
		_2024_09_12 = "o1-mini-2024-09-12"
	
	@dataclass(frozen=True)
	class o3_mini:
		"""
		Names for o3 Mini models.

		Attributes:
			latest_stable (str): The name of the latest stable o3 Mini model.
			_2025_01_31 (str): The name of o3 Mini model version from 2025-01-31.
		"""
		latest_stable = "o3-mini"
		_2025_01_31 = "o3-mini-2025-01-31"


@dataclass(frozen=True)
class OpenAIMimeTypes:
	"""
	Defines common MIME types for OpenAI.

	Attributes:
		text_plain (str): MIME type for plain text.
		application_json (str): MIME type for JSON.
		image_jpeg (str): MIME type for JPEG images.
		image_png (str): MIME type for PNG images.
		image_gif (str): MIME type for GIF images.
		audio_mpeg (str): MIME type for MPEG audio.
		audio_wav (str): MIME type for WAV audio.
		video_mpeg (str): MIME type for MPEG video.
		video_mp4 (str): MIME type for MP4 video.
	"""
	text_plain = "text/plain"
	application_json = "application/json"
	image_jpeg = "image/jpeg"
	image_png = "image/png"
	image_gif = "image/gif"
	audio_mpeg = "audio/mpeg"
	audio_wav = "audio/wav"
	video_mpeg = "video/mpeg"
	video_mp4 = "video/mp4"


@dataclass(frozen=True)
class OpenAILimits:
	"""
	Stores default limits for different OpenAI models.

	Attributes:
		context_limit (dict[str, int]): The maximum context length window for each model.
		request_per_day (dict[str, int]): The maximum number of requests allowed per day for each model.
		request_per_minute (dict[str, int]): The maximum number of requests allowed per minute for each model.
		tokens_per_minute (dict[str, int]): The maximum number of tokens allowed per minute for each model.
	"""
	context_limit = {
		"gemini-2.0-pro": 2 ** 21,
		"gemini-2.0-flash": 2 ** 20,
		"gemini-2.0-flash-lite": 2 ** 20,
		"gemini-2.0-flash-thinking": 2 ** 20,
		"gemini-1.5-pro": 2 * 10 ** 6,
		"gemini-1.5-flash": 10 ** 6,
		"gemini-1.5-flash-8b": 10 ** 6
	}
	request_per_day = {
		"gemini-2.0-pro": 50,
		"gemini-2.0-flash": 1500,
		"gemini-2.0-flash-lite": 1500,
		"gemini-2.0-flash-thinking": 1500,
		"gemini-1.5-pro": 50,
		"gemini-1.5-flash": 1500,
		"gemini-1.5-flash-8b": 1500
	}
	request_per_minute = {
		"gemini-2.0-pro": 2,
		"gemini-2.0-flash": 15,
		"gemini-2.0-flash-lite": 30,
		"gemini-2.0-flash-thinking": 10,
		"gemini-1.5-pro": 2,
		"gemini-1.5-flash": 15,
		"gemini-1.5-flash-8b": 15
	}
	tokens_per_minute = {
		"gemini-2.0-pro": 32 * 10 ** 3,
		"gemini-2.0-flash": 4 * 10 ** 6,
		"gemini-2.0-flash-lite": 4 * 10 ** 6,
		"gemini-2.0-flash-thinking": 10 ** 6,
		"gemini-1.5-pro": 32 * 10 ** 3,
		"gemini-1.5-flash": 10 ** 6,
		"gemini-1.5-flash-8b": 10 ** 6
	}


class OpenAIGenerationConfig:
	"""
	Configuration options for generating text completions using the OpenAI API.

	This class encapsulates all the optional parameters available for the 'Create completion' endpoint
	in the OpenAI API. It allows for fine-grained control over the generation process, including
	sampling parameters, penalties, and output formatting.

	Attributes:
		best_of (Optional[Union[int, NotGiven]]): Generates best_of completions server-side and returns the "best" one.
		echo (Optional[Union[bool, NotGiven]]): Echoes back the prompt in addition to the completion.
		frequency_penalty (Optional[Union[float, NotGiven]]): Penalizes new tokens based on their existing frequency in the text.
		logit_bias (Optional[Union[dict[str, int], NotGiven]]): Modifies the likelihood of specified tokens appearing in the completion.
		logprobs (Optional[Union[int, NotGiven]]): Includes log probabilities on the logprobs most likely output tokens.
		max_tokens (Optional[Union[int, NotGiven]]): The maximum number of tokens to generate in the completion.
		n (Optional[Union[int, NotGiven]]): How many completions to generate for each prompt.
		presence_penalty (Optional[Union[float, NotGiven]]): Penalizes new tokens based on whether they appear in the text so far.
		seed (Optional[Union[int, NotGiven]]): If specified, the system will attempt deterministic sampling.
		stop (Optional[Union[str, list[str], NotGiven]]): Up to 4 sequences where the API will stop generating further tokens.
		stream (Optional[Union[bool, NotGiven]]): Whether to stream back partial progress.
		stream_options (Optional[Union[ChatCompletionStreamOptionsParam, NotGiven]]): Options for streaming responses. Only set when `stream` is true.
		suffix (Optional[Union[str, NotGiven]]): The suffix that comes after a completion of inserted text.
		temperature (Optional[Union[float, NotGiven]]): Sampling temperature to use, between 0 and 2.
		top_p (Optional[Union[float, NotGiven]]): An alternative to sampling with temperature, called nucleus sampling.
		user (Union[str, NotGiven]): A unique identifier representing your end-user for monitoring and abuse detection.
		extra_headers (Optional[Headers]): Custom headers to add to the request.
		extra_query (Optional[Query]): Custom query parameters to add to the request.
		extra_body (Optional[Body]): Custom body to add to the request.
		timeout (Optional[Union[float, httpx.Timeout, NotGiven]]): The timeout duration for the request.
	"""
	
	def __init__(
			self,
			best_of: Optional[Union[int, NotGiven]] = NOT_GIVEN,
			echo: Optional[Union[bool, NotGiven]] = NOT_GIVEN,
			frequency_penalty: Optional[Union[float, NotGiven]] = NOT_GIVEN,
			logit_bias: Optional[Union[dict[str, int], NotGiven]] = NOT_GIVEN,
			logprobs: Optional[Union[int, NotGiven]] = NOT_GIVEN,
			max_tokens: Optional[Union[int, NotGiven]] = NOT_GIVEN,
			n: Optional[Union[int, NotGiven]] = NOT_GIVEN,
			presence_penalty: Optional[Union[float, NotGiven]] = NOT_GIVEN,
			seed: Optional[Union[int, NotGiven]] = NOT_GIVEN,
			stop: Optional[Union[str, list[str], NotGiven]] = NOT_GIVEN,
			stream: Optional[Union[bool, NotGiven]] = NOT_GIVEN,
			stream_options: Optional[Union[ChatCompletionStreamOptionsParam, NotGiven]] = NOT_GIVEN,
			suffix: Optional[Union[str, NotGiven]] = NOT_GIVEN,
			temperature: Optional[Union[float, NotGiven]] = NOT_GIVEN,
			top_p: Optional[Union[float, NotGiven]] = NOT_GIVEN,
			user: Union[str, NotGiven] = NOT_GIVEN,
			extra_headers: Optional[Headers] = None,
			extra_query: Optional[Query] = None,
			extra_body: Optional[Body] = None,
			timeout: Optional[Union[float, httpx.Timeout, NotGiven]] = NOT_GIVEN
	):
		"""
		Initializes an OpenAIGenerationConfig object with various parameters for controlling text generation.

		Args:
			best_of (Optional[Union[int, NotGiven]]): Generates `best_of` completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed. Defaults to 1.
			echo (Optional[Union[bool, NotGiven]]): Echo back the prompt in addition to the completion. Defaults to False.
			frequency_penalty (Optional[Union[float, NotGiven]]): Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Defaults to 0.
			logit_bias (Optional[Union[dict[str, int], NotGiven]]):  Map that modifies the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens (specified by their token ID) to an associated bias value from -100 to 100. Defaults to null.
			logprobs (Optional[Union[int, NotGiven]]): Include the log probabilities on the `logprobs` most likely output tokens, as well the chosen tokens. For example, if `logprobs` is 5, the API will return a list of the 5 most likely tokens. The API will always return the `logprobs` of the sampled tokens, so there may be up to `logprobs+1` elements in the response. Defaults to null.
			max_tokens (Optional[Union[int, NotGiven]]): The maximum number of tokens to generate in the completion. Defaults to 16.
			n (Optional[Union[int, NotGiven]]): How many completions to generate for each prompt. Defaults to 1.
			presence_penalty (Optional[Union[float, NotGiven]]): Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Defaults to 0.
			seed (Optional[Union[int, NotGiven]]): If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same `seed` and parameters should return the same result. Determinism is not guaranteed. Defaults to null.
			stop (Optional[Union[str, list[str], NotGiven]]): Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence. Defaults to null.
			stream (Optional[Union[bool, NotGiven]]): Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent events as the text becomes available, with the stream terminated by a `data: [DONE]` message. Defaults to False.
			stream_options (Optional[Union[ChatCompletionStreamOptionsParam, NotGiven]]): Options for streaming responses. Only set this when you set `stream: true`. Defaults to None.
			suffix (Optional[Union[str, NotGiven]]): The suffix that comes after a completion of inserted text. Defaults to null.
			temperature (Optional[Union[float, NotGiven]]): What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Defaults to 1.
			top_p (Optional[Union[float, NotGiven]]): An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. Defaults to 1.
			user (Union[str, NotGiven]): A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Defaults to NOT_GIVEN.
			extra_headers (Optional[Headers]): Custom headers to add to the request. Defaults to None.
			extra_query (Optional[Query]): Custom query parameters to add to the request. Defaults to None.
			extra_body (Optional[Body]): Custom body to add to the request. Defaults to None.
			timeout (Optional[Union[float, httpx.Timeout, NotGiven]]): The timeout duration for the request. Defaults to NOT_GIVEN.
		"""
		self.best_of = best_of
		self.echo = echo
		self.frequency_penalty = frequency_penalty
		self.logit_bias = logit_bias
		self.logprobs = logprobs
		self.max_tokens = max_tokens
		self.n = n
		self.presence_penalty = presence_penalty
		self.seed = seed
		self.stop = stop
		self.stream = stream
		self.stream_options = stream_options
		self.suffix = suffix
		self.temperature = temperature
		self.top_p = top_p
		self.user = user
		self.extra_headers = extra_headers
		self.extra_query = extra_query
		self.extra_body = extra_body
		self.timeout = timeout


@dataclass(frozen=True)
class OpenAIContentRoles:
	"""
	Defines the roles for OpenAI content.

	Attributes:
		user (str): Represents the user role.
		assistant (str): Represents the assistant (AI) role.
	"""
	user = "user"
	developer = "developer"
	assistant = "assistant"
