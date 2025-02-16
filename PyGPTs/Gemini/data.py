from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiModels:
	"""
	Provides a structured way to access different Gemini model names.

	This class uses nested dataclasses to organize and easily retrieve model names, including different versions and variations (e.g., latest, stable, specific versions).
	"""
	
	@dataclass(frozen=True)
	class Gemini_1_5_flash:
		"""
		Names for Gemini 1.5 Flash models.

		Attributes:
			latest (str): The name of the latest Gemini 1.5 Flash model.
			latest_stable (str): The name of the latest stable Gemini 1.5 Flash model.
			_001 (str): The name of Gemini 1.5 Flash version 001.
			_002 (str): The name of Gemini 1.5 Flash version 002.
		"""
		latest = "gemini-1.5-flash-latest"
		latest_stable = "gemini-1.5-flash"
		_001 = "gemini-1.5-flash-001"
		_002 = "gemini-1.5-flash-002"
	
	@dataclass(frozen=True)
	class Gemini_1_5_flash_8b:
		"""
		Names for Gemini 1.5 Flash 8b models.

		Attributes:
			latest (str): The name of the latest Gemini 1.5 Flash 8b model.
			latest_stable (str): The name of the latest stable Gemini 1.5 Flash 8b model.
			_001 (str): The name of Gemini 1.5 Flash 8b version 001.
		"""
		latest = "gemini-1.5-flash-8b-latest"
		latest_stable = "gemini-1.5-flash-8b"
		_001 = "gemini-1.5-flash-8b-001"
	
	@dataclass(frozen=True)
	class Gemini_1_5_pro:
		"""
		Names for Gemini 1.5 Pro models.

		Attributes:
			latest (str): The name of the latest Gemini 1.5 Pro model.
			latest_stable (str): The name of the latest stable Gemini 1.5 Pro model.
			_001 (str): The name of Gemini 1.5 Pro version 001.
			_002 (str): The name of Gemini 1.5 Pro version 002.
		"""
		latest = "gemini-1.5-pro-latest"
		latest_stable = "gemini-1.5-pro"
		_001 = "gemini-1.5-pro-001"
		_002 = "gemini-1.5-pro-002"
	
	@dataclass(frozen=True)
	class Gemini_2_0_flash:
		"""
		Names for Gemini 2.0 Flash models.

		Attributes:
			latest (str): The name of the latest Gemini 2.0 Flash model.
			latest_stable (str): The name of the latest stable Gemini 2.0 Flash model.
			_001 (str): The name of Gemini 2.0 Flash version 001.
		"""
		latest = "gemini-2.0-flash-latest"
		latest_stable = "gemini-2.0-flash"
		_001 = "gemini-2.0-flash-001"
	
	@dataclass(frozen=True)
	class Gemini_2_0_flash_lite:
		"""
		Names for Gemini 2.0 Flash Lite models.

		Attributes:
			preview (str): The name of the experimental Gemini 2.0 Flash Lite model.
		"""
		preview = "gemini-2.0-flash-lite-preview-02-05"
	
	@dataclass(frozen=True)
	class Gemini_2_0_flash_thinking:
		"""
		Names for Gemini 2.0 Flash Thinking models.

		Attributes:
			exp (str): The name of the experimental Gemini 2.0 Flash Thinking model.
		"""
		exp = "gemini-2.0-flash-thinking-exp"
	
	@dataclass(frozen=True)
	class Gemini_2_0_pro:
		"""
		Names for Gemini 2.0 Pro models.

		Attributes:
			exp (str): The name of the experimental Gemini 2.0 Pro model.
		"""
		exp = "gemini-2.0-pro-exp"


@dataclass(frozen=True)
class GeminiMimeTypes:
	"""
	Defines common MIME types for Gemini.

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
class GeminiLimits:
	"""
	Stores default limits for different Gemini models.

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


@dataclass(frozen=True)
class GeminiContentRoles:
	"""
	Defines the roles for Gemini content.

	Attributes:
		user (str): Represents the user role.
		model (str): Represents the model (AI) role.
	"""
	user = "user"
	model = "model"
