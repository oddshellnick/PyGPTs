class GeminiMinuteLimitException(Exception):
	"""
	Raised when a Gemini model has reached its per-minute rate limit.
	"""
	
	def __init__(self):
		"""Initializes the exception with a default message."""
		super().__init__("Minute limit reached")


class GeminiDayLimitException(Exception):
	"""
	Raised when a Gemini model has reached its per-day rate limit.
	"""
	
	def __init__(self):
		"""Initializes the exception with a default message."""
		super().__init__("Day limit reached")


class GeminiContextLimitException(Exception):
	"""
	Exception raised when the model's context window limit is reached.
	"""
	
	def __init__(self):
		"""
		Initializes a new instance of `GeminiContextLimitException`.
		"""
		super().__init__("Model context limit reached")


class GeminiChatTypeException(Exception):
	"""
	Exception raised when a chat session is accessed with an incorrect type assumption (e.g., trying to access an async chat as a sync chat).
	"""
	
	def __init__(self, index: int, type_: str):
		"""
		Initializes a new instance of `GeminiChatTypeException`.

		Args:
			index (int): The index of the chat session.
			type_ (str): The expected type of the chat session.
		"""
		super().__init__(f"Chat with index {index} is not {type_}")
