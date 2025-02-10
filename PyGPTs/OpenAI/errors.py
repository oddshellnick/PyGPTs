class OpenAIMinuteLimitException(Exception):
	"""
	Raised when a OpenAI model has reached its per-minute rate limit.
	"""
	
	def __init__(self):
		"""Initializes the exception with a default message."""
		super().__init__("Minute limit reached")


class OpenAIDayLimitException(Exception):
	"""
	Raised when a OpenAI model has reached its per-day rate limit.
	"""
	
	def __init__(self):
		"""Initializes the exception with a default message."""
		super().__init__("Day limit reached")


class OpenAIContextLimitException(Exception):
	"""
	Exception raised when the model's context window limit is reached.
	"""
	
	def __init__(self):
		"""Initializes the exception with a default message."""
		super().__init__("Model context limit reached")


class OpenAIChatTypeException(Exception):
	"""
	Exception raised when a chat session is accessed with an incorrect type assumption (e.g., trying to access an async chat as a sync chat).
	"""
	
	def __init__(self, index: int, type_: str):
		"""
		Initializes a new instance of `OpenAIChatTypeException`.

		Args:
			index (int): The index of the chat session.
			type_ (str): The expected type of the chat session.
		"""
		super().__init__(f"Chat with index {index} is not {type_}")
