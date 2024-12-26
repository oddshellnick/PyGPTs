class GeminiNoUsefulModelsException(Exception):
    """
    Raised when all Gemini models managed by GeminiManager have reached their rate limits.
    """

    def __init__(self):
        """Initializes the exception with a default message."""
        super().__init__("All Gemini limits reached")


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
