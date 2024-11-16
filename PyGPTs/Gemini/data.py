class GeminiLimits:
    """
    Stores default rate limits for different Gemini models.

    Attributes:
        request_per_day (dict[str, int]): The maximum number of requests allowed per day for each model.
        request_per_minute (dict[str, int]): The maximum number of requests allowed per minute for each model.
        tokens_per_minute (dict[str, int]): The maximum number of tokens allowed per minute for each model.
    """

    request_per_day = {"gemini-1.5-pro": 50, "gemini-1.5-flash": 1000, "gemini-1.0-pro": 1000}
    request_per_minute = {"gemini-1.5-pro": 2, "gemini-1.5-flash": 10, "gemini-1.0-pro": 10}
    tokens_per_minute = {"gemini-1.5-pro": 32 * 10**3, "gemini-1.5-flash": 10**6, "gemini-1.0-pro": 32 * 10**3}


class GeminiModels:
    """
    Stores string identifiers for different Gemini models.

    Attributes:
        gemini_1_5_pro (str): gemini 1.5 pro name.
        gemini_1_5_flash (str): gemini 1.5 flash name.
        gemini_1_0_pro (str): gemini 1.0 pro name.
    """

    gemini_1_5_pro = "gemini-1.5-pro"
    gemini_1_5_flash = "gemini-1.5-flash"
    gemini_1_0_pro = "gemini-1.0-pro"
