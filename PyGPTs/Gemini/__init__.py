import datetime
import time

import pytz

import google.generativeai as genai
import google.generativeai.types as genai_types
from google.ai.generativelanguage_v1 import GenerateContentResponse

import PyGPTs.Gemini.data as data
import PyGPTs.Gemini.types as types
import PyGPTs.Gemini.errors as errors


class GeminiSettings:
    """
    Stores settings for interacting with Gemini models.

    Attributes:
        api_key (str): The API key for authenticating with Gemini.
        model_name (str): The name of the Gemini model to use. Defaults to "gemini_1_5_flash".
        safety_settings (dict[genai_types.HarmCategory, genai_types.HarmBlockThreshold]): Safety settings for the model. Defaults to blocking no harmful content.
        generation_config (GenerationConfig): Configuration for text generation. Defaults to a conservative configuration.
        start_day (datetime.datetime): The start day for tracking usage limits. Defaults to the current day in the US/Eastern time zone.
        request_per_day_used (int): The number of requests used so far today. Defaults to 0.
        request_per_day_limit (int | None): The maximum number of requests allowed per day. Defaults to the limit specified in "data.GeminiLimits" for the chosen model.
        request_per_minute_limit (int | None): The maximum number of requests allowed per minute. Defaults to the limit specified in "data.GeminiLimits" for the chosen model.
        tokens_per_minute_limit (int | None): The maximum number of tokens allowed per minute. Defaults to the limit specified in "data.GeminiLimits" for the chosen model.
        raise_error_on_minute_limit (bool): Whether to raise an error when a rate limit is exceeded. Defaults to True.

    :Usage:
        from google.generativeai import GenerationConfig
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        # Basic usage with default settings
        settings = GeminiSettings(api_key="YOUR_API_KEY")

        # Custom settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ALL,
        }
        generation_config = GenerationConfig(temperature=0.5, top_k=10)

        settings = GeminiSettings(
            api_key="YOUR_API_KEY",
            model_name="gemini-pro",
            safety_settings=safety_settings,
            generation_config=generation_config,
            request_per_day_limit=500,
            request_per_minute_limit=10,
            tokens_per_minute_limit=500,
        )
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = data.GeminiModels.gemini_1_5_flash,
        safety_settings: dict[genai_types.HarmCategory, genai_types.HarmBlockThreshold] | None = None,
        generation_config: genai.GenerationConfig = genai.GenerationConfig(
            candidate_count=1, temperature=0.7, top_p=0.5, top_k=40, response_mime_type="text/plain"
        ),
        start_day: datetime.datetime = datetime.datetime.now(tz=pytz.timezone("America/New_York")),
        request_per_day_used: int = 0,
        request_per_day_limit: int | None = None,
        request_per_minute_limit: int | None = None,
        tokens_per_minute_limit: int | None = None,
        raise_error_on_minute_limit: bool = True,
    ):
        """
        Initializes an instance of the GeminiSettings class.

        Args:
            api_key (str): The API key for authenticating with Gemini.
            model_name (str): The name of the Gemini model to use. Defaults to "gemini_1_5_flash".
            safety_settings (dict[genai_types.HarmCategory, genai_types.HarmBlockThreshold]): Safety settings for the model. Defaults to blocking no harmful content.
            generation_config (genai.GenerationConfig): Configuration for text generation. Defaults to a conservative configuration.
            start_day (datetime.datetime): The start day for tracking usage limits. Defaults to the current day in the America/New_York time zone.
            request_per_day_used (int): The number of requests used so far today. Defaults to 0.
            request_per_day_limit (int | None): The maximum number of requests allowed per day. Defaults to the limit specified in "data.GeminiLimits" for the chosen model.
            request_per_minute_limit (int | None): The maximum number of requests allowed per minute. Defaults to the limit specified in "data.GeminiLimits" for the chosen model.
            tokens_per_minute_limit (int | None): The maximum number of tokens allowed per minute. Defaults to the limit specified in "data.GeminiLimits" for the chosen model.
            raise_error_on_minute_limit (bool): Whether to raise an error when a minute rate limit is exceeded. Defaults to True.
        """
        if safety_settings is None:
            safety_settings = {
                genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai_types.HarmBlockThreshold.BLOCK_NONE,
                genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai_types.HarmBlockThreshold.BLOCK_NONE,
                genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai_types.HarmBlockThreshold.BLOCK_NONE,
                genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai_types.HarmBlockThreshold.BLOCK_NONE,
            }

        self.api_key = api_key
        self.model_name = model_name
        self.safety_settings = safety_settings
        self.generation_config = generation_config
        self.raise_error_on_minute_limit = raise_error_on_minute_limit
        self.request_per_day_used = request_per_day_used

        if start_day is not None:
            start_day = start_day.astimezone(pytz.timezone("America/New_York"))

        self.start_day = datetime.datetime(
            year=start_day.year,
            month=start_day.month,
            day=start_day.day,
            tzinfo=start_day.tzinfo,
        )

        if request_per_day_limit is None:
            self.request_per_day_limit = data.GeminiLimits.request_per_day[model_name]
        else:
            self.request_per_day_limit = request_per_day_limit

        if request_per_minute_limit is None:
            self.request_per_minute_limit = data.GeminiLimits.request_per_minute[model_name]
        else:
            self.request_per_minute_limit = request_per_minute_limit

        if tokens_per_minute_limit is None:
            self.tokens_per_minute_limit = data.GeminiLimits.tokens_per_minute[model_name]
        else:
            self.tokens_per_minute_limit = tokens_per_minute_limit


class GeminiLimiter:
    """
    Manages rate limiting for Gemini API requests.

    Attributes:
        start_day (datetime.datetime): The start day for tracking daily usage limits.
        request_per_day_used (int): The number of requests used so far today.
        request_per_day_limit (int): The maximum number of requests allowed per day.
        request_per_minute_limit (int): The maximum number of requests allowed per minute.
        tokens_per_minute_limit (int): The maximum number of tokens allowed per minute.
        raise_error_on_minute_limit (bool): Whether to raise an error when a rate limit is exceeded. Defaults to True.
        request_per_minute_used (int): The number of requests used so far this minute.
        tokens_per_minute_used (int): The number of tokens used so far this minute.
        start_time (float): The timestamp of the start of the current minute.

    :Usage:
        from datetime import datetime
        from pytz import timezone

        start_day = datetime.now(tz=timezone("America/New_York"))
        limiter = GeminiLimiter(start_day, 0, 1000, 10, 1000)
        limiter.add_data(50)  # Adds 50 tokens and 1 request to the usage count
        limiter.check_limits(50) # check for the usage of the limits
    """

    def __init__(
        self,
        start_day: datetime.datetime,
        request_per_day_used: int,
        request_per_day_limit: int,
        request_per_minute_limit: int,
        tokens_per_minute_limit: int,
        raise_error_on_minute_limit: bool = True,
    ):
        """
        Initializes an instance of the GeminiLimiter class.

        Args:
            start_day (datetime.datetime): The start day for tracking daily usage.
            request_per_day_used (int): Initial count of requests used per day.
            request_per_day_limit (int): Maximum requests allowed per day.
            request_per_minute_limit (int): Maximum requests allowed per minute.
            tokens_per_minute_limit (int): Maximum tokens allowed per minute.
            raise_error_on_minute_limit (bool): Whether to raise exceptions when hitting minute limits.
        """
        self.start_day = start_day
        self.request_per_day_limit = request_per_day_limit
        self.request_per_minute_limit = request_per_minute_limit
        self.tokens_per_minute_limit = tokens_per_minute_limit
        self.raise_error_on_minute_limit = raise_error_on_minute_limit
        self.request_per_day_used = request_per_day_used
        self.request_per_minute_used = 0
        self.tokens_per_minute_used = 0
        self.start_time = time.time()

    def close_minute_limit(self):
        """
        Sets the per-minute usage counters to their limits, effectively blocking further requests for the current minute.
        """
        self.request_per_minute_used = self.request_per_minute_limit
        self.tokens_per_minute_used = self.tokens_per_minute_limit

    def close_day_limit(self):
        """
        Sets the per-day usage counter to its limit, effectively blocking further requests for the current day.
        """
        self.request_per_day_used = self.request_per_day_limit

    def check_limits(self, last_tokens: int):
        """
        Checks if any rate limits have been exceeded. Resets minute counters if a minute has passed.
        Pauses execution or raises an error if a limit is exceeded, depending on raise_error_on_limit.

        Args:
            last_tokens (int): The number of tokens used in the last request.

        Raises:
            GeminiDayLimitException: If the daily request limit has been exceeded.
            GeminiMinuteLimitException: If the per-minute request or token limit has been exceeded and raise_error_on_limit is True.
        """
        elapsed_time = time.time() - self.start_time
        current_date = datetime.datetime.now(tz=pytz.timezone("America/New_York"))

        if current_date.date() == self.start_day.date() and self.request_per_day_used > self.request_per_day_limit:
            raise errors.GeminiDayLimitException()

        if elapsed_time < 60:
            if self.request_per_day_used > self.request_per_day_limit:
                self.request_per_day_used = 1
                self.start_day = datetime.datetime(
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

    def add_data(self, tokens: int):
        """
        Increments the usage counters for requests and tokens.

        Args:
            tokens (int): The number of tokens used in the last request.
        """
        self.request_per_day_used += 1
        self.request_per_minute_used += 1
        self.tokens_per_minute_used += tokens

        self.check_limits(tokens)


class Gemini:
    """
    A wrapper class for interacting with Google Gemini models using the `genai` library.

    Attributes:
        api_key (str): The API key used for authentication.
        model_name (str): The name of the Gemini model being used.
        model (genai.GenerativeModel): The underlying `genai` model instance.
        limiter (GeminiLimiter): A rate limiter instance to manage API usage.
        chats (List[genai.ChatSession]): A list of active chat sessions.

    :Usage:
        settings = GeminiSettings(api_key="YOUR_API_KEY")
        gemini = Gemini(settings)

        gemini.start_chat()
        response = gemini.send_message("Hello, Gemini!", chat_index=0)
        print(response.text)

        content_response = gemini.generate_content("Write a short story.")
        print(content_response.text)
    """

    def __init__(self, gemini_settings: GeminiSettings):
        """
        Initializes a new Gemini instance.

        Args:
            gemini_settings (GeminiSettings): An instance of GeminiSettings containing configuration parameters.
        """
        genai.configure(api_key=gemini_settings.api_key)

        self.api_key = gemini_settings.api_key
        self.model_name = gemini_settings.model_name

        self.model = genai.GenerativeModel(
            model_name=gemini_settings.model_name,
            safety_settings=gemini_settings.safety_settings,
            generation_config=gemini_settings.generation_config,
        )

        self.limiter = GeminiLimiter(
            start_day=gemini_settings.start_day,
            request_per_day_used=gemini_settings.request_per_day_used,
            request_per_day_limit=gemini_settings.request_per_day_limit,
            request_per_minute_limit=gemini_settings.request_per_minute_limit,
            tokens_per_minute_limit=gemini_settings.tokens_per_minute_limit,
            raise_error_on_minute_limit=gemini_settings.raise_error_on_minute_limit,
        )

        self.chats: list[genai.ChatSession] = []

    def start_chat(self):
        """Starts a new chat session."""
        self.chats.append(self.model.start_chat())

    def send_message(
        self,
        message: types.gemini_message_input,
        stream: bool = False,
        request_options: genai_types.RequestOptions = genai_types.RequestOptions(),
        chat_index: int = -1,
    ) -> GenerateContentResponse:
        """
        Sends a message to a chat session.

        Args:
            message (str): The message to send.
            stream (bool): Whether to stream the response. Defaults to False.
            request_options (genai_types.RequestOptions): Optional request options. Defaults to an empty RequestOptions object.
            chat_index (int): The index of the chat session. Defaults to -1 (the last chat session).

        Returns:
            GenerateContentResponse: The response from the Gemini model.
        """
        return self.chats[chat_index].send_message(content=message, stream=stream, request_options=request_options)

    def get_model_name(self) -> str:
        """
        Returns the name of the Gemini model being used.

        Returns:
            str: Gemini model name.
        """
        return self.model_name

    def get_minute_limits_used(self) -> dict[str, int]:
        """
        Returns the current per-minute usage and limits.

        Returns:
            dict[str, int]: current per-minute usage and limits.
        """
        return {
            "used_requests": self.limiter.request_per_minute_used,
            "used_tokens": self.limiter.tokens_per_minute_used,
            "requests_limit": self.limiter.request_per_minute_limit,
            "tokens_limit": self.limiter.tokens_per_minute_limit,
        }

    def get_day_limits_used(self) -> dict[str, int | datetime.datetime]:
        """
        Returns the current per-day usage and limits.

        Returns:
            dict[str, int | datetime.datetime]: current per-day usage and limits.
        """
        return {
            "used_requests": self.limiter.request_per_day_used,
            "requests_limit": self.limiter.request_per_day_limit,
            "date": self.limiter.start_day,
        }

    def get_current_limit_day(self) -> datetime.datetime:
        """
        Returns the current day being used for tracking daily limits.

        Returns:
            datetime.datetime: Current day being used for tracking daily limits.
        """
        return self.limiter.start_day

    def get_api_key(self) -> str:
        """
        Returns the API key being used.

        Returns:
            str: Gemini API key.
        """
        return self.api_key

    def generate_content(
        self,
        message: types.gemini_generate_input,
        stream: bool = False,
        request_options: genai_types.RequestOptions = genai_types.RequestOptions(),
    ) -> GenerateContentResponse:
        """
        Generates content without a chat session.

        Args:
            message (str): The prompt for content generation.
            stream (bool): Whether to stream the response. Defaults to False.
            request_options (genai_types.RequestOptions): Optional request options. Defaults to an empty RequestOptions object.

        Returns:
            GenerateContentResponse: The generated content from the Gemini model.
        """
        self.limiter.add_data(self.model.count_tokens(contents=message, request_options=request_options).total_tokens)

        return self.model.generate_content(contents=message, stream=stream, request_options=request_options)

    def close_minute_limit(self):
        """Manually closes the minute limit."""
        self.limiter.close_minute_limit()

    def close_day_limit(self):
        """Manually closes the day limit."""
        self.limiter.close_day_limit()

    def close_chat(self, chat_index: int = -1):
        """
        Closes a chat session.

        Args:
            chat_index (int): The index of the chat session to close. Defaults to -1 (the last chat session).
        """
        self.chats.pop(chat_index)


class GeminiManager:
    """
    Manages multiple Gemini instances and their rate limits, switching between them as needed.

    Attributes:
        models_settings (List[GeminiSettings]): A list of GeminiSettings objects, one for each model.
        current_model_index (int): The index of the currently used Gemini model.
        current_model (Gemini): The currently used Gemini instance.

    :Usage:
        settings1 = GeminiSettings(api_key="YOUR_API_KEY_1")
        settings2 = GeminiSettings(api_key="YOUR_API_KEY_2")
        manager = GeminiManager([settings1, settings2])

        model = manager.use_current_model()
        response = model.generate_content("Write a poem.")
        print(response.text)

        manager.use_next_model() # Switches to the next available model
        model = manager.use_current_model()

        manager.use_model(model_api_key="YOUR_API_KEY_1") # switches to the model by api key
        model = manager.use_current_model()
    """

    def __init__(self, geminis_settings: list[GeminiSettings]):
        """
        Initializes a new GeminiManager instance.

        Args:
            geminis_settings (List[GeminiSettings]): A list of GeminiSettings objects.

        Raises:
            GeminiNoUsefulModelsException: If none of the provided models have available quota.
        """
        self.models_settings = geminis_settings
        self.current_model_index = self.get_lowest_useful_model_index()
        self.current_model = Gemini(self.models_settings[self.current_model_index])

    def check_models_limits(self) -> bool:
        """
        Checks if any of the managed models have available quota.

        Returns:
            bool: True if any model has available quota, False otherwise.
        """
        current_date = datetime.datetime.now(tz=pytz.timezone("America/New_York"))

        return any(
            model_settings.request_per_day_used < model_settings.request_per_day_limit
            or current_date.day != model_settings.start_day.day
            for model_settings in self.models_settings
        )

    def get_lowest_useful_model_index(self) -> int:
        """
        Finds the index of the first model with available quota.

        Returns:
            int: The index of the first available model.

        Raises:
            GeminiNoUsefulModelsException: If no models have available quota.
        """
        if self.check_models_limits():
            current_date = datetime.datetime.now(tz=pytz.timezone("America/New_York"))
            index = 0

            for i in range(len(self.models_settings)):
                if (
                    self.models_settings[i].request_per_day_used < self.models_settings[i].request_per_day_limit
                    or current_date.day != self.models_settings[i].start_day.day
                ):
                    break

                index += 1

            return index

        raise errors.GeminiNoUsefulModelsException()

    def use_next_model(self) -> Gemini:
        """
        Switches to the next available Gemini model.

        Returns:
            Gemini: The next available Gemini instance.

        Raises:
            GeminiNoUsefulModelsException: If no models have available quota.
        """
        if self.check_models_limits():
            self.current_model_index = (self.current_model_index + 1) % len(self.models_settings)
            self.current_model = Gemini(self.models_settings[self.current_model_index])

            return self.current_model

        raise errors.GeminiNoUsefulModelsException()

    def get_model_index(self, model_api_key: str | None = None) -> int:
        """
        Retrieves the index of a model based on its API KEY.

        Args:
            model_api_key (str | None): The API key of the model to search for. (Optional if you use current model)

        Returns:
           int: The index of the model if found.

        Raises:
            AttributeError: If the API key is not found in the managed models.
        """
        if model_api_key:
            for i in range(len(self.models_settings)):
                if self.models_settings[i].api_key == model_api_key:
                    return i

        raise AttributeError("This API key doesn't found")

    def use_model(self, model_index: int | None = None, model_api_key: str | None = None) -> Gemini:
        """
        Switches to a specific Gemini model by index or API key.

        Args:
            model_index (int | None): The index of the model to use. (Optional)
            model_api_key (str | None): The API key of the model to use. (Optional)

        Returns:
            Gemini: The selected Gemini instance.

        Raises:
            AttributeError: If both `model_index` and `model_api_key` are provided, or if neither are provided.
            GeminiNoUsefulModelsException: If the selected model has no available quota.
        """
        if model_index is not None and model_api_key is not None:
            raise AttributeError("You can't use both model_index and model_api_key")

        if model_index is None and model_api_key is None:
            raise AttributeError("You must provide model_index or model_api_key")

        if self.check_models_limits():
            self.current_model_index = model_index if model_index is not None else self.get_model_index(model_api_key)
            self.current_model = Gemini(self.models_settings[self.current_model_index])

            return self.current_model

        raise errors.GeminiNoUsefulModelsException()

    def use_current_model(self) -> Gemini:
        """
        Returns the currently active Gemini model.

        Returns:
            Gemini: The currently active Gemini model.
        """
        return self.current_model

    def reset_models_settings(self, gemini_settings: list[GeminiSettings]):
        """
        Resets the managed models and their settings.

        Args:
            gemini_settings (List[GeminiSettings]): A new list of GeminiSettings objects.

        Raises:
            GeminiNoUsefulModelsException: if there are no models with available quota
        """
        self.models_settings = gemini_settings
        self.current_model_index = self.get_lowest_useful_model_index()
