import typing


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


class GeminiMimeTypes:
    """
    Defines common MIME types for Gemini.

    Attributes:
        image_jpeg (str): MIME type for JPEG images.
        image_png (str): MIME type for PNG images.
        image_gif (str): MIME type for GIF images.
        audio_mpeg (str): MIME type for MPEG audio.
        audio_wav (str): MIME type for WAV audio.
        video_mpeg (str): MIME type for MPEG video.
        video_mp4 (str): MIME type for MP4 video.

    :Usage:
        GeminiMimeTypes.image_jpeg
        "image/jpeg"

        GeminiMimeTypes.audio_mpeg
        "audio/mpeg"
    """
    image_jpeg = "image/jpeg"
    image_png = "image/png"
    image_gif = "image/gif"
    audio_mpeg = "audio/mpeg"
    audio_wav = "audio/wav"
    video_mpeg = "video/mpeg"
    video_mp4 = "video/mp4"


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


class GeminiFileData(typing.TypedDict):
    """
    Represents file data for Gemini.

    Attributes:
        mime_type (str): The MIME type of the file.
        file_uri (str): The URI of the file.

    :Usage:
        file_data: GeminiFileData = {"mime_type": GeminiMimeTypes.image_jpeg, "file_uri": "gs://my-bucket/image.jpg"}
    """
    mime_type: str
    file_uri: str


class GeminiContentRoles:
    """
    Defines the roles for Gemini content.

    Attributes:
        user (str): Represents the user role.
        model (str): Represents the model (AI) role.

    :Usage:
        GeminiContentRoles.user
        "user"

        GeminiContentRoles.model
        "model"
    """
    user = "user"
    model = "model"


class GeminiContentDict(typing.TypedDict):
    """
    Represents a dictionary for Gemini content.

    Attributes:
        content (str): The actual content string.
        role (str): The role of the content (e.g., user or model).

    :Usage:
        content_dict: GeminiContentDict = {"content": "Hello, Gemini!", "role": GeminiContentRoles.user}
    """
    content: str
    role: str
