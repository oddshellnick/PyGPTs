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
