import typing


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


class GeminiContentDict(typing.TypedDict):
    """
    Represents a dictionary for Gemini content.

    Attributes:
        content (str): The actual content string.
        role (str): The role of the content (e.g., user or model).

    :Usage:
        content_dict: GeminiContentDict = {"content": "Hello, Gemini!", "role": GeminiContentRoles.user}
    """
    content: str | GeminiFileData
    role: str


gemini_message_input = typing.Union[str, GeminiContentDict, GeminiFileData]
gemini_generate_input = typing.Union[str, GeminiContentDict, GeminiFileData, typing.Iterable[GeminiContentDict, GeminiFileData]]
