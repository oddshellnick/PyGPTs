import re
import typing
from google.genai.types import GenerateContentResponse


def find_base_model(model_version: str) -> typing.Optional[str]:
	"""
	Extracts the base model name from a given model version string.

	This function uses a regular expression to identify and extract the base model name from a model version string.
	The base model name is expected to be at the beginning of the string and follow a pattern like:
	"model-version-variant" or "model-version".

	Args:
		model_version (str): The model version string to parse.

	Returns:
		typing.Optional[str]: The extracted base model name, or None if no base model name is found in the string.

	:Usage:
		find_base_model("gemini-1.0-pro") # returns "gemini-1.0-pro"
		find_base_model("gemini-1.5-pro-001") # returns "gemini-1.5-pro"
		find_base_model("gemini-2.0-flash-latest") # returns "gemini-2.0-flash"
		find_base_model("some-invalid-model-name") # returns None
	"""
	found = re.search(
			r"\A[a-z]+-[0-9.]+-[a-z]+(?:-\b(?:\d+b|it|lite|thinking)\b)*",
			model_version
	)
	
	return found.group(0) if found else None


def extract_token_count_from_gemini_response(gemini_response: GenerateContentResponse) -> int:
	"""
	Extracts the total token count from a Gemini API response object.

	This function sums up the `token_count` attribute of each candidate within a `GenerateContentResponse`.
	If a candidate does not have a `token_count` (is None), it's treated as having 0 tokens.

	Args:
		gemini_response (GenerateContentResponse): The Gemini API response object from which to extract the token count.

	Returns:
		int: The total token count from the Gemini response.

	:Usage:
		response = gemini_client.generate_content("Write a short description")
		token_count = extract_token_count_from_gemini_response(response)
		print(f"Token count: {token_count}")
	"""
	if gemini_response.candidates is not None:
		return sum(
				candidate.token_count
				for candidate in gemini_response.candidates
				if candidate.token_count is not None
		)
	
	return 0


def extract_text_from_gemini_response(gemini_response: GenerateContentResponse) -> str:
	"""
	Extracts the text content from a Gemini API response object.

	This function iterates through the candidates and parts within a `GenerateContentResponse` object
	to concatenate and return the text content.

	Args:
		gemini_response (GenerateContentResponse): The Gemini API response object from which to extract text.

	Returns:
		str: The extracted text content from the Gemini response.

	:Usage:
		response = gemini_client.generate_content("Write a short description")
		text_content = extract_text_from_gemini_response(response)
		print(text_content)
	"""
	if gemini_response.candidates is not None:
		return "".join(
				part.text
				for candidate in gemini_response.candidates
				if candidate.content is not None
				and candidate.content.parts is not None
				for part in candidate.content.parts
				if part.text is not None
		)
	
	return ""
