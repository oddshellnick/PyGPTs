import re
import typing


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
