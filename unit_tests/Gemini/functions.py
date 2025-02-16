from unittest.mock import MagicMock
from parameterized import parameterized
from unittest import (
	TestCase,
	TestLoader,
	TestSuite,
	TextTestRunner
)
from google.genai.types import (
	Candidate,
	Content,
	GenerateContentResponse,
	Part
)
from PyGPTs.Gemini.functions import (
	extract_text_from_gemini_response,
	extract_token_count_from_gemini_response,
	find_base_model
)


class TestGeminiResponseTokenCountExtraction(TestCase):
	@parameterized.expand([(None, 0), ([], 0), ([None], 0), ([0], 0), ([15, None], 15), ([10, 20], 30)])
	def test_extract_token_count_from_gemini_response(self, candidates, expected_count):
		"""Test extract_token_count_from_gemini_response function."""
		if candidates is not None:
			mock_candidates = [
				MagicMock(spec=Candidate, token_count=candidate) for candidate in candidates
			]
		else:
			mock_candidates = None
		
		mock_gemini_response = MagicMock(spec=GenerateContentResponse, candidates=mock_candidates)
		
		token_count = extract_token_count_from_gemini_response(mock_gemini_response)
		self.assertEqual(token_count, expected_count)


class TestGeminiResponseTextExtraction(TestCase):
	@parameterized.expand(
			[
				(None, ""),
				([], ""),
				([{"content": None}], ""),
				([{"content": {"parts": None}}], ""),
				(
						[{"content": {"parts": ["This is part 1. ", "This is part 2."]}}],
						"This is part 1. This is part 2."
				)
			]
	)
	def test_extract_text_from_gemini_response(self, candidates, expected_text):
		"""Test extract_text_from_gemini_response function."""
		if candidates is not None:
			mock_candidates = []
		
			for candidate in candidates:
				if candidate["content"] is not None:
					if candidate["content"]["parts"] is not None:
						mock_parts = []
						for part in candidate["content"]["parts"]:
							mock_parts.append(MagicMock(spec=Part, text=part))
					else:
						mock_parts = None
		
					mock_content = MagicMock(spec=Content, parts=mock_parts)
				else:
					mock_content = None
		
				mock_candidates.append(MagicMock(spec=Candidate, content=mock_content))
		else:
			mock_candidates = None
		
		mock_gemini_response = MagicMock(spec=GenerateContentResponse, candidates=mock_candidates)
		
		extracted_text = extract_text_from_gemini_response(mock_gemini_response)
		self.assertEqual(extracted_text, expected_text)


class TestFindBaseModel(TestCase):
	@parameterized.expand(
			[
				("gemini-1.0-pro", "gemini-1.0-pro"),
				("gemini-1.5-pro-001", "gemini-1.5-pro"),
				("gemini-2.0-flash-latest", "gemini-2.0-flash"),
				("gemini-2.0-pro-exp", "gemini-2.0-pro"),
				("gemini-2.0-flash-lite", "gemini-2.0-flash-lite"),
				("gemini-2.0-flash-thinking", "gemini-2.0-flash-thinking"),
				("gemini-1.5-flash-8b", "gemini-1.5-flash-8b"),
				("gemini-1.5-flash-8b-001", "gemini-1.5-flash-8b"),
				("gemini-1.5-flash-8b-it-001", "gemini-1.5-flash-8b-it"),
				("some-invalid-model-name", None),
				("invalid-1.0", None),
				("", None),
				("gemini-1.5-pro-extra-hyphens", "gemini-1.5-pro"),
				("gemini-2.0-flash-lite-02-05", "gemini-2.0-flash-lite"),
			]
	)
	def test_base_model_name(self, model_version, expected_base_model):
		"""Test find_base_model with various model name formats."""
		actual_base_model = find_base_model(model_version)
		self.assertEqual(actual_base_model, expected_base_model)


def functions_test_suite() -> TestSuite:
	suite = TestSuite()
	test_loader = TestLoader()
	
	suite.addTest(test_loader.loadTestsFromTestCase(TestFindBaseModel))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiResponseTextExtraction))
	suite.addTest(
			test_loader.loadTestsFromTestCase(TestGeminiResponseTokenCountExtraction)
	)
	
	return suite


if __name__ == "__main__":
	runner = TextTestRunner()
	runner.run(functions_test_suite())
