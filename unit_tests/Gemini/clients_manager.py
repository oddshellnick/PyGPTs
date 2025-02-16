from typing import Optional, Union
from parameterized import parameterized
from unittest.mock import MagicMock, patch
from PyGPTs.Gemini.clients_manager import GeminiClientsManager
from PyGPTs.Gemini.client import (
	GeminiClient,
	GeminiClientSettings
)
from unittest import (
	TestCase,
	TestLoader,
	TestSuite,
	TextTestRunner
)


class TestGeminiClientsManager(TestCase):
	def setUp(self):
		self.mock_client1 = MagicMock(spec=GeminiClient)
		self.mock_client1.api_key = "api_key_1"
		self.mock_client_settings1 = MagicMock(spec=GeminiClientSettings)
		self.mock_client_settings1.has_day_limits = True
		
		self.mock_client2 = MagicMock(spec=GeminiClient)
		self.mock_client2.api_key = "api_key_2"
		self.mock_client_settings2 = MagicMock(spec=GeminiClientSettings)
		self.mock_client_settings2.has_day_limits = False
		
		with patch(
				"PyGPTs.Gemini.clients_manager.GeminiClient",
				side_effect=[self.mock_client1, self.mock_client2]
		):
			self.manager = GeminiClientsManager([self.mock_client_settings1, self.mock_client_settings2])
	
	@parameterized.expand(
			[
				({"model_index": 1}, 1),
				({"model_api_key": "api_key_1"}, 0),
				({"model_index": 5}, None),
				({}, 0)
			]
	)
	def test_client(
			self,
			input_params: dict[str, Union[int, str]],
			expected_index: Optional[int]
	):
		expected_client = [self.mock_client1, self.mock_client2][expected_index] if expected_index is not None else None
		
		client = self.manager.client(**input_params)
		
		self.assertEqual(client, expected_client)
		self.assertEqual(self.manager.current_model_index, expected_index)
	
	def test_client_both_params_raises_error(self):
		with self.assertRaises(ValueError) as err:
			self.manager.client(model_index=0, model_api_key="api_key_1")
		self.assertEqual(
				str(err.exception),
				"You can't use both 'model_index' and 'model_api_key'"
		)
	
	@parameterized.expand([("api_key_1", 0), ("api_key_2", 1), ("unknown_api_key", None)])
	def test_get_client_index(self, api_key, expected_index):
		index = self.manager.get_client_index(api_key)
		
		self.assertEqual(index, expected_index)
	
	@parameterized.expand(
			[
				(True, True, True),
				(False, True, True),
				(True, False, True),
				(False, False, False)
			]
	)
	def test_has_useful_model(
			self,
			has_day_limits1: bool,
			has_day_limits2: bool,
			expected_result: bool
	):
		self.mock_client1.has_day_limits = has_day_limits1
		self.mock_client2.has_day_limits = has_day_limits2
		
		self.assertEqual(self.manager.has_useful_model, expected_result)
	
	def test_init(self):
		self.assertEqual(len(self.manager.clients), 2)
		self.assertEqual(self.manager.clients[0], self.mock_client1)
		self.assertEqual(self.manager.clients[1], self.mock_client2)
		self.assertEqual(self.manager.current_model_index, 0)
	
	@parameterized.expand([(True, True, 0), (False, True, 1), (True, False, 0), (False, False, None)])
	def test_lowest_useful_client_index(
			self,
			has_day_limits1: bool,
			has_day_limits2: bool,
			expected_result: int
	):
		self.mock_client1.has_day_limits = has_day_limits1
		self.mock_client2.has_day_limits = has_day_limits2
		
		index = self.manager.lowest_useful_client_index
		
		self.assertEqual(index, expected_result)
	
	def test_next_client(self):
		self.manager.current_model_index = 0
		next_client = self.manager.next_client
		
		self.assertEqual(next_client, self.mock_client2)
		self.assertEqual(self.manager.current_model_index, 1)
		
		next_client_again = self.manager.next_client
		
		self.assertEqual(next_client_again, self.mock_client1)
		self.assertEqual(self.manager.current_model_index, 0)
	
	def test_reset_clients(self):
		mock_client3 = MagicMock(spec=GeminiClient)
		mock_client_settings3 = MagicMock(spec=GeminiClientSettings)
		mock_client_settings3.has_day_limits = True
		new_settings_list = [mock_client_settings3]
		
		with patch(
				"PyGPTs.Gemini.clients_manager.GeminiClient",
				return_value=mock_client3
		) as mock_gemini_client:
			self.manager.reset_clients(new_settings_list)
		
		mock_gemini_client.assert_called_with(mock_client_settings3)
		self.assertEqual(len(self.manager.clients), 1)
		self.assertEqual(self.manager.clients[0], mock_client3)
		self.assertEqual(self.manager.current_model_index, 0)


def clients_manager_test_suite() -> TestSuite:
	suite = TestSuite()
	test_loader = TestLoader()
	
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiClientsManager))
	
	return suite


if __name__ == "__main__":
	runner = TextTestRunner()
	runner.run(clients_manager_test_suite())
