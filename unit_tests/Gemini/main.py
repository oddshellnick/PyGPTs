import pytz
import typing
import datetime
import unittest
from google.genai import Client
from parameterized import parameterized
import google.genai.types as genai_types
from unittest.mock import (
	AsyncMock,
	MagicMock,
	patch
)
from PyGPTs.Gemini import (
	BaseGeminiChat,
	GeminiAsyncChat,
	GeminiChat,
	GeminiClient,
	GeminiClientSettings,
	GeminiClientsManager,
	GeminiLimiter,
	GeminiModel,
	GeminiModelSettings,
	data,
	errors
)


class TestGeminiClientsManager(unittest.TestCase):
	@patch("PyGPTs.Gemini.GeminiClient")
	def test_check_models_limits(self, mock_gemini_client):
		"""Test check_models_limits property returns the manager has a useful model."""
		client_api_keys_list = [f"api_key_{i}" for i in range(2)]
		client_settings_list = [
			GeminiClientSettings(api_key=api_key)
			for api_key in client_api_keys_list
		]
		client_mocks_list = [MagicMock(spec=GeminiClient) for _ in range(len(client_settings_list))]
		
		for i in range(len(client_mocks_list)):
			client_mocks_list[i].check_day_limits.return_value = True
			client_mocks_list[i].api_key = client_api_keys_list[i]
		
		mock_gemini_client.side_effect = client_mocks_list
		
		manager = GeminiClientsManager(gemini_clients_settings=client_settings_list)
		
		self.assertTrue(manager.has_useful_model)
		
		for i in range(len(client_mocks_list)):
			client_mocks_list[i].check_day_limits.return_value = False
		
		self.assertFalse(manager.has_useful_model)
		
		for i in range(len(client_mocks_list)):
			client_mocks_list[i].check_day_limits.return_value = i != 0
		
		self.assertTrue(manager.has_useful_model)
	
	@patch("PyGPTs.Gemini.GeminiClient")
	def test_client(self, mock_gemini_client):
		"""Test the client method of GeminiClientsManager to retrieve clients by index and api_key."""
		client_api_keys_list = [f"api_key_{i}" for i in range(3)]
		client_settings_list = [
			GeminiClientSettings(api_key=api_key)
			for api_key in client_api_keys_list
		]
		client_mocks_list = [MagicMock(spec=GeminiClient) for _ in range(len(client_settings_list))]
		
		for i in range(len(client_mocks_list)):
			client_mocks_list[i].check_day_limits.return_value = i != 0
			client_mocks_list[i].api_key = client_api_keys_list[i]
		
		mock_gemini_client.side_effect = client_mocks_list
		
		manager = GeminiClientsManager(gemini_clients_settings=client_settings_list)
		
		selected_client = manager.client(model_index=1)
		self.assertIs(selected_client, client_mocks_list[1])
		self.assertEqual(manager.current_model_index, 1)
		
		selected_client = manager.client(model_api_key=client_api_keys_list[2])
		self.assertIs(selected_client, client_mocks_list[2])
		self.assertEqual(manager.current_model_index, 2)
		
		selected_client = manager.client(model_api_key="wrong_api_key")
		self.assertIs(selected_client, None)
		
		selected_client = manager.client(model_index=100)
		self.assertIs(selected_client, None)
		
		with self.assertRaises(ValueError) as err:
			manager.client(model_index=0, model_api_key=client_api_keys_list[0])
		self.assertEqual(
				str(err.exception),
				"You can't use both 'model_index' and 'model_api_key'"
		)
	
	@patch("PyGPTs.Gemini.GeminiClient")
	def test_get_client_index(self, mock_gemini_client):
		"""Test get_client_index method of GeminiClientsManager to find client index by api_key."""
		client_api_keys_list = [f"api_key_{i}" for i in range(3)]
		client_settings_list = [
			GeminiClientSettings(api_key=api_key)
			for api_key in client_api_keys_list
		]
		client_mocks_list = [MagicMock(spec=GeminiClient) for _ in range(len(client_settings_list))]
		
		for i in range(len(client_mocks_list)):
			client_mocks_list[i].check_day_limits.return_value = i != 0
			client_mocks_list[i].api_key = client_api_keys_list[i]
		
		mock_gemini_client.side_effect = client_mocks_list
		
		manager = GeminiClientsManager(gemini_clients_settings=client_settings_list)
		
		index = manager.get_client_index(client_api_keys_list[1])
		self.assertEqual(index, 1)
		
		index = manager.get_client_index("wrong_api_key")
		self.assertIs(index, None)
	
	@patch("PyGPTs.Gemini.GeminiClient")
	def test_init(self, mock_gemini_client):
		"""Test initialization of GeminiClientsManager."""
		client_api_keys_list = [f"api_key_{i}" for i in range(2)]
		client_settings_list = [
			GeminiClientSettings(api_key=api_key)
			for api_key in client_api_keys_list
		]
		client_mocks_list = [MagicMock(spec=GeminiClient) for _ in range(len(client_settings_list))]
		
		for i in range(len(client_mocks_list)):
			client_mocks_list[i].check_day_limits.return_value = True
			client_mocks_list[i].api_key = client_api_keys_list[i]
		
		mock_gemini_client.side_effect = client_mocks_list
		
		manager = GeminiClientsManager(gemini_clients_settings=client_settings_list)
		
		self.assertEqual(mock_gemini_client.call_count, 2)
		self.assertEqual(len(manager.clients), 2)
		self.assertIs(manager.clients[0], client_mocks_list[0])
		self.assertIs(manager.clients[1], client_mocks_list[1])
	
	@patch("PyGPTs.Gemini.GeminiClient")
	def test_lowest_useful_model_index(self, mock_gemini_client):
		"""Test lowest_useful_model_index property returns the index of the first useful model."""
		client_api_keys_list = [f"api_key_{i}" for i in range(2)]
		client_settings_list = [
			GeminiClientSettings(api_key=api_key)
			for api_key in client_api_keys_list
		]
		client_mocks_list = [MagicMock(spec=GeminiClient) for _ in range(len(client_settings_list))]
		
		for i in range(len(client_mocks_list)):
			client_mocks_list[i].check_day_limits.return_value = i != 0
			client_mocks_list[i].api_key = client_api_keys_list[i]
		
		mock_gemini_client.side_effect = client_mocks_list
		
		manager = GeminiClientsManager(gemini_clients_settings=client_settings_list)
		
		lowest_model_index = manager.lowest_useful_client_index
		self.assertEqual(lowest_model_index, 1)
		
		for i in range(len(client_mocks_list)):
			client_mocks_list[i].check_day_limits.return_value = False
		
		lowest_model_index = manager.lowest_useful_client_index
		self.assertEqual(lowest_model_index, None)
	
	@patch("PyGPTs.Gemini.GeminiClient")
	def test_next_client(self, mock_gemini_client):
		"""Test next_client property returns next available client."""
		client_api_keys_list = [f"api_key_{i}" for i in range(3)]
		client_settings_list = [
			GeminiClientSettings(api_key=api_key)
			for api_key in client_api_keys_list
		]
		client_mocks_list = [MagicMock(spec=GeminiClient) for _ in range(len(client_settings_list))]
		
		for i in range(len(client_mocks_list)):
			client_mocks_list[i].check_day_limits.return_value = i != 0
			client_mocks_list[i].api_key = client_api_keys_list[i]
		
		mock_gemini_client.side_effect = client_mocks_list
		
		manager = GeminiClientsManager(gemini_clients_settings=client_settings_list)
		
		next_client = manager.next_client
		self.assertIs(next_client, client_mocks_list[2])
		self.assertEqual(manager.current_model_index, 2)
		
		next_client = manager.next_client
		self.assertIs(next_client, client_mocks_list[0])
		self.assertEqual(manager.current_model_index, 0)
		
		next_client = manager.next_client
		self.assertIs(next_client, client_mocks_list[1])
		self.assertEqual(manager.current_model_index, 1)
	
	@patch("PyGPTs.Gemini.GeminiClient")
	def test_reset_clients(self, mock_gemini_client):
		"""Test reset_clients method to update clients."""
		false_client_api_keys_list = [f"api_key_{i}" for i in range(2)]
		false_client_settings_list = [
			GeminiClientSettings(api_key=api_key)
			for api_key in false_client_api_keys_list
		]
		false_client_mocks_list = [MagicMock(spec=GeminiClient) for _ in range(len(false_client_settings_list))]
		
		for i in range(len(false_client_mocks_list)):
			false_client_mocks_list[i].check_day_limits.return_value = False
			false_client_mocks_list[i].api_key = false_client_api_keys_list[i]
		
		true_client_api_keys_list = [f"api_key_{i}" for i in range(2, 4)]
		true_client_settings_list = [
			GeminiClientSettings(api_key=api_key)
			for api_key in true_client_api_keys_list
		]
		true_client_mocks_list = [MagicMock(spec=GeminiClient) for _ in range(len(true_client_settings_list))]
		
		for i in range(len(true_client_mocks_list)):
			true_client_mocks_list[i].check_day_limits.return_value = True
			true_client_mocks_list[i].api_key = true_client_api_keys_list[i]
		
		mock_gemini_client.side_effect = false_client_mocks_list + true_client_mocks_list
		
		manager = GeminiClientsManager(gemini_clients_settings=[])
		
		manager.reset_clients(gemini_clients_settings=false_client_mocks_list)
		self.assertEqual(len(manager.clients), 2)
		self.assertIs(manager.clients[0], false_client_mocks_list[0])
		self.assertIs(manager.clients[1], false_client_mocks_list[1])
		self.assertIs(manager.current_model_index, None)
		
		manager.reset_clients(gemini_clients_settings=true_client_mocks_list)
		self.assertEqual(len(manager.clients), 2)
		self.assertIs(manager.clients[0], true_client_mocks_list[0])
		self.assertIs(manager.clients[1], true_client_mocks_list[1])
		self.assertIs(manager.current_model_index, 0)


class TestGeminiClient(unittest.IsolatedAsyncioTestCase):
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiModel")
	async def test_async_send_message(self, mock_gemini_model_init, mock_genai_client_init):
		"""Test async_send_message method calls chat"s send_message."""
		mock_chat_session1 = MagicMock(spec=GeminiAsyncChat)
		mock_chat_session1.send_message_stream = MagicMock()
		mock_chat_session2 = MagicMock(spec=GeminiChat)
		mock_chat_session2.send_message_stream = MagicMock()
		
		client_settings = GeminiClientSettings(
				api_key="test_api_key",
				model_settings=GeminiModelSettings(),
				chats=[mock_chat_session1, mock_chat_session2]
		)
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_gemini_model = MagicMock(spec=GeminiModel)
		mock_gemini_model_init.return_value = mock_gemini_model
		
		message = "Send message to chat"
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		await client_instance.async_send_message(message=message, chat_index=0)
		
		mock_chat_session1.send_message.assert_called_once_with(message=message)
		
		with self.assertRaises(errors.GeminiChatTypeException) as err:
			await client_instance.async_send_message(message=message, chat_index=1)
		self.assertEqual(str(err.exception), f"Chat with index 1 is not asynchronous")
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiModel")
	async def test_async_send_message_stream(self, mock_gemini_model_init, mock_genai_client_init):
		"""Test async_send_message_stream method calls chat"s send_message_stream."""
		mock_chat_session1 = MagicMock(spec=GeminiAsyncChat)
		mock_chat_session1.send_message_stream = MagicMock()
		mock_chat_session2 = MagicMock(spec=GeminiChat)
		mock_chat_session2.send_message_stream = MagicMock()
		
		client_settings = GeminiClientSettings(
				api_key="test_api_key",
				model_settings=GeminiModelSettings(),
				chats=[mock_chat_session1, mock_chat_session2]
		)
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_gemini_model = MagicMock(spec=GeminiModel)
		mock_gemini_model_init.return_value = mock_gemini_model
		
		message = "Send message to chat"
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		await client_instance.async_send_message_stream(message=message, chat_index=0)
		
		mock_chat_session1.send_message_stream.assert_called_once_with(message=message)
		
		with self.assertRaises(errors.GeminiChatTypeException) as err:
			await client_instance.async_send_message_stream(message=message, chat_index=1)
		self.assertEqual(str(err.exception), f"Chat with index 1 is not asynchronous")
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiModel")
	def test_chat_method(self, mock_gemini_model_init, mock_genai_client_init):
		"""Test chat method to access chat session."""
		mock_chat_session = MagicMock(spec=GeminiChat)
		client_settings = GeminiClientSettings(
				api_key="test_api_key",
				model_settings=GeminiModelSettings(),
				chats=[mock_chat_session]
		)
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_gemini_model = MagicMock(spec=GeminiModel)
		mock_gemini_model_init.return_value = mock_gemini_model
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		retrieved_chat = client_instance.chat(chat_index=0)
		
		self.assertIsInstance(retrieved_chat, GeminiChat)
		self.assertEqual(retrieved_chat, mock_chat_session)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	def test_check_day_limits(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test check_day_limits method."""
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_gemini_limiter_init.return_value = mock_limiter
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		client_instance.check_day_limits()
		
		mock_limiter.check_day_limits.assert_called_once()
		self.assertIsInstance(client_instance.limiter, GeminiLimiter)
		self.assertEqual(client_instance.limiter, mock_limiter)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiModel")
	def test_close_chat(self, mock_gemini_model_init, mock_genai_client_init):
		"""Test close_chat method removes chat session."""
		mock_chat_session = MagicMock(spec=GeminiChat)
		client_settings = GeminiClientSettings(
				api_key="test_api_key",
				model_settings=GeminiModelSettings(),
				chats=[mock_chat_session]
		)
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_gemini_model = MagicMock(spec=GeminiModel)
		mock_gemini_model_init.return_value = mock_gemini_model
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		client_instance.close_chat(chat_index=0)
		
		self.assertEqual(len(client_instance.chats), 0)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	def test_close_day_limit(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test close_day_limit method calls limiter method."""
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_gemini_limiter_init.return_value = mock_limiter
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		client_instance.close_day_limit()
		
		mock_limiter.close_day_limit.assert_called_once()
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	def test_close_minute_limit(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test close_minute_limit method calls limiter method."""
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_gemini_limiter_init.return_value = mock_limiter
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		client_instance.close_minute_limit()
		
		mock_limiter.close_minute_limit.assert_called_once()
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	def test_context_usage_property(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test context_usage property returns limiter"s context usage."""
		context_used = 1
		context_limit = 10
		
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_limiter.context_used = context_used
		mock_limiter.context_limit = context_limit
		mock_gemini_limiter_init.return_value = mock_limiter
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		context_usage = client_instance.context_usage
		
		self.assertEqual(
				context_usage,
				{"context_used": context_used, "context_limit": context_limit}
		)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	def test_current_limit_day_property(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test current_limit_day property returns limiter"s current limit day."""
		limit_day = datetime.datetime(2024, 1, 1, tzinfo=pytz.timezone("America/New_York"))
		
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_limiter.limit_day = limit_day
		mock_gemini_limiter_init.return_value = mock_limiter
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		current_limit_day = client_instance.current_limit_day
		
		self.assertEqual(current_limit_day, limit_day)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	def test_day_usage_property(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test day_usage property returns limiter"s day usage."""
		request_per_day_used = 1
		request_per_day_limit = 10
		limit_day = datetime.datetime(2024, 1, 1, tzinfo=pytz.timezone("America/New_York"))
		
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_limiter.request_per_day_used = request_per_day_used
		mock_limiter.request_per_day_limit = request_per_day_limit
		mock_limiter.limit_day = limit_day
		mock_gemini_limiter_init.return_value = mock_limiter
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		day_usage = client_instance.day_usage
		
		self.assertEqual(
				day_usage,
				{
					"used_requests": request_per_day_used,
					"requests_limit": request_per_day_limit,
					"date": limit_day
				}
		)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	async def test_generate_async_content(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test generate_async_content method."""
		mock_response = genai_types.GenerateContentResponse()
		mock_response.candidates = [MagicMock(token_count=5)]
		
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
		mock_genai_client.models.count_tokens.return_value.total_tokens = 10
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_gemini_limiter_init.return_value = mock_limiter
		
		message = "Generate async content"
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		await client_instance.async_generate_content(message=message)
		
		mock_limiter.async_add_data.assert_called_once_with(10)
		mock_genai_client.aio.models.generate_content.assert_called_once()
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	async def test_generate_async_content_stream(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test generate_async_content_stream method."""
		mock_stream_part1 = genai_types.GenerateContentResponse()
		mock_stream_part1.candidates = [MagicMock(token_count=3)]
		mock_stream_part2 = genai_types.GenerateContentResponse()
		mock_stream_part2.candidates = [MagicMock(token_count=2)]
		mock_stream_parts = [mock_stream_part1, mock_stream_part2]
		
		mock_generator = AsyncMock()
		mock_generator.__aiter__.return_value = mock_stream_parts
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_generator)
		mock_genai_client.models.count_tokens.return_value.total_tokens = 8
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_gemini_limiter_init.return_value = mock_limiter
		
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		client_instance.client = mock_genai_client
		
		message = "Generate async content stream"
		stream = client_instance.async_generate_content_stream(message=message)
		
		parts_generated = []
		async for part in stream:
			parts_generated.append(part)
		self.assertEqual(len(parts_generated), 2)
		
		mock_limiter.async_add_data.assert_called_once_with(8)
		mock_genai_client.aio.models.generate_content_stream.assert_called_once_with(
				model=client_instance.model_name,
				contents=message,
				config=client_instance.generation_config
		)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	def test_generate_content(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test generate_content method."""
		mock_response = genai_types.GenerateContentResponse()
		mock_response.candidates = [MagicMock(token_count=5)]
		
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client.models = MagicMock()
		mock_genai_client.models.generate_content = MagicMock(return_value=mock_response)
		mock_genai_client.models.count_tokens.return_value.total_tokens = 10
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_gemini_limiter_init.return_value = mock_limiter
		
		message = "Generate content"
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		client_instance.generate_content(message=message)
		
		mock_limiter.add_data.assert_called_once_with(10)
		mock_genai_client.models.generate_content.assert_called_once()
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	def test_generate_content_stream(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test generate_content_stream method."""
		mock_stream_part1 = genai_types.GenerateContentResponse()
		mock_stream_part1.candidates = [MagicMock(token_count=3)]
		mock_stream_part2 = genai_types.GenerateContentResponse()
		mock_stream_part2.candidates = [MagicMock(token_count=2)]
		mock_stream_parts = [mock_stream_part1, mock_stream_part2]
		
		mock_generator = MagicMock()
		mock_generator.__iter__.return_value = mock_stream_parts
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client.models = MagicMock()
		mock_genai_client.models.generate_content_stream = MagicMock(return_value=mock_generator)
		mock_genai_client.models.count_tokens.return_value.total_tokens = 8
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_gemini_limiter_init.return_value = mock_limiter
		
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		client_instance.client = mock_genai_client
		
		message = "Generate content stream"
		stream = client_instance.generate_content_stream(message=message)
		
		parts_generated = []
		for part in stream:
			parts_generated.append(part)
		self.assertEqual(len(parts_generated), 2)
		
		mock_limiter.add_data.assert_called_once_with(8)
		mock_genai_client.models.generate_content_stream.assert_called_once_with(
				model=client_instance.model_name,
				contents=message,
				config=client_instance.generation_config
		)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiModel")
	def test_generation_config_property(self, mock_gemini_model_init, mock_genai_client_init):
		"""Test generation_config property returns model"s generation config."""
		generation_config_ = genai_types.GenerateContentConfigDict(
				temperature=0.7,
				top_p=0.5,
				top_k=40,
				candidate_count=1,
				response_mime_type=data.GeminiMimeTypes.text_plain
		)
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_gemini_model = MagicMock(spec=GeminiModel)
		mock_gemini_model.generation_config = generation_config_
		mock_gemini_model_init.return_value = mock_gemini_model
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		generation_config = client_instance.generation_config
		
		self.assertEqual(generation_config, generation_config_)
	
	@patch("google.genai.Client")
	def test_get_chats(self, mock_genai_client_init):
		"""Test get_chats method yields chat histories."""
		mock_chat_session1 = MagicMock(spec=GeminiChat)
		mock_chat_session2 = MagicMock(spec=GeminiChat)
		mock_chats = [mock_chat_session1, mock_chat_session2]
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		client_settings = GeminiClientSettings(
				api_key="test_api_key",
				model_settings=GeminiModelSettings(),
				chats=mock_chats
		)
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		
		chats_history = list(client_instance.get_chats())
		self.assertEqual(len(chats_history), 2)
		self.assertEqual(chats_history[0], mock_chat_session1)
		self.assertEqual(chats_history[1], mock_chat_session2)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiModel")
	def test_init(self, mock_gemini_model_init, mock_genai_client_init):
		"""Test initialization of GeminiClient."""
		mock_client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_gemini_model = MagicMock(spec=GeminiModel)
		mock_gemini_model_init.return_value = mock_gemini_model
		
		client_instance = GeminiClient(gemini_client_settings=mock_client_settings)
		
		mock_genai_client_init.assert_called_once_with(api_key=mock_client_settings.api_key)
		mock_gemini_model_init.assert_called_once_with(mock_client_settings.model_settings)
		self.assertEqual(client_instance.api_key, mock_client_settings.api_key)
		self.assertIsInstance(client_instance.chats, typing.List)
		self.assertEqual(len(client_instance.chats), 0)
		self.assertIsInstance(client_instance.client, Client)
		self.assertEqual(client_instance.client, mock_genai_client)
		self.assertIsInstance(client_instance.model, GeminiModel)
		self.assertEqual(client_instance.model, mock_gemini_model)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	def test_limiter_property(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test limiter property returns model"s limiter."""
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_gemini_limiter_init.return_value = mock_limiter
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		limiter = client_instance.limiter
		
		self.assertEqual(limiter, mock_limiter)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiLimiter")
	def test_minute_usage_property(self, mock_gemini_limiter_init, mock_genai_client_init):
		"""Test minute_usage property returns limiter"s minute usage."""
		request_per_minute_used = 1
		request_per_minute_limit = 10
		tokens_per_minute_used = 1
		tokens_per_minute_limit = 10
		
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_limiter = MagicMock(spec=GeminiLimiter)
		mock_limiter.request_per_minute_used = request_per_minute_used
		mock_limiter.request_per_minute_limit = request_per_minute_limit
		mock_limiter.tokens_per_minute_used = tokens_per_minute_used
		mock_limiter.tokens_per_minute_limit = tokens_per_minute_limit
		mock_gemini_limiter_init.return_value = mock_limiter
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		minute_usage = client_instance.minute_usage
		
		self.assertEqual(
				minute_usage,
				{
					"used_requests": request_per_minute_used,
					"requests_limit": request_per_minute_limit,
					"used_tokens": tokens_per_minute_used,
					"tokens_limit": tokens_per_minute_limit
				}
		)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiModel")
	def test_model_name_property(self, mock_gemini_model_init, mock_genai_client_init):
		"""Test model_name property returns model"s model name."""
		model_name_ = "gemini-2.0-pro"
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_gemini_model = MagicMock(spec=GeminiModel)
		mock_gemini_model.model_name = model_name_
		mock_gemini_model_init.return_value = mock_gemini_model
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		model_name = client_instance.model_name
		
		self.assertEqual(model_name, model_name_)
	
	@patch("google.genai.Client")
	def test_model_settings_property(self, mock_genai_client_init):
		"""Test model_settings property returns model"s settings."""
		model_settings_ = GeminiModelSettings()
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=model_settings_)
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		model_settings = client_instance.model_settings
		
		self.assertEqual(model_settings, model_settings_)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiModel")
	def test_send_message(self, mock_gemini_model_init, mock_genai_client_init):
		"""Test send_message method calls chat"s send_message."""
		mock_chat_session1 = MagicMock(spec=GeminiChat)
		mock_chat_session1.send_message_stream = MagicMock()
		mock_chat_session2 = MagicMock(spec=GeminiAsyncChat)
		mock_chat_session2.send_message_stream = MagicMock()
		
		client_settings = GeminiClientSettings(
				api_key="test_api_key",
				model_settings=GeminiModelSettings(),
				chats=[mock_chat_session1, mock_chat_session2]
		)
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_gemini_model = MagicMock(spec=GeminiModel)
		mock_gemini_model_init.return_value = mock_gemini_model
		
		message = "Send message to chat"
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		client_instance.send_message(message=message, chat_index=0)
		
		mock_chat_session1.send_message.assert_called_once_with(message=message)
		
		with self.assertRaises(errors.GeminiChatTypeException) as err:
			client_instance.send_message(message=message, chat_index=1)
		self.assertEqual(str(err.exception), f"Chat with index 1 is not synchronous")
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiModel")
	def test_send_message_stream(self, mock_gemini_model_init, mock_genai_client_init):
		"""Test send_message_stream method calls chat"s send_message_stream."""
		mock_chat_session1 = MagicMock(spec=GeminiChat)
		mock_chat_session1.send_message_stream = MagicMock()
		mock_chat_session2 = MagicMock(spec=GeminiAsyncChat)
		mock_chat_session2.send_message_stream = MagicMock()
		
		client_settings = GeminiClientSettings(
				api_key="test_api_key",
				model_settings=GeminiModelSettings(),
				chats=[mock_chat_session1, mock_chat_session2]
		)
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_gemini_model = MagicMock(spec=GeminiModel)
		mock_gemini_model_init.return_value = mock_gemini_model
		
		message = "Send message to chat"
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		client_instance.send_message_stream(message=message, chat_index=0)
		
		mock_chat_session1.send_message_stream.assert_called_once_with(message=message)
		
		with self.assertRaises(errors.GeminiChatTypeException) as err:
			client_instance.send_message_stream(message=message, chat_index=1)
		self.assertEqual(str(err.exception), f"Chat with index 1 is not synchronous")
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiAsyncChat")
	def test_start_async_chat(self, mock_gemini_async_chat_init, mock_genai_client_init):
		"""Test start_async_chat method appends GeminiAsyncChat to chats."""
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client.settings = None
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_gemini_async_chat = MagicMock(spec=GeminiAsyncChat)
		mock_gemini_async_chat_init.return_value = mock_gemini_async_chat
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		client_instance.start_async_chat()
		
		self.assertEqual(len(client_instance.chats), 1)
		mock_gemini_async_chat_init.assert_called_once_with(
				client=mock_genai_client,
				model_settings=client_instance.model_settings,
				history=[]
		)
		self.assertIsInstance(client_instance.chats[0], GeminiAsyncChat)
	
	@patch("google.genai.Client")
	@patch("PyGPTs.Gemini.GeminiChat")
	def test_start_chat(self, mock_gemini_chat_init, mock_genai_client_init):
		"""Test start_chat method appends GeminiChat to chats."""
		client_settings = GeminiClientSettings(api_key="test_api_key", model_settings=GeminiModelSettings())
		
		mock_genai_client = MagicMock(spec=Client)
		mock_genai_client.settings = None
		mock_genai_client_init.return_value = mock_genai_client
		
		mock_gemini_chat = MagicMock(spec=GeminiChat)
		mock_gemini_chat_init.return_value = mock_gemini_chat
		
		client_instance = GeminiClient(gemini_client_settings=client_settings)
		client_instance.start_chat()
		
		self.assertEqual(len(client_instance.chats), 1)
		mock_gemini_chat_init.assert_called_once_with(
				client=mock_genai_client,
				model_settings=client_instance.model_settings,
				history=[]
		)
		self.assertIsInstance(client_instance.chats[0], GeminiChat)


class TestGeminiChat(unittest.TestCase):
	def test_create_chat(self):
		"""Test create_chat method in GeminiChat."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		history = [{"role": "user", "parts": ["Hello"]}, {"role": "model", "parts": ["Hi"]}]
		mock_sync_chat = MagicMock()
		mock_client.chats.create = MagicMock(return_value=mock_sync_chat)
		
		chat_instance = GeminiChat(client=mock_client, model_settings=model_settings, history=history)
		created_chat, model_name = chat_instance.create_chat(model_settings=model_settings, history=history)
		
		mock_client.chats.create.assert_called_with(
				model=model_settings.model_name,
				config=model_settings.generation_config,
				history=history
		)
		self.assertEqual(created_chat, mock_sync_chat)
		self.assertEqual(model_name, model_settings.model_name)
	
	def test_init(self):
		"""Test initialization of GeminiChat."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		history = [{"role": "user", "parts": ["Hello"]}, {"role": "model", "parts": ["Hi"]}]
		
		with patch("PyGPTs.Gemini.BaseGeminiChat.__init__") as mock_base_init:
			GeminiChat(client=mock_client, model_settings=model_settings, history=history)
			mock_base_init.assert_called_once_with(client=mock_client, model_settings=model_settings, history=history)
	
	def test_send_message(self):
		"""Test send_message method in GeminiChat."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		
		chat_instance = GeminiChat(client=mock_client, model_settings=model_settings)
		chat_instance.chat = MagicMock()
		chat_instance.limiter = MagicMock()
		
		mock_response = genai_types.GenerateContentResponse()
		mock_response.candidates = [MagicMock(token_count=5)]
		
		chat_instance.chat.send_message = MagicMock()
		chat_instance.chat.send_message.return_value = mock_response
		
		mock_client.models.count_tokens.return_value.total_tokens = 10
		
		message = "Hello Gemini"
		response = chat_instance.send_message(message=message)
		
		chat_instance.limiter.add_data.assert_called_once_with(10)
		chat_instance.chat.send_message.assert_called_once_with(message=message)
		chat_instance.limiter.add_context.assert_called_once_with(5)
		self.assertEqual(response, mock_response)
	
	def test_send_message_stream(self):
		"""Test send_message_stream method in GeminiChat."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		chat_instance = GeminiChat(client=mock_client, model_settings=model_settings)
		chat_instance.chat = MagicMock()
		chat_instance.limiter = MagicMock()
		
		mock_stream_part1 = genai_types.GenerateContentResponse()
		mock_stream_part1.candidates = [MagicMock(token_count=3)]
		mock_stream_part2 = genai_types.GenerateContentResponse()
		mock_stream_part2.candidates = [MagicMock(token_count=2)]
		mock_stream_parts = [mock_stream_part1, mock_stream_part2]
		
		mock_generator = MagicMock()
		mock_generator.__iter__.return_value = mock_stream_parts
		
		chat_instance.chat.send_message_stream = MagicMock()
		chat_instance.chat.send_message_stream.return_value = mock_generator
		
		mock_client.models.count_tokens.return_value.total_tokens = 8
		
		message = "Tell me a story"
		stream = chat_instance.send_message_stream(message=message)
		
		context_add_calls = []
		for part in stream:
			context_add_calls.append(part)
			chat_instance.limiter.add_context.assert_called_with(sum(candidate.token_count for candidate in part.candidates))
		self.assertEqual(len(context_add_calls), 2)
		
		chat_instance.limiter.add_data.assert_called_once_with(8)
		chat_instance.chat.send_message_stream.assert_called_once_with(message=message)


class TestGeminiAsyncChat(unittest.IsolatedAsyncioTestCase):
	def test_create_chat(self):
		"""Test create_chat method in GeminiAsyncChat."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		history = [{"role": "user", "parts": ["Hello"]}, {"role": "model", "parts": ["Hi"]}]
		mock_async_chat = AsyncMock()
		mock_client.aio.chats.create = MagicMock(return_value=mock_async_chat)
		
		chat_instance = GeminiAsyncChat(client=mock_client, model_settings=model_settings, history=history)
		created_chat, model_name = chat_instance.create_chat(model_settings=model_settings, history=history)
		
		mock_client.aio.chats.create.assert_called_with(
				model=model_settings.model_name,
				config=model_settings.generation_config,
				history=history
		)
		self.assertEqual(created_chat, mock_async_chat)
		self.assertEqual(model_name, model_settings.model_name)
	
	def test_init(self):
		"""Test initialization of GeminiAsyncChat."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		history = [{"role": "user", "parts": ["Hello"]}, {"role": "model", "parts": ["Hi"]}]
		
		with patch("PyGPTs.Gemini.BaseGeminiChat.__init__") as mock_base_init:
			GeminiAsyncChat(client=mock_client, model_settings=model_settings, history=history)
			mock_base_init.assert_called_once_with(client=mock_client, model_settings=model_settings, history=history)
	
	async def test_send_message(self):
		"""Test send_message method in GeminiAsyncChat."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		
		chat_instance = GeminiAsyncChat(client=mock_client, model_settings=model_settings)
		chat_instance.chat = AsyncMock()
		chat_instance.limiter.async_add_data = AsyncMock()
		chat_instance.limiter.add_context = MagicMock()
		
		mock_response = genai_types.GenerateContentResponse()
		mock_response.candidates = [MagicMock(token_count=5)]
		
		chat_instance.chat.send_message = AsyncMock()
		chat_instance.chat.send_message.return_value = mock_response
		
		mock_client.models.count_tokens.return_value.total_tokens = 10
		
		message = "Hello Gemini"
		response = await chat_instance.send_message(message=message)
		
		chat_instance.limiter.async_add_data.assert_called_once_with(10)
		chat_instance.chat.send_message.assert_called_once_with(message=message)
		chat_instance.limiter.add_context.assert_called_once_with(5)
		self.assertEqual(response, mock_response)
	
	async def test_send_message_stream(self):
		"""Test send_message_stream method in GeminiAsyncChat."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		
		chat_instance = GeminiAsyncChat(client=mock_client, model_settings=model_settings)
		chat_instance.limiter.async_add_data = AsyncMock()
		chat_instance.limiter.add_context = MagicMock()
		
		mock_stream_part1 = genai_types.GenerateContentResponse()
		mock_stream_part1.candidates = [MagicMock(token_count=3)]
		mock_stream_part2 = genai_types.GenerateContentResponse()
		mock_stream_part2.candidates = [MagicMock(token_count=2)]
		mock_stream_parts = [mock_stream_part1, mock_stream_part2]
		
		mock_async_generator = AsyncMock()
		mock_async_generator.__aiter__.return_value = mock_stream_parts
		
		chat_instance.chat.send_message_stream = AsyncMock()
		chat_instance.chat.send_message_stream.return_value = mock_async_generator
		
		mock_client.models.count_tokens.return_value.total_tokens = 8
		
		message = "Tell me a story"
		stream = chat_instance.send_message_stream(message=message)
		
		context_add_calls = []
		async for part in stream:
			context_add_calls.append(part)
			chat_instance.limiter.add_context.assert_called_with(sum(candidate.token_count for candidate in part.candidates))
		self.assertEqual(len(context_add_calls), 2)
		
		chat_instance.limiter.async_add_data.assert_called_once_with(8)
		chat_instance.chat.send_message_stream.assert_called_once_with(message=message)


class TestBaseGeminiChat(unittest.TestCase):
	@patch("PyGPTs.Gemini.BaseGeminiChat.create_chat")
	def test_change_settings(self, mock_create_chat):
		"""Test change_settings method updates settings."""
		initial_model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		new_model_settings = GeminiModelSettings(model_name="gemini-2.0-pro-latest")
		
		mock_chat_instance_initial = MagicMock()
		mock_chat_instance_new = MagicMock()
		
		mock_history_list = ["message1", "message2"]
		mock_chat_instance_initial._curated_history = mock_history_list
		mock_create_chat.side_effect = [
			(mock_chat_instance_initial, initial_model_settings.model_name),
			(mock_chat_instance_new, new_model_settings.model_name)
		]
		
		mock_client = MagicMock(spec=Client)
		chat_instance = BaseGeminiChat(client=mock_client, model_settings=initial_model_settings)
		chat_instance.change_settings(new_model_settings)
		
		mock_create_chat.assert_called_with(model_settings=new_model_settings, history=mock_history_list)
		self.assertEqual(chat_instance.model_name, new_model_settings.model_name)
		self.assertIsInstance(chat_instance.limiter, GeminiLimiter)
	
	@patch('PyGPTs.Gemini.BaseGeminiChat.create_chat')
	def test_clear_chat_history(self, mock_create_chat):
		"""Test clear_chat_history method resets chat and context."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		
		mock_chat_instance = MagicMock()
		mock_create_chat.return_value = (mock_chat_instance, model_settings.model_name)
		
		chat_instance = BaseGeminiChat(client=mock_client, model_settings=model_settings)
		chat_instance.settings = model_settings
		chat_instance.limiter = MagicMock(spec=GeminiLimiter)
		chat_instance.clear_chat_history()
		
		mock_create_chat.assert_called_with(model_settings=model_settings, history=[])
		self.assertEqual(chat_instance.chat, mock_chat_instance)
		self.assertEqual(chat_instance.model_name, model_settings.model_name)
		chat_instance.limiter.clear_context.assert_called_once()
	
	@patch("PyGPTs.Gemini.BaseGeminiChat.create_chat")
	def test_context_usage_property(self, mock_create_chat):
		"""Test context_usage property returns correct dictionary."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro", context_limit=2048)
		mock_chat_instance = MagicMock()
		mock_create_chat.return_value = (mock_chat_instance, model_settings.model_name)
		
		chat_instance = BaseGeminiChat(client=mock_client, model_settings=model_settings)
		chat_instance.limiter.context_used = 1000
		
		context_usage = chat_instance.context_usage
		self.assertEqual(context_usage, {"context_used": 1000, "context_limit": 2048})
	
	def test_create_chat_abstract(self):
		"""Test that create_chat is defined but raises NotImplementedError if called directly."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		
		with self.assertRaises(TypeError) as err:
			BaseGeminiChat(client=mock_client, model_settings=model_settings)
		self.assertEqual(str(err.exception), "cannot unpack non-iterable NoneType object")
	
	@patch("PyGPTs.Gemini.BaseGeminiChat.create_chat")
	def test_current_limit_day_property(self, mock_create_chat):
		"""Test current_limit_day property returns start_day from limiter."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		mock_chat_instance = MagicMock()
		mock_create_chat.return_value = (mock_chat_instance, model_settings.model_name)
		
		chat_instance = BaseGeminiChat(client=mock_client, model_settings=model_settings)
		expected_day = chat_instance.limiter.limit_day
		current_limit_day = chat_instance.current_limit_day
		
		self.assertEqual(current_limit_day, expected_day)
	
	@patch("PyGPTs.Gemini.BaseGeminiChat.create_chat")
	def test_day_usage_property(self, mock_create_chat):
		"""Test day_usage property returns correct dictionary."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		mock_chat_instance = MagicMock()
		mock_create_chat.return_value = (mock_chat_instance, model_settings.model_name)
		
		chat_instance = BaseGeminiChat(client=mock_client, model_settings=model_settings)
		chat_instance.limiter.request_per_day_used = 50
		chat_instance.limiter.request_per_day_limit = 100
		expected_day = chat_instance.limiter.limit_day
		
		day_usage = chat_instance.day_usage
		self.assertEqual(
				day_usage,
				{"used_requests": 50, "requests_limit": 100, "date": expected_day}
		)
	
	@patch("PyGPTs.Gemini.BaseGeminiChat.create_chat")
	def test_history_property(self, mock_create_chat):
		"""Test history property returns the chat history."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		mock_chat_instance = MagicMock()
		
		mock_history_list = ["message1", "message2"]
		mock_chat_instance._curated_history = mock_history_list
		mock_create_chat.return_value = (mock_chat_instance, model_settings.model_name)
		
		chat_instance = BaseGeminiChat(client=mock_client, model_settings=model_settings)
		history = chat_instance.history
		
		self.assertEqual(history, mock_history_list)
	
	@patch("PyGPTs.Gemini.BaseGeminiChat.create_chat")
	def test_init(self, mock_create_chat):
		"""Test initialization of BaseGeminiChat."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		history = [{"role": "user", "parts": ["Hello"]}, {"role": "model", "parts": ["Hi"]}]
		mock_chat_instance = MagicMock()
		mock_create_chat.return_value = (mock_chat_instance, model_settings.model_name)
		
		chat_instance = BaseGeminiChat(client=mock_client, model_settings=model_settings, history=history)
		
		self.assertEqual(chat_instance.client, mock_client)
		mock_create_chat.assert_called_once_with(model_settings=model_settings, history=history)
		self.assertEqual(chat_instance.chat, mock_chat_instance)
		self.assertIsInstance(chat_instance.limiter, GeminiLimiter)
	
	@patch("PyGPTs.Gemini.BaseGeminiChat.create_chat")
	def test_init_limiter_context_usage(self, mock_create_chat):
		"""Test that limiter"s context_used is correctly initialized based on history."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro", context_limit=2048)
		history = [{"role": "user", "parts": ["Hello"]}, {"role": "model", "parts": ["Hi"]}]
		mock_chat_instance = MagicMock()
		mock_create_chat.return_value = (mock_chat_instance, model_settings.model_name)
		
		mock_client.models.count_tokens.return_value.total_tokens = 10
		
		chat_instance = BaseGeminiChat(client=mock_client, model_settings=model_settings, history=history)
		
		self.assertEqual(chat_instance.limiter.context_used, 20)
	
	@patch("PyGPTs.Gemini.BaseGeminiChat.create_chat")
	def test_minute_usage_property(self, mock_create_chat):
		"""Test minute_usage property returns correct dictionary."""
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		mock_chat_instance = MagicMock()
		mock_create_chat.return_value = (mock_chat_instance, model_settings.model_name)
		
		chat_instance = BaseGeminiChat(client=mock_client, model_settings=model_settings)
		chat_instance.limiter.request_per_minute_used = 5
		chat_instance.limiter.request_per_minute_limit = 10
		chat_instance.limiter.tokens_per_minute_used = 500
		chat_instance.limiter.tokens_per_minute_limit = 1000
		
		minute_usage = chat_instance.minute_usage
		self.assertEqual(
				minute_usage,
				{
					"used_requests": 5,
					"requests_limit": 10,
					"used_tokens": 500,
					"tokens_limit": 1000
				}
		)
	
	@patch('PyGPTs.Gemini.BaseGeminiChat.create_chat')
	def test_reset_history(self, mock_create_chat):
		"""Test reset_history method updates chat and context usage."""
		mock_chat_instance = MagicMock()
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		mock_create_chat.return_value = (mock_chat_instance, model_settings.model_name)
		
		new_history = [
			{"role": "user", "parts": ["New Hello"]},
			{"role": "model", "parts": ["New Hi"]}
		]
		mock_client.models.count_tokens.return_value.total_tokens = 30
		
		chat_instance = BaseGeminiChat(client=mock_client, model_settings=model_settings)
		chat_instance.settings = model_settings
		chat_instance.limiter = MagicMock(spec=GeminiLimiter)
		chat_instance.reset_history(new_history)
		
		mock_create_chat.assert_called_with(model_settings=model_settings, history=new_history)
		self.assertEqual(chat_instance.chat, mock_chat_instance)
		self.assertEqual(chat_instance.model_name, model_settings.model_name)
		self.assertEqual(chat_instance.limiter.context_used, 30)
		mock_client.models.count_tokens.assert_called_with(model=model_settings.model_name, contents=new_history, config=None)
	
	@parameterized.expand([(1, 3), (None, 2), (2, None), (None, None)])
	@patch('PyGPTs.Gemini.BaseGeminiChat.reset_history')
	@patch('PyGPTs.Gemini.BaseGeminiChat.create_chat')
	def test_slice_history(self, start_slice, end_slice, mock_create_chat, mock_reset_history):
		"""Test slice_history method slices history and calls reset_history."""
		initial_history = [
			{"role": "user", "parts": ["Message 1"]},
			{"role": "model", "parts": ["Response 1"]},
			{"role": "user", "parts": ["Message 2"]},
			{"role": "model", "parts": ["Response 2"]},
			{"role": "user", "parts": ["Message 3"]},
		]
		
		mock_chat_instance = MagicMock()
		mock_chat_instance._curated_history = initial_history
		
		mock_client = MagicMock(spec=Client)
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		mock_create_chat.return_value = (mock_chat_instance, model_settings.model_name)
		
		chat_instance = BaseGeminiChat(
				client=mock_client,
				model_settings=model_settings,
				history=initial_history
		)
		chat_instance.slice_history(start=start_slice, end=end_slice)
		
		mock_reset_history.assert_called_with(initial_history[slice(start_slice, end_slice)], None)


class TestGeminiModel(unittest.TestCase):
	def test_init(self):
		"""Test initialization of GeminiModel."""
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		gemini_model = GeminiModel(gemini_model_settings=model_settings)
		
		self.assertEqual(gemini_model.model_name, model_settings.model_name)
		self.assertEqual(gemini_model.generation_config, model_settings.generation_config)
		self.assertIsInstance(gemini_model.limiter, GeminiLimiter)
		self.assertEqual(gemini_model.settings, model_settings)
	
	def test_limiter_initialization(self):
		"""Test that GeminiLimiter is initialized correctly within GeminiModel."""
		model_settings = GeminiModelSettings(
				model_name="gemini-2.0-pro",
				start_day=datetime.datetime.now(),
				request_per_day_used=10,
				request_per_day_limit=100,
				request_per_minute_limit=5,
				tokens_per_minute_limit=500,
				context_used=100,
				context_limit=1000,
				raise_error_on_minute_limit=False
		)
		gemini_model = GeminiModel(gemini_model_settings=model_settings)
		limiter = gemini_model.limiter
		
		self.assertEqual(limiter.limit_day, model_settings.start_day)
		self.assertEqual(limiter.request_per_day_used, model_settings.request_per_day_used)
		self.assertEqual(limiter.request_per_day_limit, model_settings.request_per_day_limit)
		self.assertEqual(
				limiter.request_per_minute_limit,
				model_settings.request_per_minute_limit
		)
		self.assertEqual(
				limiter.tokens_per_minute_limit,
				model_settings.tokens_per_minute_limit
		)
		self.assertEqual(limiter.context_used, model_settings.context_used)
		self.assertEqual(limiter.context_limit, model_settings.context_limit)
		self.assertEqual(
				limiter.raise_error_on_minute_limit,
				model_settings.raise_error_on_minute_limit
		)


class TestGeminiLimiter(unittest.TestCase):
	def setUp(self):
		"""Setup method to create a GeminiLimiter instance for each test."""
		self.start_day = datetime.datetime.now(tz=pytz.timezone("America/New_York"))
		self.request_per_day_used = 0
		self.request_per_day_limit = 100
		self.request_per_minute_limit = 10
		self.tokens_per_minute_limit = 1000
		self.context_used = 10
		self.context_limit = 2048
		self.raise_error_on_minute_limit = True
		
		self.limiter = GeminiLimiter(
				limit_day=self.start_day,
				request_per_day_used=self.request_per_day_used,
				request_per_day_limit=self.request_per_day_limit,
				request_per_minute_limit=self.request_per_minute_limit,
				tokens_per_minute_limit=self.tokens_per_minute_limit,
				context_used=self.context_used,
				context_limit=self.context_limit,
				raise_error_on_minute_limit=self.raise_error_on_minute_limit
		)
	
	def test_add_context(self):
		"""Test adding context."""
		tokens = 1
		self.limiter.add_context(tokens)
		
		self.assertEqual(self.limiter.context_used, self.context_used + tokens)
	
	def test_add_context_context_limit_exceeded(self):
		"""Test exceeding the context limit."""
		with self.assertRaises(errors.GeminiContextLimitException) as err:
			self.limiter.add_context(self.limiter.context_limit + 1)
		self.assertEqual(str(err.exception), "Model context limit reached")
	
	@patch("PyGPTs.Gemini.GeminiLimiter.check_limits")
	def test_add_data(self, mock_check_limits):
		"""Test adding data increments the usage counters and calls check_limits."""
		tokens = 100
		self.limiter.add_data(tokens)
		
		self.assertEqual(self.limiter.request_per_day_used, self.request_per_day_used + 1)
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, tokens)
		self.assertEqual(self.limiter.context_used, self.context_used + tokens)
		mock_check_limits.assert_called_once_with(tokens)
	
	@patch("PyGPTs.Gemini.GeminiLimiter.check_limits")
	def test_async_add_data(self, mock_check_limits):
		"""Test adding data increments the usage counters and calls check_limits."""
		tokens = 100
		self.limiter.add_data(tokens)
		
		self.assertEqual(self.limiter.request_per_day_used, self.request_per_day_used + 1)
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, tokens)
		self.assertEqual(self.limiter.context_used, self.context_used + tokens)
		mock_check_limits.assert_called_once_with(tokens)
	
	def test_async_check_day_limits(self):
		"""Test check_day_limits method."""
		self.limiter.request_per_day_used = self.request_per_day_limit - 1
		self.assertTrue(self.limiter.check_day_limits())
		
		self.limiter.request_per_day_used = self.request_per_day_limit
		self.assertFalse(self.limiter.check_day_limits())
		
		new_day = datetime.datetime.now(tz=pytz.timezone("America/New_York")).replace(day=self.start_day.day + 1)
		self.limiter.limit_day = new_day
		self.assertTrue(self.limiter.check_day_limits())
	
	def test_async_check_limits_context_limit_exceeded(self):
		"""Test that GeminiContextLimitException is raised when context limit is exceeded."""
		self.limiter.context_used = self.context_limit + 1
		
		with self.assertRaises(errors.GeminiContextLimitException) as err:
			self.limiter.check_limits(10)
		self.assertEqual(str(err.exception), "Model context limit reached")
	
	def test_async_check_limits_day_limit_exceeded(self):
		"""Test that GeminiDayLimitException is raised when daily limit is exceeded."""
		self.limiter.request_per_day_used = self.request_per_day_limit + 1
		
		with self.assertRaises(errors.GeminiDayLimitException) as err:
			self.limiter.check_limits(self.request_per_day_limit + 1)
		self.assertEqual(str(err.exception), "Day limit reached")
	
	def test_async_check_limits_minute_limit_exceeded(self):
		"""Test that GeminiMinuteLimitException is raised when minute limit is exceeded."""
		self.limiter.raise_error_on_minute_limit = True
		self.limiter.request_per_minute_used = self.request_per_minute_limit + 1
		
		with self.assertRaises(errors.GeminiMinuteLimitException) as err:
			self.limiter.check_limits(10)
		self.assertEqual(str(err.exception), "Minute limit reached")
		
		self.limiter.request_per_minute_used = self.request_per_minute_limit - 1
		self.limiter.tokens_per_minute_used = self.tokens_per_minute_limit + 1
		
		with self.assertRaises(errors.GeminiMinuteLimitException) as err:
			self.limiter.check_limits(10)
		self.assertEqual(str(err.exception), "Minute limit reached")
	
	@patch("time.sleep")
	def test_async_check_limits_minute_limit_exceeded_sleep(self, mock_sleep):
		"""Test that time.sleep is called when minute limit is exceeded and raise_error is False."""
		self.limiter.request_per_minute_used = self.request_per_minute_limit + 1
		self.limiter.raise_error_on_minute_limit = False
		tokens = 10
		self.limiter.check_limits(tokens)
		
		mock_sleep.assert_called()
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, tokens)
	
	@patch("time.time")
	def test_async_check_limits_minute_reset(self, mock_time):
		"""Test that minute counters are reset if a minute has passed."""
		mock_time.return_value = self.limiter.start_time + 61
		tokens = 10
		self.limiter.check_limits(tokens)
		
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, tokens)
	
	def test_async_check_limits_within_limits(self):
		"""Test that check_limits does nothing when within all limits."""
		self.limiter.request_per_day_used = self.request_per_day_limit - 1
		self.limiter.request_per_minute_used = self.request_per_minute_limit - 1
		self.limiter.tokens_per_minute_used = self.tokens_per_minute_limit - 1
		self.limiter.context_used = self.context_limit - 1
		
		try:
			self.limiter.check_limits(10)
		except errors.GeminiDayLimitException:
			self.fail("GeminiDayLimitException raised unexpectedly")
		except errors.GeminiMinuteLimitException:
			self.fail("GeminiMinuteLimitException raised unexpectedly")
		except errors.GeminiContextLimitException:
			self.fail("GeminiContextLimitException raised unexpectedly")
	
	def test_check_day_limits(self):
		"""Test check_day_limits method."""
		self.limiter.request_per_day_used = self.request_per_day_limit - 1
		self.assertTrue(self.limiter.check_day_limits())
		
		self.limiter.request_per_day_used = self.request_per_day_limit
		self.assertFalse(self.limiter.check_day_limits())
		
		new_day = datetime.datetime.now(tz=pytz.timezone("America/New_York")).replace(day=self.start_day.day + 1)
		self.limiter.limit_day = new_day
		self.assertTrue(self.limiter.check_day_limits())
	
	def test_check_limits_context_limit_exceeded(self):
		"""Test that GeminiContextLimitException is raised when context limit is exceeded."""
		self.limiter.context_used = self.context_limit + 1
		
		with self.assertRaises(errors.GeminiContextLimitException) as err:
			self.limiter.check_limits(10)
		self.assertEqual(str(err.exception), "Model context limit reached")
	
	def test_check_limits_day_limit_exceeded(self):
		"""Test that GeminiDayLimitException is raised when daily limit is exceeded."""
		self.limiter.request_per_day_used = self.request_per_day_limit + 1
		
		with self.assertRaises(errors.GeminiDayLimitException) as err:
			self.limiter.check_limits(self.request_per_day_limit + 1)
		self.assertEqual(str(err.exception), "Day limit reached")
	
	def test_check_limits_minute_limit_exceeded(self):
		"""Test that GeminiMinuteLimitException is raised when minute limit is exceeded."""
		self.limiter.raise_error_on_minute_limit = True
		self.limiter.request_per_minute_used = self.request_per_minute_limit + 1
		
		with self.assertRaises(errors.GeminiMinuteLimitException) as err:
			self.limiter.check_limits(10)
		self.assertEqual(str(err.exception), "Minute limit reached")
		
		self.limiter.request_per_minute_used = self.request_per_minute_limit - 1
		self.limiter.tokens_per_minute_used = self.tokens_per_minute_limit + 1
		
		with self.assertRaises(errors.GeminiMinuteLimitException) as err:
			self.limiter.check_limits(10)
		self.assertEqual(str(err.exception), "Minute limit reached")
	
	@patch("time.sleep")
	def test_check_limits_minute_limit_exceeded_sleep(self, mock_sleep):
		"""Test that time.sleep is called when minute limit is exceeded and raise_error is False."""
		self.limiter.request_per_minute_used = self.request_per_minute_limit + 1
		self.limiter.raise_error_on_minute_limit = False
		tokens = 10
		self.limiter.check_limits(tokens)
		
		mock_sleep.assert_called()
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, tokens)
	
	@patch("time.time")
	def test_check_limits_minute_reset(self, mock_time):
		"""Test that minute counters are reset if a minute has passed."""
		mock_time.return_value = self.limiter.start_time + 61
		tokens = 10
		self.limiter.check_limits(tokens)
		
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, tokens)
	
	def test_check_limits_within_limits(self):
		"""Test that check_limits does nothing when within all limits."""
		self.limiter.request_per_day_used = self.request_per_day_limit - 1
		self.limiter.request_per_minute_used = self.request_per_minute_limit - 1
		self.limiter.tokens_per_minute_used = self.tokens_per_minute_limit - 1
		self.limiter.context_used = self.context_limit - 1
		
		try:
			self.limiter.check_limits(10)
		except errors.GeminiDayLimitException:
			self.fail("GeminiDayLimitException raised unexpectedly")
		except errors.GeminiMinuteLimitException:
			self.fail("GeminiMinuteLimitException raised unexpectedly")
		except errors.GeminiContextLimitException:
			self.fail("GeminiContextLimitException raised unexpectedly")
	
	def test_clear_context(self):
		"""Test clearing context usage."""
		self.limiter.clear_context()
		self.assertEqual(self.limiter.context_used, 0)
	
	def test_close_day_limit(self):
		"""Test close_day_limit method."""
		self.limiter.close_day_limit()
		
		self.assertEqual(self.limiter.request_per_day_used, self.request_per_day_limit)
	
	def test_close_minute_limit(self):
		"""Test close_minute_limit method."""
		self.limiter.close_minute_limit()
		
		self.assertEqual(self.limiter.request_per_minute_used, self.request_per_minute_limit)
		self.assertEqual(self.limiter.tokens_per_minute_used, self.tokens_per_minute_limit)
	
	def test_decrease_context(self):
		"""Test decreasing context usage."""
		tokens_to_decrease = 5
		decreased_context = self.context_used - tokens_to_decrease
		self.limiter.decrease_context(tokens_to_decrease)
		self.assertEqual(self.limiter.context_used, decreased_context)
		
		tokens_to_decrease = self.context_used + 1
		with self.assertRaises(ValueError) as context:
			self.limiter.decrease_context(tokens_to_decrease)
		self.assertEqual(str(context.exception), "Cannot decrease context below 0")
		self.assertEqual(self.limiter.context_used, decreased_context)
	
	def test_init(self):
		"""Test the initialization of GeminiLimiter."""
		self.assertEqual(self.limiter.limit_day, self.start_day)
		self.assertEqual(self.limiter.request_per_day_limit, self.request_per_day_limit)
		self.assertEqual(self.limiter.request_per_minute_limit, self.request_per_minute_limit)
		self.assertEqual(self.limiter.tokens_per_minute_limit, self.tokens_per_minute_limit)
		self.assertEqual(
				self.limiter.raise_error_on_minute_limit,
				self.raise_error_on_minute_limit
		)
		self.assertEqual(self.limiter.request_per_day_used, self.request_per_day_used)
		self.assertEqual(self.limiter.request_per_minute_used, 0)
		self.assertEqual(self.limiter.tokens_per_minute_used, 0)
		self.assertEqual(self.limiter.context_used, self.context_used)
		self.assertEqual(self.limiter.context_limit, self.context_limit)
		self.assertIsInstance(self.limiter.start_time, float)


class TestGeminiClientSettings(unittest.TestCase):
	def test_init_with_api_key_and_model_settings(self):
		"""Test initialization with both API key and custom model settings."""
		api_key = "test_api_key"
		model_settings = GeminiModelSettings(model_name="gemini-2.0-pro")
		settings = GeminiClientSettings(api_key=api_key, model_settings=model_settings)
		
		self.assertEqual(settings.api_key, api_key)
		self.assertEqual(settings.model_settings, model_settings)
	
	def test_init_with_api_key_only(self):
		"""Test initialization with only the API key."""
		api_key = "test_api_key"
		settings = GeminiClientSettings(api_key=api_key)
		
		self.assertEqual(settings.api_key, api_key)
		self.assertIsInstance(settings.model_settings, GeminiModelSettings)


class TestGeminiModelSettings(unittest.TestCase):
	def test_init_default(self):
		"""Test default initialization of GeminiModelSettings."""
		settings = GeminiModelSettings()
		model_name = data.GeminiModels.Gemini_2_0_flash.latest_stable
		
		self.assertEqual(settings.model_name, model_name)
		self.assertIsInstance(settings.generation_config, dict)
		self.assertEqual(settings.request_per_day_used, 0)
		self.assertEqual(
				settings.request_per_day_limit,
				data.GeminiLimits.request_per_day[model_name]
		)
		self.assertEqual(
				settings.request_per_minute_limit,
				data.GeminiLimits.request_per_minute[model_name]
		)
		self.assertEqual(
				settings.tokens_per_minute_limit,
				data.GeminiLimits.tokens_per_minute[model_name]
		)
		self.assertEqual(settings.context_used, 0)
		self.assertEqual(settings.context_limit, data.GeminiLimits.context_limit[model_name])
		self.assertTrue(settings.raise_error_on_minute_limit)
		self.assertIsInstance(settings.start_day, datetime.datetime)
	
	def test_init_default_limits_from_data(self):
		"""Test that default limits are fetched from data.GeminiLimits."""
		model_name = "gemini-2.0-flash-latest"
		base_model_name = "gemini-2.0-flash"
		settings = GeminiModelSettings(model_name=model_name)
		
		self.assertEqual(
				settings.request_per_day_limit,
				data.GeminiLimits.request_per_day[base_model_name]
		)
		self.assertEqual(
				settings.request_per_minute_limit,
				data.GeminiLimits.request_per_minute[base_model_name]
		)
		self.assertEqual(
				settings.tokens_per_minute_limit,
				data.GeminiLimits.tokens_per_minute[base_model_name]
		)
		self.assertEqual(settings.context_limit, data.GeminiLimits.context_limit[base_model_name])
	
	def test_init_raises_value_error_when_model_not_in_limits(self):
		"""Test that ValueError is raised when model is not in GeminiLimits."""
		model_name = "non-existing-model"
		
		with self.assertRaises(ValueError) as err:
			GeminiModelSettings(model_name=model_name)
		self.assertEqual(
				str(err.exception),
				f"{model_name} is not a default model name. Specify 'request_per_day_limit'."
		)
		
		with self.assertRaises(ValueError) as err:
			GeminiModelSettings(model_name=model_name, request_per_day_limit=1)
		self.assertEqual(
				str(err.exception),
				f"{model_name} is not a default model name. Specify 'request_per_minute_limit'."
		)
		
		with self.assertRaises(ValueError) as err:
			GeminiModelSettings(
					model_name=model_name,
					request_per_day_limit=1,
					request_per_minute_limit=1
			)
		self.assertEqual(
				str(err.exception),
				f"{model_name} is not a default model name. Specify 'tokens_per_minute_limit'."
		)
		
		with self.assertRaises(ValueError) as err:
			GeminiModelSettings(
					model_name=model_name,
					request_per_day_limit=1,
					request_per_minute_limit=1,
					tokens_per_minute_limit=1
			)
		self.assertEqual(
				str(err.exception),
				f"{model_name} is not a default model name. Specify 'context_limit'."
		)
	
	def test_init_with_custom_safety_settings(self):
		"""Test initialization with custom safety settings."""
		model_name = "gemini-2.0-pro"
		safety_settings = [
			genai_types.SafetySettingDict(
					category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
					threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
			)
		]
		generation_config = genai_types.GenerateContentConfigDict(safety_settings=safety_settings)
		
		settings = GeminiModelSettings(model_name=model_name, generation_config=generation_config,)
		
		self.assertEqual(settings.model_name, model_name)
		self.assertEqual(settings.generation_config, generation_config)
		self.assertEqual(settings.generation_config["safety_settings"], safety_settings)
	
	def test_init_with_params(self):
		"""Test initialization with custom parameters."""
		model_name = "gemini-2.0-pro"
		generation_config = genai_types.GenerateContentConfigDict(temperature=0.9)
		start_day = datetime.datetime(2024, 1, 1, tzinfo=pytz.utc)
		request_per_day_used = 5
		request_per_day_limit = 100
		request_per_minute_limit = 10
		tokens_per_minute_limit = 1000
		context_used = 50
		context_limit = 2048
		raise_error_on_minute_limit = False
		
		settings = GeminiModelSettings(
				model_name=model_name,
				generation_config=generation_config,
				start_day=start_day,
				request_per_day_used=request_per_day_used,
				request_per_day_limit=request_per_day_limit,
				request_per_minute_limit=request_per_minute_limit,
				tokens_per_minute_limit=tokens_per_minute_limit,
				context_used=context_used,
				context_limit=context_limit,
				raise_error_on_minute_limit=raise_error_on_minute_limit
		)
		
		start_day = start_day.astimezone(pytz.timezone("America/New_York"))
		start_day_formatted = datetime.datetime(
				year=start_day.year,
				month=start_day.month,
				day=start_day.day,
				tzinfo=start_day.tzinfo
		)
		
		self.assertEqual(settings.model_name, model_name)
		self.assertEqual(settings.generation_config, generation_config)
		self.assertEqual(settings.start_day, start_day_formatted)
		self.assertEqual(settings.request_per_day_used, request_per_day_used)
		self.assertEqual(settings.request_per_day_limit, request_per_day_limit)
		self.assertEqual(settings.request_per_minute_limit, request_per_minute_limit)
		self.assertEqual(settings.tokens_per_minute_limit, tokens_per_minute_limit)
		self.assertEqual(settings.context_used, context_used)
		self.assertEqual(settings.context_limit, context_limit)
		self.assertEqual(settings.raise_error_on_minute_limit, raise_error_on_minute_limit)


def main_test_suite():
	suite = unittest.TestSuite()
	test_loader = unittest.TestLoader()
	
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiModelSettings))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiClientSettings))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiLimiter))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiModel))
	suite.addTest(test_loader.loadTestsFromTestCase(TestBaseGeminiChat))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiAsyncChat))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiChat))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiClient))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiClientsManager))
	
	return suite
