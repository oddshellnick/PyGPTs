from google.genai import Client
from PyGPTs.Gemini.data import GeminiModels
from PyGPTs.Gemini.model import GeminiModelSettings
from google.genai.types import GenerateContentResponse
from PyGPTs.Gemini.errors import GeminiChatTypeException
from unittest.mock import (
	AsyncMock,
	MagicMock,
	patch
)
from PyGPTs.Gemini.chat import (
	GeminiAsyncChat,
	GeminiChat
)
from PyGPTs.Gemini.client import (
	GeminiClient,
	GeminiClientSettings
)
from unittest import (
	IsolatedAsyncioTestCase,
	TestCase,
	TestLoader,
	TestSuite,
	TextTestRunner
)


class TestGeminiClient(IsolatedAsyncioTestCase):
	def setUp(self):
		self.mock_client = MagicMock(spec=Client)
		self.mock_client_settings = GeminiClientSettings(api_key="test_api_key")
		
		self.gemini_client = GeminiClient(self.mock_client_settings)
		self.gemini_client.client = self.mock_client
		
		self.mock_gemini_response = MagicMock(spec=GenerateContentResponse)
	
	@patch("PyGPTs.Gemini.client.GeminiClient.async_add_data")
	async def test_async_generate_content(self, mock_async_add_data: MagicMock):
		self.gemini_client.client.models.count_tokens = MagicMock()
		self.gemini_client.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
		
		self.mock_client.aio.models.generate_content = AsyncMock()
		self.mock_client.aio.models.generate_content.return_value = self.mock_gemini_response
		
		response = await self.gemini_client.async_generate_content(message="Test async generate")
		
		mock_async_add_data.assert_called_with(10)
		self.mock_client.aio.models.generate_content.assert_called_with(
				model=self.gemini_client.model_name,
				contents="Test async generate",
				config=self.gemini_client.generation_config
		)
		self.assertEqual(response, self.mock_gemini_response)
	
	@patch("PyGPTs.Gemini.client.GeminiClient.async_add_data")
	async def test_async_generate_content_stream(self, mock_async_add_data: MagicMock):
		mock_count_tokens_response = MagicMock(total_tokens=5)
		self.mock_client.models.count_tokens.return_value = mock_count_tokens_response
		
		mock_stream = AsyncMock()
		mock_stream.__aiter__.return_value = [self.mock_gemini_response]
		
		self.mock_client.aio.models.generate_content_stream = AsyncMock()
		self.mock_client.aio.models.generate_content_stream.return_value = mock_stream
		
		stream_generator = self.gemini_client.async_generate_content_stream(message="Test async stream")
		responses = [response async for response in stream_generator]
		
		mock_async_add_data.assert_called_with(5)
		self.mock_client.aio.models.generate_content_stream.assert_called_with(
				model=self.gemini_client.model_name,
				contents="Test async stream",
				config=self.gemini_client.generation_config
		)
		self.assertEqual(responses, [self.mock_gemini_response])
	
	async def test_async_send_message_async_chat(self):
		mock_async_chat = MagicMock(spec=GeminiAsyncChat, is_async=True)
		mock_async_chat.send_message.return_value = self.mock_gemini_response
		self.gemini_client.chats = [mock_async_chat]
		
		response = await self.gemini_client.async_send_message(message="Async chat message")
		
		mock_async_chat.send_message.assert_called_with(message="Async chat message")
		self.assertEqual(response, self.mock_gemini_response)
	
	async def test_async_send_message_stream_async_chat(self):
		mock_stream = AsyncMock()
		mock_stream.__aiter__.return_value = [self.mock_gemini_response]
		
		mock_async_chat = MagicMock(spec=GeminiAsyncChat, is_async=True)
		mock_async_chat.send_message_stream = MagicMock()
		mock_async_chat.send_message_stream.return_value = mock_stream
		
		self.gemini_client.chats = [mock_async_chat]
		
		stream_generator = await self.gemini_client.async_send_message_stream(message="Async stream message")
		responses = [response async for response in stream_generator]
		
		mock_async_chat.send_message_stream.assert_called_with(message="Async stream message")
		self.assertEqual(responses, [self.mock_gemini_response])
	
	async def test_async_send_message_stream_sync_chat_raises_error(self):
		mock_sync_chat = MagicMock(spec=GeminiChat, is_async=False)
		self.gemini_client.chats = [mock_sync_chat]
		
		with self.assertRaises(GeminiChatTypeException) as err:
			await self.gemini_client.async_send_message_stream(message="Should raise error")
		self.assertEqual(str(err.exception), "Chat with index -1 is not asynchronous")
	
	async def test_async_send_message_sync_chat_raises_error(self):
		mock_sync_chat = MagicMock(spec=GeminiChat, is_async=False)
		self.gemini_client.chats = [mock_sync_chat]
		
		with self.assertRaises(GeminiChatTypeException) as err:
			await self.gemini_client.async_send_message(message="Should raise error")
		self.assertEqual(str(err.exception), "Chat with index -1 is not asynchronous")
	
	def test_chat_custom_index(self):
		mock_chat1 = MagicMock(spec=GeminiChat)
		mock_chat2 = MagicMock(spec=GeminiAsyncChat)
		self.gemini_client.chats = [mock_chat1, mock_chat2]
		
		retrieved_chat = self.gemini_client.chat(0)
		
		self.assertEqual(retrieved_chat, mock_chat1)
	
	def test_chat_default_index(self):
		mock_chat = MagicMock(spec=GeminiChat)
		self.gemini_client.chats = [mock_chat]
		
		retrieved_chat = self.gemini_client.chat()
		
		self.assertEqual(retrieved_chat, mock_chat)
	
	def test_client_settings_getter(self):
		retrieved_settings = self.gemini_client.client_settings
		
		self.assertIsInstance(retrieved_settings, GeminiClientSettings)
		self.assertEqual(retrieved_settings.api_key, self.gemini_client.api_key)
		self.assertEqual(retrieved_settings.chats, self.gemini_client.chats)
		self.assertEqual(retrieved_settings.model_name, self.gemini_client.model_name)
		self.assertEqual(
				retrieved_settings.generation_config,
				self.gemini_client.generation_config
		)
		self.assertEqual(
				retrieved_settings.count_tokens_config,
				self.gemini_client.count_tokens_config
		)
		self.assertEqual(
				retrieved_settings.request_per_day_used,
				self.gemini_client.request_per_day_used
		)
		self.assertEqual(
				retrieved_settings.request_per_day_limit,
				self.gemini_client.request_per_day_limit
		)
		self.assertEqual(
				retrieved_settings.request_per_minute_limit,
				self.gemini_client.request_per_minute_limit
		)
		self.assertEqual(
				retrieved_settings.tokens_per_minute_limit,
				self.gemini_client.tokens_per_minute_limit
		)
		self.assertEqual(retrieved_settings.context_used, self.gemini_client.context_used)
		self.assertEqual(retrieved_settings.context_limit, self.gemini_client.context_limit)
		self.assertEqual(
				retrieved_settings.raise_error_on_minute_limit,
				self.gemini_client.raise_error_on_minute_limit
		)
	
	def test_client_settings_setter(self):
		model_name = GeminiModels.Gemini_1_5_flash_8b.latest
		new_model_settings = GeminiModelSettings(model_name=model_name)
		new_settings = GeminiClientSettings(
				api_key="new_api_key",
				chats=[MagicMock(spec=GeminiChat)],
				model_settings=new_model_settings
		)
		self.gemini_client.client_settings = new_settings
		
		self.assertEqual(self.gemini_client.api_key, "new_api_key")
		self.assertEqual(self.gemini_client.chats, new_settings.chats)
		self.assertIsInstance(self.gemini_client.model_settings, GeminiModelSettings)
	
	def test_close_chat(self):
		mock_chat1 = MagicMock(spec=GeminiChat)
		mock_chat2 = MagicMock(spec=GeminiAsyncChat)
		self.gemini_client.chats = [mock_chat1, mock_chat2]
		
		self.gemini_client.close_chat(0)
		
		self.assertEqual(self.gemini_client.chats, [mock_chat2])
	
	@patch("PyGPTs.Gemini.client.GeminiClient.add_data")
	def test_generate_content(self, mock_add_data: MagicMock):
		mock_count_tokens_response = MagicMock(total_tokens=8)
		self.mock_client.models.count_tokens.return_value = mock_count_tokens_response
		
		self.mock_client.models.generate_content.return_value = self.mock_gemini_response
		
		response = self.gemini_client.generate_content(message="Test generate")
		
		mock_add_data.assert_called_once_with(8)
		self.mock_client.models.generate_content.assert_called_with(
				model=self.gemini_client.model_name,
				contents="Test generate",
				config=self.gemini_client.generation_config
		)
		self.assertEqual(response, self.mock_gemini_response)
	
	@patch("PyGPTs.Gemini.client.GeminiClient.add_data")
	def test_generate_content_stream(self, mock_add_data: MagicMock):
		mock_count_tokens_response = MagicMock(total_tokens=6)
		self.mock_client.models.count_tokens.return_value = mock_count_tokens_response
		
		mock_stream = MagicMock()
		mock_stream.__iter__.return_value = [self.mock_gemini_response]
		
		self.mock_client.models.generate_content_stream = MagicMock()
		self.mock_client.models.generate_content_stream.return_value = mock_stream
		
		stream_generator = self.gemini_client.generate_content_stream(message="Test stream")
		responses = list(stream_generator)
		
		mock_add_data.assert_called_once_with(6)
		self.mock_client.models.generate_content_stream.assert_called_with(
				model=self.gemini_client.model_name,
				contents="Test stream",
				config=self.gemini_client.generation_config
		)
		self.assertEqual(responses, [self.mock_gemini_response])
	
	def test_get_chats(self):
		mock_chats = [MagicMock(spec=GeminiChat), MagicMock(spec=GeminiAsyncChat)]
		self.gemini_client.chats = mock_chats
		retrieved_chats = self.gemini_client.get_chats()
		self.assertEqual(retrieved_chats, mock_chats)
	
	def test_init(self):
		self.assertEqual(self.gemini_client.api_key, "test_api_key")
		self.assertEqual(self.gemini_client.client, self.mock_client)
		self.assertEqual(self.gemini_client.chats, [])
	
	def test_send_message_async_chat_raises_error(self):
		mock_async_chat = MagicMock(spec=GeminiAsyncChat, is_async=True)
		self.gemini_client.chats = [mock_async_chat]
		
		with self.assertRaises(GeminiChatTypeException) as err:
			self.gemini_client.send_message(message="Should raise error")
		self.assertEqual(str(err.exception), "Chat with index -1 is not synchronous")
	
	def test_send_message_stream_async_chat_raises_error(self):
		mock_async_chat = MagicMock(spec=GeminiAsyncChat, is_async=True)
		self.gemini_client.chats = [mock_async_chat]
		
		with self.assertRaises(GeminiChatTypeException) as err:
			self.gemini_client.send_message_stream(message="Should raise error")
		self.assertEqual(str(err.exception), "Chat with index -1 is not synchronous")
	
	def test_send_message_stream_sync_chat(self):
		mock_stream = MagicMock()
		mock_stream.__iter__.return_value = [self.mock_gemini_response]
		
		mock_sync_chat = MagicMock(spec=GeminiChat, is_async=False)
		mock_sync_chat.send_message_stream = MagicMock()
		mock_sync_chat.send_message_stream.return_value = mock_stream
		self.gemini_client.chats = [mock_sync_chat]
		
		stream_generator = self.gemini_client.send_message_stream(message="Sync stream message")
		responses = list(stream_generator)
		
		mock_sync_chat.send_message_stream.assert_called_with(message="Sync stream message")
		self.assertEqual(responses, [self.mock_gemini_response])
	
	def test_send_message_sync_chat(self):
		mock_sync_chat = MagicMock(spec=GeminiChat)
		mock_sync_chat.is_async = False
		mock_sync_chat.send_message.return_value = self.mock_gemini_response
		self.gemini_client.chats = [mock_sync_chat]
		
		response = self.gemini_client.send_message(message="Sync chat message")
		self.assertEqual(response, self.mock_gemini_response)
		mock_sync_chat.send_message.assert_called_with(message="Sync chat message")
	
	def test_start_async_chat(self):
		self.gemini_client.chats = MagicMock()
		self.gemini_client.chats.append = MagicMock()
		
		self.gemini_client.start_async_chat()
		appended_chat = self.gemini_client.chats.append.call_args[0][0]
		
		self.gemini_client.chats.append.assert_called_once()
		self.assertIsInstance(appended_chat, GeminiAsyncChat)
	
	def test_start_chat(self):
		self.gemini_client.chats = MagicMock()
		self.gemini_client.chats.append = MagicMock()
		
		self.gemini_client.start_chat()
		appended_chat = self.gemini_client.chats.append.call_args[0][0]
		
		self.gemini_client.chats.append.assert_called_once()
		self.assertIsInstance(appended_chat, GeminiChat)


class TestGeminiClientSettings(TestCase):
	def test_init_custom(self):
		model_name = GeminiModels.Gemini_1_5_flash_8b.latest
		model_settings = GeminiModelSettings(model_name=model_name)
		chats = [MagicMock(spec=GeminiChat), MagicMock(spec=GeminiAsyncChat)]
		settings = GeminiClientSettings(api_key="custom_key", chats=chats, model_settings=model_settings)
		
		self.assertEqual(settings.api_key, "custom_key")
		self.assertEqual(settings.chats, chats)
		self.assertEqual(settings.model_settings, model_settings)
	
	def test_init_default(self):
		settings = GeminiClientSettings(api_key="test_key")
		
		self.assertEqual(settings.api_key, "test_key")
		self.assertEqual(settings.chats, [])
		self.assertIsInstance(settings.model_settings, GeminiModelSettings)


def client_test_suite() -> TestSuite:
	suite = TestSuite()
	test_loader = TestLoader()
	
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiClientSettings))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiClient))
	
	return suite


if __name__ == "__main__":
	runner = TextTestRunner()
	runner.run(client_test_suite())
