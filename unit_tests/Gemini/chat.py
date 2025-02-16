from google.genai import Client
from google.genai.chats import Chat
from PyGPTs.Gemini.data import GeminiModels
from PyGPTs.Gemini.types import GeminiContentDict
from PyGPTs.Gemini.model import GeminiModelSettings
from google.genai.types import GenerateContentResponse
from unittest.mock import (
	AsyncMock,
	MagicMock,
	patch
)
from unittest import (
	IsolatedAsyncioTestCase,
	TestCase,
	TestLoader,
	TestSuite,
	TextTestRunner
)
from PyGPTs.Gemini.chat import (
	BaseGeminiChat,
	GeminiAsyncChat,
	GeminiAsyncChatSettings,
	GeminiBaseChatSettings,
	GeminiChat,
	GeminiChatSettings
)


class TestGeminiChat(TestCase):
	def setUp(self):
		self.mock_client = MagicMock(spec=Client)
		self.mock_chat_settings = GeminiChatSettings(client=self.mock_client)
		self.gemini_chat = GeminiChat(self.mock_chat_settings)
		self.mock_gemini_response = MagicMock(spec=GenerateContentResponse)
	
	def test_create_chat(self):
		self.gemini_chat.client.chats.create = MagicMock()
		
		self.gemini_chat.client.models.count_tokens = MagicMock()
		self.gemini_chat.client.models.count_tokens.return_value = MagicMock(total_tokens=3)
		
		model_name = GeminiModels.Gemini_1_5_flash_8b.latest
		model_settings = GeminiModelSettings(model_name=model_name)
		history = [GeminiContentDict(role="user", parts=["Hello"])]
		self.gemini_chat.create_chat(model_settings=model_settings, history=history)
		
		self.gemini_chat.client.chats.create.assert_called_with(
				model=model_name,
				config=model_settings.generation_config,
				history=history
		)
		self.gemini_chat.client.models.count_tokens.assert_called_with(
				model=model_name,
				contents=history,
				config=self.gemini_chat.count_tokens_config
		)
		self.assertEqual(self.gemini_chat.context_used, 3)
	
	@patch("PyGPTs.Gemini.chat.extract_token_count_from_gemini_response")
	@patch("PyGPTs.Gemini.chat.GeminiChat.add_data")
	@patch("PyGPTs.Gemini.chat.GeminiChat.add_context")
	def test_send_message(
			self,
			mock_add_context: MagicMock,
			mock_add_data: MagicMock,
			mock_extract_tokens: MagicMock
	):
		self.gemini_chat.client.models.count_tokens = MagicMock()
		self.gemini_chat.client.models.count_tokens.return_value = MagicMock(total_tokens=7)
		
		self.gemini_chat.chat.send_message = MagicMock()
		self.gemini_chat.chat.send_message.return_value = self.mock_gemini_response
		
		mock_extract_tokens.return_value = 2
		
		response = self.gemini_chat.send_message("Test message")
		
		mock_add_data.assert_called_once_with(7)
		self.gemini_chat.chat.send_message.assert_called_with(message="Test message")
		mock_add_context.assert_called_with(2)
		self.assertEqual(response, self.mock_gemini_response)
	
	@patch("PyGPTs.Gemini.chat.extract_token_count_from_gemini_response")
	@patch("PyGPTs.Gemini.chat.GeminiChat.add_data")
	@patch("PyGPTs.Gemini.chat.GeminiChat.add_context")
	def test_send_message_stream(
			self,
			mock_add_context: MagicMock,
			mock_add_data: MagicMock,
			mock_extract_tokens: MagicMock
	):
		self.gemini_chat.client.models.count_tokens = MagicMock()
		self.gemini_chat.client.models.count_tokens.return_value = MagicMock(total_tokens=7)
		
		mock_stream = MagicMock()
		mock_stream.__iter__.return_value = [self.mock_gemini_response]
		
		self.gemini_chat.chat.send_message_stream = MagicMock()
		self.gemini_chat.chat.send_message_stream.return_value = mock_stream
		
		mock_extract_tokens.return_value = 3
		
		stream_generator = self.gemini_chat.send_message_stream("Stream message")
		responses = list(stream_generator)
		
		mock_add_data.assert_called_once_with(7)
		self.gemini_chat.chat.send_message_stream.assert_called_with(message="Stream message")
		mock_add_context.assert_called_with(3)
		self.assertEqual(responses, [self.mock_gemini_response])


class TestGeminiAsyncChat(IsolatedAsyncioTestCase):
	def setUp(self):
		self.mock_client = MagicMock(spec=Client)
		self.mock_async_chat_settings = GeminiAsyncChatSettings(client=self.mock_client)
		self.gemini_async_chat = GeminiAsyncChat(self.mock_async_chat_settings)
		self.mock_gemini_response = MagicMock(spec=GenerateContentResponse)
	
	async def test_create_chat(self):
		self.gemini_async_chat.client.aio.chats.create = MagicMock()
		
		self.gemini_async_chat.client.models.count_tokens = MagicMock()
		self.gemini_async_chat.client.models.count_tokens.return_value = MagicMock(total_tokens=2)
		
		model_name = GeminiModels.Gemini_1_5_flash_8b.latest
		model_settings = GeminiModelSettings(model_name=model_name)
		history = [GeminiContentDict(role="user", parts=["Hello"])]
		self.gemini_async_chat.create_chat(model_settings=model_settings, history=history)
		
		self.gemini_async_chat.client.aio.chats.create.assert_called_with(
				model=model_name,
				config=model_settings.generation_config,
				history=history
		)
		self.gemini_async_chat.client.models.count_tokens.assert_called_with(
				model=model_name,
				contents=history,
				config=self.gemini_async_chat.count_tokens_config
		)
		self.assertEqual(self.gemini_async_chat.context_used, 2)
	
	@patch("PyGPTs.Gemini.chat.extract_token_count_from_gemini_response")
	@patch("PyGPTs.Gemini.chat.GeminiAsyncChat.async_add_data")
	@patch("PyGPTs.Gemini.chat.GeminiAsyncChat.add_context")
	async def test_send_message(
			self,
			mock_add_context: MagicMock,
			mock_async_add_data: MagicMock,
			mock_extract_tokens: MagicMock
	):
		self.gemini_async_chat.client.models.count_tokens = MagicMock()
		self.gemini_async_chat.client.models.count_tokens.return_value = MagicMock(total_tokens=6)
		
		self.gemini_async_chat.chat.send_message = AsyncMock()
		self.gemini_async_chat.chat.send_message.return_value = self.mock_gemini_response
		
		mock_extract_tokens.return_value = 1
		
		response = await self.gemini_async_chat.send_message("Async test message")
		
		mock_async_add_data.assert_called()
		self.gemini_async_chat.chat.send_message.assert_called_with(message="Async test message")
		mock_add_context.assert_called_with(1)
		self.assertEqual(response, self.mock_gemini_response)
	
	@patch("PyGPTs.Gemini.chat.extract_token_count_from_gemini_response")
	@patch("PyGPTs.Gemini.chat.GeminiAsyncChat.async_add_data")
	@patch("PyGPTs.Gemini.chat.GeminiAsyncChat.add_context")
	async def test_send_message_stream(
			self,
			mock_add_context: MagicMock,
			mock_async_add_data: MagicMock,
			mock_extract_tokens: MagicMock
	):
		self.gemini_async_chat.client.models.count_tokens = MagicMock()
		self.gemini_async_chat.client.models.count_tokens.return_value = MagicMock(total_tokens=5)
		
		mock_stream = AsyncMock()
		mock_stream.__aiter__.return_value = [self.mock_gemini_response]
		
		self.gemini_async_chat.chat.send_message_stream = AsyncMock()
		self.gemini_async_chat.chat.send_message_stream.return_value = mock_stream
		
		mock_extract_tokens.return_value = 4
		
		stream_generator = self.gemini_async_chat.send_message_stream("Async stream message")
		responses = [response async for response in stream_generator]
		
		mock_async_add_data.assert_called()
		self.gemini_async_chat.chat.send_message_stream.assert_called_with(message="Async stream message")
		mock_add_context.assert_called_with(4)
		self.assertEqual(responses, [self.mock_gemini_response])


class TestBaseGeminiChat(TestCase):
	def setUp(self):
		self.mock_client = MagicMock(spec=Client)
		self.chat_settings = GeminiBaseChatSettings(client=self.mock_client)
		self.base_chat = BaseGeminiChat(self.chat_settings)
	
	def test_chat_settings_property_getter(self):
		retrieved_settings = self.base_chat.chat_settings
		
		self.assertIsInstance(retrieved_settings, GeminiBaseChatSettings)
		self.assertEqual(retrieved_settings.client, self.base_chat.client)
		self.assertEqual(retrieved_settings.is_async, self.base_chat.is_async)
		self.assertEqual(retrieved_settings.history, self.base_chat.history)
		self.assertEqual(retrieved_settings.model_name, self.base_chat.model_name)
		self.assertEqual(retrieved_settings.generation_config, self.base_chat.generation_config)
		self.assertEqual(
				retrieved_settings.count_tokens_config,
				self.base_chat.count_tokens_config
		)
		self.assertEqual(
				retrieved_settings.request_per_day_used,
				self.base_chat.request_per_day_used
		)
		self.assertEqual(
				retrieved_settings.request_per_day_limit,
				self.base_chat.request_per_day_limit
		)
		self.assertEqual(
				retrieved_settings.request_per_minute_limit,
				self.base_chat.request_per_minute_limit
		)
		self.assertEqual(
				retrieved_settings.tokens_per_minute_limit,
				self.base_chat.tokens_per_minute_limit
		)
		self.assertEqual(retrieved_settings.context_used, self.base_chat.context_used)
		self.assertEqual(retrieved_settings.context_limit, self.base_chat.context_limit)
		self.assertEqual(
				retrieved_settings.raise_error_on_minute_limit,
				self.base_chat.raise_error_on_minute_limit
		)
	
	@patch("PyGPTs.Gemini.chat.BaseGeminiChat.history")
	@patch("PyGPTs.Gemini.chat.BaseGeminiChat.create_chat")
	def test_chat_settings_property_setter(self, mock_create_chat: MagicMock, mock_history: MagicMock):
		new_model_settings = GeminiModelSettings(model_name=GeminiModels.Gemini_1_5_flash_8b.latest)
		self.base_chat.chat_settings = new_model_settings
		
		mock_create_chat.assert_called_with(model_settings=new_model_settings, history=self.base_chat.history)
	
	@patch("PyGPTs.Gemini.chat.BaseGeminiChat.model_settings")
	@patch("PyGPTs.Gemini.chat.BaseGeminiChat.clear_context")
	@patch("PyGPTs.Gemini.chat.BaseGeminiChat.create_chat")
	def test_clear_chat_history(
			self,
			mock_create_chat: MagicMock,
			mock_clear_context: MagicMock,
			mock_model_settings: MagicMock
	):
		self.base_chat.clear_chat_history()
		
		mock_create_chat.assert_called_with(model_settings=self.base_chat.model_settings, history=[])
		mock_clear_context.assert_called_once()
	
	@patch("PyGPTs.Gemini.chat.BaseGeminiChat.model_settings")
	def test_create_chat(self, mock_model_settings: MagicMock):
		self.mock_client.models.count_tokens = MagicMock()
		self.mock_client.models.count_tokens.return_value = MagicMock(total_tokens=5)
		
		model_settings = GeminiModelSettings(model_name=GeminiModels.Gemini_1_5_flash_8b.latest)
		history = [GeminiContentDict(role="user", parts=["Hi"])]
		chat = self.base_chat.create_chat(model_settings=model_settings, history=history)
		
		self.assertIsNone(chat)
		self.assertEqual(self.base_chat.model_settings, model_settings)
		self.mock_client.models.count_tokens.assert_called_with(
				model=GeminiModels.Gemini_1_5_flash_8b.latest,
				contents=history,
				config=self.base_chat.count_tokens_config
		)
		self.assertEqual(self.base_chat.context_used, 5)
	
	def test_history_property(self):
		self.base_chat.chat = MagicMock()
		self.base_chat.chat._curated_history = [GeminiContentDict(role="user", parts=["Hello"])]
		
		self.assertEqual(self.base_chat.history, self.base_chat.chat._curated_history)
	
	def test_init(self):
		self.assertEqual(self.base_chat.client, self.chat_settings.client)
		self.assertEqual(self.base_chat.is_async, self.chat_settings.is_async)
		self.assertIsNone(self.base_chat.chat)
	
	@patch("PyGPTs.Gemini.chat.BaseGeminiChat.model_settings")
	@patch("PyGPTs.Gemini.chat.BaseGeminiChat.create_chat")
	def test_reset_history(self, mock_create_chat: MagicMock, mock_model_settings: MagicMock):
		self.base_chat.client.models.count_tokens = MagicMock()
		self.base_chat.client.models.count_tokens.return_value = MagicMock(total_tokens=8)
		
		new_history = [GeminiContentDict(role="user", parts=["New history"])]
		self.base_chat.reset_history(new_history)
		
		mock_create_chat.assert_called_with(model_settings=self.base_chat.model_settings, history=new_history)
		self.base_chat.client.models.count_tokens.assert_called_with(
				model=self.base_chat.model_name,
				contents=new_history,
				config=self.base_chat.count_tokens_config
		)
		self.assertEqual(self.base_chat.context_used, 8)
	
	@patch("PyGPTs.Gemini.chat.BaseGeminiChat.reset_history")
	def test_slice_history(self, mock_reset_history: MagicMock):
		current_history = [
			GeminiContentDict(role="user", parts=["Msg 1"]),
			GeminiContentDict(role="model", parts=["Resp 1"]),
			GeminiContentDict(role="user", parts=["Msg 2"])
		]
		self.base_chat.chat = MagicMock(spec=Chat)
		self.base_chat.chat._curated_history = current_history
		
		self.base_chat.slice_history(start=1, end=3)
		expected_sliced_history = current_history[1:3]
		
		mock_reset_history.assert_called_with(expected_sliced_history)


class TestGeminiAsyncChatSettings(TestCase):
	def setUp(self):
		self.mock_client = MagicMock(spec=Client)
		self.model_settings = GeminiModelSettings()
	
	def test_init_custom(self):
		custom_history = [GeminiContentDict(role="user", parts=["Hello"])]
		settings = GeminiAsyncChatSettings(
				client=self.mock_client,
				history=custom_history,
				model_settings=self.model_settings
		)
		
		self.assertEqual(settings.client, self.mock_client)
		self.assertTrue(settings.is_async)
		self.assertEqual(settings.history, custom_history)
		self.assertEqual(settings.model_settings, self.model_settings)
	
	def test_init_default(self):
		settings = GeminiAsyncChatSettings(client=self.mock_client)
		
		self.assertEqual(settings.client, self.mock_client)
		self.assertTrue(settings.is_async)
		self.assertIsNone(settings.history)
		self.assertIsInstance(settings.model_settings, GeminiModelSettings)


class TestGeminiChatSettings(TestCase):
	def setUp(self):
		self.mock_client = MagicMock(spec=Client)
		self.mock_model_settings = GeminiModelSettings()
	
	def test_init_custom(self):
		custom_history = [GeminiContentDict(role="user", parts=["Hello"])]
		settings = GeminiChatSettings(
				client=self.mock_client,
				history=custom_history,
				model_settings=self.mock_model_settings
		)
		
		self.assertEqual(settings.client, self.mock_client)
		self.assertFalse(settings.is_async)
		self.assertEqual(settings.history, custom_history)
		self.assertEqual(settings.model_settings, self.mock_model_settings)
	
	def test_init_default(self):
		settings = GeminiChatSettings(client=self.mock_client)
		
		self.assertEqual(settings.client, self.mock_client)
		self.assertFalse(settings.is_async)
		self.assertIsNone(settings.history)
		self.assertIsInstance(settings.model_settings, GeminiModelSettings)


class TestGeminiBaseChatSettings(TestCase):
	def setUp(self):
		self.mock_client = MagicMock(spec=Client)
		self.model_settings = GeminiModelSettings()
	
	def test_init_custom(self):
		custom_history = [GeminiContentDict(role="user", parts=["Hello"])]
		settings = GeminiBaseChatSettings(
				client=self.mock_client,
				is_async=True,
				history=custom_history,
				model_settings=self.model_settings
		)
		
		self.assertEqual(settings.client, self.mock_client)
		self.assertTrue(settings.is_async)
		self.assertEqual(settings.history, custom_history)
		self.assertEqual(settings.model_settings, self.model_settings)
	
	def test_init_default(self):
		settings = GeminiBaseChatSettings(client=self.mock_client)
		
		self.assertEqual(settings.client, self.mock_client)
		self.assertIsNone(settings.is_async)
		self.assertIsNone(settings.history)
		self.assertIsInstance(settings.model_settings, GeminiModelSettings)
	
	def test_init_history_context_used(self):
		self.mock_client.models.count_tokens = MagicMock()
		self.mock_client.models.count_tokens.return_value = MagicMock(total_tokens=10)
		
		history = [GeminiContentDict(role="user", parts=["Test message"])]
		settings = GeminiBaseChatSettings(
				client=self.mock_client,
				history=history,
				model_settings=self.model_settings
		)
		
		self.mock_client.models.count_tokens.assert_called_once()
		self.assertEqual(settings.limiter_settings.context_used, 10)
	
	def test_to_dict(self):
		settings = GeminiBaseChatSettings(client=self.mock_client, is_async=True)
		settings_dict = settings.to_dict()
		
		self.assertEqual(settings_dict["client"], self.mock_client)
		self.assertEqual(settings_dict["is_async"], True)
		self.assertIsNone(settings_dict["history"])
		self.assertIsInstance(settings_dict["model_settings"], GeminiModelSettings)


def chat_test_suite() -> TestSuite:
	suite = TestSuite()
	test_loader = TestLoader()
	
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiBaseChatSettings))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiChatSettings))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiAsyncChatSettings))
	suite.addTest(test_loader.loadTestsFromTestCase(TestBaseGeminiChat))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiAsyncChat))
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiChat))
	
	return suite


if __name__ == "__main__":
	runner = TextTestRunner()
	runner.run(chat_test_suite())
