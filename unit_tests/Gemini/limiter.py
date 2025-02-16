import time
import pytz
from unittest.mock import patch
from datetime import datetime, timedelta
from unittest.async_case import IsolatedAsyncioTestCase
from unittest import (
	TestCase,
	TestLoader,
	TestSuite,
	TextTestRunner
)
from PyGPTs.Gemini.limiter import (
	GeminiLimiter,
	GeminiLimiterSettings
)
from PyGPTs.Gemini.errors import (
	GeminiContextLimitException,
	GeminiDayLimitException,
	GeminiMinuteLimitException
)


class TestGeminiLimiter(IsolatedAsyncioTestCase):
	def setUp(self):
		self.settings = GeminiLimiterSettings(
				request_per_day_limit=10,
				request_per_minute_limit=2,
				tokens_per_minute_limit=100,
				context_limit=1000
		)
		self.limiter = GeminiLimiter(self.settings)
	
	def test_add_context_limit_exceeded(self):
		with self.assertRaises(GeminiContextLimitException):
			self.limiter.add_context(1001)
	
	def test_add_context_within_limit(self):
		self.limiter.add_context(500)
		
		self.assertEqual(self.limiter.context_used, 500)
	
	def test_add_data_context_limit_exceeded(self):
		self.limiter.context_used = 951
		
		with self.assertRaises(GeminiContextLimitException):
			self.limiter.add_data(50)
	
	def test_add_data_day_limit_exceeded(self):
		self.limiter.request_per_day_used = 9
		
		with self.assertRaises(GeminiDayLimitException):
			self.limiter.add_data(50)
	
	def test_add_data_within_limits(self):
		self.limiter.add_data(50)
		
		self.assertEqual(self.limiter.request_per_day_used, 1)
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, 50)
		self.assertEqual(self.limiter.context_used, 50)
	
	async def test_async_add_data_context_limit_exceeded(self):
		self.limiter.context_used = 951
		
		with self.assertRaises(GeminiContextLimitException):
			await self.limiter.async_add_data(50)
	
	async def test_async_add_data_day_limit_exceeded(self):
		self.limiter.request_per_day_used = 10
		
		with self.assertRaises(GeminiDayLimitException):
			await self.limiter.async_add_data(50)
	
	async def test_async_add_data_within_limits(self):
		await self.limiter.async_add_data(50)
		
		self.assertEqual(self.limiter.request_per_day_used, 1)
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, 50)
		self.assertEqual(self.limiter.context_used, 50)
	
	async def test_async_check_limits_context_limit_exceeded(self):
		self.limiter.context_used = 1000
		
		with self.assertRaises(GeminiContextLimitException):
			await self.limiter.async_check_limits(10)
	
	async def test_async_check_limits_day_exceeded_restarts_counters(self):
		with patch("PyGPTs.Gemini.limiter.datetime") as mock_datetime:
			mock_datetime.now.return_value = datetime(
					self.limiter.limit_day.year + 1,
					self.limiter.limit_day.month,
					self.limiter.limit_day.day,
					tzinfo=pytz.timezone("America/New_York")
			)
			await self.limiter.async_check_limits(10)
		
		self.assertEqual(self.limiter.request_per_day_used, 1)
	
	async def test_async_check_limits_day_limit_exceeded(self):
		self.limiter.request_per_day_used = 10
		
		with self.assertRaises(GeminiDayLimitException):
			await self.limiter.async_check_limits(10)
	
	async def test_async_check_limits_minute_exceeded_restarts_counters(self):
		with patch("PyGPTs.Gemini.limiter.time.time") as mock_time:
			mock_time.return_value = self.limiter.start_time + 60
			await self.limiter.async_check_limits(20)
		
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, 20)
	
	async def test_async_check_limits_minute_limit_exceeded_pauses_execution(self):
		self.limiter.raise_error_on_minute_limit = False
		self.limiter.request_per_minute_used = 2
		
		start_time = time.time()
		with patch("PyGPTs.Gemini.limiter.time.time") as mock_time:
			mock_time.return_value = start_time + 59.89
			await self.limiter.async_check_limits(10)
		end_time = time.time()
		
		self.assertGreaterEqual(end_time - start_time, 0.1)
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, 10)
	
	async def test_async_check_limits_minute_limit_exceeded_raises_error(self):
		self.limiter.request_per_minute_used = 2
		
		with self.assertRaises(GeminiMinuteLimitException):
			await self.limiter.async_check_limits(10)
	
	async def test_async_check_limits_within_limits(self):
		await self.limiter.async_check_limits(10)
		
		self.assertEqual(self.limiter.request_per_day_used, 0)
		self.assertEqual(self.limiter.request_per_minute_used, 0)
		self.assertEqual(self.limiter.tokens_per_minute_used, 0)
		self.assertEqual(self.limiter.context_used, 0)
	
	def test_check_limits_context_limit_exceeded(self):
		self.limiter.context_used = 1000
		
		with self.assertRaises(GeminiContextLimitException):
			self.limiter.check_limits(10)
	
	def test_check_limits_day_exceeded_restarts_counters(self):
		with patch("PyGPTs.Gemini.limiter.datetime") as mock_datetime:
			mock_datetime.now.return_value = datetime(
					self.limiter.limit_day.year + 1,
					self.limiter.limit_day.month,
					self.limiter.limit_day.day,
					tzinfo=pytz.timezone("America/New_York")
			)
			self.limiter.check_limits(10)
		
		self.assertEqual(self.limiter.request_per_day_used, 1)
	
	def test_check_limits_day_limit_exceeded(self):
		self.limiter.request_per_day_used = 10
		
		with self.assertRaises(GeminiDayLimitException):
			self.limiter.check_limits(10)
	
	def test_check_limits_minute_exceeded_restarts_counters(self):
		with patch("PyGPTs.Gemini.limiter.time.time") as mock_time:
			mock_time.return_value = self.limiter.start_time + 60
			self.limiter.check_limits(20)
		
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, 20)
	
	def test_check_limits_minute_limit_exceeded_pauses_execution(self):
		self.limiter.raise_error_on_minute_limit = False
		self.limiter.request_per_minute_used = 2
		
		start_time = time.time()
		
		with patch("PyGPTs.Gemini.limiter.time.time") as mock_time:
			mock_time.return_value = start_time + 59.89
			self.limiter.check_limits(10)
		
		end_time = time.time()
		
		self.assertGreaterEqual(end_time - start_time, 0.1)
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, 10)
	
	def test_check_limits_minute_limit_exceeded_raises_error(self):
		self.limiter.request_per_minute_used = 2
		
		with self.assertRaises(GeminiMinuteLimitException):
			self.limiter.check_limits(10)
	
	def test_check_limits_within_limits(self):
		self.limiter.check_limits(10)
		
		self.assertEqual(self.limiter.request_per_day_used, 0)
		self.assertEqual(self.limiter.request_per_minute_used, 0)
		self.assertEqual(self.limiter.tokens_per_minute_used, 0)
		self.assertEqual(self.limiter.context_used, 0)
	
	def test_clear_context(self):
		self.limiter.context_used = 700
		self.limiter.clear_context()
		
		self.assertEqual(self.limiter.context_used, 0)
	
	def test_close_day_limit(self):
		self.limiter.close_day_limit()
		
		self.assertEqual(self.limiter.request_per_day_used, self.limiter.request_per_day_limit)
		self.assertFalse(self.limiter.has_day_limits)
	
	def test_close_minute_limit(self):
		self.limiter.close_minute_limit()
		
		self.assertEqual(
				self.limiter.request_per_minute_used,
				self.limiter.request_per_minute_limit
		)
		self.assertEqual(
				self.limiter.tokens_per_minute_used,
				self.limiter.tokens_per_minute_limit
		)
		self.assertFalse(self.limiter.has_minute_limits)
	
	def test_context_usage(self):
		self.limiter.context_used = 600
		usage = self.limiter.context_usage
		
		self.assertEqual(usage["context_used"], 600)
		self.assertEqual(usage["context_limit"], 1000)
	
	def test_day_usage(self):
		self.limiter.request_per_day_used = 5
		usage = self.limiter.day_usage
		
		self.assertEqual(usage["used_requests"], 5)
		self.assertEqual(usage["requests_limit"], 10)
		self.assertEqual(usage["date"], self.limiter.limit_day)
	
	def test_decrease_context_invalid(self):
		self.limiter.context_used = 100
		
		with self.assertRaises(ValueError):
			self.limiter.decrease_context(200)
	
	def test_decrease_context_valid(self):
		self.limiter.context_used = 500
		self.limiter.decrease_context(200)
		
		self.assertEqual(self.limiter.context_used, 300)
	
	def test_has_context_limit_exceeded(self):
		self.limiter.context_used = 1000
		
		self.assertFalse(self.limiter.has_context)
	
	def test_has_context_within_limit(self):
		self.assertTrue(self.limiter.has_context)
	
	def test_has_day_limits_limit_exceeded(self):
		self.limiter.request_per_day_used = 10
		
		self.assertFalse(self.limiter.has_day_limits)
	
	def test_has_day_limits_within_limit(self):
		self.assertTrue(self.limiter.has_day_limits)
	
	def test_has_minute_limits_both_limits_exceeded(self):
		self.limiter.request_per_minute_used = 2
		self.limiter.tokens_per_minute_used = 100
		
		self.assertFalse(self.limiter.has_minute_limits)
	
	def test_has_minute_limits_request_limit_exceeded(self):
		self.limiter.request_per_minute_used = 2
		
		self.assertFalse(self.limiter.has_minute_limits)
	
	def test_has_minute_limits_token_limit_exceeded(self):
		self.limiter.tokens_per_minute_used = 100
		
		self.assertFalse(self.limiter.has_minute_limits)
	
	def test_has_minute_limits_within_limit(self):
		self.assertTrue(self.limiter.has_minute_limits)
	
	def test_init(self):
		self.assertEqual(self.limiter.limit_day, self.settings.limit_day)
		self.assertEqual(self.limiter.request_per_day_used, self.settings.request_per_day_used)
		self.assertEqual(
				self.limiter.request_per_day_limit,
				self.settings.request_per_day_limit
		)
		self.assertEqual(
				self.limiter.request_per_minute_limit,
				self.settings.request_per_minute_limit
		)
		self.assertEqual(
				self.limiter.tokens_per_minute_limit,
				self.settings.tokens_per_minute_limit
		)
		self.assertEqual(self.limiter.context_used, self.settings.context_used)
		self.assertEqual(self.limiter.context_limit, self.settings.context_limit)
		self.assertEqual(
				self.limiter.raise_error_on_minute_limit,
				self.settings.raise_error_on_minute_limit
		)
		self.assertEqual(self.limiter.request_per_minute_used, 0)
		self.assertEqual(self.limiter.tokens_per_minute_used, 0)
		self.assertIsInstance(self.limiter.start_time, float)
	
	def test_limit_day_exceeded_false(self):
		self.assertFalse(self.limiter.limit_day_exceeded)
	
	def test_limit_day_exceeded_true(self):
		with patch("PyGPTs.Gemini.limiter.datetime") as mock_datetime:
			mock_datetime.now.return_value = datetime(
					self.limiter.limit_day.year + 1,
					self.limiter.limit_day.month,
					self.limiter.limit_day.day,
					tzinfo=pytz.timezone("America/New_York")
			)
			self.assertTrue(self.limiter.limit_day_exceeded)
	
	def test_limiter_settings_getter(self):
		settings = self.limiter.limiter_settings
		
		self.assertIsInstance(settings, GeminiLimiterSettings)
		self.assertEqual(settings.limit_day, self.limiter.limit_day)
		self.assertEqual(settings.request_per_day_limit, self.limiter.request_per_day_limit)
	
	def test_limiter_settings_setter(self):
		new_settings = GeminiLimiterSettings(
				request_per_day_limit=20,
				request_per_minute_limit=5,
				tokens_per_minute_limit=200,
				context_limit=1500,
				raise_error_on_minute_limit=False
		)
		self.limiter.limiter_settings = new_settings
		
		self.assertEqual(self.limiter.request_per_day_limit, 20)
		self.assertEqual(self.limiter.request_per_minute_limit, 5)
		self.assertEqual(self.limiter.tokens_per_minute_limit, 200)
		self.assertEqual(self.limiter.context_limit, 1500)
		self.assertFalse(self.limiter.raise_error_on_minute_limit)
		self.assertEqual(self.limiter.request_per_minute_used, 0)
		self.assertEqual(self.limiter.tokens_per_minute_used, 0)
	
	def test_minute_exceeded_false(self):
		self.assertFalse(self.limiter.minute_exceeded)
	
	def test_minute_exceeded_true(self):
		with patch("PyGPTs.Gemini.limiter.time.time") as mock_time:
			mock_time.return_value = self.limiter.start_time + 60
			self.assertTrue(self.limiter.minute_exceeded)
	
	def test_minute_usage(self):
		self.limiter.request_per_minute_used = 1
		self.limiter.tokens_per_minute_used = 60
		usage = self.limiter.minute_usage
		
		self.assertEqual(usage["used_requests"], 1)
		self.assertEqual(usage["requests_limit"], 2)
		self.assertEqual(usage["used_tokens"], 60)
		self.assertEqual(usage["tokens_limit"], 100)
	
	def test_restart_day_counters(self):
		self.limiter.request_per_day_used = 10
		initial_limit_day = self.limiter.limit_day - timedelta(days=1)
		
		self.limiter.restart_day_counters()
		
		self.assertEqual(self.limiter.request_per_day_used, 1)
		self.assertNotEqual(self.limiter.limit_day, initial_limit_day)
	
	def test_restart_minute_counters(self):
		self.limiter.request_per_minute_used = 2
		self.limiter.tokens_per_minute_used = 50
		initial_start_time = self.limiter.start_time
		time.sleep(0.01)
		self.limiter.restart_minute_counters(20)
		
		self.assertEqual(self.limiter.request_per_minute_used, 1)
		self.assertEqual(self.limiter.tokens_per_minute_used, 20)
		self.assertGreater(self.limiter.start_time, initial_start_time)


def limiter_test_suite() -> TestSuite:
	suite = TestSuite()
	test_loader = TestLoader()
	
	suite.addTest(test_loader.loadTestsFromTestCase(TestGeminiLimiter))
	
	return suite


class TestGeminiLimiterSettings(TestCase):
	def test_init_custom(self):
		limit_day = datetime(2023, 1, 1, tzinfo=pytz.utc)
		settings = GeminiLimiterSettings(
				limit_day=limit_day,
				request_per_day_used=10,
				request_per_day_limit=100,
				request_per_minute_limit=5,
				tokens_per_minute_limit=1000,
				context_used=500,
				context_limit=2000,
				raise_error_on_minute_limit=False
		)
		
		self.assertIsInstance(settings.limit_day, datetime)
		self.assertEqual(settings.limit_day.year, 2022)
		self.assertEqual(settings.limit_day.month, 12)
		self.assertEqual(settings.limit_day.day, 31)
		self.assertIsInstance(settings.limit_day.tzinfo, pytz.tzinfo.DstTzInfo)
		self.assertEqual(str(settings.limit_day.tzinfo), "America/New_York")
		self.assertEqual(settings.request_per_day_used, 10)
		self.assertEqual(settings.request_per_day_limit, 100)
		self.assertEqual(settings.request_per_minute_limit, 5)
		self.assertEqual(settings.tokens_per_minute_limit, 1000)
		self.assertEqual(settings.context_used, 500)
		self.assertEqual(settings.context_limit, 2000)
		self.assertFalse(settings.raise_error_on_minute_limit)
	
	def test_init_default(self):
		settings = GeminiLimiterSettings()
		
		self.assertIsInstance(settings.limit_day, datetime)
		self.assertIsInstance(settings.limit_day.tzinfo, pytz.tzinfo.DstTzInfo)
		self.assertEqual(str(settings.limit_day.tzinfo), "America/New_York")
		self.assertEqual(settings.limit_day.hour, 0)
		self.assertEqual(settings.limit_day.minute, 0)
		self.assertEqual(settings.limit_day.second, 0)
		self.assertEqual(settings.limit_day.microsecond, 0)
		self.assertEqual(settings.request_per_day_used, 0)
		self.assertIsNone(settings.request_per_day_limit)
		self.assertIsNone(settings.request_per_minute_limit)
		self.assertIsNone(settings.tokens_per_minute_limit)
		self.assertEqual(settings.context_used, 0)
		self.assertIsNone(settings.context_limit)
		self.assertTrue(settings.raise_error_on_minute_limit)
	
	def test_to_dict(self):
		limit_day = datetime(2023, 1, 1, tzinfo=pytz.utc)
		settings = GeminiLimiterSettings(
				limit_day=limit_day,
				request_per_day_used=10,
				request_per_day_limit=100,
				request_per_minute_limit=5,
				tokens_per_minute_limit=1000,
				context_used=500,
				context_limit=2000,
				raise_error_on_minute_limit=False
		)
		settings_dict = settings.to_dict()
		
		self.assertEqual(settings_dict["limit_day"], settings.limit_day)
		self.assertEqual(settings_dict["request_per_day_used"], 10)
		self.assertEqual(settings_dict["request_per_day_limit"], 100)
		self.assertEqual(settings_dict["request_per_minute_limit"], 5)
		self.assertEqual(settings_dict["tokens_per_minute_limit"], 1000)
		self.assertEqual(settings_dict["context_used"], 500)
		self.assertEqual(settings_dict["context_limit"], 2000)
		self.assertEqual(settings_dict["raise_error_on_minute_limit"], False)


if __name__ == "__main__":
	runner = TextTestRunner()
	runner.run(limiter_test_suite())
