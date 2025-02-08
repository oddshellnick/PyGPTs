import os
import torch
import typing
from PyGPTs.HuggingFace.base_objects import ObjectTypeKwargs
from PyVarTools.python_instances_tools import get_class_fields
from transformers import (
	AutoTokenizer,
	PreTrainedTokenizer,
	PretrainedConfig
)


class TokenizerTypeKwargs(ObjectTypeKwargs):
	"""
	Keyword arguments for tokenizer instantiation. Extends ObjectTypeKwargs.
	"""
	
	def __init__(self, **kwargs):
		"""Initializes TokenizerTypeKwargs with given keyword arguments."""
		super().__init__(**kwargs)


class HF_TokenizerSettings:
	"""
	Stores settings for initializing a Hugging Face tokenizer.

	Attributes:
		pretrained_model_name_or_path (typing.Union[str, PathLike]): The path or name of the pretrained model.
		cache_dir (typing.Optional[typing.Union[str, PathLike]]): Path to a directory in which a downloaded pretrained model configuration
			 should be cached if the standard cache should not be used. Defaults to None.
		config (typing.Optional[PretrainedConfig]): An instance of a configuration object to use instead of loading the
			 configuration from the pretrained model configuration file. Defaults to None.
		force_download (typing.Optional[bool]): Whether to force the (re-)download the model weights and configuration files and
			 override the cached versions if they exist. Defaults to None.
		proxies (typing.Optional[dict[str, str]]): A dictionary of proxy servers to use by protocol or endpoint, e.g.,
			 `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.`. The proxies are used on each request. Defaults to None.
		sub_folder (typing.Optional[str]): In case the relevant files are located inside a sub directory of the model repo on
			 huggingface.co, you can specify the folder name here. Defaults to None.
		token (typing.Optional[str]): An authentication token for private repositories on huggingface.co. Defaults to None.
		tokenizer_type (typing.Optional[str]): The tokenizer type. Defaults to None.
		torch_dtype (typing.Optional[torch.dtype]): The torch datatype. Defaults to None.
		trust_remote_code (typing.Optional[bool]): Whether to allow loading user-provided code contained in the downloaded model. Defaults to None.
		use_fast (typing.Optional[bool]): Whether to use the fast tokenizer. Defaults to None.

	:Usage:
		settings = HF_TokenizerSettings(
			pretrained_model_name_or_path="gpt2", tokenizer_type="gpt2"
		)
	"""
	
	def __init__(
			self,
			pretrained_model_name_or_path: typing.Union[str, os.PathLike],
			cache_dir: typing.Optional[typing.Union[str, os.PathLike]] = None,
			config: typing.Optional[PretrainedConfig] = None,
			force_download: typing.Optional[bool] = None,
			proxies: typing.Optional[dict[str, str]] = None,
			sub_folder: typing.Optional[str] = None,
			tokenizer_type: typing.Optional[str] = None,
			torch_dtype: typing.Optional[typing.Union[str, torch.dtype]] = None,
			trust_remote_code: typing.Optional[bool] = None,
			token: typing.Optional[str] = None,
			use_fast: typing.Optional[bool] = None,
			tokenizer_type_kwargs: typing.Optional[TokenizerTypeKwargs] = None,
	):
		"""
		Initializes HF_TokenizerSettings with the provided parameters.

		Args:
			pretrained_model_name_or_path (typing.Union[str, os.PathLike]): Path to pretrained model or model identifier from huggingface.co/models.
			cache_dir (typing.Optional[typing.Union[str, os.PathLike]]): Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used. Defaults to None.
			config (typing.Optional[PretrainedConfig]): An instance of a configuration object to use instead of loading the configuration from the pretrained model configuration file. Defaults to None.
			force_download (typing.Optional[bool]): Whether to force the (re-)download the model weights and configuration files and override the cached versions if they exist. Defaults to None.
			proxies (typing.Optional[dict[str, str]]): A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request. Defaults to None.
			sub_folder (typing.Optional[str]): In case the relevant files are located inside a sub directory of the model repo on huggingface.co, you can specify the folder name here. Defaults to None.
			token (typing.Optional[str]): An authentication token (See https://huggingface.co/settings/token) for private repositories on huggingface.co. Defaults to None.
			tokenizer_type (typing.Optional[str]): The tokenizer type to use. Defaults to None.
			torch_dtype (typing.Optional[typing.Union[str, torch.dtype]]): torch.dtype or string that can be converted to a torch.dtype. Defaults to None.
			trust_remote_code (typing.Optional[bool]): Whether or not to allow for loading user-provided code contained in the downloaded model. Defaults to None.
			use_fast (typing.Optional[bool]): Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. Defaults to None.
			tokenizer_type_kwargs (typing.Optional[TokenizerTypeKwargs]): Additional keyword arguments passed along to the specific tokenizer type. Defaults to None.
		"""
		self.pretrained_model_name_or_path = pretrained_model_name_or_path
		self.cache_dir = cache_dir
		self.config = config
		self.force_download = force_download
		self.proxies = proxies
		self.sub_folder = sub_folder
		self.token = token
		self.torch_dtype = torch_dtype
		self.tokenizer_type = tokenizer_type
		self.trust_remote_code = trust_remote_code
		self.use_fast = use_fast
		
		if isinstance(tokenizer_type_kwargs, TokenizerTypeKwargs):
			for field, value in get_class_fields(tokenizer_type_kwargs).items():
				if value is not None:
					setattr(self, field, value)
		elif tokenizer_type_kwargs is not None:
			raise ValueError('"tokenizer_type_kwargs" must be of type TokenizerTypeKwargs')


class HF_Tokenizer:
	"""
	Wraps a Hugging Face tokenizer for easier initialization and access.

	Attributes:
		tokenizer (PreTrainedTokenizer): The initialized Hugging Face tokenizer.

	:Usage:
		from PyGPTs.HuggingFace.Tokenizers import HF_Tokenizer, HF_TokenizerSettings

		settings = HF_TokenizerSettings(pretrained_model_name_or_path="gpt2")
		tokenizer = HF_Tokenizer(settings)

		tokens = tokenizer.tokenizer("Hello, world!")
	"""
	
	def __init__(self, tokenizer_settings: HF_TokenizerSettings):
		"""
		Initializes a HF_Tokenizer with the given settings.

		Args:
			tokenizer_settings (HF_TokenizerSettings): The settings for the tokenizer.
		"""
		self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
				**{
					name: value
					for name, value in get_class_fields(tokenizer_settings).items()
					if value is not None
				}
		)
