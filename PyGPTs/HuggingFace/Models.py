import os
import torch
import typing
from PyGPTs.HuggingFace.base_objects import ObjectTypeKwargs
from PyVarTools.python_instances_tools import get_class_fields
from transformers import (
	PreTrainedModel,
	PretrainedConfig
)
from transformers.utils.quantization_config import QuantizationConfigMixin


class ModelTypeKwargs(ObjectTypeKwargs):
	"""
	Keyword arguments for model instantiation. Extends ObjectTypeKwargs.
	"""
	
	def __init__(self, **kwargs):
		"""Initializes ModelTypeKwargs with given keyword arguments."""
		super().__init__(**kwargs)


class HF_ModelSettings:
	"""
	Stores settings for initializing a Hugging Face model.

	Attributes:
		pretrained_model_name_or_path (typing.Union[str, os.PathLike]): Path to pretrained model or model identifier from huggingface.co/models.
		model_class (typing.Any): The model class to use.
		attn_implementation (typing.Optional[str]): The attention implementation to use. Defaults to None.
		cache_dir (typing.Optional[typing.Union[str, os.PathLike]]): Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used. Defaults to None.
		code_revision (typing.Optional[str]): The specific version of the code repository to use as a revision. Defaults to None.
		config (typing.Optional[PretrainedConfig]): An instance of a configuration object to use instead of loading the configuration from the pretrained model configuration file. Defaults to None.
		device_map (typing.Optional[typing.Union[typing.Union[int, str, torch.device], dict[str, typing.Union[int, str, torch.device]]]]): A map that specifies where each submodule should go. Defaults to None.
		force_download (bool): Whether to force the (re-)download the model weights and configuration files and override the cached versions if they exist. Defaults to False.
		from_flax (typing.Optional[bool]): Whether to load the model weights from Flax. Defaults to None.
		from_tf (typing.Optional[bool]): Whether to load the model weights from TensorFlow. Defaults to None.
		ignore_mismatched_sizes (typing.Optional[bool]): Whether or not to raise an error if some of the weights from the checkpoint do not have the same size in the current model (if the model is script-able). Defaults to None.
		local_files_only (typing.Optional[bool]): Whether or not to only look at local files (i.e., do not try to download the model). Defaults to None.
		low_cpu_mem_usage (typing.Optional[bool]): Whether to try to load the model in 8bit and fp16 to save memory at the cost of a slower first inference. Defaults to None.
		max_memory (typing.Optional[dict]): A dictionary device identifier to maximum usable memory. Defaults to None.
		mirror (typing.Optional[str]): Mirror source to resolve accessibility issues if needed. Defaults to None.
		offload_buffers (typing.Optional[bool]): Whether to automatically offload model weights to the CPU. Defaults to None.
		offload_folder (typing.Optional[typing.Union[str, PathLike]]): Path to the folder to offload weights to when `offload_state_dict=True`. Defaults to None.
		offload_state_dict (typing.Optional[bool]): Whether to offload the state dict to the CPU or disk depending on `offload_folder`. Defaults to None.
		output_loading_info (typing.Optional[bool]): Whether to also return additional information about the model loading. Defaults to None.
		proxies (typing.Optional[dict[str, str]]): A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.`. Defaults to None.
		quantization_config (typing.Optional[typing.Union[QuantizationConfigMixin, dict]]): The quantization configuration. Defaults to None.
		revision (typing.Optional[str]): Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git. Defaults to None.
		state_dict (typing.Optional[dict[str, str]]): A state dictionary to use instead of loading the state dict from the model file. Defaults to None.
		sub_folder (typing.Optional[str]): In case the relevant files are located inside a sub directory of the model repo on huggingface.co, you can specify the folder name here. Defaults to None.
		token (typing.Optional[str]): An authentication token (See https://huggingface.co/settings/token) for private repositories on huggingface.co. Defaults to None.
		torch_dtype (typing.Optional[typing.Union[str, torch.dtype]]): `torch.dtype` or string that can be converted to a `torch.dtype`. Defaults to None.
		trust_remote_code (typing.Optional[bool]): Whether or not to allow for loading user-provided code contained in the downloaded model. Defaults to None.
		use_safetensors (typing.Optional[bool]): If True, will try to load the safetensors version of the weights and configuration files if both are available (otherwise, it defaults to loading the `pytorch_model.bin` weights). If False, will try to load the standard weights. Defaults to None.
		variant (typing.Optional[str]): Model variant to use. Defaults to None.
		_fast_init (typing.Optional[bool]): Whether or not to disable fast initialization. Defaults to None.

	:Usage:
		from transformers import AutoConfig
		from PyGPTs.HuggingFace.Models import HF_ModelSettings

		config = AutoConfig.from_pretrained("gpt2") # Example config
		settings = HF_ModelSettings(
			pretrained_model_name_or_path="gpt2", model_class="gpt2", config=config
		)
	"""
	
	def __init__(
			self,
			pretrained_model_name_or_path: typing.Union[str, os.PathLike],
			model_class: typing.Any,
			attn_implementation: typing.Optional[str] = None,
			cache_dir: typing.Optional[typing.Union[str, os.PathLike]] = None,
			code_revision: typing.Optional[str] = None,
			config: typing.Optional[PretrainedConfig] = None,
			device_map: typing.Optional[
				typing.Union[
					typing.Union[int, str, torch.device],
					dict[str, typing.Union[int, str, torch.device]]
				]
			] = None,
			force_download: bool = False,
			from_flax: typing.Optional[bool] = None,
			from_tf: typing.Optional[bool] = None,
			ignore_mismatched_sizes: typing.Optional[bool] = None,
			local_files_only: typing.Optional[bool] = None,
			low_cpu_mem_usage: typing.Optional[bool] = None,
			max_memory: typing.Optional[dict] = None,
			mirror: typing.Optional[str] = None,
			offload_buffers: typing.Optional[bool] = None,
			offload_folder: typing.Optional[typing.Union[str, os.PathLike]] = None,
			output_loading_info: typing.Optional[bool] = None,
			offload_state_dict: typing.Optional[bool] = None,
			proxies: typing.Optional[dict[str, str]] = None,
			quantization_config: typing.Optional[typing.Union[QuantizationConfigMixin, dict]] = None,
			revision: typing.Optional[str] = None,
			state_dict: typing.Optional[dict[str, str]] = None,
			sub_folder: typing.Optional[str] = None,
			token: typing.Optional[str] = None,
			torch_dtype: typing.Optional[typing.Union[str, torch.dtype]] = None,
			trust_remote_code: typing.Optional[bool] = None,
			use_safetensors: typing.Optional[bool] = None,
			variant: typing.Optional[str] = None,
			_fast_init: typing.Optional[bool] = None,
			model_type_kwargs: typing.Optional[ModelTypeKwargs] = None,
	):
		"""
		Initializes HF_ModelSettings with provided parameters.

		Args:
			pretrained_model_name_or_path (typing.Union[str, os.PathLike]): Path to pretrained model or model identifier from huggingface.co/models.
			model_class (typing.Any): The model class to use.
			attn_implementation (typing.Optional[str]): The attention implementation to use. Defaults to None.
			cache_dir (typing.Optional[typing.Union[str, os.PathLike]]): Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used. Defaults to None.
			code_revision (typing.Optional[str]): The specific version of the code repository to use as a revision. Defaults to None.
			config (typing.Optional[PretrainedConfig]): An instance of a configuration object to use instead of loading the configuration from the pretrained model configuration file. Defaults to None.
			device_map (typing.Optional[typing.Union[typing.Union[int, str, torch.device], dict[str, typing.Union[int, str, torch.device]]]]): A map that specifies where each submodule should go. Defaults to None.
			force_download (bool): Whether to force the (re-)download the model weights and configuration files and override the cached versions if they exist. Defaults to False.
			from_flax (typing.Optional[bool]): Whether to load the model weights from Flax. Defaults to None.
			from_tf (typing.Optional[bool]): Whether to load the model weights from TensorFlow. Defaults to None.
			ignore_mismatched_sizes (typing.Optional[bool]): Whether or not to raise an error if some of the weights from the checkpoint do not have the same size in the current model (if the model is script-able). Defaults to None.
			local_files_only (typing.Optional[bool]): Whether or not to only look at local files (i.e., do not try to download the model). Defaults to None.
			low_cpu_mem_usage (typing.Optional[bool]): Whether to try to load the model in 8bit and fp16 to save memory at the cost of a slower first inference. Defaults to None.
			max_memory (typing.Optional[dict]): A dictionary device identifier to maximum usable memory. Defaults to None.
			mirror (typing.Optional[str]): Mirror source to resolve accessibility issues if needed. Defaults to None.
			offload_buffers (typing.Optional[bool]): Whether to automatically offload model weights to the CPU. Defaults to None.
			offload_folder (typing.Optional[typing.Union[str, os.PathLike]]): Path to the folder to offload weights to when `offload_state_dict=True`. Defaults to None.
			offload_state_dict (typing.Optional[bool]): Whether to offload the state dict to the CPU or disk depending on `offload_folder`. Defaults to None.
			output_loading_info (typing.Optional[bool]): Whether to also return additional information about the model loading. Defaults to None.
			proxies (typing.Optional[dict[str, str]]): A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.`. The proxies are used on each request. Defaults to None.
			quantization_config (typing.Optional[typing.Union[QuantizationConfigMixin, dict]]): The quantization configuration. Defaults to None.
			revision (typing.Optional[str]): Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git. Defaults to None.
			state_dict (typing.Optional[dict[str, str]]): A state dictionary to use instead of loading the state dict from the model file. Defaults to None.
			sub_folder (typing.Optional[str]): In case the relevant files are located inside a sub directory of the model repo on huggingface.co, you can specify the folder name here. Defaults to None.
			token (typing.Optional[str]): An authentication token for private repositories on huggingface.co. Defaults to None.
			torch_dtype (typing.Optional[typing.Union[str, torch.dtype]]): `torch.dtype` or string that can be converted to a `torch.dtype`. Defaults to None.
			trust_remote_code (typing.Optional[bool]): Whether or not to allow for loading user-provided code contained in the downloaded model. Defaults to None.
			use_safetensors (typing.Optional[bool]): If True, will try to load the safetensors version of the weights and configuration files if both are available (otherwise it defaults to loading the `pytorch_model.bin` weights). If False, will try to load the standard weights. Defaults to None.
			variant (typing.Optional[str]): Model variant to use. Defaults to None.
			_fast_init (typing.Optional[bool]): Whether or not to disable fast initialization. Defaults to None.
			model_type_kwargs (typing.Optional[ModelTypeKwargs]): Additional keyword arguments passed along to the specific model type. Defaults to None.

		Raises:
			ValueError: if model_type_kwargs is not None and not an instance of ModelTypeKwargs
		"""
		self.pretrained_model_name_or_path = pretrained_model_name_or_path
		self.model_class = model_class
		self.attn_implementation = attn_implementation
		self.cache_dir = cache_dir
		self.code_revision = code_revision
		self.config = config
		self.device_map = device_map
		self.force_download = force_download
		self.from_flax = from_flax
		self.from_tf = from_tf
		self.ignore_mismatched_sizes = ignore_mismatched_sizes
		self.local_files_only = local_files_only
		self.low_cpu_mem_usage = low_cpu_mem_usage
		self.max_memory = max_memory
		self.mirror = mirror
		self.offload_buffers = offload_buffers
		self.offload_folder = offload_folder
		self.output_loading_info = output_loading_info
		self.offload_state_dict = offload_state_dict
		self.proxies = proxies
		self.quantization_config = quantization_config
		self.revision = revision
		self.state_dict = state_dict
		self.sub_folder = sub_folder
		self.token = token
		self.torch_dtype = torch_dtype
		self.trust_remote_code = trust_remote_code
		self.use_safetensors = use_safetensors
		self.variant = variant
		self._fast_init = _fast_init
		
		if isinstance(model_type_kwargs, ModelTypeKwargs):
			for field, value in get_class_fields(model_type_kwargs).items():
				if value is not None:
					setattr(self, field, value)
		elif model_type_kwargs is not None:
			raise ValueError('"model_type_kwargs" must be of type ModelTypeKwargs')


class HF_Model:
	"""
	Wraps a Hugging Face model for easier initialization and access.

	Attributes:
		model (PreTrainedModel): The initialized Hugging Face model.

	:Usage:
		from transformers import AutoModel
		from PyGPTs.HuggingFace.Models import HF_Model, HF_ModelSettings

		settings = HF_ModelSettings(
			pretrained_model_name_or_path="gpt2", model_class=AutoModel
		)

		model = HF_Model(settings)
	"""
	
	def __init__(self, model_settings: HF_ModelSettings):
		"""
		Initializes a new HF_Model instance.

		Args:
			model_settings (HF_ModelSettings): The settings for the model.
		"""
		self.model: PreTrainedModel = model_settings.model_class.from_pretrained(
				**{
					name: value
					for name, value in get_class_fields(model_settings).items()
					if value is not None
					and name != "model_class"
				}
		)
