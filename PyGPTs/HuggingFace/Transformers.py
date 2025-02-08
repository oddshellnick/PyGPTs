import os
import numpy
import torch
import typing
from transformers import (
	BaseImageProcessor,
	PretrainedConfig
)
from PyVarTools.python_instances_tools import get_function_parameters
from transformers.utils.quantization_config import QuantizationConfigMixin
from PyGPTs.HuggingFace.Models import (
	HF_Model,
	HF_ModelSettings,
	ModelTypeKwargs
)
from PyGPTs.HuggingFace.Pipelines import (
	HF_Pipeline,
	HF_PipelineSettings,
	PipelineTypeKwargs
)
from PyGPTs.HuggingFace.Tokenizers import (
	HF_Tokenizer,
	HF_TokenizerSettings,
	TokenizerTypeKwargs
)


class HF_TransformerSettings:
	"""
	Stores settings for a Hugging Face Transformer model, tokenizer, and pipeline.

	Attributes:
		model_settings (HF_ModelSettings): Settings for the model.
		tokenizer_settings (HF_TokenizerSettings): Settings for the tokenizer.
		pipeline_settings (HF_PipelineSettings): Settings for the pipeline.

	:Usage:
		from transformers import AutoConfig
		from PyGPTs.HuggingFace import HF_TransformerSettings

		config = AutoConfig.from_pretrained('gpt2') # Example config
		settings = HF_TransformerSettings(
			pretrained_model_name_or_path="gpt2",
			model_class = "gpt2",
			config = config
		)

	"""
	
	def __init__(
			self,
			pretrained_model_name_or_path: typing.Union[str, os.PathLike],
			model_class: typing.Any,
			pipeline_class: typing.Optional[typing.Any] = None,
			attn_implementation: typing.Optional[str] = None,
			cache_dir: typing.Optional[typing.Union[str, os.PathLike]] = None,
			code_revision: typing.Optional[str] = None,
			config: typing.Optional[PretrainedConfig] = None,
			device: typing.Optional[typing.Union[int, str, torch.device]] = None,
			device_map: typing.Optional[
				typing.Union[
					typing.Union[int, str, torch.device],
					dict[str, typing.Union[int, str, torch.device]]
				]
			] = None,
			feature_extractor: typing.Optional[str] = None,
			force_download: bool = False,
			from_flax: typing.Optional[bool] = None,
			from_tf: typing.Optional[bool] = None,
			framework: typing.Optional[str] = None,
			image_processor: typing.Optional[typing.Union[str, BaseImageProcessor]] = None,
			ignore_mismatched_sizes: typing.Optional[bool] = None,
			local_files_only: typing.Optional[bool] = None,
			low_cpu_mem_usage: typing.Optional[bool] = None,
			max_memory: typing.Optional[dict] = None,
			mirror: typing.Optional[str] = None,
			offload_buffers: typing.Optional[bool] = None,
			offload_folder: typing.Optional[typing.Union[str, os.PathLike]] = None,
			offload_state_dict: typing.Optional[bool] = None,
			output_loading_info: typing.Optional[bool] = None,
			proxies: typing.Optional[dict[str, str]] = None,
			quantization_config: typing.Optional[typing.Union[QuantizationConfigMixin, dict]] = None,
			revision: typing.Optional[str] = None,
			state_dict: typing.Optional[dict[str, str]] = None,
			sub_folder: typing.Optional[str] = None,
			task: typing.Optional[str] = None,
			token: typing.Optional[str] = None,
			torch_dtype: typing.Optional[typing.Union[str, torch.dtype]] = None,
			trust_remote_code: typing.Optional[bool] = None,
			tokenizer_type: typing.Optional[str] = None,
			use_fast: typing.Optional[bool] = None,
			use_safetensors: typing.Optional[bool] = None,
			variant: typing.Optional[str] = None,
			_fast_init: typing.Optional[bool] = None,
			model_type_kwargs: typing.Optional[ModelTypeKwargs] = None,
			tokenizer_type_kwargs: typing.Optional[TokenizerTypeKwargs] = None,
			pipeline_type_kwargs: typing.Optional[PipelineTypeKwargs] = None,
	):
		"""
		Initializes HF_TransformerSettings with provided parameters.

		Args:
			pretrained_model_name_or_path (typing.Union[str, os.PathLike]): Path to pretrained model or model identifier from huggingface.co/models.
			model_class (typing.Any): The model class to use.
			pipeline_class (typing.Optional[typing.Any]): The pipeline class to use. Defaults to None.
			attn_implementation (typing.Optional[str]): The attention implementation to use. Defaults to None.
			cache_dir (typing.Optional[typing.Union[str, os.PathLike]]): Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used. Defaults to None.
			code_revision (typing.Optional[str]): The specific version of the code repository to use as a revision. This is useful when dealing with bleeding-edge code. Defaults to None.
			config (typing.Optional[PretrainedConfig]): An instance of a configuration object to use instead of loading the configuration from the pretrained model configuration file. Defaults to None.
			device (typing.Optional[typing.Union[int, str, torch.device]]): Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too. Defaults to None.
			device_map (typing.Optional[typing.Union[typing.Union[int, str, torch.device], dict[str, typing.Union[int, str, torch.device]]]]): A map that specifies where each submodule should go. It doesn't need to be refined to each individual layers. For example, '{'layer1': 0, 'layer2': 'cpu', 'layer3': 'cuda:1'}' will map layers with the name 'layer1' to GPU 0, 'layer2' to CPU and 'layer3' to GPU 1. More precisely, if the map is specified, it will map the layers that have names containing the keys (even partially) to the associated device. Also, you can specify a default device by using the key "default", and specify which devices should not be used by the key "offload". Defaults to None.
			feature_extractor (typing.Optional[str]): Name or path of the feature extractor to use. Defaults to None.
			force_download (typing.Optional[bool]): Whether to force the (re-)download the model weights and configuration files and override the cached versions if they exist. Defaults to False.
			from_flax (typing.Optional[bool]): Whether to load the model weights from Flax. Defaults to None.
			from_tf (typing.Optional[bool]): Whether to load the model weights from TensorFlow. Defaults to None.
			framework (typing.Optional[str]): Explicitly specify the framework to use (`"pt"` or `"tf"`). Defaults to None.
			image_processor (typing.Optional[typing.Union[str, BaseImageProcessor]]): Name or path of the image processor to use. Defaults to None.
			ignore_mismatched_sizes (typing.Optional[bool]): Whether or not to raise an error if some of the weights from the checkpoint do not have the same size in the current model (if the model is script-able). Defaults to None.
			local_files_only (typing.Optional[bool]): Whether or not to only look at local files (i.e., do not try to download the model). Defaults to None.
			low_cpu_mem_usage (typing.Optional[bool]): Whether to try to load the model in 8bit and fp16 to save memory at the cost of a slower first inference. Defaults to None.
			max_memory (typing.Optional[dict]): A dictionary device identifier to maximum usable memory. Will default to the maximum memory available if no values are given. Defaults to None.
			mirror (typing.Optional[str]): Mirror source to resolve accessibility issues if needed. Defaults to None.
			offload_buffers (typing.Optional[bool]): Whether to automatically offload model weights to the CPU. Defaults to None.
			offload_folder (typing.Optional[typing.Union[str, os.PathLike]]): Path to the folder to offload weights to when `offload_state_dict=True`. Defaults to None.
			offload_state_dict (typing.Optional[bool]): Whether to offload the state dict to the CPU or disk depending on `offload_folder`. Defaults to None.
			output_loading_info (typing.Optional[bool]): Whether to also return additional information about the model loading. Defaults to None.
			proxies (typing.Optional[dict[str, str]]): A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request. Defaults to None.
			quantization_config (typing.Optional[typing.Union[QuantizationConfigMixin, dict]]): The quantization configuration. Defaults to None.
			revision (typing.Optional[str]): Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git. Defaults to None.
			state_dict (typing.Optional[dict[str, str]]): A state dictionary to use instead of loading the state dict from the model file. Defaults to None.
			sub_folder (typing.Optional[str]): In case the relevant files are located inside a sub directory of the model repo on huggingface.co, you can specify the folder name here. Defaults to None.
			task (typing.Optional[str]): The task defining which pipeline will be returned. Defaults to None.
			token (typing.Optional[str]): An authentication token (See https://huggingface.co/settings/token) for private repositories on huggingface.co. Defaults to None.
			torch_dtype (typing.Optional[typing.Union[str, torch.dtype]]): torch.dtype or string that can be converted to a torch.dtype. Defaults to None.
			trust_remote_code (typing.Optional[bool]): Whether to allow the loading of user-provided code contained in the downloaded model. Defaults to None.
			tokenizer_type (typing.Optional[str]): The tokenizer class to use. Defaults to None.
			use_fast (typing.Optional[bool]): Whether to use the fast tokenizer or not. Defaults to None.
			use_safetensors (typing.Optional[bool]): If True, will try to load the safetensors version of the weights and configuration files if both are available (otherwise it defaults to loading the `pytorch_model.bin` weights). If False, will try to load the standard weights. Defaults to None.
			variant (typing.Optional[str]): Model variant to use. Defaults to None.
			_fast_init (typing.Optional[bool]): Whether or not to disable fast initialization. Defaults to None.
			model_type_kwargs (typing.Optional[ModelTypeKwargs]): Keyword arguments for the model type. Defaults to None.
			tokenizer_type_kwargs (typing.Optional[TokenizerTypeKwargs]): Keyword arguments for the tokenizer type. Defaults to None.
			pipeline_type_kwargs (typing.Optional[PipelineTypeKwargs]): Keyword arguments for the pipeline type. Defaults to None.
		"""
		parameters = locals()
		
		self.model_settings = HF_ModelSettings(
				**{
					name: parameters[name]
					for name in get_function_parameters(function_=HF_ModelSettings.__init__, excluding_parameters=["self"]).keys()
				}
		)
		
		self.tokenizer_settings = HF_TokenizerSettings(
				**{
					name: parameters[name]
					for name in get_function_parameters(function_=HF_TokenizerSettings.__init__, excluding_parameters=["self"]).keys()
				}
		)
		
		self.pipeline_settings = HF_PipelineSettings(
				**{
					name: parameters[name]
					for name in get_function_parameters(
							function_=HF_PipelineSettings.__init__,
							excluding_parameters=["self", "model", "tokenizer"]
					).keys()
				}
		)


class HF_Transformer:
	"""
	Combines a Hugging Face model, tokenizer, and pipeline for text generation.

	Attributes:
		model (HF_Model): The Hugging Face model instance.
		tokenizer (HF_Tokenizer): The Hugging Face tokenizer instance.
		pipeline (HF_Pipeline): The Hugging Face pipeline instance.

	:Usage:
		from PyGPTs.HuggingFace import HF_Transformer, HF_TransformerSettings

		settings = HF_TransformerSettings(
		   pretrained_model_name_or_path="gpt2", model_class="gpt2"
		)
		transformer = HF_Transformer(settings)

		output = transformer.generate_content(inputs="Write a short story.")

	"""
	
	def __init__(self, huggingface_transformer_settings: HF_TransformerSettings):
		"""
		Initializes a new HF_Transformer instance.

		Args:
			huggingface_transformer_settings (HF_TransformerSettings): Settings for the transformer.
		"""
		self.model = HF_Model(model_settings=huggingface_transformer_settings.model_settings)
		
		self.tokenizer = HF_Tokenizer(tokenizer_settings=huggingface_transformer_settings.tokenizer_settings)
		
		huggingface_transformer_settings.pipeline_settings.model = self.model.model
		huggingface_transformer_settings.pipeline_settings.tokenizer = self.tokenizer.tokenizer
		self.pipeline = HF_Pipeline(pipeline_settings=huggingface_transformer_settings.pipeline_settings)
	
	def generate_content(
			self,
			inputs: typing.Union[numpy.ndarray, bytes, str, dict],
			max_length: typing.Optional[int] = None,
			max_new_tokens: typing.Optional[int] = None,
			return_timestamps: typing.Optional[typing.Union[str, bool]] = None,
	) -> typing.Any:
		"""
		Generates content using the pipeline.

		Args:
			inputs (typing.Union[np.ndarray, bytes, str, dict]): The input for the pipeline.
			max_length (typing.Optional[int]): Maximum length of the generated sequence. Defaults to None.
			max_new_tokens (typing.Optional[int]): Maximum number of new tokens to generate. Defaults to None.
			return_timestamps (typing.Optional[typing.Union[str, bool]]): Whether to return timestamps. Defaults to None.

		Returns:
			typing.Any: The output of the pipeline, which depends on the specific task and model.
		"""
		return self.pipeline.pipe(
				inputs=inputs,
				max_length=max_length,
				max_new_tokens=max_new_tokens,
				return_timestamps=return_timestamps
		)
