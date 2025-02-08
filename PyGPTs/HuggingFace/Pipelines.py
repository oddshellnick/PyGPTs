import numpy
import torch
import typing
from PyGPTs.HuggingFace.base_objects import ObjectTypeKwargs
from PyVarTools.python_instances_tools import get_class_fields
from transformers.pipelines import (
	ArgumentHandler,
	pipeline
)
from transformers import (
	BaseImageProcessor,
	ModelCard,
	PreTrainedModel,
	PreTrainedTokenizer,
	PreTrainedTokenizerFast,
	PretrainedConfig,
	TFPreTrainedModel
)


class PipelineTypeKwargs(ObjectTypeKwargs):
	"""
	Keyword arguments for pipeline instantiation. Extends ObjectTypeKwargs.
	"""
	
	def __init__(self, **kwargs):
		"""Initializes PipelineTypeKwargs with given keyword arguments."""
		super().__init__(**kwargs)


class TextGenerationPipelineKwargs(PipelineTypeKwargs):
	"""Keyword arguments specifically for text generation pipelines. Extends PipelineTypeKwargs."""
	
	def __init__(
			self,
			args_parser: typing.Optional[ArgumentHandler] = None,
			batch_size: typing.Optional[int] = None,
			binary_output: typing.Optional[bool] = None,
			model_card: typing.Optional[typing.Union[str, ModelCard]] = None,
			num_workers: typing.Optional[int] = None,
	):
		"""
		Initializes TextGenerationPipelineKwargs with specific keyword arguments.

		Args:
			args_parser (typing.Optional[ArgumentHandler]): Defaults to None.
			batch_size (typing.Optional[int]): Defaults to None.
			binary_output (typing.Optional[bool]): Defaults to None.
			model_card (typing.Optional[typing.Union[str, ModelCard]]): Defaults to None.
			num_workers (typing.Optional[int]): Defaults to None.
		"""
		super().__init__(
				args_parser=args_parser,
				batch_size=batch_size,
				binary_output=binary_output,
				model_card=model_card,
				num_workers=num_workers,
		)


class HF_PipelineSettings:
	"""
	Stores settings for initializing a Hugging Face pipeline.

	Attributes:
		pipeline_class (typing.Optional[typing.Any]): The pipeline class to use. Defaults to None.
		config (typing.Optional[typing.Union[str, PretrainedConfig]]): The configuration to use. Defaults to None.
		device (typing.Optional[typing.Union[int, str, torch.device]]): The device to use. Defaults to None.
		device_map (typing.Optional[typing.Union[typing.Union[int, str, torch.device], dict[str, typing.Union[int, str, torch.device]]]]): The device map to use. Defaults to None.
		feature_extractor (typing.Optional[str]): The feature extractor to use. Defaults to None.
		framework (typing.Optional[str]): The framework to use. Defaults to None.
		image_processor (typing.Optional[typing.Union[str, BaseImageProcessor]]): The image processor to use. Defaults to None.
		model (typing.Optional[typing.Union[str, PreTrainedModel, TFPreTrainedModel]]): The model to use. Defaults to None.
		revision (typing.Optional[str]): The revision to use. Defaults to None.
		task (typing.Optional[str]): The task to use. Defaults to None.
		token (typing.Optional[typing.Union[str, bool]]): The token to use. Defaults to None.
		tokenizer (typing.Optional[typing.Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]]): The tokenizer to use. Defaults to None.
		torch_dtype (typing.Optional[typing.Union[str, torch.dtype]]): The torch dtype to use. Defaults to None.
		trust_remote_code (typing.Optional[bool]): Whether to trust remote code. Defaults to None.
		use_fast (typing.Optional[bool]): Whether to use fast tokenizer. Defaults to None.

	:Usage:
		from transformers import AutoConfig, AutoModel, AutoTokenizer
		from PyGPTs.HuggingFace.Pipelines import HF_PipelineSettings

		config = AutoConfig.from_pretrained("gpt2")
		model = AutoModel.from_pretrained("gpt2")
		tokenizer = AutoTokenizer.from_pretrained("gpt2")
		settings = HF_PipelineSettings(
			model="gpt2", tokenizer=tokenizer, model=model, task="text-generation"
		)
	"""
	
	def __init__(
			self,
			pipeline_class: typing.Optional[typing.Any] = None,
			device: typing.Optional[typing.Union[int, str, torch.device]] = None,
			device_map: typing.Optional[
				typing.Union[
					typing.Union[int, str, torch.device],
					dict[str, typing.Union[int, str, torch.device]]
				]
			] = None,
			feature_extractor: typing.Optional[str] = None,
			framework: typing.Optional[str] = None,
			image_processor: typing.Optional[typing.Union[str, BaseImageProcessor]] = None,
			model: typing.Optional[typing.Union[str, PreTrainedModel, TFPreTrainedModel]] = None,
			revision: typing.Optional[str] = None,
			task: typing.Optional[str] = None,
			token: typing.Optional[typing.Union[str, bool]] = None,
			torch_dtype: typing.Optional[typing.Union[str, torch.dtype]] = None,
			trust_remote_code: typing.Optional[bool] = None,
			use_fast: typing.Optional[bool] = None,
			config: typing.Optional[typing.Union[str, PretrainedConfig]] = None,
			tokenizer: typing.Optional[typing.Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
			pipeline_type_kwargs: typing.Optional[PipelineTypeKwargs] = None,
	):
		"""
		Initializes HF_PipelineSettings with the given settings.

		Args:
			pipeline_class (typing.Optional[typing.Any]): The class to use for the pipeline, e.g., `transformers.pipeline`. Defaults to None.
			device (typing.Optional[typing.Union[int, str, torch.device]]): Device ordinal for CPU/GPU supports. Defaults to None.
			device_map (typing.Optional[typing.Union[typing.Union[int, str, torch.device], dict[str, typing.Union[int, str, torch.device]]]]): A map that specifies where each submodule should go. Defaults to None.
			feature_extractor (typing.Optional[str]): Name or path of the feature extractor to use. Defaults to None.
			framework (typing.Optional[str]): Explicitly specify the framework to use (`"pt"` or `"tf"`). Defaults to None.
			image_processor (typing.Optional[typing.Union[str, BaseImageProcessor]]): Name or path of the image processor to use. Defaults to None.
			model (typing.Optional[typing.Union[str, PreTrainedModel, TFPreTrainedModel]]): The model to use. Defaults to None.
			revision (typing.Optional[str]): Revision is the specific model version to use. Defaults to None.
			task (typing.Optional[str]): The task defining which pipeline will be returned. Defaults to None.
			token (typing.Optional[typing.Union[str, bool]]): An authentication token (See https://huggingface.co/settings/token) for private repositories on huggingface.co or, if `local_files_only=True`, path to the credentials file. If `token=True`, the token will be retrieved from the cache. Defaults to None.
			torch_dtype (typing.Optional[typing.Union[str, torch.dtype]]): `torch.dtype` or string that can be converted to a `torch.dtype`. Defaults to None.
			trust_remote_code (typing.Optional[bool]): Whether or not to allow for loading user-provided code contained in the downloaded model. Defaults to None.
			use_fast (typing.Optional[bool]): Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. Defaults to None.
			config (typing.Optional[PretrainedConfig]): An instance of a configuration object to use instead of loading the configuration from the pretrained model configuration file. Defaults to None.
			tokenizer (typing.Optional[PreTrainedTokenizer]): An instance of a tokenizer to use instead of loading it from `pretrained_model_name_or_path`. Defaults to None.
			pipeline_type_kwargs (typing.Optional[PipelineTypeKwargs]): Additional keyword arguments passed along to the specific pipeline type. Defaults to None.

		Raises:
			ValueError: if pipeline_type_kwargs is not None or not of type PipelineTypeKwargs
		"""
		self.pipeline_class = pipeline_class
		self.config = config
		self.device = device
		self.device_map = device_map
		self.feature_extractor = feature_extractor
		self.framework = framework
		self.image_processor = image_processor
		self.model = model
		self.revision = revision
		self.task = task
		self.token = token
		self.tokenizer = tokenizer
		self.torch_dtype = torch_dtype
		self.trust_remote_code = trust_remote_code
		self.use_fast = use_fast
		
		if isinstance(pipeline_type_kwargs, PipelineTypeKwargs):
			for field, value in get_class_fields(pipeline_type_kwargs).items():
				if value is not None:
					setattr(self, field, value)
		elif pipeline_type_kwargs is not None:
			raise ValueError('"pipeline_type_kwargs" must be of type PipelineTypeKwargs')


class HF_Pipeline:
	"""
	Wraps a Hugging Face pipeline for text generation.

	Attributes:
		pipeline_ (transformers.pipelines.base.Pipeline): The wrapped Hugging Face pipeline.

	:Usage:
		 from transformers import pipeline, AutoTokenizer, AutoModel
		 from PyGPTs.HuggingFace.Pipelines import HF_Pipeline, HF_PipelineSettings

		 tokenizer = AutoTokenizer.from_pretrained("gpt2")
		 model = AutoModel.from_pretrained("gpt2")

		 settings = HF_PipelineSettings(model=model, tokenizer=tokenizer, task="text-generation")
		 pipe = HF_Pipeline(settings)
		 pipe.pipe(inputs="Write a short story.")
	"""
	
	def __init__(self, pipeline_settings: HF_PipelineSettings):
		"""
		Initializes a new HF_Pipeline instance.

		Args:
			pipeline_settings (HF_PipelineSettings): The settings to use for initializing the pipeline.
		"""
		self.pipeline_ = pipeline(
				**{
					name: value
					for name, value in get_class_fields(pipeline_settings).items()
					if value is not None
					and name != "pipeline_type_kwargs"
				}
		)
	
	def pipe(
			self,
			inputs: typing.Union[numpy.ndarray, bytes, str, dict],
			max_length: typing.Optional[int] = None,
			max_new_tokens: typing.Optional[int] = None,
			return_timestamps: typing.Optional[typing.Union[str, bool]] = None,
	) -> typing.Any:
		"""
		Runs the pipeline with the given inputs and parameters.

		Args:
			inputs (typing.Union[numpy.ndarray, bytes, str, dict]): The input to the pipeline.
			max_length (typing.Optional[int]): The maximum length to generate. Defaults to None.
			max_new_tokens (typing.Optional[int]): The maximum number of new tokens to generate. Defaults to None.
			return_timestamps (typing.Optional[typing.Union[str, bool]]): Whether to return timestamps of each token. Defaults to None.

		Returns:
			typing.Any: The generated text.
		"""
		return self.pipeline_(
				inputs,
				**{
					name: value
					for name, value in locals().items()
					if value is not None
					and name != "inputs"
				}
		)[0]["generated_text"][-1]["content"]
