import typing

import numpy
import torch
from PyVarTools.python_instances_tools import get_class_fields
from transformers import (
    BaseImageProcessor,
    ModelCard,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TFPreTrainedModel,
)
from transformers.pipelines import ArgumentHandler, pipeline

from PyGPTs.HuggingFace.base_objects import ObjectTypeKwargs


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
        args_parser: ArgumentHandler | None = None,
        batch_size: int | None = None,
        binary_output: bool | None = None,
        model_card: str | ModelCard | None = None,
        num_workers: int | None = None,
    ):
        """
        Initializes TextGenerationPipelineKwargs with specific keyword arguments.

        Args:
            args_parser (ArgumentHandler | None): Defaults to None.
            batch_size (int | None): Defaults to None.
            binary_output (bool | None): Defaults to None.
            model_card (str | ModelCard | None): Defaults to None.
            num_workers (int | None): Defaults to None.
        """
        super().__init__(
            args_parser=args_parser,
            batch_size=batch_size,
            binary_output=binary_output,
            model_card=model_card,
            num_workers=num_workers,
        )


class HuggingFacePipelineSettings:
    """
    Stores settings for initializing a Hugging Face pipeline.

    Attributes:
        pipeline_class (typing.Any | None): The pipeline class to use. Defaults to None.
        config (str | PretrainedConfig | None): The configuration to use. Defaults to None.
        device (int | str | torch.device | None): The device to use. Defaults to None.
        device_map (int | str | torch.device | dict[str, int | str | torch.device] | None): The device map to use. Defaults to None.
        feature_extractor (str | None): The feature extractor to use. Defaults to None.
        framework (str | None): The framework to use. Defaults to None.
        image_processor (str | BaseImageProcessor | None): The image processor to use. Defaults to None.
        model (str | PreTrainedModel | TFPreTrainedModel | None): The model to use. Defaults to None.
        revision (str | None): The revision to use. Defaults to None.
        task (str | None): The task to use. Defaults to None.
        token (str | bool | None): The token to use. Defaults to None.
        tokenizer (str | PreTrainedTokenizer | PreTrainedTokenizerFast | None): The tokenizer to use. Defaults to None.
        torch_dtype (str | torch.dtype | None): The torch dtype to use. Defaults to None.
        trust_remote_code (bool | None): Whether to trust remote code. Defaults to None.
        use_fast (bool | None): Whether to use fast tokenizer. Defaults to None.

    :Usage:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from PyGPTs.HuggingFace.Pipelines import HuggingFacePipelineSettings

        config = AutoConfig.from_pretrained("gpt2")
        model = AutoModel.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        settings = HuggingFacePipelineSettings(
            model="gpt2", tokenizer=tokenizer, model=model, task="text-generation"
        )
    """

    def __init__(
        self,
        pipeline_class: typing.Any | None = None,
        device: int | str | torch.device | None = None,
        device_map: int | str | torch.device | dict[str, int | str | torch.device] | None = None,
        feature_extractor: str | None = None,
        framework: str | None = None,
        image_processor: str | BaseImageProcessor | None = None,
        model: str | PreTrainedModel | TFPreTrainedModel | None = None,
        revision: str | None = None,
        task: str | None = None,
        token: str | bool | None = None,
        torch_dtype: str | torch.dtype | None = None,
        trust_remote_code: bool | None = None,
        use_fast: bool | None = None,
        config: str | PretrainedConfig | None = None,
        tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        pipeline_type_kwargs: PipelineTypeKwargs | None = None,
    ):
        """
        Initializes HuggingFacePipelineSettings with the given settings.

        Args:
            pipeline_class (typing.Any | None): The class to use for the pipeline, e.g., `transformers.pipeline`. Defaults to None.
            device (int | str | torch.device | None): Device ordinal for CPU/GPU supports. Defaults to None.
            device_map (int | str | torch.device | dict[str, int | str | torch.device] | None):  A map that specifies where each submodule should go. Defaults to None.
            feature_extractor (str | None):  Name or path of the feature extractor to use. Defaults to None.
            framework (str | None): Explicitly specify the framework to use (`"pt"` or `"tf"`). Defaults to None.
            image_processor (str | BaseImageProcessor | None): Name or path of the image processor to use. Defaults to None.
            model (str | PreTrainedModel | TFPreTrainedModel | None): The model to use. Defaults to None.
            revision (str | None): Revision is the specific model version to use. Defaults to None.
            task (str | None): The task defining which pipeline will be returned. Defaults to None.
            token (str | bool | None): An authentication token (See https://huggingface.co/settings/token) for private repositories on huggingface.co or, if `local_files_only=True`, path to the credentials file. If `token=True`, the token will be retrieved from the cache. Defaults to None.
            torch_dtype (str | torch.dtype | None): `torch.dtype` or string that can be converted to a `torch.dtype`. Defaults to None.
            trust_remote_code (bool | None): Whether or not to allow for loading user-provided code contained in the downloaded model. Defaults to None.
            use_fast (bool | None): Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. Defaults to None.
            config (PretrainedConfig | None): An instance of a configuration object to use instead of loading the configuration from the pretrained model configuration file. Defaults to None.
            tokenizer (PreTrainedTokenizer | None): An instance of a tokenizer to use instead of loading it from `pretrained_model_name_or_path`. Defaults to None.
            pipeline_type_kwargs (PipelineTypeKwargs | None):  Additional keyword arguments passed along to the specific pipeline type. Defaults to None.

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


class HuggingFacePipeline:
    """
    Wraps a Hugging Face pipeline for text generation.

    Attributes:
        pipeline_ (transformers.pipelines.base.Pipeline): The wrapped Hugging Face pipeline.

    :Usage:
         from transformers import pipeline, AutoTokenizer, AutoModel
         from PyGPTs.HuggingFace.Pipelines import HuggingFacePipeline, HuggingFacePipelineSettings

         tokenizer = AutoTokenizer.from_pretrained("gpt2")
         model = AutoModel.from_pretrained("gpt2")

         settings = HuggingFacePipelineSettings(model=model, tokenizer=tokenizer, task="text-generation")
         pipe = HuggingFacePipeline(settings)
         pipe.pipe(inputs="Write a short story.")
    """

    def __init__(self, pipeline_settings: HuggingFacePipelineSettings):
        """
        Initializes a new HuggingFacePipeline instance.

        Args:
            pipeline_settings (HuggingFacePipelineSettings): The settings to use for initializing the pipeline.
        """
        self.pipeline_ = pipeline(
            **{
                name: value
                for name, value in get_class_fields(pipeline_settings).items()
                if value is not None and name != "pipeline_type_kwargs"
            }
        )

    def pipe(
        self,
        inputs: numpy.ndarray | bytes | str | dict,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        return_timestamps: str | bool | None = None,
    ) -> typing.Any:
        """
        Runs the pipeline with the given inputs and parameters.

        Args:
            inputs (numpy.ndarray | bytes | str | dict): The input to the pipeline.
            max_length (int | None): The maximum length to generate. Defaults to None.
            max_new_tokens (int | None):  The maximum number of new tokens to generate. Defaults to None.
            return_timestamps (str | bool | None): Whether to return timestamps of each token. Defaults to None.

        Returns:
            typing.Any: The generated text.
        """
        return self.pipeline_(
            inputs, **{name: value for name, value in locals().items() if value is not None and name != "inputs"}
        )[0]["generated_text"][-1]["content"]
