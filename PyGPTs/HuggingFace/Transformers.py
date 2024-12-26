import os
import typing

import numpy
import torch
from PyVarTools.python_instances_tools import get_function_parameters
from transformers import BaseImageProcessor, PretrainedConfig
from transformers.utils.quantization_config import QuantizationConfigMixin

from PyGPTs.HuggingFace.Models import (
    HuggingFaceModel,
    HuggingFaceModelSettings,
    ModelTypeKwargs,
)
from PyGPTs.HuggingFace.Pipelines import (
    HuggingFacePipeline,
    HuggingFacePipelineSettings,
    PipelineTypeKwargs,
)
from PyGPTs.HuggingFace.Tokenizers import (
    HuggingFaceTokenizer,
    HuggingFaceTokenizerSettings,
    TokenizerTypeKwargs,
)


class HuggingFaceTransformerSettings:
    """
    Stores settings for a Hugging Face Transformer model, tokenizer, and pipeline.

    Attributes:
        model_settings (HuggingFaceModelSettings): Settings for the model.
        tokenizer_settings (HuggingFaceTokenizerSettings): Settings for the tokenizer.
        pipeline_settings (HuggingFacePipelineSettings): Settings for the pipeline.

    :Usage:
        from transformers import AutoConfig
        from PyGPTs.HuggingFace import HuggingFaceTransformerSettings

        config = AutoConfig.from_pretrained('gpt2')  # Example config
        settings = HuggingFaceTransformerSettings(
            pretrained_model_name_or_path="gpt2",
            model_class = "gpt2",
            config = config
        )

    """

    def __init__(
        self,
        pretrained_model_name_or_path: str | os.PathLike,
        model_class: typing.Any,
        pipeline_class: typing.Any | None = None,
        attn_implementation: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        code_revision: str | None = None,
        config: PretrainedConfig | None = None,
        device: int | str | torch.device | None = None,
        device_map: int | str | torch.device | dict[str, int | str | torch.device] | None = None,
        feature_extractor: str | None = None,
        force_download: bool = False,
        from_flax: bool | None = None,
        from_tf: bool | None = None,
        framework: str | None = None,
        image_processor: str | BaseImageProcessor | None = None,
        ignore_mismatched_sizes: bool | None = None,
        local_files_only: bool | None = None,
        low_cpu_mem_usage: bool | None = None,
        max_memory: dict | None = None,
        mirror: str | None = None,
        offload_buffers: bool | None = None,
        offload_folder: str | os.PathLike | None = None,
        offload_state_dict: bool | None = None,
        output_loading_info: bool | None = None,
        proxies: dict[str, str] | None = None,
        quantization_config: QuantizationConfigMixin | dict | None = None,
        revision: str | None = None,
        state_dict: dict[str, str] | None = None,
        sub_folder: str | None = None,
        task: str | None = None,
        token: str | None = None,
        torch_dtype: str | torch.dtype | None = None,
        trust_remote_code: bool | None = None,
        tokenizer_type: str | None = None,
        use_fast: bool | None = None,
        use_safetensors: bool | None = None,
        variant: str | None = None,
        _fast_init: bool | None = None,
        model_type_kwargs: ModelTypeKwargs | None = None,
        tokenizer_type_kwargs: TokenizerTypeKwargs | None = None,
        pipeline_type_kwargs: PipelineTypeKwargs | None = None,
    ):
        """
        Initializes HuggingFaceTransformerSettings with provided parameters.

        Args:
            pretrained_model_name_or_path (str | os.PathLike): Path to pretrained model or model identifier from huggingface.co/models.
            model_class (Any): The model class to use.
            pipeline_class (Any | None): The pipeline class to use. Defaults to None.
            attn_implementation (str | None): The attention implementation to use. Defaults to None.
            cache_dir (str | os.PathLike | None): Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used. Defaults to None.
            code_revision (str | None): The specific version of the code repository to use as a revision. This is useful when dealing with bleeding-edge code. Defaults to None.
            config (PretrainedConfig | None): An instance of a configuration object to use instead of loading the configuration from the pretrained model configuration file. Defaults to None.
            device (int | str | torch.device | None): Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too. Defaults to None.
            device_map (int | str | torch.device | dict[str, int | str | torch.device] | None): A map that specifies where each submodule should go. It doesn't need to be refined to each individual layers. For example, '{'layer1': 0, 'layer2': 'cpu', 'layer3': 'cuda:1'}' will map layers with the name 'layer1' to GPU 0, 'layer2' to CPU and 'layer3' to GPU 1.  More precisely, if the map is specified, it will map the layers that have names containing the keys (even partially) to the associated device. Also, you can specify a default device by using the key "default", and specify which devices should not be used by the key "offload". Defaults to None.
            feature_extractor (str | None): Name or path of the feature extractor to use. Defaults to None.
            force_download (bool | None): Whether to force the (re-)download the model weights and configuration files and override the cached versions if they exist. Defaults to False.
            from_flax (bool | None): Whether to load the model weights from Flax. Defaults to None.
            from_tf (bool | None): Whether to load the model weights from TensorFlow. Defaults to None.
            framework (str | None): Explicitly specify the framework to use (`"pt"` or `"tf"`). Defaults to None.
            image_processor (str | BaseImageProcessor | None): Name or path of the image processor to use. Defaults to None.
            ignore_mismatched_sizes (bool | None): Whether or not to raise an error if some of the weights from the checkpoint do not have the same size in the current model (if the model is script-able). Defaults to None.
            local_files_only (bool | None): Whether or not to only look at local files (i.e., do not try to download the model). Defaults to None.
            low_cpu_mem_usage (bool | None): Whether to try to load the model in 8bit and fp16 to save memory at the cost of a slower first inference. Defaults to None.
            max_memory (dict | None): A dictionary device identifier to maximum usable memory.  Will default to the maximum memory available if no values are given. Defaults to None.
            mirror (str | None): Mirror source to resolve accessibility issues if needed. Defaults to None.
            offload_buffers (bool | None): Whether to automatically offload model weights to the CPU. Defaults to None.
            offload_folder (str | os.PathLike | None): Path to the folder to offload weights to when `offload_state_dict=True`. Defaults to None.
            offload_state_dict (bool | None): Whether to offload the state dict to the CPU or disk depending on `offload_folder`. Defaults to None.
            output_loading_info (bool | None): Whether to also return additional information about the model loading. Defaults to None.
            proxies (dict[str, str] | None): A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request. Defaults to None.
            quantization_config (QuantizationConfigMixin | dict | None): The quantization configuration. Defaults to None.
            revision (str | None): Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git. Defaults to None.
            state_dict (dict[str, str] | None): A state dictionary to use instead of loading the state dict from the model file. Defaults to None.
            sub_folder (str | None): In case the relevant files are located inside a sub directory of the model repo on huggingface.co, you can specify the folder name here. Defaults to None.
            task (str | None):  The task defining which pipeline will be returned. Defaults to None.
            token (str | None): An authentication token (See https://huggingface.co/settings/token) for private repositories on huggingface.co. Defaults to None.
            torch_dtype (str | torch.dtype | None): torch.dtype or string that can be converted to a torch.dtype. Defaults to None.
            trust_remote_code (bool | None): Whether to allow the loading of user-provided code contained in the downloaded model. Defaults to None.
            tokenizer_type (str | None): The tokenizer class to use. Defaults to None.
            use_fast (bool | None): Whether to use the fast tokenizer or not. Defaults to None.
            use_safetensors (bool | None): If True, will try to load the safetensors version of the weights and configuration files if both are available (otherwise it defaults to loading the `pytorch_model.bin` weights). If False, will try to load the standard weights. Defaults to None.
            variant (str | None): Model variant to use. Defaults to None.
            _fast_init (bool | None): Whether or not to disable fast initialization. Defaults to None.
            model_type_kwargs (ModelTypeKwargs | None): Keyword arguments for the model type. Defaults to None.
            tokenizer_type_kwargs (TokenizerTypeKwargs | None): Keyword arguments for the tokenizer type. Defaults to None.
            pipeline_type_kwargs (PipelineTypeKwargs | None): Keyword arguments for the pipeline type. Defaults to None.
        """
        parameters = locals()
        self.model_settings = HuggingFaceModelSettings(
            **{
                name: parameters[name]
                for name in get_function_parameters(
                    function_=HuggingFaceModelSettings.__init__, excluding_parameters=["self"]
                ).keys()
            }
        )
        self.tokenizer_settings = HuggingFaceTokenizerSettings(
            **{
                name: parameters[name]
                for name in get_function_parameters(
                    function_=HuggingFaceTokenizerSettings.__init__, excluding_parameters=["self"]
                ).keys()
            }
        )
        self.pipeline_settings = HuggingFacePipelineSettings(
            **{
                name: parameters[name]
                for name in get_function_parameters(
                    function_=HuggingFacePipelineSettings.__init__, excluding_parameters=["self", "model", "tokenizer"]
                ).keys()
            }
        )


class HuggingFaceTransformer:
    """
    Combines a Hugging Face model, tokenizer, and pipeline for text generation.

    Attributes:
        model (HuggingFaceModel): The Hugging Face model instance.
        tokenizer (HuggingFaceTokenizer): The Hugging Face tokenizer instance.
        pipeline (HuggingFacePipeline): The Hugging Face pipeline instance.

    :Usage:
        from PyGPTs.HuggingFace import HuggingFaceTransformer, HuggingFaceTransformerSettings

        settings = HuggingFaceTransformerSettings(
           pretrained_model_name_or_path="gpt2", model_class="gpt2"
        )
        transformer = HuggingFaceTransformer(settings)

        output = transformer.generate_content(inputs="Write a short story.")

    """

    def __init__(self, huggingface_transformer_settings: HuggingFaceTransformerSettings):
        """
        Initializes a new HuggingFaceTransformer instance.

        Args:
            huggingface_transformer_settings (HuggingFaceTransformerSettings): Settings for the transformer.
        """
        self.model = HuggingFaceModel(model_settings=huggingface_transformer_settings.model_settings)
        self.tokenizer = HuggingFaceTokenizer(tokenizer_settings=huggingface_transformer_settings.tokenizer_settings)
        huggingface_transformer_settings.pipeline_settings.model = self.model.model
        huggingface_transformer_settings.pipeline_settings.tokenizer = self.tokenizer.tokenizer
        self.pipeline = HuggingFacePipeline(pipeline_settings=huggingface_transformer_settings.pipeline_settings)

    def generate_content(
        self,
        inputs: numpy.ndarray | bytes | str | dict,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        return_timestamps: str | bool | None = None,
    ) -> typing.Any:
        """
        Generates content using the pipeline.

        Args:
            inputs (np.ndarray | bytes | str | dict): The input for the pipeline.
            max_length (int | None): Maximum length of the generated sequence. Defaults to None.
            max_new_tokens (int | None): Maximum number of new tokens to generate. Defaults to None.
            return_timestamps (str | bool | None): Whether to return timestamps. Defaults to None.

        Returns:
            typing.Any: The output of the pipeline, which depends on the specific task and model.
        """
        return self.pipeline.pipe(
            inputs=inputs, max_length=max_length, max_new_tokens=max_new_tokens, return_timestamps=return_timestamps
        )
