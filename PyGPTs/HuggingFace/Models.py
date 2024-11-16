import os
import typing

import torch
from PyVarTools.python_instances_tools import get_class_fields
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.quantization_config import QuantizationConfigMixin

from PyGPTs.HuggingFace.base_objects import ObjectTypeKwargs


class ModelTypeKwargs(ObjectTypeKwargs):
    """
    Keyword arguments for model instantiation. Extends ObjectTypeKwargs.
    """

    def __init__(self, **kwargs):
        """Initializes ModelTypeKwargs with given keyword arguments."""
        super().__init__(**kwargs)


class HuggingFaceModelSettings:
    """
    Stores settings for initializing a Hugging Face model.

    Attributes:
        pretrained_model_name_or_path (str | os.PathLike): Path to pretrained model or model identifier from huggingface.co/models.
        model_class (typing.Any): The model class to use.
        attn_implementation (str | None):  The attention implementation to use. Defaults to None.
        cache_dir (str | os.PathLike | None):  Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used. Defaults to None.
        code_revision (str | None): The specific version of the code repository to use as a revision. Defaults to None.
        config (PretrainedConfig | None):  An instance of a configuration object to use instead of loading the configuration from the pretrained model configuration file. Defaults to None.
        device_map (int | str | torch.device | dict[str, int | str | torch.device] | None): A map that specifies where each submodule should go. Defaults to None.
        force_download (bool): Whether to force the (re-)download the model weights and configuration files and override the cached versions if they exist. Defaults to False.
        from_flax (bool | None): Whether to load the model weights from Flax. Defaults to None.
        from_tf (bool | None): Whether to load the model weights from TensorFlow. Defaults to None.
        ignore_mismatched_sizes (bool | None): Whether or not to raise an error if some of the weights from the checkpoint do not have the same size in the current model (if the model is script-able). Defaults to None.
        local_files_only (bool | None): Whether or not to only look at local files (i.e., do not try to download the model). Defaults to None.
        low_cpu_mem_usage (bool | None): Whether to try to load the model in 8bit and fp16 to save memory at the cost of a slower first inference. Defaults to None.
        max_memory (dict | None): A dictionary device identifier to maximum usable memory. Defaults to None.
        mirror (str | None): Mirror source to resolve accessibility issues if needed. Defaults to None.
        offload_buffers (bool | None): Whether to automatically offload model weights to the CPU. Defaults to None.
        offload_folder (str | PathLike | None): Path to the folder to offload weights to when `offload_state_dict=True`. Defaults to None.
        offload_state_dict (bool | None): Whether to offload the state dict to the CPU or disk depending on `offload_folder`. Defaults to None.
        output_loading_info (bool | None): Whether to also return additional information about the model loading. Defaults to None.
        proxies (dict[str, str] | None): A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.`. Defaults to None.
        quantization_config (QuantizationConfigMixin | dict | None): The quantization configuration. Defaults to None.
        revision (str | None): Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git. Defaults to None.
        state_dict (dict[str, str] | None): A state dictionary to use instead of loading the state dict from the model file. Defaults to None.
        sub_folder (str | None): In case the relevant files are located inside a sub directory of the model repo on huggingface.co, you can specify the folder name here. Defaults to None.
        token (str | None): An authentication token (See https://huggingface.co/settings/token) for private repositories on huggingface.co. Defaults to None.
        torch_dtype (str | torch.dtype | None): `torch.dtype` or string that can be converted to a `torch.dtype`. Defaults to None.
        trust_remote_code (bool | None): Whether or not to allow for loading user-provided code contained in the downloaded model. Defaults to None.
        use_safetensors (bool | None): If True, will try to load the safetensors version of the weights and configuration files if both are available (otherwise, it defaults to loading the `pytorch_model.bin` weights). If False, will try to load the standard weights. Defaults to None.
        variant (str | None): Model variant to use. Defaults to None.
        _fast_init (bool | None): Whether or not to disable fast initialization. Defaults to None.

    :Usage:
        from transformers import AutoConfig
        from PyGPTs.HuggingFace.Models import HuggingFaceModelSettings

        config = AutoConfig.from_pretrained("gpt2") # Example config
        settings = HuggingFaceModelSettings(
            pretrained_model_name_or_path="gpt2", model_class="gpt2", config=config
        )
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str | os.PathLike,
        model_class: typing.Any,
        attn_implementation: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        code_revision: str | None = None,
        config: PretrainedConfig | None = None,
        device_map: int | str | torch.device | dict[str, int | str | torch.device] | None = None,
        force_download: bool = False,
        from_flax: bool | None = None,
        from_tf: bool | None = None,
        ignore_mismatched_sizes: bool | None = None,
        local_files_only: bool | None = None,
        low_cpu_mem_usage: bool | None = None,
        max_memory: dict | None = None,
        mirror: str | None = None,
        offload_buffers: bool | None = None,
        offload_folder: str | os.PathLike | None = None,
        output_loading_info: bool | None = None,
        offload_state_dict: bool | None = None,
        proxies: dict[str, str] | None = None,
        quantization_config: QuantizationConfigMixin | dict | None = None,
        revision: str | None = None,
        state_dict: dict[str, str] | None = None,
        sub_folder: str | None = None,
        token: str | None = None,
        torch_dtype: str | torch.dtype | None = None,
        trust_remote_code: bool | None = None,
        use_safetensors: bool | None = None,
        variant: str | None = None,
        _fast_init: bool | None = None,
        model_type_kwargs: ModelTypeKwargs | None = None,
    ):
        """
        Initializes HuggingFaceModelSettings with provided parameters.

        Args:
            pretrained_model_name_or_path (str | os.PathLike): Path to pretrained model or model identifier from huggingface.co/models.
            model_class (typing.Any): The model class to use.
            attn_implementation (str | None): The attention implementation to use. Defaults to None.
            cache_dir (str | os.PathLike | None): Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used. Defaults to None.
            code_revision (str | None):  The specific version of the code repository to use as a revision. Defaults to None.
            config (PretrainedConfig | None): An instance of a configuration object to use instead of loading the configuration from the pretrained model configuration file. Defaults to None.
            device_map (int | str | torch.device | dict[str, int | str | torch.device] | None):  A map that specifies where each submodule should go. Defaults to None.
            force_download (bool): Whether to force the (re-)download the model weights and configuration files and override the cached versions if they exist. Defaults to False.
            from_flax (bool | None): Whether to load the model weights from Flax. Defaults to None.
            from_tf (bool | None): Whether to load the model weights from TensorFlow. Defaults to None.
            ignore_mismatched_sizes (bool | None): Whether or not to raise an error if some of the weights from the checkpoint do not have the same size in the current model (if the model is script-able). Defaults to None.
            local_files_only (bool | None): Whether or not to only look at local files (i.e., do not try to download the model). Defaults to None.
            low_cpu_mem_usage (bool | None): Whether to try to load the model in 8bit and fp16 to save memory at the cost of a slower first inference. Defaults to None.
            max_memory (dict | None): A dictionary device identifier to maximum usable memory. Defaults to None.
            mirror (str | None): Mirror source to resolve accessibility issues if needed. Defaults to None.
            offload_buffers (bool | None): Whether to automatically offload model weights to the CPU. Defaults to None.
            offload_folder (str | os.PathLike | None): Path to the folder to offload weights to when `offload_state_dict=True`. Defaults to None.
            offload_state_dict (bool | None):  Whether to offload the state dict to the CPU or disk depending on `offload_folder`. Defaults to None.
            output_loading_info (bool | None): Whether to also return additional information about the model loading. Defaults to None.
            proxies (dict[str, str] | None): A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.`. The proxies are used on each request. Defaults to None.
            quantization_config (QuantizationConfigMixin | dict | None): The quantization configuration. Defaults to None.
            revision (str | None): Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git. Defaults to None.
            state_dict (dict[str, str] | None): A state dictionary to use instead of loading the state dict from the model file. Defaults to None.
            sub_folder (str | None): In case the relevant files are located inside a sub directory of the model repo on huggingface.co, you can specify the folder name here. Defaults to None.
            token (str | None): An authentication token for private repositories on huggingface.co. Defaults to None.
            torch_dtype (str | torch.dtype | None): `torch.dtype` or string that can be converted to a `torch.dtype`. Defaults to None.
            trust_remote_code (bool | None): Whether or not to allow for loading user-provided code contained in the downloaded model. Defaults to None.
            use_safetensors (bool | None): If True, will try to load the safetensors version of the weights and configuration files if both are available (otherwise it defaults to loading the `pytorch_model.bin` weights). If False, will try to load the standard weights. Defaults to None.
            variant (str | None): Model variant to use. Defaults to None.
            _fast_init (bool | None): Whether or not to disable fast initialization. Defaults to None.
            model_type_kwargs (ModelTypeKwargs | None):  Additional keyword arguments passed along to the specific model type. Defaults to None.

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


class HuggingFaceModel:
    """
    Wraps a Hugging Face model for easier initialization and access.

    Attributes:
        model (PreTrainedModel): The initialized Hugging Face model.

    :Usage:
        from transformers import AutoModel
        from PyGPTs.HuggingFace.Models import HuggingFaceModel, HuggingFaceModelSettings

        settings = HuggingFaceModelSettings(
            pretrained_model_name_or_path="gpt2", model_class=AutoModel
        )

        model = HuggingFaceModel(settings)

    """

    def __init__(self, model_settings: HuggingFaceModelSettings):
        """
        Initializes a new HuggingFaceModel instance.

        Args:
            model_settings (HuggingFaceModelSettings): The settings for the model.
        """
        self.model: PreTrainedModel = model_settings.model_class.from_pretrained(
            **{
                name: value
                for name, value in get_class_fields(model_settings).items()
                if value is not None and name != "model_class"
            }
        )
