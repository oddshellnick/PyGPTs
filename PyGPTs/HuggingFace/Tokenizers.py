import os
import torch
from PyVarTools.python_instances_tools import get_class_fields
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizer

from PyGPTs.HuggingFace.base_objects import ObjectTypeKwargs


class TokenizerTypeKwargs(ObjectTypeKwargs):
    """
    Keyword arguments for tokenizer instantiation. Extends ObjectTypeKwargs.
    """

    def __init__(self, **kwargs):
        """Initializes TokenizerTypeKwargs with given keyword arguments."""
        super().__init__(**kwargs)


class HuggingFaceTokenizerSettings:
    """
    Stores settings for initializing a Hugging Face tokenizer.

    Attributes:
        pretrained_model_name_or_path (str | PathLike): The path or name of the pretrained model.
        cache_dir (str | PathLike | None): Path to a directory in which a downloaded pretrained model configuration
             should be cached if the standard cache should not be used. Defaults to None.
        config (PretrainedConfig | None): An instance of a configuration object to use instead of loading the
             configuration from the pretrained model configuration file. Defaults to None.
        force_download (bool | None): Whether to force the (re-)download the model weights and configuration files and
             override the cached versions if they exist. Defaults to None.
        proxies (dict[str, str] | None): A dictionary of proxy servers to use by protocol or endpoint, e.g.,
             `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.`. The proxies are used on each request. Defaults to None.
        sub_folder (str | None): In case the relevant files are located inside a sub directory of the model repo on
             huggingface.co, you can specify the folder name here. Defaults to None.
        token (str | None):  An authentication token for private repositories on huggingface.co. Defaults to None.
        tokenizer_type (str | None):  The tokenizer type. Defaults to None.
        torch_dtype (torch.dtype | None): The torch datatype. Defaults to None.
        trust_remote_code (bool | None): Whether to allow loading user-provided code contained in the downloaded model. Defaults to None.
        use_fast (bool | None): Whether to use the fast tokenizer. Defaults to None.

    :Usage:
        settings = HuggingFaceTokenizerSettings(
            pretrained_model_name_or_path="gpt2", tokenizer_type="gpt2"
        )
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        config: PretrainedConfig | None = None,
        force_download: bool | None = None,
        proxies: dict[str, str] | None = None,
        sub_folder: str | None = None,
        tokenizer_type: str | None = None,
        torch_dtype: str | torch.dtype | None = None,
        trust_remote_code: bool | None = None,
        token: str | None = None,
        use_fast: bool | None = None,
        tokenizer_type_kwargs: TokenizerTypeKwargs | None = None,
    ):
        """
        Initializes HuggingFaceTokenizerSettings with the provided parameters.

        Args:
            pretrained_model_name_or_path (str | os.PathLike): Path to pretrained model or model identifier from huggingface.co/models.
            cache_dir (str | os.PathLike | None): Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used. Defaults to None.
            config (PretrainedConfig | None): An instance of a configuration object to use instead of loading the configuration from the pretrained model configuration file. Defaults to None.
            force_download (bool | None): Whether to force the (re-)download the model weights and configuration files and override the cached versions if they exist. Defaults to None.
            proxies (dict[str, str] | None):  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request. Defaults to None.
            sub_folder (str | None): In case the relevant files are located inside a sub directory of the model repo on huggingface.co, you can specify the folder name here. Defaults to None.
            token (str | None): An authentication token (See https://huggingface.co/settings/token) for private repositories on huggingface.co. Defaults to None.
            tokenizer_type (str | None): The tokenizer type to use. Defaults to None.
            torch_dtype (str | torch.dtype | None): torch.dtype or string that can be converted to a torch.dtype. Defaults to None.
            trust_remote_code (bool | None): Whether or not to allow for loading user-provided code contained in the downloaded model. Defaults to None.
            use_fast (bool | None): Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. Defaults to None.
            tokenizer_type_kwargs (TokenizerTypeKwargs | None): Additional keyword arguments passed along to the specific tokenizer type. Defaults to None.
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


class HuggingFaceTokenizer:
    """
    Wraps a Hugging Face tokenizer for easier initialization and access.

    Attributes:
        tokenizer (PreTrainedTokenizer): The initialized Hugging Face tokenizer.

    :Usage:
        from PyGPTs.HuggingFace.Tokenizers import HuggingFaceTokenizer, HuggingFaceTokenizerSettings

        settings = HuggingFaceTokenizerSettings(pretrained_model_name_or_path="gpt2")
        tokenizer = HuggingFaceTokenizer(settings)

        tokens = tokenizer.tokenizer("Hello, world!")
    """

    def __init__(self, tokenizer_settings: HuggingFaceTokenizerSettings):
        """
        Initializes a HuggingFaceTokenizer with the given settings.

        Args:
            tokenizer_settings (HuggingFaceTokenizerSettings): The settings for the tokenizer.
        """
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            **{name: value for name, value in get_class_fields(tokenizer_settings).items() if value is not None}
        )
