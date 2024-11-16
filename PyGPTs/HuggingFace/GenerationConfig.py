import os

from PyVarTools.python_instances_tools import get_class_fields
from transformers import Constraint, GenerationConfig, WatermarkingConfig


class HuggingFaceGenerationTokensUsedSettings:
    """
    Settings for tokens used in Hugging Face text generation.

    Attributes:
        bos_token_id (int | None): The ID of the beginning-of-sequence token. Defaults to None.
        eos_token_id (int | list[int] | None): The ID of the end-of-sequence token. Defaults to None.
        pad_token_id (int | None): The ID of the padding token. Defaults to None.
    """

    def __init__(self, bos_token_id: int | None = None, eos_token_id: int | list[int] | None = None, pad_token_id: int | None = None):
        """
        Initializes HuggingFaceGenerationTokensUsedSettings with token IDs.

        Args:
            bos_token_id (int | None): The ID of the beginning-of-sequence token. Defaults to None.
            eos_token_id (int | list[int] | None): The ID of the end-of-sequence token. Defaults to None.
            pad_token_id (int | None): The ID of the padding token. Defaults to None.
        """
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id


class HuggingFaceGenerationStrategySettings:
    """
    Settings for the generation strategy in Hugging Face text generation.

    Attributes:
        do_sample (bool | None): Whether to use sampling. Defaults to None.
        num_beam_groups (int | None): Number of beam groups for beam search. Defaults to None.
        num_beams (int | None): Number of beams for beam search. Defaults to None.
        penalty_alpha (float | None): Penalty alpha for contrastive search. Defaults to None.
        use_cache (bool | None): Whether to use caching. Defaults to None.
    """

    def __init__(
        self,
        do_sample: bool | None = None,
        num_beam_groups: int | None = None,
        num_beams: int | None = None,
        penalty_alpha: float | None = None,
        use_cache: bool | None = None,
    ):
        """
        Initializes HuggingFaceGenerationStrategySettings with strategy parameters.

        Args:
            do_sample (bool | None): Whether to use sampling. Defaults to None.
            num_beam_groups (int | None): Number of beam groups for beam search. Defaults to None.
            num_beams (int | None): Number of beams for beam search. Defaults to None.
            penalty_alpha (float | None): Penalty alpha for contrastive search. Defaults to None.
            use_cache (bool | None): Whether to use caching. Defaults to None.
        """
        self.do_sample = do_sample
        self.num_beam_groups = num_beam_groups
        self.num_beams = num_beams
        self.penalty_alpha = penalty_alpha
        self.use_cache = use_cache


class HuggingFaceGenerationOutputVariablesSettings:
    """
    Settings for output variables in Hugging Face text generation.

    Attributes:
        num_return_sequences (int | None): The number of sequences to return. Defaults to None.
        output_attentions (bool | None): Whether to output attention weights. Defaults to None.
        output_hidden_states (bool | None): Whether to output hidden states. Defaults to None.
        output_scores (bool | None): Whether to output scores. Defaults to None.
        return_dict_in_generate (bool | None): Whether to return a dictionary. Defaults to None.
        output_logits (bool | None): Whether to output logits. Defaults to None.
    """

    def __init__(
        self,
        num_return_sequences: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_scores: bool | None = None,
        output_logits: bool | None = None,
        return_dict_in_generate: bool | None = None,
    ):
        """
        Initializes HuggingFaceGenerationOutputVariablesSettings with output parameters.

        Args:
            num_return_sequences (int | None): The number of sequences to return. Defaults to None.
            output_attentions (bool | None): Whether to output attention weights. Defaults to None.
            output_hidden_states (bool | None): Whether to output hidden states. Defaults to None.
            output_scores (bool | None): Whether to output scores. Defaults to None.
            return_dict_in_generate (bool | None): Whether to return a dictionary. Defaults to None.
            output_logits (bool | None): Whether to output logits. Defaults to None.
        """
        self.num_return_sequences = num_return_sequences
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.output_scores = output_scores
        self.output_logits = output_logits
        self.return_dict_in_generate = return_dict_in_generate


class HuggingFaceGenerationOutputSettings:
    """
    Settings for the output of Hugging Face text generation.

    Attributes:
        early_stopping (bool | None): Whether to stop generation early. Defaults to None.
        max_length (int | None): The maximum generation length. Defaults to None.
        max_new_tokens (int | None): The maximum number of new tokens to generate. Defaults to None.
        max_time (float | None): Maximum time allowed for generation. Defaults to None.
        min_length (int | None): Minimum generation length. Defaults to None.
        min_new_tokens (int | None):  Minimum number of new tokens to generate. Defaults to None.
        stop_strings (str | list[str] | None): String or list of strings to stop on. Defaults to None.
    """

    def __init__(
        self,
        early_stopping: bool | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        max_time: float | None = None,
        min_length: int | None = None,
        min_new_tokens: int | None = None,
        stop_strings: str | list[str] | None = None,
    ):
        """
        Initializes HuggingFaceGenerationOutputSettings with output constraints.

        Args:
            early_stopping (bool | None): Whether to stop generation early. Defaults to None.
            max_length (int | None): The maximum generation length. Defaults to None.
            max_new_tokens (int | None): The maximum number of new tokens to generate. Defaults to None.
            max_time (float | None): Maximum time allowed for generation. Defaults to None.
            min_length (int | None): Minimum generation length. Defaults to None.
            min_new_tokens (int | None):  Minimum number of new tokens to generate. Defaults to None.
            stop_strings (str | list[str] | None): String or list of strings to stop on. Defaults to None.
        """
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.max_time = max_time
        self.min_length = min_length
        self.min_new_tokens = min_new_tokens
        self.stop_strings = stop_strings


class HuggingFaceGenerationOutputLogitsSettings:
    """
    Settings for controlling the logits during text generation.

    Attributes:
        bad_words_ids (list[list[int]] | None): List of token ids that are not allowed to be generated. Defaults to None.
        begin_suppress_tokens (list[int] | None): List of token ids that should not be generated at the beginning of the sequence. Defaults to None.
        constraints (list[Constraint] | None): Custom constraints to use. Defaults to None.
        diversity_penalty (float | None): The diversity penalty to apply. Defaults to None.
        encoder_repetition_penalty (float | None):  The repetition penalty for the encoder. Defaults to None.
        epsilon_cutoff (float | None): The epsilon cutoff to use. Defaults to None.
        eta_cutoff (float | None): The eta cutoff to use. Defaults to None.
        exponential_decay_length_penalty (tuple[int, float] | None): The exponential decay length penalty to use. Defaults to None.
        force_words_ids (list[list[int]] | None): List of token ids that must be generated. Defaults to None.
        forced_bos_token_id (int | None): The id of the token to force as the beginning of the sequence. Defaults to None.
        forced_decoder_ids (list[list[int]] | None):  List of token ids to force the decoder to generate. Defaults to None.
        forced_eos_token_id (int | list[int] | None): The id of the token to force as the end of the sequence. Defaults to None.
        guidance_scale (float | None): The guidance scale to use. Defaults to None.
        length_penalty (float | None): The length penalty to use. Defaults to None.
        low_memory (bool | None): Whether to use low memory generation. Defaults to None.
        min_p (float | None): The minimal probability for generating a token. Defaults to None.
        no_repeat_ngram_size (int | None): The size of n-grams to avoid repeating. Defaults to None.
        remove_invalid_values (bool | None): Whether to remove invalid values. Defaults to None.
        renormalize_logits (bool | None):  Whether to renormalize logits. Defaults to None.
        repetition_penalty (float | None):  The repetition penalty to apply. Defaults to None.
        sequence_bias (dict[tuple[int], float] | None):  A dictionary mapping sequences of token ids to a bias value. Defaults to None.
        suppress_tokens (list[int] | None): A list of tokens or token ids to suppress. Defaults to None.
        temperature (float | None):  The temperature to use. Defaults to None.
        token_healing (bool | None): Whether to enable token healing. Defaults to None.
        top_k (int | None): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
        top_p (float | None): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to None.
        typical_p (float | None): The typical_p to use. Defaults to None.
        watermarking_config (WatermarkingConfig | dict | None): The watermarking configuration. Defaults to None.

    :Usage:
        from transformers import WatermarkingConfig
        from PyGPTs.HuggingFace.Generations import HuggingFaceGenerationOutputLogitsSettings
        watermarking_config = WatermarkingConfig() # Example config
        settings10 = HuggingFaceGenerationOutputLogitsSettings(watermarking_config=watermarking_config)

    """

    def __init__(
        self,
        bad_words_ids: list[list[int]] | None = None,
        begin_suppress_tokens: list[int] | None = None,
        constraints: list[Constraint] | None = None,
        diversity_penalty: float | None = None,
        encoder_repetition_penalty: float | None = None,
        epsilon_cutoff: float | None = None,
        eta_cutoff: float | None = None,
        exponential_decay_length_penalty: tuple[int, float] | None = None,
        force_words_ids: list[list[int]] | None = None,
        forced_bos_token_id: int | None = None,
        forced_decoder_ids: list[list[int]] | None = None,
        forced_eos_token_id: int | list[int] | None = None,
        guidance_scale: float | None = None,
        length_penalty: float | None = None,
        low_memory: bool | None = None,
        min_p: float | None = None,
        no_repeat_ngram_size: int | None = None,
        remove_invalid_values: bool | None = None,
        renormalize_logits: bool | None = None,
        repetition_penalty: float | None = None,
        sequence_bias: dict[tuple[int], float] | None = None,
        suppress_tokens: list[int] | None = None,
        temperature: float | None = None,
        token_healing: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        typical_p: float | None = None,
        watermarking_config: WatermarkingConfig | dict | None = None,
    ):
        """
        Initializes HuggingFaceGenerationOutputLogitsSettings.

        Args:
            bad_words_ids (list[list[int]] | None): List of token ids that are not allowed to be generated. Defaults to None.
            begin_suppress_tokens (list[int] | None): List of token ids that should not be generated at the beginning of the sequence. Defaults to None.
            constraints (list[Constraint] | None): Custom constraints to use. Defaults to None.
            diversity_penalty (float | None): The diversity penalty to apply. Defaults to None.
            encoder_repetition_penalty (float | None): The repetition penalty for the encoder. Defaults to None.
            epsilon_cutoff (float | None): The epsilon cutoff to use. Defaults to None.
            eta_cutoff (float | None): The eta cutoff to use. Defaults to None.
            exponential_decay_length_penalty (tuple[int, float] | None): The exponential decay length penalty to use. Defaults to None.
            force_words_ids (list[list[int]] | None): List of token ids that must be generated. Defaults to None.
            forced_bos_token_id (int | None):  The id of the token to force as the beginning of the sequence. Defaults to None.
            forced_decoder_ids (list[list[int]] | None): List of token ids to force the decoder to generate. Defaults to None.
            forced_eos_token_id (int | list[int] | None): The id of the token to force as the end of the sequence. Defaults to None.
            guidance_scale (float | None): The guidance scale to use. Defaults to None.
            length_penalty (float | None): The length penalty to apply during generation. Defaults to None.
            low_memory (bool | None): Whether to use low memory during generation. Defaults to None.
            min_p (float | None):  The minimal probability for generating a token. Defaults to None.
            no_repeat_ngram_size (int | None): The size of n-grams to avoid repeating. Defaults to None.
            remove_invalid_values (bool | None): Whether to remove invalid values during generation. Defaults to None.
            renormalize_logits (bool | None): Whether to renormalize the logits during generation. Defaults to None.
            repetition_penalty (float | None): The repetition penalty to apply during generation. Defaults to None.
            sequence_bias (dict[tuple[int], float] | None): A dictionary mapping sequences of token ids to a bias value. Defaults to None.
            suppress_tokens (list[int] | None):  A list of tokens or token ids to suppress. Defaults to None.
            temperature (float | None): The temperature to use during generation. Defaults to None.
            token_healing (bool | None):  Whether to enable token healing during generation. Defaults to None.
            top_k (int | None): The number of highest probability vocabulary tokens to keep for top-k-filtering during generation. Defaults to None.
            top_p (float | None): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to None.
            typical_p (float | None): The typical_p to use. Defaults to None.
            watermarking_config (WatermarkingConfig | dict | None): The watermarking configuration. Defaults to None.
        """
        self.bad_words_ids = bad_words_ids
        self.begin_suppress_tokens = begin_suppress_tokens
        self.constraints = constraints
        self.diversity_penalty = diversity_penalty
        self.encoder_repetition_penalty = encoder_repetition_penalty
        self.epsilon_cutoff = epsilon_cutoff
        self.eta_cutoff = eta_cutoff
        self.exponential_decay_length_penalty = exponential_decay_length_penalty
        self.force_words_ids = force_words_ids
        self.forced_bos_token_id = forced_bos_token_id
        self.forced_decoder_ids = forced_decoder_ids
        self.forced_eos_token_id = forced_eos_token_id
        self.guidance_scale = guidance_scale
        self.length_penalty = length_penalty
        self.low_memory = low_memory
        self.min_p = min_p
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.remove_invalid_values = remove_invalid_values
        self.renormalize_logits = renormalize_logits
        self.repetition_penalty = repetition_penalty
        self.sequence_bias = sequence_bias
        self.suppress_tokens = suppress_tokens
        self.temperature = temperature
        self.token_healing = token_healing
        self.top_k = top_k
        self.top_p = top_p
        self.typical_p = typical_p
        self.watermarking_config = watermarking_config


class HuggingFaceGenerationConfigSettings:
    """
    Stores settings for creating a Hugging Face GenerationConfig.

    Attributes:
        pretrained_model_name_or_path (str | os.PathLike):  The pretrained model name or path.
        config_file_name (str | os.PathLike | None): The config file name. Defaults to None.
        cache_dir (str | os.PathLike | None):  The cache directory. Defaults to None.
        force_download (bool | None): Whether to force download. Defaults to None.
        proxies (dict[str, str] | None): The proxies to use. Defaults to None.
        return_unused_kwargs (bool | None): Whether to return unused kwargs. Defaults to None.
        sub_folder (str | None): The subfolder to use. Defaults to None.
        token (str | bool | None): The token to use or path to the credentials file. Defaults to None.

    :Usage:
        from PyGPTs.HuggingFace.Generations import HuggingFaceGenerationConfigSettings

        settings = HuggingFaceGenerationConfigSettings(pretrained_model_name_or_path="gpt2")

    """

    def __init__(
        self,
        pretrained_model_name_or_path: str | os.PathLike,
        config_file_name: str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool | None = None,
        proxies: dict[str, str] | None = None,
        token: str | bool | None = None,
        return_unused_kwargs: bool | None = None,
        sub_folder: str | None = None,
        generation_output_logits_settings: HuggingFaceGenerationOutputLogitsSettings | None = None,
        generation_output_settings: HuggingFaceGenerationOutputSettings | None = None,
        generation_output_variables_settings: HuggingFaceGenerationOutputVariablesSettings | None = None,
        generation_strategy_settings: HuggingFaceGenerationStrategySettings | None = None,
        generation_tokens_used_settings: HuggingFaceGenerationTokensUsedSettings | None = None,
    ):
        """
        Initializes HuggingFaceGenerationConfigSettings.

        Args:
            pretrained_model_name_or_path (str | os.PathLike): Path to pretrained model or model identifier from huggingface.co/models.
            config_file_name (str | os.PathLike | None): The config file name. Defaults to None.
            cache_dir (str | os.PathLike | None):  Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used. Defaults to None.
            force_download (bool | None): Whether to force the (re-)download the model weights and configuration files and override the cached versions if they exist. Defaults to None.
            proxies (dict[str, str] | None): A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.`. The proxies are used on each request. Defaults to None.
            token (str | bool | None): An authentication token (See https://huggingface.co/settings/token) for private repositories on huggingface.co or, if `local_files_only=True`, path to the credentials file. If `token=True`, the token will be retrieved from the cache. Defaults to None.
            return_unused_kwargs (bool | None): Whether or not to return unused keyword arguments. Defaults to None.
            sub_folder (str | None):  In case the relevant files are located inside a sub directory of the model repo on huggingface.co, you can specify the folder name here. Defaults to None.
            generation_output_logits_settings (HuggingFaceGenerationOutputLogitsSettings | None): Settings for the logits. Defaults to None.
            generation_output_settings (HuggingFaceGenerationOutputSettings | None): Settings for the output. Defaults to None.
            generation_output_variables_settings (HuggingFaceGenerationOutputVariablesSettings | None):  Settings for the variables. Defaults to None.
            generation_strategy_settings (HuggingFaceGenerationStrategySettings | None): Settings for the strategy. Defaults to None.
            generation_tokens_used_settings (HuggingFaceGenerationTokensUsedSettings | None):  Settings for the tokens used. Defaults to None.
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.config_file_name = config_file_name
        self.cache_dir = cache_dir
        self.force_download = force_download
        self.proxies = proxies
        self.token = token
        self.return_unused_kwargs = return_unused_kwargs
        self.sub_folder = sub_folder

        if generation_output_logits_settings is not None:
            for name, value in get_class_fields(generation_output_logits_settings).items():
                if value is not None:
                    setattr(self, name, value)

        if generation_output_settings is not None:
            for name, value in get_class_fields(generation_output_settings).items():
                if value is not None:
                    setattr(self, name, value)

        if generation_output_variables_settings is not None:
            for name, value in get_class_fields(generation_output_variables_settings).items():
                if value is not None:
                    setattr(self, name, value)

        if generation_strategy_settings is not None:
            for name, value in get_class_fields(generation_strategy_settings).items():
                if value is not None:
                    setattr(self, name, value)

        if generation_tokens_used_settings is not None:
            for name, value in get_class_fields(generation_tokens_used_settings).items():
                if value is not None:
                    setattr(self, name, value)


class HuggingFaceGenerationConfig:
    """
    Wraps a Hugging Face GenerationConfig for easier initialization.


    Attributes:
        generation_config (GenerationConfig): The initialized Hugging Face GenerationConfig.

    :Usage:
        from transformers import GenerationConfig
        from PyGPTs.HuggingFace.Generations import HuggingFaceGenerationConfig, HuggingFaceGenerationConfigSettings

        settings = HuggingFaceGenerationConfigSettings(pretrained_model_name_or_path='gpt2')
        generation_config = HuggingFaceGenerationConfig(settings)
    """

    def __init__(self, generation_config_settings: HuggingFaceGenerationConfigSettings):
        """
        Initializes a new HuggingFaceGenerationConfig instance.

        Args:
            generation_config_settings (HuggingFaceGenerationConfigSettings): The settings to use for initializing the
                 GenerationConfig.
        """
        self.generation_config = GenerationConfig.from_pretrained(
            **{name: value for name, value in get_class_fields(generation_config_settings).items() if value is not None}
        )
