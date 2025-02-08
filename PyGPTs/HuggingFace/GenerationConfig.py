import os
import typing
from PyVarTools.python_instances_tools import get_class_fields
from transformers import (
	Constraint,
	GenerationConfig,
	WatermarkingConfig
)


class HF_GenerationTokensUsedSettings:
	"""
	Settings for tokens used in Hugging Face text generation.

	Attributes:
		bos_token_id (typing.Optional[int]): The ID of the beginning-of-sequence token. Defaults to None.
		eos_token_id (typing.Optional[typing.Union[int, list[int]]]): The ID of the end-of-sequence token. Defaults to None.
		pad_token_id (typing.Optional[int]): The ID of the padding token. Defaults to None.
	"""
	
	def __init__(
			self,
			bos_token_id: typing.Optional[int] = None,
			eos_token_id: typing.Optional[typing.Union[int, list[int]]] = None,
			pad_token_id: typing.Optional[int] = None
	):
		"""
		Initializes HF_GenerationTokensUsedSettings with token IDs.

		Args:
			bos_token_id (typing.Optional[int]): The ID of the beginning-of-sequence token. Defaults to None.
			eos_token_id (typing.Optional[typing.Union[int, list[int]]]): The ID of the end-of-sequence token. Defaults to None.
			pad_token_id (typing.Optional[int]): The ID of the padding token. Defaults to None.
		"""
		self.bos_token_id = bos_token_id
		self.eos_token_id = eos_token_id
		self.pad_token_id = pad_token_id


class HF_GenerationStrategySettings:
	"""
	Settings for the generation strategy in Hugging Face text generation.

	Attributes:
		do_sample (typing.Optional[bool]): Whether to use sampling. Defaults to None.
		num_beam_groups (typing.Optional[int]): Number of beam groups for beam search. Defaults to None.
		num_beams (typing.Optional[int]): Number of beams for beam search. Defaults to None.
		penalty_alpha (typing.Optional[float]): Penalty alpha for contrastive search. Defaults to None.
		use_cache (typing.Optional[bool]): Whether to use caching. Defaults to None.
	"""
	
	def __init__(
			self,
			do_sample: typing.Optional[bool] = None,
			num_beam_groups: typing.Optional[int] = None,
			num_beams: typing.Optional[int] = None,
			penalty_alpha: typing.Optional[float] = None,
			use_cache: typing.Optional[bool] = None,
	):
		"""
		Initializes HF_GenerationStrategySettings with strategy parameters.

		Args:
			do_sample (typing.Optional[bool]): Whether to use sampling. Defaults to None.
			num_beam_groups (typing.Optional[int]): Number of beam groups for beam search. Defaults to None.
			num_beams (typing.Optional[int]): Number of beams for beam search. Defaults to None.
			penalty_alpha (typing.Optional[float]): Penalty alpha for contrastive search. Defaults to None.
			use_cache (typing.Optional[bool]): Whether to use caching. Defaults to None.
		"""
		self.do_sample = do_sample
		self.num_beam_groups = num_beam_groups
		self.num_beams = num_beams
		self.penalty_alpha = penalty_alpha
		self.use_cache = use_cache


class HF_GenerationOutputVariablesSettings:
	"""
	Settings for output variables in Hugging Face text generation.

	Attributes:
		num_return_sequences (typing.Optional[int]): The number of sequences to return. Defaults to None.
		output_attentions (typing.Optional[bool]): Whether to output attention weights. Defaults to None.
		output_hidden_states (typing.Optional[bool]): Whether to output hidden states. Defaults to None.
		output_scores (typing.Optional[bool]): Whether to output scores. Defaults to None.
		return_dict_in_generate (typing.Optional[bool]): Whether to return a dictionary. Defaults to None.
		output_logits (typing.Optional[bool]): Whether to output logits. Defaults to None.
	"""
	
	def __init__(
			self,
			num_return_sequences: typing.Optional[int] = None,
			output_attentions: typing.Optional[bool] = None,
			output_hidden_states: typing.Optional[bool] = None,
			output_scores: typing.Optional[bool] = None,
			output_logits: typing.Optional[bool] = None,
			return_dict_in_generate: typing.Optional[bool] = None,
	):
		"""
		Initializes HF_GenerationOutputVariablesSettings with output parameters.

		Args:
			num_return_sequences (typing.Optional[int]): The number of sequences to return. Defaults to None.
			output_attentions (typing.Optional[bool]): Whether to output attention weights. Defaults to None.
			output_hidden_states (typing.Optional[bool]): Whether to output hidden states. Defaults to None.
			output_scores (typing.Optional[bool]): Whether to output scores. Defaults to None.
			return_dict_in_generate (typing.Optional[bool]): Whether to return a dictionary. Defaults to None.
			output_logits (typing.Optional[bool]): Whether to output logits. Defaults to None.
		"""
		self.num_return_sequences = num_return_sequences
		self.output_attentions = output_attentions
		self.output_hidden_states = output_hidden_states
		self.output_scores = output_scores
		self.output_logits = output_logits
		self.return_dict_in_generate = return_dict_in_generate


class HF_GenerationOutputSettings:
	"""
	Settings for the output of Hugging Face text generation.

	Attributes:
		early_stopping (typing.Optional[bool]): Whether to stop generation early. Defaults to None.
		max_length (typing.Optional[int]): The maximum generation length. Defaults to None.
		max_new_tokens (typing.Optional[int]): The maximum number of new tokens to generate. Defaults to None.
		max_time (typing.Optional[float]): Maximum time allowed for generation. Defaults to None.
		min_length (typing.Optional[int]): Minimum generation length. Defaults to None.
		min_new_tokens (typing.Optional[int]): Minimum number of new tokens to generate. Defaults to None.
		stop_strings (typing.Optional[typing.Union[str, list[str]]]): String or list of strings to stop on. Defaults to None.
	"""
	
	def __init__(
			self,
			early_stopping: typing.Optional[bool] = None,
			max_length: typing.Optional[int] = None,
			max_new_tokens: typing.Optional[int] = None,
			max_time: typing.Optional[float] = None,
			min_length: typing.Optional[int] = None,
			min_new_tokens: typing.Optional[int] = None,
			stop_strings: typing.Optional[typing.Union[str, list[str]]] = None,
	):
		"""
		Initializes HF_GenerationOutputSettings with output constraints.

		Args:
			early_stopping (typing.Optional[bool]): Whether to stop generation early. Defaults to None.
			max_length (typing.Optional[int]): The maximum generation length. Defaults to None.
			max_new_tokens (typing.Optional[int]): The maximum number of new tokens to generate. Defaults to None.
			max_time (typing.Optional[float]): Maximum time allowed for generation. Defaults to None.
			min_length (typing.Optional[int]): Minimum generation length. Defaults to None.
			min_new_tokens (typing.Optional[int]): Minimum number of new tokens to generate. Defaults to None.
			stop_strings (typing.Optional[typing.Union[str, list[str]]]): String or list of strings to stop on. Defaults to None.
		"""
		self.early_stopping = early_stopping
		self.max_length = max_length
		self.max_new_tokens = max_new_tokens
		self.max_time = max_time
		self.min_length = min_length
		self.min_new_tokens = min_new_tokens
		self.stop_strings = stop_strings


class HF_GenerationOutputLogitsSettings:
	"""
	Settings for controlling the logits during text generation.

	Attributes:
		bad_words_ids (typing.Optional[list[list[int]]]): List of token ids that are not allowed to be generated. Defaults to None.
		begin_suppress_tokens (typing.Optional[list[int]]): List of token ids that should not be generated at the beginning of the sequence. Defaults to None.
		constraints (typing.Optional[list[Constraint]]): Custom constraints to use. Defaults to None.
		diversity_penalty (typing.Optional[float]): The diversity penalty to apply. Defaults to None.
		encoder_repetition_penalty (typing.Optional[float]): The repetition penalty for the encoder. Defaults to None.
		epsilon_cutoff (typing.Optional[float]): The epsilon cutoff to use. Defaults to None.
		eta_cutoff (typing.Optional[float]): The eta cutoff to use. Defaults to None.
		exponential_decay_length_penalty (typing.Optional[tuple[int, float]]): The exponential decay length penalty to use. Defaults to None.
		force_words_ids (typing.Optional[list[list[int]]]): List of token ids that must be generated. Defaults to None.
		forced_bos_token_id (typing.Optional[int]): The id of the token to force as the beginning of the sequence. Defaults to None.
		forced_decoder_ids (typing.Optional[list[list[int]]]): List of token ids to force the decoder to generate. Defaults to None.
		forced_eos_token_id (typing.Optional[typing.Union[int, list[int]]]): The id of the token to force as the end of the sequence. Defaults to None.
		guidance_scale (typing.Optional[float]): The guidance scale to use. Defaults to None.
		length_penalty (typing.Optional[float]): The length penalty to use. Defaults to None.
		low_memory (typing.Optional[bool]): Whether to use low memory generation. Defaults to None.
		min_p (typing.Optional[float]): The minimal probability for generating a token. Defaults to None.
		no_repeat_ngram_size (typing.Optional[int]): The size of n-grams to avoid repeating. Defaults to None.
		remove_invalid_values (typing.Optional[bool]): Whether to remove invalid values. Defaults to None.
		renormalize_logits (typing.Optional[bool]): Whether to renormalize logits. Defaults to None.
		repetition_penalty (typing.Optional[float]): The repetition penalty to apply. Defaults to None.
		sequence_bias (typing.Optional[dict[tuple[int], float]]): A dictionary mapping sequences of token ids to a bias value. Defaults to None.
		suppress_tokens (typing.Optional[list[int]]): A list of tokens or token ids to suppress. Defaults to None.
		temperature (typing.Optional[float]): The temperature to use. Defaults to None.
		token_healing (typing.Optional[bool]): Whether to enable token healing. Defaults to None.
		top_k (typing.Optional[int]): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
		top_p (typing.Optional[float]): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to None.
		typical_p (typing.Optional[float]): The typical_p to use. Defaults to None.
		watermarking_config (typing.Optional[typing.Union[WatermarkingConfig, dict]]): The watermarking configuration. Defaults to None.

	:Usage:
		from transformers import WatermarkingConfig
		from PyGPTs.HuggingFace.Generations import HF_GenerationOutputLogitsSettings
		watermarking_config = WatermarkingConfig() # Example config
		settings10 = HF_GenerationOutputLogitsSettings(watermarking_config=watermarking_config)

	"""
	
	def __init__(
			self,
			bad_words_ids: typing.Optional[list[list[int]]] = None,
			begin_suppress_tokens: typing.Optional[list[int]] = None,
			constraints: typing.Optional[list[Constraint]] = None,
			diversity_penalty: typing.Optional[float] = None,
			encoder_repetition_penalty: typing.Optional[float] = None,
			epsilon_cutoff: typing.Optional[float] = None,
			eta_cutoff: typing.Optional[float] = None,
			exponential_decay_length_penalty: typing.Optional[tuple[int, float]] = None,
			force_words_ids: typing.Optional[list[list[int]]] = None,
			forced_bos_token_id: typing.Optional[int] = None,
			forced_decoder_ids: typing.Optional[list[list[int]]] = None,
			forced_eos_token_id: typing.Optional[typing.Union[int, list[int]]] = None,
			guidance_scale: typing.Optional[float] = None,
			length_penalty: typing.Optional[float] = None,
			low_memory: typing.Optional[bool] = None,
			min_p: typing.Optional[float] = None,
			no_repeat_ngram_size: typing.Optional[int] = None,
			remove_invalid_values: typing.Optional[bool] = None,
			renormalize_logits: typing.Optional[bool] = None,
			repetition_penalty: typing.Optional[float] = None,
			sequence_bias: typing.Optional[dict[tuple[int], float]] = None,
			suppress_tokens: typing.Optional[list[int]] = None,
			temperature: typing.Optional[float] = None,
			token_healing: typing.Optional[bool] = None,
			top_k: typing.Optional[int] = None,
			top_p: typing.Optional[float] = None,
			typical_p: typing.Optional[float] = None,
			watermarking_config: typing.Optional[typing.Union[WatermarkingConfig, dict]] = None,
	):
		"""
		Initializes HF_GenerationOutputLogitsSettings.

		Args:
			bad_words_ids (typing.Optional[list[list[int]]]): List of token ids that are not allowed to be generated. Defaults to None.
			begin_suppress_tokens (typing.Optional[list[int]]): List of token ids that should not be generated at the beginning of the sequence. Defaults to None.
			constraints (typing.Optional[list[Constraint]]): Custom constraints to use. Defaults to None.
			diversity_penalty (typing.Optional[float]): The diversity penalty to apply. Defaults to None.
			encoder_repetition_penalty (typing.Optional[float]): The repetition penalty for the encoder. Defaults to None.
			epsilon_cutoff (typing.Optional[float]): The epsilon cutoff to use. Defaults to None.
			eta_cutoff (typing.Optional[float]): The eta cutoff to use. Defaults to None.
			exponential_decay_length_penalty (typing.Optional[tuple[int, float]]): The exponential decay length penalty to use. Defaults to None.
			force_words_ids (typing.Optional[list[list[int]]]): List of token ids that must be generated. Defaults to None.
			forced_bos_token_id (typing.Optional[int]): The id of the token to force as the beginning of the sequence. Defaults to None.
			forced_decoder_ids (typing.Optional[list[list[int]]]): List of token ids to force the decoder to generate. Defaults to None.
			forced_eos_token_id (typing.Optional[typing.Union[int, list[int]]]): The id of the token to force as the end of the sequence. Defaults to None.
			guidance_scale (typing.Optional[float]): The guidance scale to use. Defaults to None.
			length_penalty (typing.Optional[float]): The length penalty to apply during generation. Defaults to None.
			low_memory (typing.Optional[bool]): Whether to use low memory during generation. Defaults to None.
			min_p (typing.Optional[float]): The minimal probability for generating a token. Defaults to None.
			no_repeat_ngram_size (typing.Optional[int]): The size of n-grams to avoid repeating. Defaults to None.
			remove_invalid_values (typing.Optional[bool]): Whether to remove invalid values during generation. Defaults to None.
			renormalize_logits (typing.Optional[bool]): Whether to renormalize the logits during generation. Defaults to None.
			repetition_penalty (typing.Optional[float]): The repetition penalty to apply during generation. Defaults to None.
			sequence_bias (typing.Optional[dict[tuple[int], float]]): A dictionary mapping sequences of token ids to a bias value. Defaults to None.
			suppress_tokens (typing.Optional[list[int]]): A list of tokens or token ids to suppress. Defaults to None.
			temperature (typing.Optional[float]): The temperature to use during generation. Defaults to None.
			token_healing (typing.Optional[bool]): Whether to enable token healing during generation. Defaults to None.
			top_k (typing.Optional[int]): The number of highest probability vocabulary tokens to keep for top-k-filtering during generation. Defaults to None.
			top_p (typing.Optional[float]): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to None.
			typical_p (typing.Optional[float]): The typical_p to use. Defaults to None.
			watermarking_config (typing.Optional[typing.Union[WatermarkingConfig, dict]]): The watermarking configuration. Defaults to None.
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


class HF_GenerationConfigSettings:
	"""
	Stores settings for creating a Hugging Face GenerationConfig.

	Attributes:
		pretrained_model_name_or_path (typing.Union[str, os.PathLike]): The pretrained model name or path.
		config_file_name (typing.Optional[typing.Union[str, os.PathLike]]): The config file name. Defaults to None.
		cache_dir (typing.Optional[typing.Union[str, os.PathLike]]): The cache directory. Defaults to None.
		force_download (typing.Optional[bool]): Whether to force download. Defaults to None.
		proxies (typing.Optional[dict[str, str]]): The proxies to use. Defaults to None.
		return_unused_kwargs (typing.Optional[bool]): Whether to return unused kwargs. Defaults to None.
		sub_folder (typing.Optional[str]): The subfolder to use. Defaults to None.
		token (typing.Optional[typing.Union[str, bool]]): The token to use or path to the credentials file. Defaults to None.

	:Usage:
		from PyGPTs.HuggingFace.Generations import HF_GenerationConfigSettings

		settings = HF_GenerationConfigSettings(pretrained_model_name_or_path="gpt2")

	"""
	
	def __init__(
			self,
			pretrained_model_name_or_path: typing.Union[str, os.PathLike],
			config_file_name: typing.Optional[typing.Union[str, os.PathLike]] = None,
			cache_dir: typing.Optional[typing.Union[str, os.PathLike]] = None,
			force_download: typing.Optional[bool] = None,
			proxies: typing.Optional[dict[str, str]] = None,
			token: typing.Optional[typing.Union[str, bool]] = None,
			return_unused_kwargs: typing.Optional[bool] = None,
			sub_folder: typing.Optional[str] = None,
			generation_output_logits_settings: typing.Optional[HF_GenerationOutputLogitsSettings] = None,
			generation_output_settings: typing.Optional[HF_GenerationOutputSettings] = None,
			generation_output_variables_settings: typing.Optional[HF_GenerationOutputVariablesSettings] = None,
			generation_strategy_settings: typing.Optional[HF_GenerationStrategySettings] = None,
			generation_tokens_used_settings: typing.Optional[HF_GenerationTokensUsedSettings] = None,
	):
		"""
		Initializes HF_GenerationConfigSettings.

		Args:
			pretrained_model_name_or_path (typing.Union[str, os.PathLike]): Path to pretrained model or model identifier from huggingface.co/models.
			config_file_name (typing.Optional[typing.Union[str, os.PathLike]]): The config file name. Defaults to None.
			cache_dir (typing.Optional[typing.Union[str, os.PathLike]]): Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used. Defaults to None.
			force_download (typing.Optional[bool]): Whether to force the (re-)download the model weights and configuration files and override the cached versions if they exist. Defaults to None.
			proxies (typing.Optional[dict[str, str]]): A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.`. The proxies are used on each request. Defaults to None.
			token (typing.Optional[typing.Union[str, bool]]): An authentication token (See https://huggingface.co/settings/token) for private repositories on huggingface.co or, if `local_files_only=True`, path to the credentials file. If `token=True`, the token will be retrieved from the cache. Defaults to None.
			return_unused_kwargs (typing.Optional[bool]): Whether or not to return unused keyword arguments. Defaults to None.
			sub_folder (typing.Optional[str]): In case the relevant files are located inside a sub directory of the model repo on huggingface.co, you can specify the folder name here. Defaults to None.
			generation_output_logits_settings (typing.Optional[HF_GenerationOutputLogitsSettings]): Settings for the logits. Defaults to None.
			generation_output_settings (typing.Optional[HF_GenerationOutputSettings]): Settings for the output. Defaults to None.
			generation_output_variables_settings (typing.Optional[HF_GenerationOutputVariablesSettings]): Settings for the variables. Defaults to None.
			generation_strategy_settings (typing.Optional[HF_GenerationStrategySettings]): Settings for the strategy. Defaults to None.
			generation_tokens_used_settings (typing.Optional[HF_GenerationTokensUsedSettings]): Settings for the tokens used. Defaults to None.
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


class HF_GenerationConfig:
	"""
	Wraps a Hugging Face GenerationConfig for easier initialization.


	Attributes:
		generation_config (GenerationConfig): The initialized Hugging Face GenerationConfig.

	:Usage:
		from transformers import GenerationConfig
		from PyGPTs.HuggingFace.Generations import HF_GenerationConfig, HF_GenerationConfigSettings

		settings = HF_GenerationConfigSettings(pretrained_model_name_or_path='gpt2')
		generation_config = HF_GenerationConfig(settings)
	"""
	
	def __init__(self, generation_config_settings: HF_GenerationConfigSettings):
		"""
		Initializes a new HF_GenerationConfig instance.

		Args:
			generation_config_settings (HF_GenerationConfigSettings): The settings to use for initializing the
				 GenerationConfig.
		"""
		self.generation_config = GenerationConfig.from_pretrained(
				**{
					name: value
					for name, value in get_class_fields(generation_config_settings).items()
					if value is not None
				}
		)
