import os
import typing
from transformers import PretrainedConfig
from PyVarTools.python_instances_tools import get_class_fields


class HF_TensorFlowSpecificConfigSettings:
	"""
	TensorFlow-specific settings for Hugging Face configurations.

	Attributes:
		tf_legacy_loss (typing.Optional[bool]): Defaults to None.
		use_bfloat16 (typing.Optional[bool]): Defaults to None.
	"""
	
	def __init__(
			self,
			tf_legacy_loss: typing.Optional[bool] = None,
			use_bfloat16: typing.Optional[bool] = None
	):
		"""
		Initializes HF_TensorFlowSpecificConfigSettings.

		Args:
			tf_legacy_loss (typing.Optional[bool]): Whether or not to use the legacy loss function. If set to `False` (the default), uses the new loss function. Defaults to None.
			use_bfloat16 (typing.Optional[bool]): Whether to use bfloat16. Defaults to None.
		"""
		self.tf_legacy_loss = tf_legacy_loss
		self.use_bfloat16 = use_bfloat16


class HF_PyTorchSpecificConfigSettings:
	"""
	PyTorch-specific settings for Hugging Face configurations.

	Attributes:
		tie_word_embeddings (typing.Optional[bool]): Defaults to None.
		torchscript (typing.Optional[bool]): Defaults to None.
		torch_dtype (typing.Optional[str]): Defaults to None.
	"""
	
	def __init__(
			self,
			tie_word_embeddings: typing.Optional[bool] = None,
			torchscript: typing.Optional[bool] = None,
			torch_dtype: typing.Optional[str] = None
	):
		"""
		Initializes HF_PyTorchSpecificConfigSettings.

		Args:
			tie_word_embeddings (typing.Optional[bool]): Whether to tie the input and output embeddings. Defaults to None.
			torchscript (typing.Optional[bool]): Whether to generate torchscript. Defaults to None.
			torch_dtype (typing.Optional[str]): The torch dtype to use. Defaults to None.
		"""
		self.tie_word_embeddings = tie_word_embeddings
		self.torchscript = torchscript
		self.torch_dtype = torch_dtype


class HF_ConfigSettings:
	"""
	Stores settings for creating a Hugging Face configuration.

	Attributes:
		pretrained_model_name_or_path (typing.Union[str, os.PathLike]): The pretrained model name or path.
		add_cross_attention (typing.Optional[bool]): Whether to add cross-attention. Defaults to None.
		architectures (typing.Optional[list[str]]): The model architectures. Defaults to None.
		bad_words_ids (typing.Optional[list[int]]): A list of bad word IDs. Defaults to None.
		chunk_size_feed_forward (typing.Optional[int]): The chunk size of the feed-forward layer. Defaults to None.
		cross_attention_hidden_size (typing.Optional[bool]): The size of the cross-attention hidden state. Defaults to None.
		diversity_penalty (typing.Optional[float]): The diversity penalty. Defaults to None.
		do_sample (typing.Optional[bool]): Whether to perform sampling during generation. Defaults to None.
		early_stopping (typing.Optional[bool]): Whether to stop generation early. Defaults to None.
		encoder_no_repeat_ngram_size (typing.Optional[int]): The n-gram size to avoid repetition in the encoder. Defaults to None.
		finetuning_task (typing.Optional[str]): The finetuning task. Defaults to None.
		forced_bos_token_id (typing.Optional[int]): The forced beginning-of-sentence token ID. Defaults to None.
		forced_eos_token_id (typing.Optional[typing.Union[int, list[int]]]): The forced end-of-sentence token ID. Defaults to None.
		id2label (typing.Optional[dict[int, str]]): Mapping from label IDs to labels. Defaults to None.
		is_decoder (typing.Optional[bool]): Whether the model is a decoder. Defaults to None.
		is_encoder_decoder (typing.Optional[bool]): Whether the model is an encoder-decoder. Defaults to None.
		label2id (typing.Optional[dict[str, int]]): Mapping from labels to label IDs. Defaults to None.
		length_penalty (typing.Optional[float]): The length penalty. Defaults to None.
		max_length (typing.Optional[int]): The maximum length of the generated sequence. Defaults to None.
		min_length (typing.Optional[int]): The minimum length of the generated sequence. Defaults to None.
		no_repeat_ngram_size (typing.Optional[int]): The n-gram size to avoid repetition. Defaults to None.
		num_beam_groups (typing.Optional[int]): The number of beam groups. Defaults to None.
		num_beams (typing.Optional[int]): The number of beams. Defaults to None.
		num_labels (typing.Optional[int]): The number of labels. Defaults to None.
		num_return_sequences (typing.Optional[int]): The number of sequences to return. Defaults to None.
		output_attentions (typing.Optional[bool]): Whether to output attention weights. Defaults to None.
		output_hidden_states (typing.Optional[bool]): Whether to output hidden states. Defaults to None.
		output_scores (typing.Optional[bool]): Whether to output scores. Defaults to None.
		problem_type (typing.Optional[str]): The type of problem to solve. Defaults to None.
		prune_heads (typing.Optional[dict[int, list[int]]]): The heads to prune. Defaults to None.
		remove_invalid_values (typing.Optional[bool]): Whether to remove invalid values. Defaults to None.
		repetition_penalty (typing.Optional[float]): The repetition penalty. Defaults to None.
		return_dict (typing.Optional[bool]): Whether to return a dictionary. Defaults to None.
		return_dict_in_generate (typing.Optional[bool]): Whether to return a dictionary during generation. Defaults to None.
		task_specific_params (typing.Optional[dict[str, typing.Any]]): Task-specific parameters. Defaults to None.
		temperature (typing.Optional[float]): The temperature for sampling. Defaults to None.
		tie_encoder_decoder (typing.Optional[bool]): Whether to tie the encoder and decoder. Defaults to None.
		top_k (typing.Optional[int]): The top-k value for sampling. Defaults to None.
		top_p (typing.Optional[float]): The top-p value for sampling. Defaults to None.
		typical_p (typing.Optional[float]): The typical p value for sampling. Defaults to None.
	"""
	
	def __init__(
			self,
			pretrained_model_name_or_path: typing.Union[str, os.PathLike],
			add_cross_attention: typing.Optional[bool] = None,
			architectures: typing.Optional[list[str]] = None,
			bad_words_ids: typing.Optional[list[int]] = None,
			chunk_size_feed_forward: typing.Optional[int] = None,
			cross_attention_hidden_size: typing.Optional[bool] = None,
			diversity_penalty: typing.Optional[float] = None,
			do_sample: typing.Optional[bool] = None,
			early_stopping: typing.Optional[bool] = None,
			encoder_no_repeat_ngram_size: typing.Optional[int] = None,
			finetuning_task: typing.Optional[str] = None,
			forced_bos_token_id: typing.Optional[int] = None,
			forced_eos_token_id: typing.Optional[typing.Union[int, list[int]]] = None,
			id2label: typing.Optional[dict[int, str]] = None,
			is_decoder: typing.Optional[bool] = None,
			is_encoder_decoder: typing.Optional[bool] = None,
			label2id: typing.Optional[dict[str, int]] = None,
			length_penalty: typing.Optional[float] = None,
			max_length: typing.Optional[int] = None,
			min_length: typing.Optional[int] = None,
			no_repeat_ngram_size: typing.Optional[int] = None,
			num_beam_groups: typing.Optional[int] = None,
			num_beams: typing.Optional[int] = None,
			num_labels: typing.Optional[int] = None,
			num_return_sequences: typing.Optional[int] = None,
			output_attentions: typing.Optional[bool] = None,
			output_hidden_states: typing.Optional[bool] = None,
			output_scores: typing.Optional[bool] = None,
			problem_type: typing.Optional[str] = None,
			prune_heads: typing.Optional[dict[int, list[int]]] = None,
			remove_invalid_values: typing.Optional[bool] = None,
			repetition_penalty: typing.Optional[float] = None,
			return_dict: typing.Optional[bool] = None,
			return_dict_in_generate: typing.Optional[bool] = None,
			task_specific_params: typing.Optional[dict[str, typing.Any]] = None,
			temperature: typing.Optional[float] = None,
			tie_encoder_decoder: typing.Optional[bool] = None,
			top_k: typing.Optional[int] = None,
			top_p: typing.Optional[float] = None,
			typical_p: typing.Optional[float] = None,
			specific_config_settings: typing.Optional[
				typing.Union[HF_PyTorchSpecificConfigSettings, HF_TensorFlowSpecificConfigSettings]
			] = None,
	):
		"""
		Initializes HF_ConfigSettings with provided parameters.

		Args:
			pretrained_model_name_or_path (typing.Union[str, os.PathLike]): Path to pretrained model or model identifier from huggingface.co/models.
			add_cross_attention (typing.Optional[bool]): Whether to add cross-attention. Defaults to None.
			architectures (typing.Optional[list[str]]): Model architectures. Defaults to None.
			bad_words_ids (typing.Optional[list[int]]): List of token ids that are not allowed to be generated. Defaults to None.
			chunk_size_feed_forward (typing.Optional[int]): The chunk size of feed forward blocks. Defaults to None.
			cross_attention_hidden_size (typing.Optional[bool]): The hidden size of the cross-attention layer. Defaults to None.
			diversity_penalty (typing.Optional[float]): The diversity penalty to apply during beam search. Defaults to None.
			do_sample (typing.Optional[bool]): Whether or not to use sampling ; if set to `False` greedy decoding is used. Defaults to None.
			early_stopping (typing.Optional[bool]): Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not. Defaults to None.
			encoder_no_repeat_ngram_size (typing.Optional[int]): Value of n-gram size for first repetition control in the encoder. Defaults to None.
			finetuning_task (typing.Optional[str]): Name of the task used for finetuning if trained for a task other than language modeling. Defaults to None.
			forced_bos_token_id (typing.Optional[int]): The id of the token to force as the first generated token after the decoder_start_token_id. Defaults to None.
			forced_eos_token_id (typing.Optional[typing.Union[int, list[int]]]): The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a list to set multiple *end-of-sequence* tokens when multiple sequences are generated. Defaults to None.
			id2label (typing.Optional[dict[int, str]]): Dictionary mapping an assigned label id to the corresponding label. Defaults to None.
			is_decoder (typing.Optional[bool]): Whether or not the model is used as decoder. Defaults to None.
			is_encoder_decoder (typing.Optional[bool]): Whether the model is used as an encoder/decoder or not. Defaults to None.
			label2id (typing.Optional[dict[str, int]]): Dictionary mapping a label to the corresponding assigned label id. Defaults to None.
			length_penalty (typing.Optional[float]): Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to generate longer sequences. Defaults to None.
			max_length (typing.Optional[int]): The maximum length of the sequence to be generated. Defaults to None.
			min_length (typing.Optional[int]): The minimum length of the sequence to be generated. Defaults to None.
			no_repeat_ngram_size (typing.Optional[int]): If set to int > 0, all ngrams of that size can only appear once. Defaults to None.
			num_beam_groups (typing.Optional[int]): Number of groups to divide `num_beams` into in order to use DIVERSE BEAM SEARCH. See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details. Defaults to None.
			num_beams (typing.Optional[int]): Number of beams for beam search. 1 means no beam search. Defaults to None.
			num_labels (typing.Optional[int]): The number of labels to use in the classification and tagging tasks. Defaults to None.
			num_return_sequences (typing.Optional[int]): The number of independently computed returned sequences for each element in the batch. Defaults to None.
			output_attentions (typing.Optional[bool]): Whether the model should return attentions weights. Defaults to None.
			output_hidden_states (typing.Optional[bool]): Whether the model should return hidden states. Defaults to None.
			output_scores (typing.Optional[bool]): Whether or not the model should return the log probabilities. Defaults to None.
			problem_type (typing.Optional[str]): Problem type to use when running multiple choice classification. Defaults to None.
			prune_heads (typing.Optional[dict[int, list[int]]]): Dictionary containing heads to prune in each layer. Defaults to None.
			remove_invalid_values (typing.Optional[bool]): Whether or not to remove any values from the predicted logits. Defaults to None.
			repetition_penalty (typing.Optional[float]): The penalty applied to repeated n-grams. 1.0 means no penalty. Defaults to None.
			return_dict (typing.Optional[bool]): Whether or not the model should return a `ModelOutput` instead of a plain tuple. Defaults to None.
			return_dict_in_generate (typing.Optional[bool]): Whether or not to return a `ModelOutput` instead of a plain tuple. Defaults to None.
			task_specific_params (typing.Optional[dict[str, typing.Any]]): A dictionary of parameters to be passed to the task. Defaults to None.
			temperature (typing.Optional[float]): The value used to module the next token probabilities. Defaults to None.
			tie_encoder_decoder (typing.Optional[bool]): Whether or not to tie the weights of the encoder and decoder of the transformer. Defaults to None.
			top_k (typing.Optional[int]): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
			top_p (typing.Optional[float]): If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation. Defaults to None.
			typical_p (typing.Optional[float]): Typical Decoding mass. Defaults to None.
			specific_config_settings (typing.Optional[typing.Union[HF_PyTorchSpecificConfigSettings, HF_TensorFlowSpecificConfigSettings]]): PyTorch- or TensorFlow-specific configuration settings. Defaults to None.
		"""
		self.pretrained_model_name_or_path = pretrained_model_name_or_path
		self.add_cross_attention = add_cross_attention
		self.architectures = architectures
		self.bad_words_ids = bad_words_ids
		self.chunk_size_feed_forward = chunk_size_feed_forward
		self.cross_attention_hidden_size = cross_attention_hidden_size
		self.diversity_penalty = diversity_penalty
		self.do_sample = do_sample
		self.early_stopping = early_stopping
		self.encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size
		self.finetuning_task = finetuning_task
		self.forced_bos_token_id = forced_bos_token_id
		self.forced_eos_token_id = forced_eos_token_id
		self.id2label = id2label
		self.is_decoder = is_decoder
		self.is_encoder_decoder = is_encoder_decoder
		self.label2id = label2id
		self.length_penalty = length_penalty
		self.max_length = max_length
		self.min_length = min_length
		self.no_repeat_ngram_size = no_repeat_ngram_size
		self.num_beam_groups = num_beam_groups
		self.num_beams = num_beams
		self.num_labels = num_labels
		self.num_return_sequences = num_return_sequences
		self.output_attentions = output_attentions
		self.output_hidden_states = output_hidden_states
		self.output_scores = output_scores
		self.problem_type = problem_type
		self.prune_heads = prune_heads
		self.remove_invalid_values = remove_invalid_values
		self.repetition_penalty = repetition_penalty
		self.return_dict = return_dict
		self.return_dict_in_generate = return_dict_in_generate
		self.task_specific_params = task_specific_params
		self.temperature = temperature
		self.tie_encoder_decoder = tie_encoder_decoder
		self.top_k = top_k
		self.top_p = top_p
		self.typical_p = typical_p
		
		if isinstance(
				specific_config_settings,
				(HF_PyTorchSpecificConfigSettings, HF_TensorFlowSpecificConfigSettings),
		):
			for field, value in get_class_fields(specific_config_settings).items():
				if value is not None:
					setattr(self, field, value)
		elif specific_config_settings is not None:
			raise ValueError(
					'"specific_config_settings" must be of type HF_PyTorchSpecificConfigSettings or HF_TensorFlowSpecificConfigSettings'
			)


class HF_Config:
	"""
	Wraps a Hugging Face PretrainedConfig for easier initialization.

	Attributes:
		config (transformers.PretrainedConfig): The initialized Hugging Face config.

	:Usage:
		from transformers import AutoConfig
		from PyGPTs.HuggingFace import HF_Config, HF_ConfigSettings

		settings = HF_ConfigSettings(
			pretrained_model_name_or_path="gpt2"
		)
		config = HF_Config(settings)
	"""
	
	def __init__(self, generation_config_settings: HF_ConfigSettings):
		"""
		Initializes HF_Config with given settings.

		Args:
			generation_config_settings (HF_ConfigSettings): Settings for creating the config.
		"""
		self.config = PretrainedConfig.from_pretrained(
				**{
					name: value
					for name, value in get_class_fields(generation_config_settings).items()
					if value is not None
				}
		)
