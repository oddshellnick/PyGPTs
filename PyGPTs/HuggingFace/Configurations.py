import os
import typing

from PyVarTools.python_instances_tools import get_class_fields
from transformers import PretrainedConfig


class HuggingFaceTensorFlowSpecificConfigSettings:
    """
    TensorFlow-specific settings for Hugging Face configurations.

    Attributes:
        tf_legacy_loss (bool | None): Defaults to None.
        use_bfloat16 (bool | None): Defaults to None.
    """

    def __init__(self, tf_legacy_loss: bool | None = None, use_bfloat16: bool | None = None):
        """
        Initializes HuggingFaceTensorFlowSpecificConfigSettings.

        Args:
            tf_legacy_loss (bool | None): Whether or not to use the legacy loss function. If set to `False` (the default), uses the new loss function. Defaults to None.
            use_bfloat16 (bool | None): Whether to use bfloat16. Defaults to None.
        """
        self.tf_legacy_loss = tf_legacy_loss
        self.use_bfloat16 = use_bfloat16


class HuggingFacePyTorchSpecificConfigSettings:
    """
    PyTorch-specific settings for Hugging Face configurations.

    Attributes:
        tie_word_embeddings (bool | None): Defaults to None.
        torchscript (bool | None): Defaults to None.
        torch_dtype (str | None): Defaults to None.
    """

    def __init__(
        self, tie_word_embeddings: bool | None = None, torchscript: bool | None = None, torch_dtype: str | None = None
    ):
        """
        Initializes HuggingFacePyTorchSpecificConfigSettings.

        Args:
            tie_word_embeddings (bool | None): Whether to tie the input and output embeddings. Defaults to None.
            torchscript (bool | None): Whether to generate torchscript. Defaults to None.
            torch_dtype (str | None): The torch dtype to use. Defaults to None.
        """
        self.tie_word_embeddings = tie_word_embeddings
        self.torchscript = torchscript
        self.torch_dtype = torch_dtype


class HuggingFaceConfigSettings:
    """
    Stores settings for creating a Hugging Face configuration.

    Attributes:
        pretrained_model_name_or_path (str | os.PathLike): The pretrained model name or path.
        add_cross_attention (bool | None): Whether to add cross-attention. Defaults to None.
        architectures (list[str] | None): The model architectures. Defaults to None.
        bad_words_ids (list[int] | None):  A list of bad word IDs. Defaults to None.
        chunk_size_feed_forward (int | None): The chunk size of the feed-forward layer. Defaults to None.
        cross_attention_hidden_size (bool | None): The size of the cross-attention hidden state. Defaults to None.
        diversity_penalty (float | None): The diversity penalty. Defaults to None.
        do_sample (bool | None): Whether to perform sampling during generation. Defaults to None.
        early_stopping (bool | None): Whether to stop generation early. Defaults to None.
        encoder_no_repeat_ngram_size (int | None): The n-gram size to avoid repetition in the encoder. Defaults to None.
        finetuning_task (str | None): The finetuning task. Defaults to None.
        forced_bos_token_id (int | None): The forced beginning-of-sentence token ID. Defaults to None.
        forced_eos_token_id (int | list[int] | None): The forced end-of-sentence token ID. Defaults to None.
        id2label (dict[int, str] | None): Mapping from label IDs to labels. Defaults to None.
        is_decoder (bool | None): Whether the model is a decoder. Defaults to None.
        is_encoder_decoder (bool | None): Whether the model is an encoder-decoder. Defaults to None.
        label2id (dict[str, int] | None): Mapping from labels to label IDs. Defaults to None.
        length_penalty (float | None): The length penalty. Defaults to None.
        max_length (int | None): The maximum length of the generated sequence. Defaults to None.
        min_length (int | None): The minimum length of the generated sequence. Defaults to None.
        no_repeat_ngram_size (int | None):  The n-gram size to avoid repetition. Defaults to None.
        num_beam_groups (int | None): The number of beam groups. Defaults to None.
        num_beams (int | None): The number of beams. Defaults to None.
        num_labels (int | None): The number of labels. Defaults to None.
        num_return_sequences (int | None): The number of sequences to return. Defaults to None.
        output_attentions (bool | None): Whether to output attention weights. Defaults to None.
        output_hidden_states (bool | None): Whether to output hidden states. Defaults to None.
        output_scores (bool | None): Whether to output scores. Defaults to None.
        problem_type (str | None): The type of problem to solve. Defaults to None.
        prune_heads (dict[int, list[int]] | None):  The heads to prune. Defaults to None.
        remove_invalid_values (bool | None): Whether to remove invalid values. Defaults to None.
        repetition_penalty (float | None): The repetition penalty. Defaults to None.
        return_dict (bool | None): Whether to return a dictionary. Defaults to None.
        return_dict_in_generate (bool | None): Whether to return a dictionary during generation. Defaults to None.
        task_specific_params (dict[str, typing.Any] | None): Task-specific parameters. Defaults to None.
        temperature (float | None): The temperature for sampling. Defaults to None.
        tie_encoder_decoder (bool | None): Whether to tie the encoder and decoder. Defaults to None.
        top_k (int | None): The top-k value for sampling. Defaults to None.
        top_p (float | None): The top-p value for sampling. Defaults to None.
        typical_p (float | None): The typical p value for sampling. Defaults to None.

    :Usage:
        from transformers import AutoConfig
        from PyGPTs.HuggingFace import HuggingFaceConfigSettings

        config = AutoConfig.from_pretrained('gpt2')  # Example config
        settings = HuggingFaceConfigSettings(
            pretrained_model_name_or_path="gpt2", config=config
        )

    """

    def __init__(
        self,
        pretrained_model_name_or_path: str | os.PathLike,
        add_cross_attention: bool | None = None,
        architectures: list[str] | None = None,
        bad_words_ids: list[int] | None = None,
        chunk_size_feed_forward: int | None = None,
        cross_attention_hidden_size: bool | None = None,
        diversity_penalty: float | None = None,
        do_sample: bool | None = None,
        early_stopping: bool | None = None,
        encoder_no_repeat_ngram_size: int | None = None,
        finetuning_task: str | None = None,
        forced_bos_token_id: int | None = None,
        forced_eos_token_id: int | list[int] | None = None,
        id2label: dict[int, str] | None = None,
        is_decoder: bool | None = None,
        is_encoder_decoder: bool | None = None,
        label2id: dict[str, int] | None = None,
        length_penalty: float | None = None,
        max_length: int | None = None,
        min_length: int | None = None,
        no_repeat_ngram_size: int | None = None,
        num_beam_groups: int | None = None,
        num_beams: int | None = None,
        num_labels: int | None = None,
        num_return_sequences: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_scores: bool | None = None,
        problem_type: str | None = None,
        prune_heads: dict[int, list[int]] | None = None,
        remove_invalid_values: bool | None = None,
        repetition_penalty: float | None = None,
        return_dict: bool | None = None,
        return_dict_in_generate: bool | None = None,
        task_specific_params: dict[str, typing.Any] | None = None,
        temperature: float | None = None,
        tie_encoder_decoder: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        typical_p: float | None = None,
        specific_config_settings: HuggingFacePyTorchSpecificConfigSettings
        | HuggingFaceTensorFlowSpecificConfigSettings
        | None = None,
    ):
        """
        Initializes HuggingFaceConfigSettings with provided parameters.

        Args:
            pretrained_model_name_or_path (str | os.PathLike): Path to pretrained model or model identifier from huggingface.co/models.
            add_cross_attention (bool | None): Whether to add cross-attention. Defaults to None.
            architectures (list[str] | None):  Model architectures. Defaults to None.
            bad_words_ids (list[int] | None): List of token ids that are not allowed to be generated. Defaults to None.
            chunk_size_feed_forward (int | None): The chunk size of feed forward blocks. Defaults to None.
            cross_attention_hidden_size (bool | None):  The hidden size of the cross-attention layer. Defaults to None.
            diversity_penalty (float | None): The diversity penalty to apply during beam search. Defaults to None.
            do_sample (bool | None): Whether or not to use sampling ; if set to `False` greedy decoding is used. Defaults to None.
            early_stopping (bool | None): Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not. Defaults to None.
            encoder_no_repeat_ngram_size (int | None): Value of n-gram size for first repetition control in the encoder. Defaults to None.
            finetuning_task (str | None): Name of the task used for finetuning if trained for a task other than language modeling. Defaults to None.
            forced_bos_token_id (int | None): The id of the token to force as the first generated token after the decoder_start_token_id. Defaults to None.
            forced_eos_token_id (int | list[int] | None): The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a list to set multiple *end-of-sequence* tokens when multiple sequences are generated. Defaults to None.
            id2label (dict[int, str] | None):  Dictionary mapping an assigned label id to the corresponding label. Defaults to None.
            is_decoder (bool | None): Whether or not the model is used as decoder. Defaults to None.
            is_encoder_decoder (bool | None): Whether the model is used as an encoder/decoder or not. Defaults to None.
            label2id (dict[str, int] | None): Dictionary mapping a label to the corresponding assigned label id. Defaults to None.
            length_penalty (float | None): Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to generate longer sequences. Defaults to None.
            max_length (int | None): The maximum length of the sequence to be generated. Defaults to None.
            min_length (int | None): The minimum length of the sequence to be generated. Defaults to None.
            no_repeat_ngram_size (int | None): If set to int > 0, all ngrams of that size can only appear once. Defaults to None.
            num_beam_groups (int | None):  Number of groups to divide `num_beams` into in order to use DIVERSE BEAM SEARCH. See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details. Defaults to None.
            num_beams (int | None): Number of beams for beam search. 1 means no beam search. Defaults to None.
            num_labels (int | None): The number of labels to use in the classification and tagging tasks. Defaults to None.
            num_return_sequences (int | None): The number of independently computed returned sequences for each element in the batch. Defaults to None.
            output_attentions (bool | None): Whether the model should return attentions weights. Defaults to None.
            output_hidden_states (bool | None): Whether the model should return hidden states. Defaults to None.
            output_scores (bool | None):  Whether or not the model should return the log probabilities. Defaults to None.
            problem_type (str | None):  Problem type to use when running multiple choice classification. Defaults to None.
            prune_heads (dict[int, list[int]] | None): Dictionary containing heads to prune in each layer. Defaults to None.
            remove_invalid_values (bool | None): Whether or not to remove any values from the predicted logits. Defaults to None.
            repetition_penalty (float | None): The penalty applied to repeated n-grams. 1.0 means no penalty. Defaults to None.
            return_dict (bool | None): Whether or not the model should return a `ModelOutput` instead of a plain tuple. Defaults to None.
            return_dict_in_generate (bool | None): Whether or not to return a `ModelOutput` instead of a plain tuple. Defaults to None.
            task_specific_params (dict[str, typing.Any] | None): A dictionary of parameters to be passed to the task. Defaults to None.
            temperature (float | None): The value used to module the next token probabilities. Defaults to None.
            tie_encoder_decoder (bool | None): Whether or not to tie the weights of the encoder and decoder of the transformer. Defaults to None.
            top_k (int | None): The  number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
            top_p (float | None):  If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation. Defaults to None.
            typical_p (float | None): Typical Decoding mass. Defaults to None.
            specific_config_settings (HuggingFacePyTorchSpecificConfigSettings | HuggingFaceTensorFlowSpecificConfigSettings | None): PyTorch- or TensorFlow-specific configuration settings. Defaults to None.
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
            (HuggingFacePyTorchSpecificConfigSettings, HuggingFaceTensorFlowSpecificConfigSettings),
        ):
            for field, value in get_class_fields(specific_config_settings).items():
                if value is not None:
                    setattr(self, field, value)
        elif specific_config_settings is not None:
            raise ValueError(
                '"specific_config_settings" must be of type HuggingFacePyTorchSpecificConfigSettings or HuggingFaceTensorFlowSpecificConfigSettings'
            )


class HuggingFaceConfig:
    """
    Wraps a Hugging Face PretrainedConfig for easier initialization.

    Attributes:
        config (transformers.PretrainedConfig): The initialized Hugging Face config.

    :Usage:
        from transformers import AutoConfig
        from PyGPTs.HuggingFace import HuggingFaceConfig, HuggingFaceConfigSettings

        settings = HuggingFaceConfigSettings(
            pretrained_model_name_or_path="gpt2"
        )
        config = HuggingFaceConfig(settings)
    """

    def __init__(self, generation_config_settings: HuggingFaceConfigSettings):
        """
        Initializes HuggingFaceConfig with given settings.

        Args:
            generation_config_settings (HuggingFaceConfigSettings): Settings for creating the config.
        """
        self.config = PretrainedConfig.from_pretrained(
            **{name: value for name, value in get_class_fields(generation_config_settings).items() if value is not None}
        )
