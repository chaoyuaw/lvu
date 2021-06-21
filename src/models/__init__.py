# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "2.3.0"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import logging

# from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
# from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, AutoConfig
# from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
# from .configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
# from .configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
# from .configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
# from .configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
# from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
# from .configuration_mmbt import MMBTConfig
# from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
# from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
# from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig

# # Configurations
# from .configuration_utils import PretrainedConfig
# from .configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
# from .configuration_xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
# from .configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig
# from .data import (
#     DataProcessor,
#     InputExample,
#     InputFeatures,
#     SingleSentenceClassificationProcessor,
#     SquadExample,
#     SquadFeatures,
#     SquadV1Processor,
#     SquadV2Processor,
#     glue_convert_examples_to_features,
#     glue_output_modes,
#     glue_processors,
#     glue_tasks_num_labels,
#     is_sklearn_available,
#     squad_convert_examples_to_features,
#     xnli_output_modes,
#     xnli_processors,
#     xnli_tasks_num_labels,
# )

# Files and general utilities
from .file_utils import (
#     CONFIG_NAME,
#     MODEL_CARD_NAME,
#     PYTORCH_PRETRAINED_BERT_CACHE,
#     PYTORCH_TRANSFORMERS_CACHE,
#     TF2_WEIGHTS_NAME,
#     TF_WEIGHTS_NAME,
#     TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
#     add_end_docstrings,
#     add_start_docstrings,
#     cached_path,
#     is_tf_available,
#     is_torch_available,
)

# Model Cards
# from .modelcard import ModelCard


# Pipelines
# from .pipelines import (
#     CsvPipelineDataFormat,
#     FeatureExtractionPipeline,
#     JsonPipelineDataFormat,
#     NerPipeline,
#     PipedPipelineDataFormat,
#     Pipeline,
#     PipelineDataFormat,
#     QuestionAnsweringPipeline,
#     TextClassificationPipeline,
#     pipeline,
# )
# from .tokenization_albert import AlbertTokenizer
# from .tokenization_auto import AutoTokenizer
# from .tokenization_bert import BasicTokenizer, BertTokenizer, BertTokenizerFast, WordpieceTokenizer
# from .tokenization_bert_japanese import BertJapaneseTokenizer, CharacterTokenizer, MecabTokenizer
# from .tokenization_camembert import CamembertTokenizer
# from .tokenization_ctrl import CTRLTokenizer
# from .tokenization_distilbert import DistilBertTokenizer
# from .tokenization_flaubert import FlaubertTokenizer
# from .tokenization_gpt2 import GPT2Tokenizer, GPT2TokenizerFast
# from .tokenization_openai import OpenAIGPTTokenizer
from .tokenization_roberta import RobertaTokenizer
# from .tokenization_t5 import T5Tokenizer
# from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer

# Tokenizers
from .tokenization_utils import PreTrainedTokenizer
# from .tokenization_xlm import XLMTokenizer
# from .tokenization_xlm_roberta import XLMRobertaTokenizer
# from .tokenization_xlnet import SPIECE_UNDERLINE, XLNetTokenizer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# if is_sklearn_available():
#     from .data import glue_compute_metrics, xnli_compute_metrics


# Modeling
# if is_torch_available():
from .modeling_utils import PreTrainedModel
# , prune_layer, Conv1D
    # from .modeling_auto import (
    #     AutoModel,
    #     AutoModelForPreTraining,
    #     AutoModelForSequenceClassification,
    #     AutoModelForQuestionAnswering,
    #     AutoModelWithLMHead,
    #     AutoModelForTokenClassification,
    #     ALL_PRETRAINED_MODEL_ARCHIVE_MAP,
    # )

from .modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertForPreTraining,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForSequenceClassification,
    BertForMultipleChoice,
    BertForTokenClassification,
    BertForQuestionAnswering,
    # load_tf_weights_in_bert,
    # BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
)
from .modeling_roberta import (
    RobertaForMaskedLM,
    RobertaModel,
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
)
# from .modeling_encoder_decoder import PreTrainedEncoderDecoder, Model2Model

# Optimization
from .optimization import (
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


