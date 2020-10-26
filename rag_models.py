
from typing import List, Optional
import os

import transformers
from transformers import BertTokenizer, AutoTokenizer
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers import RagTokenForGeneration
from transformers.retrieval_rag import CustomHFIndex


class BartBertTokenizer(BertJapaneseTokenizer):
    r"""
    Construct a BART tokenizer.

    :class:`~transformers.BartTokenizer` is identical to :class:`~transformers.RobertaTokenizer` and adds a new
    :meth:`~transformers.BartTokenizer.prepare_seq2seq_batch`

    Refer to superclass :class:`~transformers.RobertaTokenizer` for usage examples and documentation rag_tokenizerng
    the initialization parameters and other methods.
    """
    # merges and vocab same as Roberta
#     max_model_input_sizes = {m: 1024 for m in _all_bart_models}
#     pretrained_vocab_files_map = {
#         "vocab_file": {m: vocab_url for m in _all_bart_models},
#         "merges_file": {m: merges_url for m in _all_bart_models},
#     }

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = "None",
        truncation=True,
        **kwargs,
    ) -> BatchEncoding:
        kwargs.pop("src_lang", None)
        kwargs.pop("tgt_lang", None)
        if max_length is None:
            max_length = self.model_max_length
        model_inputs: BatchEncoding = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        if tgt_texts is None:
            return model_inputs
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        labels = self(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=truncation,
            **kwargs,
        )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

def get_bart_from_bert(bert_path, num_encoder_layers=1, num_decoder_layers=4):

    bart_tokenizer = BartBertTokenizer.from_pretrained(bert_path)
    bert_config = transformers.BertConfig.from_pretrained(bert_path)
    bart_config = transformers.BartConfig()

    bart_config.d_model = bert_config.hidden_size    
    bart_config.encoder_layers= num_encoder_layers
    bart_config.decoder_layers = num_decoder_layers
    bart_config.vocab_size = bart_tokenizer.vocab_size

    bart_model = transformers.BartForConditionalGeneration(bart_config)

    return bart_tokenizer, bart_model

def make_bart_model(args):
    tokenzier, model = get_bart_from_bert(args.bert_path)
    return tokenzier, model

def make_rag_model(args):
    hidden_dim = args.hidden_dim
    q_enc_path = args.question_model
    data_path = args.indexdata_path
    index_path = args.index_path

    q_tokenizer = transformers.tokenization_bert_japanese.BertJapaneseTokenizer.from_pretrained(q_enc_path)
    q_enc_config = transformers.DPRConfig.from_pretrained(q_enc_path)
    q_encoder = transformers.DPRQuestionEncoder.from_pretrained(q_enc_path)

    g_tokenizer, generator = get_bart_from_bert(q_enc_path)

    index = CustomHFIndex.load_from_disk(hidden_dim, dataset_path=data_path, index_path=index_path)
    config = transformers.RagConfig(question_encoder=q_enc_config.to_dict(), generator=generator.config.to_dict())
    retriever = transformers.RagRetriever(config, q_tokenizer, g_tokenizer, index=index)


    rag_tokenizer = transformers.RagTokenizer(q_tokenizer, g_tokenizer)
    model = transformers.RagTokenForGeneration(config, question_encoder=q_encoder, generator=generator, retriever=retriever)
    
    return rag_tokenizer, model

def load_rag_model(args):
    data_path = args.indexdata_path
    index_path = args.index_path

    pretrained = args.pretrained_model

    # tokenizer = transformers.RagTokenizer.from_pretrained(pretrained)
    config = transformers.RagConfig.from_pretrained(pretrained)

    q_tokenizer = transformers.tokenization_bert_japanese.BertJapaneseTokenizer.from_pretrained(os.path.join(pretrained, "question_encoder_tokenizer"), config=config.question_encoder)
    g_tokenizer = transformers.tokenization_bert_japanese.BertJapaneseTokenizer.from_pretrained(os.path.join(pretrained, "generator_tokenizer"), config=config.generator)

    tokenizer = transformers.RagTokenizer(q_tokenizer, g_tokenizer)
    model = transformers.RagTokenForGeneration.from_pretrained(pretrained)

    hidden_dim = model.config.retrieval_vector_size


    index = CustomHFIndex.load_from_disk(hidden_dim, dataset_path=data_path, index_path=index_path)
    retriever = transformers.RagRetriever(model.config, tokenizer.question_encoder, tokenizer.generator, index=index)
    model.set_retriever(retriever)
    
    return tokenizer, model
