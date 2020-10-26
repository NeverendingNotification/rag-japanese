import os
import argparse
from functools import partial

import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import BertTokenizer
import faiss

### reference
### https://github.com/huggingface/transformers/blob/master/examples/rag/use_own_knowledge_dataset.py

KNOWLEGE_DIR = "knowlege"
INDEX_FILE = "knowlege_index.faiss"

def embed(documents: dict, ctx_encoder, ctx_tokenizer, device) -> dict:
    """Compute the DPR embeddings of document passages"""
    if len(documents["title"]) == 0:
        input_ids = ctx_tokenizer(
            documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
        )["input_ids"]
    else:
        input_ids = ctx_tokenizer(
            documents["text"], truncation=True, padding="longest", return_tensors="pt"
        )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}

def run(args):
    csv_file = args.knowledge_file
    k_col = args.k_col
    title_col = args.title_col

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True) 
    passages_path = os.path.join(out_dir, KNOWLEGE_DIR) 
    index_path = os.path.join(out_dir, INDEX_FILE)

    context_model = args.context_model
    device  = "cuda" if (args.device == "cuda") and torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size
    hnsw_m = args.hnsw_m

    # load pretrained context model
    ctx_encoder = transformers.DPRContextEncoder.from_pretrained(context_model).to(device)
    ctx_tokenizer = transformers.tokenization_bert_japanese.BertJapaneseTokenizer.from_pretrained(context_model)
    embedding_dim = ctx_encoder.config.hidden_size

    # convert csv file to index
    dataset = load_dataset(
        "csv", data_files=[csv_file], split="train"
    )
    column_names = dataset.column_names[:]
    def set_text_title(doc):
        return {"text":doc[k_col], "title":doc[title_col] if title_col is not None else ""}
    dataset = dataset.map(set_text_title)
    dataset.remove_columns_(column_names)

    print(dataset[0])
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, device=device),
        batched=True,
        batch_size=batch_size
    )

    # save index fils
    dataset.save_to_disk(passages_path)
    index = faiss.IndexHNSWFlat(embedding_dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)    
    dataset.get_index("embeddings").save(index_path)    


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--context-model", type=str, required=True)
    parser.add_argument("--knowledge-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k-col", type=str, default="知識")
    parser.add_argument("--title-col", type=str, default=None)

    parser.add_argument("--hnsw-m", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    run(args)

if __name__ == "__main__":
    main()
    # from transformers.retrieval_rag import CustomHFIndex
    # import transformers
    # from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, RagConfig
    # from transformers import BartForConditionalGeneration


    # # question encoder
    # q_enc_path = "/home/naoki/Document/git/rag-japanese/models/dpr_transformers/q_encoder"

    # tokenizer = transformers.DPRQuestionEncoderTokenizer.from_pretrained(q_enc_path)
    # q_enc_config = transformers.DPRConfig.from_pretrained(q_enc_path)
    # q_encoder = transformers.DPRQuestionEncoder.from_pretrained(q_enc_path)

    # # generator
    # gen_config = transformers.BartConfig()
    # gen_config.encoder_layers = 1
    # gen_config.decoder_layers = 3
    # gen_config.d_model = 768
    # gen_config.vocab_size = tokenizer.vocab_size
    # gen_config.pad_token_id = q_enc_config.pad_token_id


    # # retriever 
    # data_path = "/home/naoki/Document/git/rag-japanese/data/dpr_knowlege_index/knowlege"
    # index_path = "/home/naoki/Document/git/rag-japanese/data/dpr_knowlege_index/knowlege_index.faiss"

    # config = RagConfig(question_encoder=q_enc_config.to_dict(), generator=gen_config.to_dict())
    # index = CustomHFIndex.load_from_disk(768, dataset_path=data_path, index_path=index_path)
    # retriever = RagRetriever(config, tokenizer, tokenizer, index=index)

    # import torch
    # q_text = "京都府の人口は100万人くらいですか？"

    # q_in = tokenizer.encode(q_text, return_tensors="pt")
    # with torch.no_grad():
    #     enc_out= q_encoder(q_in)
    # q_embs = enc_out.pooler_output
    # retrieved_doc_embeds, doc_ids, docs= retriever.retrieve(q_embs.cpu().numpy(), n_docs=5)
    # print(docs, doc_ids)
