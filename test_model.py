import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import transformers
from transformers.retrieval_rag import CustomHFIndex
from torch.optim.lr_scheduler import LambdaLR

from rag_models import get_bart_from_bert, BartBertTokenizer
from data import get_dataloader
from rag_models import load_rag_model


def run(args):
    model_type = args.model_type

    valid_csv = args.test_csv
    out_dir = args.out_dir
    out_file = args.out_file
    os.makedirs(out_dir, exist_ok=True)
    
    batch_size = args.batch_size
    device  = "cuda" if (args.device == "cuda") and torch.cuda.is_available() else "cpu"

    if model_type == "rag":
        tokenizer, model = load_rag_model(args)
        out_columns = ["質問", "回答", "返答"] + ["関連{}".format(i) for i in range(1, 6)]
    else:
        raise NotImplementedError(model_type)
    model = model.to(device)
    
    # train_loader = get_dataloader(train_csv, tokenizer, batch_size, shuffle=True)
    if valid_csv is not None:
        valid_loader = get_dataloader(valid_csv, tokenizer, batch_size, shuffle=False)
    else:
        valid_loader  = None

    results = []
    for q, a in tqdm(valid_loader.dataset):
        inputs = {k:v.to(device) for k,v in tokenizer.prepare_seq2seq_batch([q], return_tensors="pt").items()}
        question_hidden_states = model.question_encoder(inputs["input_ids"])[0]
        docs_dict = model.retriever(inputs["input_ids"].cpu().numpy(), question_hidden_states.detach().cpu().numpy(), return_tensors="pt")
        doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2).to(device)).squeeze(1)
        result = model.generate(context_input_ids=docs_dict["context_input_ids"].to(device), 
                                context_attention_mask=docs_dict["context_attention_mask"].to(device),
                                doc_scores=doc_scores,
                                num_beams=5, max_length=50, decoder_start_token_id=2)
        ans = tokenizer.batch_decode(result.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        relates = model.retriever.index.get_doc_dicts(docs_dict["doc_ids"])[0]["text"]
        reply = [ans] +  relates            
        results.append([q, a] + reply)

    result_df = pd.DataFrame(results, columns=out_columns)
    result_df.to_csv(os.path.join(out_dir, out_file))



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str,choices=["rag"])
    parser.add_argument("--pretrained-model", type=str, default=None)
    parser.add_argument("--indexdata-path", type=str, default=None)
    parser.add_argument("--index-path", type=str, default=None)

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="test.csv")
    parser.add_argument("--test-csv", type=str, default=None)

    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--q-col", type=str, default="質問")
    parser.add_argument("--a-col", type=str, default="回答")

    parser.add_argument("--hidden-dim", type=int, default=786)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    run(args)

if __name__ == "__main__":
    main()

