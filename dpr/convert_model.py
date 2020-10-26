import argparse
import os

import transformers
import torch

from dense_retriever import add_encoder_params, add_tokenizer_params, add_cuda_params
from dpr.options import set_encoder_params_from_state
from dpr.utils.model_utils import load_states_from_checkpoint
from dpr.models import init_biencoder_components

PRETRAIN_MODEL_NAME = "pytorch_model.bin"

def run(args):
    # pretrained_model = "/home/naoki/Document/git/DPR/results/dpr_long/dpr_biencoder.5.379"
    # out_root = "models/dpr_long"
    pretrained_model = args.pretrained_model
    out_root = args.out_dir 

    saved_state = load_states_from_checkpoint(pretrained_model)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    out_q = os.path.join(out_root, "q_encoder")
    out_c = os.path.join(out_root, "c_encoder")
    os.makedirs(out_q, exist_ok=True)
    os.makedirs(out_c, exist_ok=True)

    tensorizer.tokenizer.save_pretrained(out_q)
    encoder.question_model.save_pretrained(out_q)

    tensorizer.tokenizer.save_pretrained(out_c)
    encoder.ctx_model.save_pretrained(out_c)

    q_save_dict = {k.replace("question_model", "question_encoder.bert_model"):v for k, v in saved_state.model_dict.items() if k.startswith("question_model")}
    c_save_dict = {k.replace("ctx_model", "ctx_encoder.bert_model"):v for k, v in saved_state.model_dict.items() if k.startswith("ctx_model")}
    torch.save(q_save_dict, os.path.join(out_q, PRETRAIN_MODEL_NAME))
    torch.save(c_save_dict, os.path.join(out_c, PRETRAIN_MODEL_NAME))



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--pretrained-model', type=str, required=True)
    parser.add_argument('-o', '--out-dir', type=str, required=True)

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    return parser.parse_args()

def main():
    args = get_args()
    run(args)

if __name__ == "__main__":
    main()
