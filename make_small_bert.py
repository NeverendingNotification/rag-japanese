import os
import copy

import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer

def run(pretrained_model, out_dir, num_layers=3):
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = BertModel.from_pretrained(pretrained_model, return_dict=True)

    small_config = copy.deepcopy(model.config)
    small_config.num_hidden_layers = num_layers
    small_model = BertModel(small_config)
    small_model.load_state_dict(model.state_dict(), strict=False)

    tokenizer.save_pretrained(out_dir)
    small_model.save_pretrained(out_dir)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrained-model", type=str, required=True)
    parser.add_argument("-o", "--out-dir", type=str, required=True)
    parser.add_argument("-n", "--num-layers", type=int, default=3)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    run(args.pretrained_model, args.out_dir,num_layers=args.num_layers)

if __name__ == "__main__":
    main()
