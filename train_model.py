import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import transformers
from torch.optim.lr_scheduler import LambdaLR

from rag_models import make_rag_model
from data import get_dataloader


def run(args):
    model_type = args.model_type

    train_csv = args.train_csv
    valid_csv = args.valid_csv
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    device  = "cuda" if (args.device == "cuda") and torch.cuda.is_available() else "cpu"

    if model_type == "rag":
        tokenizer, model = make_rag_model(args)
        if args.train_type == "generator":
            params = model.generator.parameters()
        else:
            params = model.parameters()
    else:
        raise NotImplementedError(model_type)

    model = model.to(device)

    train_loader = get_dataloader(train_csv, tokenizer, batch_size, shuffle=True)
    if valid_csv is not None:
        valid_loader = get_dataloader(valid_csv, tokenizer, batch_size, shuffle=False)
        log_columns = ["Epoch", "train_loss", "valid_loss"]
    else:
        log_columns = ["Epoch", "train_loss"]
        
    optimizer = torch.optim.Adam(params, lr=args.lr0)
    optimizer.zero_grad()
    lr_scheduler = LambdaLR(optimizer, get_lr_func(num_epochs))

    logs = []
    for epoch in range(1, num_epochs + 1):
        losses = train_eval(model, train_loader, device, optimizer=optimizer, train=True)
        #  for batch in data_loader:
        lr_scheduler.step()
        if valid_loader is None:
            logs.append((epoch, np.mean(losses)))
            print("Epoch {} : loss {:.3f}".format(*logs[-1]))
        else:
            valid_losses = train_eval(model, valid_loader, device, train=False)
            logs.append((epoch, np.mean(losses), np.mean(valid_losses)))
            print("Epoch {} : train loss {:.3f} valid loss {:.3f}".format(*logs[-1]))

    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)

    log_df = pd.DataFrame(logs, columns=log_columns)
    log_df.set_index("Epoch").plot()
    plt.savefig(os.path.join(out_dir, "loss.jpg"))
    log_df.to_csv(os.path.join(out_dir, "log.csv"))

def get_lr_func(num_epochs):
  def lr_func(epoch):
    if epoch < num_epochs * 0.8:
      return 1.0
    else: 
      return 0.1
  return lr_func

def train_eval(model, loader, device, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()
    
    losses = []
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader):
            batch = {k:v.to(device) for k, v in batch.items()}
            result = model(**batch)
            loss = result[0].mean()
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.item())
    return losses

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str,choices=["rag"])
    parser.add_argument("--question-model", type=str, default=None)
    parser.add_argument("--indexdata-path", type=str, default=None)
    parser.add_argument("--index-path", type=str, default=None)

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--valid-csv", type=str, default=None)

    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--q-col", type=str, default="質問")
    parser.add_argument("--a-col", type=str, default="回答")

    parser.add_argument("--hidden-dim", type=int, default=786)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-type", type=str, default="generator")
    parser.add_argument("--lr0", type=float, default=1e-4)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    run(args)

if __name__ == "__main__":
    main()

