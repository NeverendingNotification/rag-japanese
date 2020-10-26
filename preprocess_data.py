import os
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


Q_OUT_COL = "question"
A_OUT_COL = "answers"

def make_json(know_df, qa_df, q_col="質問", a_col="回答", k_col="知識", title_col=None):
  txts = []
  use_title = title_col is not None
  for _, row in qa_df.iterrows():
    result = {}
    result[Q_OUT_COL] = row[q_col]
    result[A_OUT_COL] = [row[a_col]]
    result["positive_ctxs"] = [
      {"title":know_df.loc[i][k_col] if use_title else "", "text":know_df.loc[i][k_col]} for i in eval(row["Positive"])
    ]
    result["negative_ctxs"] = [
      {"title":know_df.loc[i][k_col] if use_title else "", "text":know_df.loc[i][k_col]} for i in eval(row["Negative"])
    ]
    result["hard_negative_ctxs"] = [
      {"title":know_df.loc[i][k_col] if use_title else "", "text":know_df.loc[i][k_col]} for i in eval(row["Hard-Negative"])
    ]
    txts.append(result)
  return txts

def train_valid_split(know_df, qa_df, split_col="都道府県", num_valid=5):
  trains, _ = train_test_split(know_df[split_col].unique(), test_size=5)
  know_df["train"] = know_df[split_col].isin(trains)
  valids_indices = know_df[~know_df["train"]].index

  qa_df["train"] = qa_df["Positive"].map(lambda x: np.all([i not in valids_indices for i in eval(x)]))
  return [
    ("train", know_df, qa_df[qa_df["train"]]),
    ("valid", know_df, qa_df[~qa_df["train"]]),
    ]

def run(know_file, qa_file, out_file, 
        valid_split=False, split_col="都道府県", num_valid=5, split_type="random", test_ratio=0.1,
        encoding="utf-8", index_col=0, out_csv=False, **kwargs):
  qa_df = pd.read_csv(qa_file, encoding=encoding, index_col=index_col)
  know_df = pd.read_csv(know_file, encoding=encoding, index_col=index_col)

  out_dir = os.path.dirname(out_file)
  os.makedirs(out_dir, exist_ok=True)

  if valid_split:
    print(split_type)
    if split_type == "column":
      spliter = train_valid_split(know_df, qa_df, split_col=split_col, num_valid=num_valid)
    elif split_type == "random":
      from sklearn.model_selection import train_test_split
      spliter = zip(["train", "valid"], [know_df, know_df], train_test_split(qa_df, test_size=test_ratio))

    for key, k_df, q_df in spliter:
      if out_csv:
        q_df.to_csv(out_file.replace(".json", "_{}.csv".format(key)))
      txts = make_json(k_df, q_df, **kwargs)
      print("number of {} qa texts : ".format(key), len(txts))
      with open(out_file.replace(".json", "_{}.json".format(key)), "w") as hndl:
          json.dump(txts, hndl)
  else:
    txts = make_json(know_df, qa_df, **kwargs)
    print("number of qa texts : ", len(txts))
    with open(out_file, "w") as hndl:
        json.dump(txts, hndl)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    # target files
    parser.add_argument("-k", "--knowledge-file", type=str, required=True)
    parser.add_argument("-q", "--qa-file", type=str, required=True)
    parser.add_argument("-o", "--out-file", type=str, required=True)
    # csv file encoding
    parser.add_argument("--encoding", type=str, default="utf-8")
    # csv target columns
    parser.add_argument("--q-col", type=str, default="質問")
    parser.add_argument("--a-col", type=str, default="回答")
    parser.add_argument("--k-col", type=str, default="知識")
    parser.add_argument("--title-col", type=str, default=None)
    # split train abd validation data
    parser.add_argument("--valid-split", action="store_true")
    parser.add_argument("--split-type", type=str, default="random")
    parser.add_argument("--split-col", type=str, default="都道府県")
    parser.add_argument("--num-valid", type=int, default=5)
    parser.add_argument("--out-csv", action="store_true")


    args = parser.parse_args()
    return args


def main():
    args = get_args()

    arg_dicts = vars(args)
    other_params ={
        name:arg_dicts[name] for name in ["encoding", "q_col", "a_col", "k_col", "title_col",
                                          "valid_split", "split_col", "num_valid", "out_csv", "split_type"]
    }
    run(args.knowledge_file, args.qa_file, args.out_file, **other_params)

if __name__ == "__main__":
    main()
