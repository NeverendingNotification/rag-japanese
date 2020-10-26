# rag-japanese
train and evaluation scripts for RAG[^1]

0. make small pretrain japanese BERT (optional).
1. preprocess your data.
2. training DPR[^2] (based on facebook implementation[^3]).
3. convert DPR model to transformers[^4] model.
4. make index from knowledges.
5. train model.
6. test model.




# Instalation
python 3.6
```
git clone https://github.com/NeverendingNotification/rag-japanese.git
cd pytorch-xai-analyze
pip install -r requirements.txt
```

# make small bert pretrained model

```
# rag-japanese
python make_small_bert.py --pretrained-model cl-tohoku/bert-base-japanese-whole-word-masking --out-dir models/small_bert --num-layers 3
```
You can skip this step if you already have proper pretrained model for DPR.

# preprocess data

You should put your knowledge text file "data/knowledge.csv" and queation and answer text file "data/qa.csv".
```
# rag-japanese
python preprocess_data.py --knowledge-file data/knowledge.csv --qa-file data/qa.csv --out-file data/dpr_qa.json --valid-split --out-csv
```
This generates data/dpr_qa_train.json, data/dpr_qa_valid.json data/dpr_qa_train.csv data/dpr_qa_valid.csv.

# training DPR
Move to dpr directory and run the following commands.
```
# rag-japanese/dpr
python train_dense_encoder.py --train_file ../data/dpr_qa_train.json --dev_file ../data/dpr_qa_valid.json --encoder_model_type hf_bert --pretrained_model_cfg ../models/small_bert --batch_size 8 --output_dir ../models/dpr --num_train_epochs 6
```

# Convert DPR model to transformers model

```
# rag-japanese/dpr
python convert_model.py -p ../models/dpr/dpr_biencoder.5.386 -o ../models/dpr_transformers

```

# convert knowledge to index
Make faiss index from knowledge text and trained context encoder.
```
# rag-japanese
python make_index.py  --context-model models/dpr_transformers/c_encoder --knowledge-file data/knowledge.csv --out-dir data/dpr_knowlege_index
```

# train model
```
# rag-japanese
python train_model.py  --model-type rag --question-model models/dpr_transformers/q_encoder --train-csv data/dpr_qa_train.csv --valid-csv data/dpr_qa_valid.csv --indexdata-path data/dpr_knowlege_index/knowlege --index-path data/dpr_knowlege_index/knowlege_index.faiss --out-dir  results/rag
```

# test model
```
# rag-japanese
python test_model.py  --model-type rag --pretrained-model results/rag --test-csv data/dpr_qa_valid.csv --indexdata-path data/dpr_knowlege_index/knowlege --index-path data/dpr_knowlege_index/knowlege_index.faiss --out-dir results/rag --out-file test.csv
```

# Reference
[^1]: https://arxiv.org/abs/2005.11401
[^2]: https://arxiv.org/abs/2004.04906
[^3]: https://github.com/facebookresearch/DPR
[^4]: https://huggingface.co/transformers/model_doc/dpr.html
