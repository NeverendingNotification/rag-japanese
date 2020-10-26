
import pandas as pd 
from torch.utils.data import Dataset, DataLoader

class BartDataest(Dataset):
  def __init__(self, csv_file, q_col="質問", a_col="回答"):
    df = pd.read_csv(csv_file)
    self.questions = df[q_col].values
    self.answers = df[a_col].values
    self.len = len(self.questions)
    
  def __len__(self):
    return self.len
  
  def __getitem__(self, idx):
    return self.questions[idx], self.answers[idx]
  
def get_collater(tokenizer):
  def collater(batch):
    questions, answers = [list(a) for a in zip(*batch)]
    return tokenizer.prepare_seq2seq_batch(questions, tgt_texts=answers, return_tensors="pt")
  return collater

def get_dataloader(csv_file, tokenizer, batch_size, shuffle=True, **kwargs):
    dataset = BartDataest(csv_file, **kwargs)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=get_collater(tokenizer), shuffle=shuffle)
    return data_loader
