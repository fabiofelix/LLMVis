
import os, pandas as pd, numpy as np, re, glob
from sklearn.model_selection import train_test_split
from enum import IntEnum
from topic_modeling import MyLDA

import utils

class DATASET(IntEnum): 
  KAGGLE = 0
  TINYSTORIESV2 = 1
  BBCNEWS = 2

def create_dataset(type, path):
  dataset = None

  if type == DATASET.KAGGLE:
    dataset = Kaggle(path)
  elif type == DATASET.TINYSTORIESV2:  
    dataset = TinyStories_V2(path)
  elif type == DATASET.BBCNEWS:  
    dataset = BBCNews(path)

  return dataset          

class TextDataset():
  def __init__(self, path = None):
    self.data_path = path

  def get_data(n_samples = 100):
    pass

##https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles
##https://www.kaggle.com/code/vanooshenzr/topic-modeling-using-bert-and-lda/notebook
class Kaggle(TextDataset):
  def __str__(self):
    return "Kaggle"
  
  def get_data(self, n_samples = 100):
    train = pd.read_csv(os.path.join(self.data_path, "train.csv"))

    train = train.rename(columns={"ABSTRACT": "text"})
    train["label"] = None
    train["topic"] = None
    train["processed_text"] = None

    ##Computer Science 	Physics 	Mathematics 	Statistics 	Quantitative Biology 	Quantitative Finance
    labels = [np.char.lower(label).tolist() for label in train.columns[3:9].to_numpy()]
    labels[0] = "compsci"
    labels[2] = "math"
    labels[4] = "biology"
    labels[5] = "finance"
    labels.append("other")

    for idx, row in train.iterrows():
      label = row[3:9].to_numpy()
      train.loc[idx, "label"] = labels[ np.argmax(label) ]  

    train["name"] = [ "kaggle_article_{}".format(id) for id in train["ID"].to_numpy() ]

    texts, _, _, _ = train_test_split(train, train["label"], train_size = n_samples, random_state=utils.SEED_VALUE, stratify=train["label"])

    return texts[["name", "text", "processed_text", "label", "topic"]]

## https://huggingface.co/datasets/roneneldan/TinyStories
class TinyStories_V2(TextDataset):
  def __init__(self, path = None):
    super().__init__(path)
    self.topic = MyLDA()

  def __str__(self):
    return "TinyStories-v2" 

  def get_data(self, n_samples = 100, max_text_length = None):  
    train = os.path.join(self.data_path, "TinyStoriesV2-GPT4-valid.txt")
    sep_text = "<|endoftext|>"

    filenames = []
    text = []
    data = open(train)

    try:
      row_text = ""

      for idx, row in enumerate(data, 1):
        row = re.sub("\\n", "", row)
        if row == sep_text:
          text.append(row_text)
          filenames.append(f"TinyStories_{idx}")
          row_text = ""
        else:
          row_text += row
    finally:
      data.close()

    texts = pd.DataFrame({"name": filenames, "text": text, "processed_text": None, "label": None, "topic": None})
    texts, _, _, _ = train_test_split(texts, texts.label, train_size = n_samples, random_state=utils.SEED_VALUE)  
    texts.topic = self.topic.run(texts.text.to_list())

    return texts[["name", "text", "processed_text", "label", "topic"]]

# http://mlg.ucd.ie/datasets/bbc.html
class BBCNews(TextDataset):
  def __str__(self):
    return "BBC-News"
  
  def get_data(self, n_samples = 100, max_text_length = 2000):      
    folders = glob.glob(os.path.join(self.data_path, "*"))
    filename = []
    text = []
    label = []
    sizes = []

    for fd in folders:
      if os.path.isdir(fd):
        files = glob.glob(os.path.join(fd, "*.txt"))

        for fl in files:
          desc = os.path.basename(fd)

          name = os.path.basename(fl)
          name = name.split(".")[0]
          
          file = open(fl)

          try:
            data = file.read()
            data = re.sub("\\n", "", data)
            data = data.strip()
          finally:
            file.close()

          if len(data) > 0 and (max_text_length is None or len(data) <= max_text_length):
            filename.append(f"BBCNews_{desc}_{name}")
            label.append(desc)
            text.append(data)
            sizes.append(len(data))

    texts = pd.DataFrame({"name": filename, "text": text, "processed_text": None, "label": label, "topic": None})
    texts, _, _, _ = train_test_split(texts, texts.label, train_size = n_samples, random_state=utils.SEED_VALUE)  

    return texts[["name", "text", "processed_text", "label", "topic"]]
