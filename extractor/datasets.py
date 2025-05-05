
import os, pandas as pd, numpy as np, re, pdb, glob, math
from sklearn.model_selection import train_test_split
from enum import IntEnum
from nlp import MyLDA

import utils

class DATASET(IntEnum): 
  PAPERABSTRACT = 0
  BBCNEWS = 1
  TINYSTORIESV2 = 2

def create_dataset(type, path):
  dataset = None

  if type == DATASET.PAPERABSTRACT:
    dataset = PaperAbstract(path)
  elif type == DATASET.BBCNEWS:  
    dataset = BBCNews(path)
  elif type == DATASET.TINYSTORIESV2:  
    dataset = TinyStories_V2(path)

  return dataset          

class TextDataset():
  def __init__(self, path = None):
    self.data_path = path

  def __class_sample_size(self, data, labels, sample_size):
    if sample_size > len(data):
      raise ValueError("sample_size should be less than data size", len(data))

    label_name, label_count = np.unique(labels, return_counts=True)
    n_classes = len(label_name)

    # Start with an even distribution
    sample_class = np.full(n_classes, math.ceil(sample_size / n_classes))
    sample_class[-1] += sample_size - sample_class.sum()  # fix rounding

    # Redistribute if any class has fewer samples than requested
    while np.any(label_count < sample_class):
      excess = sample_class - label_count
      excess[excess < 0] = 0
      sample_class = np.minimum(sample_class, label_count)

      remaining = sample_size - sample_class.sum()
      available = (label_count > sample_class)
      n_available = available.sum()

      if n_available > 0:
        add_per_class = remaining // n_available
        sample_class[available] += add_per_class
        sample_class[np.where(available)[0][-1]] += remaining - (add_per_class * n_available)
      else:
        break

    return label_name, sample_class  

  def sampling(self, data, labels, sample_size):
    label_names, class_sizes = self.__class_sample_size(data, labels, sample_size)
    sampled_data = []

    for label, size in zip(label_names, class_sizes):
      sampled_data.append(data[labels == label].sample(n=size, replace=False, random_state=utils.SEED_VALUE))

    return pd.concat(sampled_data).sort_index()

  def get_data(self, n_samples = 100):
    pass

  def format(self, data):
    data["text"] = data["text"].str.replace("''", '"')

    return data

##https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles
##https://www.kaggle.com/code/vanooshenzr/topic-modeling-using-bert-and-lda/notebook
class PaperAbstract(TextDataset):
  def __str__(self):
    return "PaperAbstract"
  
  def get_data(self, n_samples = 100):
    train = pd.read_csv(os.path.join(self.data_path, "train.csv"))

    train = train.rename(columns={"ABSTRACT": "text"})
    train.text = train.TITLE.str.strip() + " " + train.text.str.strip()
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

    train["name"] = [ "PaperAbstract_{}".format(id) for id in train["ID"].to_numpy() ]

    texts = self.sampling(train, train["label"], n_samples)

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
        row = re.sub(r'\n+', " ", row)
        row = row.strip()

        if row == sep_text:
          text.append(row_text)
          filenames.append(f"TinyStories_{idx}")
          row_text = ""
        else:
          row_text += row
    finally:
      data.close()

    texts = pd.DataFrame({"name": filenames, "text": text, "processed_text": None, "label": None, "topic": None})
    texts.topic = self.topic.run(texts.text.to_list())
    texts, _, _, _ = train_test_split(texts, texts.topic, train_size = n_samples, random_state=utils.SEED_VALUE)  

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
            data = re.sub(r'\n+', " ", data)
            data = data.strip()
          finally:
            file.close()

          if len(data) > 0 and (max_text_length is None or len(data) <= max_text_length):
            filename.append(f"BBCNews_{desc}_{name}")
            label.append(desc)
            text.append(data)
            sizes.append(len(data))

    texts = pd.DataFrame({"name": filename, "text": text, "processed_text": None, "label": label, "topic": None})
    texts = self.sampling(texts, texts["label"], n_samples)

    return self.format(texts[["name", "text", "processed_text", "label", "topic"]])
