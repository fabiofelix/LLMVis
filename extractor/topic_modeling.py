import os, numpy as np, string, re, tqdm, nltk, utils

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

nltk.data.path.append(os.path.join(utils.main_cache_path, "llm", "nltk_data"))
nltk.download("punkt_tab", download_dir = os.path.join(utils.main_cache_path, "llm", "nltk_data"))
nltk.download('stopwords', download_dir = os.path.join(utils.main_cache_path, "llm", "nltk_data"))
nltk.download('wordnet', download_dir = os.path.join(utils.main_cache_path, "llm", "nltk_data"))

def load_stop_words():
  stop_words = stopwords.words('english')

  if os.path.isfile("stopwords_news.spw"):
    file = open("stopwords_news.spw", "r")
    stop_words.extend([ re.sub("\\n", "", line)  for line in file  ])

  stop_words = set(stop_words)

  return stop_words    

def is_number(value):
  number_pattern = "^(?:-(?:[1-9](?:\\d{0,2}(?:,\\d{3})+|\\d*))|(?:0|(?:[1-9](?:\\d{0,2}(?:,\\d{3})+|\\d*))))(?:.\\d+|)$"

  return re.match(number_pattern, value) is not None

class TopicModeling:
  def __init__(self):
    self.vectorizer = None
    self.model = None
    self.words_per_topic = 3

  def preprocess_txt(self, texts, lem = True, stem = False, remove_stop_words = True):
    wnl = WordNetLemmatizer()
    ps  = PorterStemmer()
    txt_words = []
    stop_words = load_stop_words()
    table = str.maketrans('', '', string.punctuation)

    for txt in tqdm.tqdm(texts, desc = "Preprocessing" , total = len(texts), unit= "text"):
      # remove email
      txt = re.sub(r"([\w\.\-\_]+@[\w\.\-\_]+)", "", txt).strip()
      # split into words
      words = word_tokenize(txt)  
      # convert to lower case
      words = [wd.lower() for wd in words]
      # remove punctuation from each word
      words = [wd.translate(table) for wd in words]  
      # remove remaining tokens that are not alphabetic
      words = [wd for wd in words if wd.isalpha()]  
      # filter out stop words
      if remove_stop_words:
        words = [wd for wd in words if not wd in stop_words]  
      # stemming
      if stem:
        words = [ps.stem(wd) for wd in words]        
      # lemmatization
      if lem:
        words = [wnl.lemmatize(wd) for wd in words]
      # remove redundance
      words = np.unique(words).tolist()

      txt_words.append(" ".join(words))

    return txt_words
  
  def get_topic_words(self, components, topics):
    #components has one row per topic and columns (words) with probabilities
    #vectorizer.get_feature_names_out() has the real words
    #argsort returns the indices in the increasing order of the values
    topic_words = [ [ self.vectorizer.get_feature_names_out()[idx] for idx in topic.argsort()[-self.words_per_topic:] ] for topic in components ] 
    topic_words = [ "-".join(tw) for tw in topic_words]

    #topics has one row per text processed and columns (topics) with probabilities
    return [ topic_words[ txt.argmax() ] for txt in topics  ]

  def run(self, texts):
    texts_processed = self.preprocess_txt(texts)
    count_words = self.vectorizer.fit_transform(texts_processed)    
    topics = self.model.fit_transform(count_words)

    return self.get_topic_words(components = self.model.components_, topics = topics)

class MyLDA(TopicModeling):
  def __init__(self):
    super().__init__()
    self.vectorizer = CountVectorizer(min_df = 0.05, max_df = 0.95, max_features = 1000)
    self.model = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method="online", learning_offset=50.0, random_state=utils.SEED_VALUE)

class MyNMF(TopicModeling):  
  def __init__(self):
    super().__init__()
    self.vectorizer = TfidfVectorizer(min_df = 0.05, max_df = 0.95, max_features = 1000)
    self.model = NMF(n_components=5, init="nndsvda", random_state=utils.SEED_VALUE)
