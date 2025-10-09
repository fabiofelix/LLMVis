import os, numpy as np, string, re, tqdm, nltk, pdb, utils, json, calendar

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

nltk.data.path.append(os.path.join(utils.main_cache_path, "llm", "nltk_data"))
nltk.download("punkt_tab", download_dir = os.path.join(utils.main_cache_path, "llm", "nltk_data"))
nltk.download('stopwords', download_dir = os.path.join(utils.main_cache_path, "llm", "nltk_data"))
nltk.download('wordnet', download_dir = os.path.join(utils.main_cache_path, "llm", "nltk_data"))

#============== tagging and ner ==============#
nltk.download('averaged_perceptron_tagger_eng', download_dir = os.path.join(utils.main_cache_path, "llm", "nltk_data"))
nltk.download('maxent_ne_chunker_tab', download_dir = os.path.join(utils.main_cache_path, "llm", "nltk_data"))
nltk.download('words', download_dir = os.path.join(utils.main_cache_path, "llm", "nltk_data"))  
nltk.download('tagsets_json', download_dir = os.path.join(utils.main_cache_path, "llm", "nltk_data"))

def load_stop_words():
  stop_words = stopwords.words('english')
  stop_words.extend([ name.lower() for name in list(calendar.month_name)[1:] ])
  stop_words.extend([ name.lower() for name in list(calendar.day_name) ])

  if os.path.isfile("stopwords_news.spw"):
    file = open("stopwords_news.spw", "r")
    stop_words.extend([ re.sub("\\n", "", line)  for line in file  ])

  stop_words = set(stop_words)

  return stop_words    

def is_number(value):
  number_pattern = "^(?:-(?:[1-9](?:\\d{0,2}(?:,\\d{3})+|\\d*))|(?:0|(?:[1-9](?:\\d{0,2}(?:,\\d{3})+|\\d*))))(?:.\\d+|)$"

  return value.isnumeric() or re.match(number_pattern, value) is not None

def is_ordinal(value):
  number_pattern = r"\d+(?:st|nd|rd|th)"

  return re.match(number_pattern, value.lower()) is not None

def is_decade(value):
  number_pattern = r"(\d{2}|\d{4})s"

  return re.match(number_pattern, value.lower()) is not None 

def has_letter(value):
  letter_patern = '[a-zA-Z]'

  return re.search(letter_patern, value) is not None

def filter_stop_words(token_desc):
  filtered_token = []  
  stop_words = load_stop_words()

  for desc in token_desc:
    desc = '' if desc is None else desc.strip()
    #ignore punctuations
    desc_stop = re.sub(r'[^\w\s]', '', desc)
    
    if (desc != '' and
        not is_number(desc) and
        not is_ordinal(desc) and
        not is_decade(desc) and
        has_letter(desc) and
        desc_stop.lower() not in stop_words):
      filtered_token.append(desc)

  filtered_token.sort()
  return np.unique(filtered_token)

## https://fouadroumieh.medium.com/nlp-entity-extraction-ner-using-python-nltk-68649e65e54b
## https://stackoverflow.com/questions/31668493/get-indices-of-original-text-from-nltk-word-tokenize
def nltk_extract_entity(text):
  tokens = text

  if isinstance(text, str):
    tokens = nltk.word_tokenize(text)
    
  tagged_tokens = nltk.pos_tag(tokens)
  entity_tree = nltk.ne_chunk(tagged_tokens)
  tags = []
  entity_list = []
  first_idx = 0  
  tag_desc = open(os.path.join(utils.main_cache_path, "llm", "nltk_data", "help/tagsets_json/PY3_json/upenn_tagset.json"))
  tag_desc = json.load(tag_desc)

  for subtree in entity_tree:
    if isinstance(subtree, nltk.tree.Tree):
      for leaf in subtree.leaves():
        token_idx = text.find(leaf[0])
        text      = text[token_idx + len(leaf[0]):] 

        first_idx += token_idx

        desc = tag_desc[leaf[1]][0] + " (" + leaf[1] + ")" if leaf[1] in tag_desc else leaf[1]
        tags.append( (leaf[0], desc, (first_idx, first_idx + len(leaf[0]))) )

        entity_list.append(subtree.label())

        first_idx += len(leaf[0])
    else:
      token_aux = '"' if subtree[0] == '``' else subtree[0]

      token_idx = text.find(token_aux)
      text      = text[token_idx + len(token_aux):] 

      first_idx += token_idx

      desc = tag_desc[subtree[1]][0] + " (" + subtree[1] + ")" if subtree[1] in tag_desc else subtree[1]
      tags.append( (token_aux, desc, (first_idx, first_idx + len(token_aux))) )
      
      entity_list.append(None)

      first_idx += len(token_aux)

  return tags, entity_list

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

    for txt in tqdm.tqdm(texts, desc = "|- Preprocessing" , total = len(texts), unit= "text"):
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
