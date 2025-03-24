
import os, torch, pdb, string, numpy as np

from enum import IntEnum
from transformers import AutoTokenizer, AutoModelForCausalLM

from nlp import load_stop_words, is_number, has_letter, is_ordinal

LLAMA_HPC_PATH = ""

class MODEL(IntEnum): 
  MYBERT = 0
  MYLLAMA = 1
  MYGEMMA = 2

def create_model(type):
  model = None

  if type == MODEL.MYBERT:
    model = MyBERT()
  elif type == MODEL.MYLLAMA:  
    model = MyLlama()
  elif type == MODEL.MYGEMMA:  
    model = MyGemma()        

  return model  

class MyModelFamily():
  def __init__(self):
    self.set_model_path()

    self.create_tokenizer()
    self.create_model()

    self.model = self.model.to("cuda")
    self.model.eval()

  def create_model(self):
    self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, device_map='auto')        

  def create_tokenizer(self):    
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, truncation_side = "right")

  def get_special_token(self):
    return None
  
  def get_pad_token(self, return_id = False):
    return None
  
  def set_model_path(self):
    self.model_path = None

  def clean_token(self, token):
    return token

  def next_decode(self, token, token_pos, current_pos, text_current_pos):
    return token_pos, current_pos, text_current_pos    

  def pad_tokens(self, token_ids, token_text, token_pos, attn_mask):
    max_length = max(len(row) for row in token_ids)  
    token_ids  = np.array([np.pad(row, (max_length - len(row), 0), constant_values = self.get_pad_token(True)) for row in token_ids])
    token_text = np.array([np.pad(row, (max_length - len(row), 0), constant_values = self.get_pad_token()) for row in token_text])

    max_length = max(len(row) for row in token_pos)

    for idx in range(len(token_pos)):
      pad = [None] * (max_length - len(token_pos[idx]))
      pad.extend(token_pos[idx])
      token_pos[idx] = pad

    max_length = max(len(row) for row in attn_mask)  
    attn_mask  = np.array([np.pad(row, (max_length - len(row), 0)) for row in attn_mask])    

    return token_ids, token_text, token_pos, attn_mask

  def pad_features(self, features):
    max_length = max(len(row) for row in features)

    return np.array([np.pad(row, ((max_length - len(row), 0), (0, 0)) ) for row in features])    

  def get_token_position(self, data):
    tkn_backend = self.tokenizer.backend_tokenizer
    positions = [] 

    for item in data:
      if tkn_backend.normalizer is not None:
        item = tkn_backend.normalizer.normalize_str(item)

      positions.append(tkn_backend.pre_tokenizer.pre_tokenize_str(item))      

    return positions  
  
  def decode_tokens(self, data, token_ids):
    token_pos = self.get_token_position(data)
    decoded_token = []
    padded_token_pos = []

    for idx, (tkns, tkn_pos) in enumerate(zip(token_ids, token_pos)):
      tkn_desc = self.tokenizer.batch_decode(tkns)  
      aux_pos = [None] * len(tkn_desc)
      current_pos = None
      text_current_pos = None

      for jdx, desc in enumerate(tkn_desc):
        desc = desc.strip()
        
        if desc not in self.get_special_token():
          tkn_pos, current_pos, text_current_pos = self.next_decode(desc, tkn_pos, current_pos, text_current_pos)

          if desc != '' and self.clean_token(desc) in current_pos[0]:
            aux_pos[jdx] = current_pos

      decoded_token.append([ self.clean_token(desc.strip()) for desc in tkn_desc ])
      padded_token_pos.append(aux_pos)
            
    return decoded_token, padded_token_pos
  
  def filter_token(self, token_desc):
    filtered_token = []  
    stop_words = load_stop_words()

    for desc in token_desc:
      desc = desc if desc is None else desc.strip()
      
      if (desc is not None and
          desc != '' and
          desc not in self.get_special_token() and 
          desc not in string.punctuation and
          not is_number(desc) and
          not is_ordinal(desc) and
          not desc.lower() in stop_words and 
          has_letter(desc)):
        filtered_token.append(desc)

    filtered_token.sort()
    return filtered_token

  @torch.no_grad()
  def extract_feature(self, data, return_blocks = None):
    data_tokens = self.tokenizer(data, return_tensors = "pt", padding = True, padding_side="left", truncation = True).to("cuda")
    results     = self.model(**data_tokens, return_dict_in_generate=True, output_hidden_states=True, output_attentions = False, output_scores=False)

    return_blocks = [-1] if return_blocks is None or len(return_blocks) == 0 else return_blocks
    block_feature = []
    block_idx = []

    for idx in return_blocks:
      hidden_state = results["hidden_states"][idx].detach().cpu().tolist()
      block_feature.append(hidden_state)
      block_idx.append(idx if idx >= 0 else len(results.hidden_states) + idx)

    torch.cuda.empty_cache()
    return block_idx, block_feature, data_tokens["input_ids"].cpu().tolist(), data_tokens["attention_mask"].cpu().tolist()

class MyBERT(MyModelFamily):
  def __str__(self):
    return "BERT-base-uncased"
  
  def set_model_path(self):
    #A NOT case sensitive model pretrained with two self-supervised tasks:
    #1. Predicts masked words of a sentence
    #2. Predicts if two masked sentences are following each other or not  
    #The mode has 12 layers (+ inicial embedding layer) and 768 features in the hidden-state
    self.model_path = "bert-base-uncased"  

  def create_model(self):
    self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, is_decoder = True)

  def get_special_token(self):
    return [self.tokenizer.unk_token, self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.mask_token]
  
  def get_pad_token(self, return_id = False):
    token = self.tokenizer.pad_token

    if return_id:
      token = self.tokenizer(token, return_tensors = "pt", padding = False, truncation = True)
      token = token.input_ids[0][1].item() #0: [CLS], 1: token, 2: [SEP]

    return token

  def clean_token(self, token):
    return token.replace("##", "")

  def next_decode(self, token, token_pos, current_pos, text_current_pos):
    if "##" not in token:
      return token_pos[1:], token_pos[0], text_current_pos

    return token_pos, current_pos, text_current_pos
  
class MyLlama(MyModelFamily):  
  def __str__(self):
    return "Llama-v3.1-8B"
  
  def create_tokenizer(self):
    super().create_tokenizer()
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def get_special_token(self):
    return [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.unk_token]

  def get_pad_token(self, return_id = False):
    token = self.tokenizer.pad_token

    if return_id:
      token = self.tokenizer(token, return_tensors = "pt", padding = False, truncation = True)
      token = token.input_ids[0][1].item() #0: <|begin_of_text|>  1: <|end_of_text|>

    return token

  def set_model_path(self):
    self.model_path = os.path.join(LLAMA_HPC_PATH, "llama-3.1/huggingface", "Llama-3.1-8B") 

  def clean_token(self, token):
    return token.replace("\n", "").replace('Ġ', "").replace('Ċ', "")

  def next_decode(self, token, token_pos, current_pos, text_current_pos):
    if text_current_pos is None or len(text_current_pos) == 0:
      current_pos = token_pos[0] if len(token_pos) > 0 else current_pos

      text_current_pos = current_pos[0]
      ## SPACE
      text_current_pos = text_current_pos[1:] if 'Ġ' in text_current_pos else text_current_pos
      ## \n
      text_current_pos = text_current_pos.replace('Ċ', "")

      token_pos   = token_pos[1:]

    return token_pos, current_pos, text_current_pos[len(token):]  

class MyGemma(MyModelFamily):
  def __str__(self):
    return "Gemma-v2-9B"
  
  def get_special_token(self):
    return [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.unk_token, self.tokenizer.pad_token]

  def get_pad_token(self, return_id = False):
    token = self.tokenizer.pad_token

    if return_id:
      token = self.tokenizer(token, return_tensors = "pt", padding = False, truncation = True)
      token = token.input_ids[0][1].item() #0: <bos>  1: <pad>

    return token

  def set_model_path(self):
    self.model_path = os.path.join(LLAMA_HPC_PATH, "gemma", "gemma-2-9b") 

  def clean_token(self, token):
    return token.replace("▁", "")

  def get_token_position(self, data):
    tkn_backend = self.tokenizer.backend_tokenizer
    positions = [] 

    for text in data:
      if tkn_backend.normalizer is not None:
        text = tkn_backend.normalizer.normalize_str(text)

      text = tkn_backend.pre_tokenizer.pre_tokenize_str(text)[0][0]
      text = text.split("▁")
      text_aux = []

      for t in text:
        text_aux.extend(t.split('\n'))

      text = text_aux
      first = 0
      count_space = 0
      token = ""
      aux = []

      for item in text:
        if item == "":
          token += "▁"
          count_space += 1
        else:
          if count_space > 0:
            count_space = 0
            aux.append((token, (first, first + len(token))))
            first += len(token)
            token = ""
          else:
            token = "▁"   

          token += item
          aux.append((token, (first,  first + len(token))))
          first += len(token)

      positions.append(aux)

    return positions

  def next_decode(self, token, token_pos, current_pos, text_current_pos):
    if text_current_pos is None or len(text_current_pos) == 0:
      current_pos = token_pos[0] if len(token_pos) > 0 else current_pos
      #SPACE
      text_current_pos = current_pos[0].replace("▁", "")
      token_pos   = token_pos[1:]

    return token_pos, current_pos, text_current_pos[len(token):]  