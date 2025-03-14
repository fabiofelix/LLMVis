
import os, torch, string

from enum import IntEnum
from topic_modeling import load_stop_words, is_number
from transformers import AutoTokenizer, AutoModelForCausalLM

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

  def decode_tokens_v1(self, sequences):  
    sentences = sequences.cpu().tolist()
    decoded_tokens = []

    for st in sentences:
      tk_list = []

      for token_idx in st:
        token = self.tokenizer.decode(token_idx)

        if token not in self.get_special_token() and token not in string.punctuation:
          tk_list.append(token)
        elif token == self.tokenizer.sep_token:
          break  

      decoded_tokens.append(tk_list)

    return decoded_tokens  

  def get_token_position(self, data):
    tkn_backend = self.tokenizer.backend_tokenizer
    positions = [] 

    for item in data:
      if tkn_backend.normalizer is not None:
        item = tkn_backend.normalizer.normalize_str(item)

      positions.append(tkn_backend.pre_tokenizer.pre_tokenize_str(item))      

    return positions  

  def next_decode(self, token, token_pos, current_pos, aux_current_pos):
    return token_pos, current_pos, aux_current_pos
  
  def decode_tokens(self, data_ids, data, data_tokens, hidden_state):
    positions = self.get_token_position(data)
    stop_words = load_stop_words()
    tokens = {}

    for ids, token_list, token_pos, feat_list in zip(data_ids, data_tokens["input_ids"].cpu(), positions, hidden_state):
      token_list_decoded = self.tokenizer.batch_decode(token_list)
      current_pos = None
      aux_current_pos = None

      for token, feat in zip(token_list_decoded, feat_list):
        token = token.strip()

        if token not in self.get_special_token():
          token_pos, current_pos, aux_current_pos = self.next_decode(token, token_pos, current_pos, aux_current_pos)

          if (token != '' and
              token in current_pos[0] and
              token not in string.punctuation and
              not is_number(token) and 
              not token.lower() in stop_words):
          
              if token not in tokens:
                tokens[token] = {"text_ids":[], "features": [], "token_pos": []}

              tokens[token]["text_ids"].append(ids)
              tokens[token]["features"].append(feat)
              tokens[token]["token_pos"].append(current_pos)

    return tokens

  def set_model_path(self):
    self.model_path = None

  @torch.no_grad()
  def extract_feature_v1(self, data_ids, data, return_layers = None):
    return_layers = [-1] if return_layers is None or len(return_layers) == 0 else return_layers

    data_tokens = self.tokenizer(data, return_tensors = "pt", padding = True, padding_side="left").to("cuda")
    attn_mask = data_tokens["attention_mask"].cpu()
    ## generate is Auto regressive: results has one hidden_state per max_new_tokens
    results     = self.model.generate(**data_tokens, max_new_tokens = 10, return_dict_in_generate=True, output_hidden_states=True, output_attentions = False, output_scores=False)
    last_hidden_state = results.hidden_states[-1]

    layer_output_feature = []

    for idx in return_layers:
      trunc_out = last_hidden_state[idx].cpu()[:, :attn_mask.shape[1], :]
      layer_output_feature.append(trunc_out)

    return layer_output_feature, self.decode_tokens_v1(results.sequences), attn_mask

  @torch.no_grad()
  def extract_feature(self, data_ids, data, return_layers = None):
    return_layers = [-1] if return_layers is None or len(return_layers) == 0 else return_layers

    data_tokens = self.tokenizer(data, return_tensors = "pt", padding = True, padding_side="left", truncation = True).to("cuda")
    results     = self.model(**data_tokens, return_dict_in_generate=True, output_hidden_states=True, output_attentions = False, output_scores=False)

    layer_output_feature = []
    layer_token = []

    for idx in return_layers:
      hidden_state = results["hidden_states"][idx].detach().cpu()
      layer_output_feature.append( hidden_state )
      layer_token.append( self.decode_tokens(data_ids, data, data_tokens, hidden_state.numpy()) )

    torch.cuda.empty_cache()
    return layer_output_feature, layer_token, data_tokens["attention_mask"].cpu()

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
  
  def next_decode(self, token, token_pos, current_pos, aux_current_pos):
    if "##" not in token:
      return token_pos[1:], token_pos[0], None

    return token_pos, current_pos, None  

class MyLlama(MyModelFamily):  
  def __str__(self):
    return "Llama-v3.1-8B"
  
  def create_tokenizer(self):
    super().create_tokenizer()
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def get_special_token(self):
    return [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.unk_token]    

  def set_model_path(self):
    self.model_path = os.path.join(LLAMA_HPC_PATH, "llama-3.1/huggingface", "Llama-3.1-8B") 

  def next_decode(self, token, token_pos, current_pos, aux_current_pos):
    if aux_current_pos is None or len(aux_current_pos) == 0:
      current_pos = token_pos[0] if len(token_pos) > 0 else current_pos
      aux_current_pos = current_pos[0]
      aux_current_pos = aux_current_pos[1:] if 'Ġ' in aux_current_pos else aux_current_pos
      token_pos   = token_pos[1:]

      return token_pos, current_pos, aux_current_pos[len(token):]

    return token_pos, current_pos, aux_current_pos[len(token):]

class MyGemma(MyModelFamily):
  def __str__(self):
    return "Gemma-v2-9B"
  
  def get_special_token(self):
    return [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.unk_token, self.tokenizer.pad_token]

  def set_model_path(self):
    self.model_path = os.path.join(LLAMA_HPC_PATH, "gemma", "gemma-2-9b") 

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
  
  def next_decode(self, token, token_pos, current_pos, aux_current_pos):
    if aux_current_pos is None or len(aux_current_pos) == 0:
      current_pos = token_pos[0] if len(token_pos) > 0 else current_pos
      aux_current_pos = current_pos[0].replace("▁", "")
      token_pos   = token_pos[1:]

      return token_pos, current_pos, aux_current_pos[len(token):]

    return token_pos, current_pos, aux_current_pos[len(token):]  