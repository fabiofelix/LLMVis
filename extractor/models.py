
import os, torch, pdb, numpy as np

from enum import IntEnum
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

import utils

LLAMA_HPC_PATH = ""

class MODEL(IntEnum): 
  MYBERT = 0
  MYDEBERTA = 1
  MYLLAMA = 2
  MYGEMMA = 3

def create_model(type):
  model = None

  if type == MODEL.MYBERT:
    model = MyBERT()
  elif type == MODEL.MYDEBERTA:
    model = MyDeBERTa()    
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

  def __len__(self):
    #transformer layer + embedding layer
    return len(self.model.model.layers) + 1  

  def create_model(self):
    self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, device_map='auto', cache_dir = self.cache_path)        
    self.inc_pos = False

  def create_tokenizer(self):    
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, truncation_side = "right", cache_dir = self.cache_path)

  def get_special_token(self):
    return None
  
  def set_model_path(self):
    self.model_path = None
    self.cache_path = None

  def clean_token(self, token):
    return token
  
  def normalize_str(self, text):
    return text

  def decode_tokens(self, data, token_ids):
    decoded_token = []
    decoded_pos = []

    for text, tkn_id in zip(data, token_ids):
      text = self.normalize_str(text)
      tkn_desc = self.tokenizer.batch_decode(tkn_id)
      first_idx = 0
      aux_token = []
      aux_pos = []

      for desc in tkn_desc:
        desc = self.clean_token(desc)

        if desc not in self.get_special_token():
          token_idx = text.find(desc)
          text      = text[token_idx + len(desc):] 

          #Some models like Llama and Gemma include the space in the token but others like BERT doesn't
          #So BERT needs to offset the position
          if self.inc_pos:
            first_idx += token_idx

          aux_token.append(desc)
          aux_pos.append((desc, (first_idx, first_idx + len(desc))))

          first_idx += len(desc)

      decoded_token.append(aux_token) 
      decoded_pos.append(aux_pos)

    return decoded_token, decoded_pos

  def call_model(self, data_tokens):
    return self.model(**data_tokens, return_dict_in_generate=True, output_hidden_states=True, output_attentions = False, output_scores=False)

  @torch.no_grad()
  def extract_feature(self, data, return_blocks = None):
    data_tokens = self.tokenizer(data, return_tensors = "pt", padding = True, padding_side="left", truncation = True).to("cuda")
    results     = self.call_model(data_tokens)

    return_blocks = [-1] if return_blocks is None or len(return_blocks) == 0 else return_blocks
    block_feature = []
    block_idx = []

    for idx in return_blocks:
      hidden_state = results["hidden_states"][idx].detach().cpu().numpy()
      block_feature.append(hidden_state.astype(np.float32))
      block_idx.append(idx if idx >= 0 else len(results.hidden_states) + idx)

    torch.cuda.empty_cache()
    return block_idx, block_feature, data_tokens["input_ids"].cpu().tolist(), data_tokens["attention_mask"].cpu().tolist()

class MyBERT(MyModelFamily):
  def __str__(self):
    return "BERT-base-uncased"
  
  def __len__(self):
    #transformer layer + embedding layer
    return len(self.model._modules["bert"].encoder.layer) + 1
  
  def set_model_path(self):
    super().set_model_path()
    #A NOT case sensitive model pretrained with two self-supervised tasks:
    #1. Predicts masked words of a sentence
    #2. Predicts if two masked sentences are following each other or not  
    #The mode has 12 layers (+ inicial embedding layer) and 768 features in the hidden-state
    self.model_path = "bert-base-uncased"  

    self.cache_path = utils.hugging_path

  def create_model(self):
    self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, is_decoder = True, cache_dir = self.cache_path)
    self.inc_pos = True

  def get_special_token(self):
    return [self.tokenizer.unk_token, self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.mask_token]
  
  def normalize_str(self, text):  
    return self.tokenizer.backend_tokenizer.normalizer.normalize_str(text)

  def clean_token(self, token):
    return token.replace("##", "")

class MyDeBERTa(MyModelFamily):
  def __str__(self):
    return "DeBERTa-v2-xlarge"
  
  def __len__(self):
    #transformer layer + embedding layer
    return len(self.model._modules["deberta"].encoder.layer) + 1
  
  def set_model_path(self):
    super().set_model_path()
    self.model_path = "microsoft/deberta-v2-xlarge-mnli"
    self.cache_path = utils.hugging_path

  def create_model(self):
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, torch_dtype=torch.float16, is_decoder = True, cache_dir = self.cache_path)
    self.inc_pos = True

  def get_special_token(self):
    return [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.unk_token, self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.mask_token]

  def normalize_str(self, text):  
    return self.tokenizer.backend_tokenizer.normalizer.normalize_str(text)

  def clean_token(self, token):
    return token.replace("##", "")  
  
  def call_model(self, data_tokens):
    return self.model(**data_tokens, return_dict=True, output_hidden_states=True, output_attentions = False)  
  
class MyLlama(MyModelFamily):  
  def __str__(self):
    return "Llama-v3.1-8B"
  
  def create_tokenizer(self):
    super().create_tokenizer()
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def get_special_token(self):
    return [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.unk_token]

  def set_model_path(self):
    super().set_model_path()
    self.model_path = os.path.join(LLAMA_HPC_PATH, "llama-3.1/huggingface", "Llama-3.1-8B") 


class MyGemma(MyModelFamily):
  def __str__(self):
    return "Gemma-v2-9B"
  
  def get_special_token(self):
    return [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.unk_token, self.tokenizer.pad_token]

  def set_model_path(self):
    super().set_model_path()
    self.model_path = os.path.join(LLAMA_HPC_PATH, "gemma", "gemma-2-9b") 

