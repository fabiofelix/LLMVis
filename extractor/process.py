import numpy as np, torch, gc, pdb

from nlp import filter_token

##https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
##https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
##https://stackoverflow.com/questions/76926025/sentence-embeddings-from-llama-2-huggingface-opensource
##https://github.com/run-llama/llama_index/blob/main/llama_index/embeddings/huggingface.py#L133
##https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
def mean_pooling(features, attention_mask, axis = 1):
  if not isinstance(features, torch.Tensor):
    features = torch.Tensor(features)
  if not isinstance(attention_mask, torch.Tensor):
    attention_mask = torch.Tensor(attention_mask)    

  expanded_attention_mask = attention_mask.unsqueeze(-1).expand(features.size()).float()
  numerator = (features * expanded_attention_mask).sum(axis=axis)
  mean = numerator / expanded_attention_mask.sum(axis=axis).clamp(min=1e-9)

  return mean.numpy()

def tag_indexOf(token_pos, tags):
  for idx, tg in enumerate(tags):
    tkn_pos  = token_pos[1]
    tag_pos  = tg[2]

    if tkn_pos[1] >= tag_pos[0] and tkn_pos[1] <= tag_pos[1]:
      return idx

  return -1

def average_feature_axis(text_token_feat, token_desc):
  print("|- Averaging feature axis")
  idx2tkn = {}
  tkn2idx = {}
  aux = []

  for row in token_desc:
    aux.extend(row)

  filtered_desc = filter_token(aux)
  
  for idx, tkn_desc in enumerate(filtered_desc):
    idx2tkn[idx] = tkn_desc
    tkn2idx[tkn_desc] = idx

  text_token = {key: None for key in text_token_feat.keys()  }

  gc.collect()

  for key in text_token_feat:
    ##NOTE: never ever code this way
    ##new_tensor =  [ [ [0]  * key_shape[-1]  ] * len(idx2tkn) ] * key_shape[0]
    new_tensor = [ [0] * len(idx2tkn) for _ in range(len(text_token_feat[key])) ]

    for txt_id, (tkn_feat, tkn_desc) in enumerate(zip(text_token_feat[key], token_desc)):
      filtered = [ dsc for dsc in np.unique(tkn_desc) if dsc in tkn2idx ]

      ## One token can appear more than one time in the same text     
      ## Calculate the mean of these appearances
      ## And calculate the mean of the token features
      tkn_desc = np.array(tkn_desc)

      for desc in filtered:
        desc_idx = np.where(desc == tkn_desc)[0]
        new_tensor[txt_id][ tkn2idx[desc] ] = tkn_feat[desc_idx].mean(axis = 0).mean()

    gc.collect()
    text_token[key] = new_tensor

  return text_token, idx2tkn

