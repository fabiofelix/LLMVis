import numpy as np, torch, gc, os, pdb

from multiprocessing import Pool

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

def tag_indexOf(token, tags):
  index = -1

  for idx, tg in enumerate(tags):
    if token in tg[0] or tg[0] in token:
      return idx

  return index

def expand_token_axis(text_token_feat, token_ids, token_desc, model):
  print("|- Expanding token axis")
  # pdb.set_trace()
  idx2tkn = {}
  tkn2idx = {}

  filtered_desc = model.filter_token(np.unique(token_desc))
  
  for idx, tkn_desc in enumerate(filtered_desc):
    idx2tkn[idx] = tkn_desc
    tkn2idx[tkn_desc] = idx

  gc.collect()
  # pdb.set_trace()

  for key in text_token_feat:
    key_shape  = text_token_feat[key].shape
    ##NOTE: never ever code this way
    ##new_tensor =  [ [ [0]  * key_shape[-1]  ] * len(idx2tkn) ] * key_shape[0]
    new_tensor = [ ]

    for _ in range(key_shape[0]):
      aux = []
      for _ in range(len(idx2tkn)):
         aux.append([0] * key_shape[-1])
      new_tensor.append(aux)   

    for txt_id, (tkn_feat, tkn_desc) in enumerate(zip(text_token_feat[key], token_desc)):
      filtered = [ dsc for dsc in np.unique(tkn_desc) if dsc in tkn2idx ]

      # pdb.set_trace()
      ## One id (desc) can appear more than one time in the same text     
      ## Calculate the mean of these appearances
      for desc in filtered:
        desc_idx = np.where(desc == tkn_desc)[0]
        new_tensor[txt_id][ tkn2idx[desc] ] = tkn_feat[desc_idx].mean(axis = 0).tolist()

      # pdb.set_trace()

    # pdb.set_trace()
    gc.collect()
    text_token_feat[key] = new_tensor

  # pdb.set_trace()
  return text_token_feat, idx2tkn, tkn2idx   

def exec_mean(batch, axis):
  batch = np.array(batch, dtype=np.float32)
  batch = np.mean(batch, axis = axis)

  return batch.tolist()

def mean_list(data, axis, split_axis = 0):
  nr_batch = 5
  data_size = 0
  batch_size = data_size // nr_batch
  new_values = []    

  if axis == split_axis:
    raise Exception("axis and split_axis can NOT be the same")  
  if split_axis == 0:
    data_size = len(data)
    batch_size = data_size // nr_batch
  elif split_axis == 1:  
    data_size = len(data[0]) 
    batch_size = data_size // nr_batch
  else:
    raise Exception(f"split_aixs = {split_axis} not suported")  

  # pdb.set_trace()  

  tokenizers_parallelism = None

  if "TOKENIZERS_PARALLELISM" in os.environ:
    tokenizers_parallelism = os.environ["TOKENIZERS_PARALLELISM"]
  
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  try:  

    # for first_idx in range(0, data_size, batch_size):
    #   gc.collect()

    #   if split_axis == 0:
    #     batch = data[first_idx:(first_idx + batch_size)]
    #   else:
    #     batch = [row[first_idx:(first_idx + batch_size)] for row in data]

    #   batch = np.array(batch, dtype=np.float32)
    #   batch = np.mean(batch, axis = axis)

    #   new_values.extend(batch.tolist())

    # pdb.set_trace()        

    pool = Pool(processes=nr_batch)
    result_list = []

    for first_idx in range(0, data_size, batch_size):
      if split_axis == 0:
        batch = data[first_idx:(first_idx + batch_size)]
      else:
        batch = [row[first_idx:(first_idx + batch_size)] for row in data]

      result = pool.apply_async(exec_mean, args=[batch, axis])
      result_list.append(result)

    # pdb.set_trace()  
    pool.close()  
    pool.join()

    # pdb.set_trace()  

    for result in result_list:
      new_values.extend(result.get())
  finally:   
    # pdb.set_trace() 
    gc.collect()

    if tokenizers_parallelism is None:
      del os.environ["TOKENIZERS_PARALLELISM"]
    else:  
      os.environ["TOKENIZERS_PARALLELISM"] = tokenizers_parallelism

  return np.array(new_values)
