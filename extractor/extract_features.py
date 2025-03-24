
import sys, os, argparse, pdb, numpy as np, tqdm, torch, random, pandas as pd, gc
import datasets, models, utils
import umap

from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder

from nlp import nltk_extract_entity, spacy_extract_entity, has_letter
from process import mean_pooling, tag_indexOf, expand_token_axis, mean_list
from xai import MyLime, MyShap

def extract_token_info(token_desc, token_pos, text_data):
  tokens = {}
  text_data = text_data.set_index("name")

  for idx, (_, text) in enumerate(text_data.iterrows()):
    tags, entities = nltk_extract_entity(text.text)
    # tags, entities = spacy_extract_entity(text.text)

    for desc, pos in zip(token_desc[idx], token_pos[idx]):
      desc = desc.item()
      original_word = "" if pos is None else text.text[pos[1][0]:pos[1][1]].strip()
      index = tag_indexOf(original_word, tags)

      if desc not in tokens:
        tokens[desc] = { "sentences":[], "position":[], "postag": [], "entity": [], "word": [] }

      tokens[desc]["sentences"].append(text.name)
      tokens[desc]["position"].append(pos)
      tokens[desc]["postag"].append(None if index == -1 else tags[index])
      tokens[desc]["entity"].append(None if index == -1 else entities[index])
      tokens[desc]["word"].append(original_word)

      if index > -1:
        tags[index][0].replace(original_word, "")

  return tokens

## token:   <dataset>-<number_samples>_<model>_token_info.npz
def save_token_info(args, token_desc, token_pos, text_data, idx2tkn, pattern_file, update = False):
  print("|- Saving info (token)")
  tokens_info = extract_token_info(token_desc, token_pos, text_data)
  ##Force all information has the same order of idx2tkn
  tokens_info = {idx2tkn[idx]: tokens_info[idx2tkn[idx]] for idx in idx2tkn}

  tkn_ids = []
  stn_ids = []
  tkn_pos = []
  postag  = []
  entity  = []
  word    = []

  for k in tokens_info.keys():
    tkn_ids.append(k)
    stn_ids.append(tokens_info[k]["sentences"])
    tkn_pos.append(tokens_info[k]["position"])
    postag.append(tokens_info[k]["postag"])
    entity.append(tokens_info[k]["entity"])
    word.append(tokens_info[k]["word"])

  file_path = f"{pattern_file}_token_info.npz"
  file_path = os.path.join(args.output_path, file_path)

  if update:
    update_file = np.load(file_path, allow_pickle=True)
    update_file = dict(update_file)
    # update_file["named_entity"] = np.array(entity, dtype = object)
    # update_file["postag"] = np.array(postag, dtype = object)
    # update_file["word"] = np.array(word, dtype = object)
    # np.savez(file_path, **update_file)
  else:  
    np.savez(file_path, 
            token_ids = tkn_ids, 
            text_ids = np.array(stn_ids, dtype = object), 
            position = np.array(tkn_pos, dtype = object),
            postag = np.array(postag, dtype = object),
            named_entity = np.array(entity, dtype = object),
            word = np.array(word, dtype = object)
           )
    
  return tokens_info, tkn_ids, stn_ids    


## explanation: <dataset>-<number_samples>_<model>-b<model_block>_class_explanation-<explanation_name>.npz
def save_explanation(args, text_token, tkn_ids, stn_ids, labels, pattern_file, update = False):
  print("|- Saving explanations (class)")
  le = LabelEncoder()
  labels = le.fit_transform(labels)

  for key in text_token:
    for explain_name in ["LIME", "SHAP"]:
      if explain_name == "LIME":
        explainer = MyLime()
      elif explain_name == "SHAP":
        explainer = MyShap()        

      exp = explainer.run(text_token[key], labels, tkn_ids, le.classes_)
      predicted_label = exp.predicted_label.unique()
      exp = exp.set_index("predicted_label")
      exp_abs_mean = []

      for lb in le.classes_:
        abs_mean = pd.Series([0] * len(tkn_ids), index = tkn_ids)

        if lb in predicted_label:
          abs_mean = exp.loc[lb, tkn_ids].abs() if len(exp.loc[lb, tkn_ids].shape) == 1 else exp.loc[lb, tkn_ids].abs().mean()

        abs_mean = pd.concat([pd.Series(lb, index = ["predicted_label"]), abs_mean])
        exp_abs_mean.append(abs_mean)

      exp_abs_mean = pd.DataFrame(exp_abs_mean)

      file_path = pattern_file.format(key) + "_class_explanation-{}.npz".format(explain_name)
      file_path = os.path.join(args.output_path, file_path)

      np.savez(file_path, 
               class_ids = exp_abs_mean.predicted_label.to_numpy(), 
               token_ids = tkn_ids, 
               text_ids = np.array(stn_ids, dtype = object), 
               explanation=[ row[1:].to_numpy()  for _, row in exp_abs_mean.iterrows() ]
               )

## projection: <dataset>-<number_samples>_<model>-b<model_block>_sentence_proj-<projection_name>.npz
def save_projection(args, text_feat, sentence_name, sentence_label, pattern_file, update = False):
  print("|- Saving projections (text)")

  for key in text_feat:
    sentence_feat = np.array(text_feat[key])

    for proj_name in ["PCA", "tSNE", "UMAP"]:
      projection = PCA(n_components = 2, random_state = utils.SEED_VALUE)    

      if proj_name == "tSNE":
        projection  = TSNE(n_components = 2, random_state = utils.SEED_VALUE)
      elif proj_name == "UMAP":
        projection = umap.UMAP(n_components = 2, random_state = utils.SEED_VALUE)

      vis_proj = projection.fit_transform(sentence_feat)
      sh = silhouette_score(vis_proj, sentence_label) 

      file_path = pattern_file.format(key) + "_sentence_proj-{}.npz".format(proj_name)
      file_path = os.path.join(args.output_path, file_path)

      np.savez(file_path, text_ids = sentence_name, projection = vis_proj, silhouette_score=sh)

## text: <dataset>-<number_samples>_text.npz
def save_text(args, text_data, pattern_file_data, update = False):
  print("|- Saving data text")

  file_path = os.path.join(args.output_path, f"{pattern_file_data}_text.npz")

  np.savez(file_path, 
           text_ids = text_data.name.to_numpy(), 
           text = text_data.text.to_numpy(), 
           processed_text = text_data.processed_text.to_numpy(),
           label = text_data.label.to_numpy(),
           topic = text_data.topic.to_numpy() )

## features: <dataset>-<number_samples>_<model>-b<model_block>_<first_idx>-<last_idx>.npz
def run(args, parser):
  random.seed(utils.SEED_VALUE)
  np.random.seed(utils.SEED_VALUE)
  torch.manual_seed(utils.SEED_VALUE)

  dataset = datasets.create_dataset(args.dataset, args.source_path)
  model = models.create_model(args.model)

  print("|- Loading data")
  text_data = dataset.get_data(args.nsample)
  nr_samples = text_data.shape[0]
  pattern_size = len(str(text_data.shape[0]))

  pattern_file_data = f"{dataset}-{nr_samples}"
  pattern_file_data_model = f"{dataset}-{nr_samples}_{model}"
  pattern_file_data_model_block = f"{dataset}-{nr_samples}_{model}" + "-b{:d}"
  pattern_file = pattern_file_data_model_block + "_{" + f":0{pattern_size}d" + "}-{" + f":0{pattern_size}d" + "}.npz"

  text_feat  = {}
  token_feat = {}
  text_token = {}
  text_token_feat = {}

  token_ids = []
  token_desc = []
  token_pos = []
  attn_mask  = []

  if not os.path.isdir(os.path.join(args.output_path, "original_feat")):
    os.makedirs(os.path.join(args.output_path, "original_feat"))

  nr_batch = text_data.shape[0] // args.batch_size
  nr_batch = nr_batch if nr_batch * args.batch_size == text_data.shape[0] else nr_batch + 1

  for first_idx in tqdm.tqdm(range(0, text_data.shape[0], args.batch_size), desc = "|- " + str(dataset) + " " + str(model) + " Batch" , total = nr_batch, unit= "step"):  
    batch = text_data.iloc[first_idx:(first_idx + args.batch_size)]
    block_idx, block_feature, tkn_ids, masks = model.extract_feature(batch.text.tolist())

    for b_idx, feat in zip(block_idx, block_feature):
      file_path = os.path.join(args.output_path, "original_feat", pattern_file.format(b_idx, first_idx, first_idx + args.batch_size - 1))
      np.savez(file_path, 
               text_ids  = batch.name.to_numpy(), 
               text_feat = feat, 
               token_ids = tkn_ids, 
               attn_mask = masks)

      if b_idx not in text_feat:
        text_token_feat[b_idx] = []        
        text_feat[b_idx]  = []

      text_token_feat[b_idx].extend(feat)
      text_feat[b_idx].extend(mean_pooling(feat, masks, axis = 1))

    tkn_desc, tkn_pos = model.decode_tokens(batch.text.tolist(), tkn_ids)
    token_ids.extend(tkn_ids)
    token_desc.extend(tkn_desc)
    token_pos.extend(tkn_pos)
    attn_mask.extend(masks)

  ## It's necessary to left-pad the tokens again because batches can have different number of tokens.
  ## After running pad_tokens:
  ##   token_ids.shape  (batch_size, max_tokens)
  ##   token_desc.shape (batch_size, max_tokens)
  ##   token_pos.shape  (batch_size, max_tokens)
  ##   attn_mask.shape  (batch_size, max_tokens)
  ##   text_token_feat.shape  (batch_size, _max_tokens)
  ## Note: 'max_tokens' is the MAX number of tokens per sample
  print("|- Padding tokens")
  token_ids, token_desc, token_pos, attn_mask = model.pad_tokens(token_ids, token_desc, token_pos, attn_mask)

  for key in text_token_feat:
    text_token_feat[key] = model.pad_features(text_token_feat[key])

  ## After running expand_token_axis
  ##   text_token_feat[i].shape (batch_size, filtered_tokens, features)
  ##   len(idx2tkn)             (filtered_tokens)
  ##   len(tkn2idx)             (filtered_tokens)
  ## Note: 'tokens' is the TOTAL number of tokens
  ##        idx2tkn = { idx: token_desc  }
  ##        tkn2ids = { token_desc: idx  }  
  text_token_feat, idx2tkn, tkn2idx = expand_token_axis(text_token_feat, token_ids, token_desc, model)
  gc.collect()

  for idx in text_token_feat:
    text_token[idx] = mean_list(text_token_feat[idx], axis = 2, split_axis=0)

  gc.collect()
  labels = text_data.topic.to_numpy() if text_data.iloc[0].label is None else text_data.label.to_numpy()

  #pdb.set_trace()
  ## After running save_token_info
  ##   len(tokens_info) (tokens)
  ##   len(tkn_ids)     (tokens)
  ##   len(stn_ids)     (tokens, sentences)
  ## Note: 'tokens' is the TOTAL number of tokens
  ##       token_info = { token_desc: <INFO_OBJ>  }
  tokens_info, tkn_ids, stn_ids = save_token_info(args, token_desc, token_pos, text_data, idx2tkn, pattern_file_data_model)
  save_explanation(args, text_token, tkn_ids, stn_ids, labels, pattern_file_data_model_block)

  save_projection(args, text_feat, text_data.name.to_numpy(), labels, pattern_file_data_model_block)
  
  save_text(args, text_data, pattern_file_data)

  print("|- {} samples - {} tokens".format(text_data.shape[0], len(tkn_ids)))
def main(*args):
  parser = argparse.ArgumentParser(description="")

  parser.add_argument("-d", "--data", help = "Dataset index", dest = "dataset", choices = [d for d in datasets.DATASET ], required = True, type = int)
  parser.add_argument("-m", "--model", help = "Model index", dest = "model", choices = [m for m in models.MODEL], required = True, type = int)  
  parser.add_argument("-s", "--source", help = "Path to load the dataset", dest = "source_path", required = False, default = None)
  parser.add_argument("-o", "--output", help = "Path to save results", dest = "output_path", required = True)
  parser.add_argument("-n", "--nsample", help = "Number of samples to load from the dataset", dest = "nsample", required = False, default = 100, type = int)
  parser.add_argument("-b", "--batch", help = "Batch size to process the dataset", dest = "batch_size", required = False, default = 100, type = int)

  parser.set_defaults(func = run)
  
  args = parser.parse_args()
  args.func(args, parser) 

#This only executes when this file is executed rather than imported
if __name__ == '__main__':
  main(*sys.argv[1:])
