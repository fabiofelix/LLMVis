
import sys, os, argparse, pdb, numpy as np, tqdm, torch, random, pandas as pd
import datasets, models, utils
import umap

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from nlp import nltk_extract_entity
from process import mean_pooling, tag_indexOf, aggregate_feature_axis
from xai import MyLime, MyShap

def extract_token_info(token_desc, token_pos, text_data):
  tokens = {}
  text_data = text_data.set_index("name")

  for idx, (_, text) in enumerate(text_data.iterrows()):
    tags, entities = nltk_extract_entity(text.text)

    for desc, pos in zip(token_desc[idx], token_pos[idx]):
      desc = desc.strip()
      index = tag_indexOf(pos, tags)

      if desc not in tokens:
        tokens[desc] = { "sentences":[], "label": [], "position":[], "postag": [], "entity": []}

      tokens[desc]["sentences"].append(text.name)
      tokens[desc]["label"].append(text.topic if text.label is None else text.label)
      tokens[desc]["position"].append(pos)
      tokens[desc]["postag"].append(None if index == -1 else tags[index])
      tokens[desc]["entity"].append(None if index == -1 else entities[index])
      
    # pdb.set_trace()        

  # pdb.set_trace()
  return tokens

## token:   <dataset>-<number_samples>_<model>_token_info.npz
def save_token_info(args, token_desc, token_pos, text_data, idx2tkn, pattern_file):
  print("|- Saving info (token)")
  tokens_info = extract_token_info(token_desc, token_pos, text_data)
  #Force all information has the same order of idx2tkn
  tokens_info = {idx2tkn[idx]: tokens_info[idx2tkn[idx]] for idx in idx2tkn}

  tkn_ids = []
  stn_ids = []
  stn_label = []
  tkn_pos = []
  postag  = []
  entity  = []

  for k in tokens_info.keys():
    tkn_ids.append(k)
    stn_ids.append(tokens_info[k]["sentences"])
    stn_label.append(tokens_info[k]["label"])
    tkn_pos.append(tokens_info[k]["position"])
    postag.append(tokens_info[k]["postag"])
    entity.append(tokens_info[k]["entity"])

  file_path = f"{pattern_file}_token_info.npz"
  file_path = os.path.join(args.output_path, file_path)

  np.savez(file_path, 
          token_ids = tkn_ids, 
          text_ids = np.array(stn_ids, dtype = object), 
          text_label = np.array(stn_label, dtype = object),
          position = np.array(tkn_pos, dtype = object),
          postag = np.array(postag, dtype = object),
          named_entity = np.array(entity, dtype = object)
          )

  return np.array(tkn_ids), np.array(stn_ids, dtype = object)

## explanation: <dataset>-<number_samples>_<model>-b<model_block>_class_explanation-<explanation_name>.npz
def save_explanation(args, text_token, tkn_ids, stn_ids, labels, pattern_file):
  print("|- Saving explanations (class)")

  # Filter out outlier tokens indices based on their frequency (stn_ids)
  filter = utils.get_filtered_indices([ len(ids) for ids in stn_ids ], lower_threshold=3)
  tkn_ids = tkn_ids[filter]
  stn_ids = stn_ids[filter]   

  print("|-- Using {:10d} tokens".format(len(tkn_ids)))

  for key, txt_tokens in text_token.items():
    txt_tokens = np.array(txt_tokens)
    sh_original = silhouette_score(txt_tokens, labels)

    for explain_name, explain_class in [("LIME", MyLime), ("SHAP", MyShap)]:
      file_path = pattern_file.format(key) + "_class_explanation-{}.npz".format(explain_name)
      file_path = os.path.join(args.output_path, file_path)

      filtered_text_token = txt_tokens[:, filter]
      sh_filtered = silhouette_score(filtered_text_token, labels)

      explainer = explain_class(random_state=utils.SEED_VALUE)
      exp, class_eval = explainer.run(filtered_text_token, labels, tkn_ids, test_samples_per_class = 10 if args.dataset in [datasets.DATASET.AMAZONREVIEW, datasets.DATASET.TINYSTORIESV2] else 5)
      predicted_label = exp.predicted_label.unique()
      exp = exp.set_index("predicted_label")

      exp_abs_mean = []
      class_report = []
      exp_report = exp.infidelity.to_numpy()

      for lb in explainer.label_encoder.classes_:
        abs_mean = pd.Series([0] * len(tkn_ids), index = tkn_ids)

        if lb in predicted_label:
          row = exp.loc[lb, tkn_ids]
          abs_mean = row.abs() if row.ndim == 1 else row.abs().mean()

        abs_mean = pd.concat([pd.Series(lb, index = ["predicted_label"]), abs_mean])
        exp_abs_mean.append(abs_mean) 

        class_report.append([ class_eval[lb]["precision"], class_eval[lb]["recall"], class_eval[lb]["f1-score"], class_eval["accuracy"] ])  

      exp_abs_mean = pd.DataFrame(exp_abs_mean) 

      np.savez(file_path, 
              class_ids = exp_abs_mean.predicted_label.to_numpy(), 
              token_ids = tkn_ids, 
              text_ids = stn_ids, 
              explanation=[ row[1:].to_numpy()  for _, row in exp_abs_mean.iterrows() ],
              class_report = class_report,
              exp_report = exp_report,
              silhouette_score=sh_filtered,
              silhouette_score_original=sh_original,
              )                    

## projection: <dataset>-<number_samples>_<model>-b<model_block>_sentence_proj-<projection_name>.npz
def save_projection(args, text_feat, sentence_name, sentence_label, pattern_file, update = False):
  print("|- Saving projections (text)")

  for key, txt_feat in text_feat.items():    
    sentence_feat = np.array(txt_feat)
    sh_original = silhouette_score(sentence_feat, sentence_label)

    for proj_name in ["PCA", "tSNE", "UMAP"]:
      file_path = pattern_file.format(key) + "_sentence_proj-{}.npz".format(proj_name)
      file_path = os.path.join(args.output_path, file_path)
      projection = PCA(n_components = 2, random_state = utils.SEED_VALUE)    

      if proj_name == "tSNE":
        projection  = TSNE(n_components = 2, random_state = utils.SEED_VALUE)
      elif proj_name == "UMAP":
        projection = umap.UMAP(n_components = 2, random_state = utils.SEED_VALUE)

      vis_proj = projection.fit_transform(sentence_feat)
      sh = silhouette_score(vis_proj, sentence_label)

      np.savez(file_path, text_ids = sentence_name, projection = vis_proj, silhouette_score=sh, silhouette_score_original=sh_original)

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

  map_tokens = utils.read_map_tokens(args.map_path)
  dataset = datasets.create_dataset(args.dataset, args.source_path, map_tokens)
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
  text_token = {}
  text_token_feat = {}

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
    token_desc.extend(tkn_desc)
    token_pos.extend(tkn_pos)
    attn_mask.extend(masks)

  ## After running expand_token_axis
  ##   text_token[i].shape (batch_size, filtered_tokens)
  ##   len(idx2tkn)             (filtered_tokens)
  ## Note: 'tokens' is the TOTAL number of tokens
  ##        idx2tkn = { idx: token_desc  }
  text_token, idx2tkn = aggregate_feature_axis(text_token_feat, token_desc)

  labels = text_data.topic.to_numpy() if text_data.iloc[0].label is None else text_data.label.to_numpy()

  save_text(args, text_data, pattern_file_data)

  ## After running save_token_info
  ##   len(tkn_ids)     (tokens)
  ##   len(stn_ids)     (tokens, sentences)
  ## Note: 'tokens' is the TOTAL number of tokens
  tkn_ids, stn_ids = save_token_info(args, token_desc, token_pos, text_data, idx2tkn, pattern_file_data_model)

  save_projection(args, text_feat, text_data.name.to_numpy(), labels, pattern_file_data_model_block)
  save_explanation(args, text_token, tkn_ids, stn_ids, labels, pattern_file_data_model_block, row_ids=text_data.name.to_numpy())
  
  unique_token_desc = np.unique([desc for row in token_desc for desc in row])
  print("|- {} samples - {} tokens - {} (filtered stop-words)".format(text_data.shape[0], len(unique_token_desc), len(tkn_ids)))

def main(*args):
  parser = argparse.ArgumentParser(description="")

  parser.add_argument("-d", "--data", help = "Dataset index", dest = "dataset", choices = [d for d in datasets.DATASET ], required = True, type = int)
  parser.add_argument("-m", "--model", help = "Model index", dest = "model", choices = [m for m in models.MODEL], required = True, type = int)  
  parser.add_argument("-s", "--source", help = "Path to load the dataset", dest = "source_path", required = False, default = None)
  parser.add_argument("-o", "--output", help = "Path to save results", dest = "output_path", required = True)
  parser.add_argument("-n", "--nsample", help = "Number of samples to load from the dataset", dest = "nsample", required = False, default = 100, type = int)
  parser.add_argument("-b", "--batch", help = "Batch size to process the dataset", dest = "batch_size", required = False, default = 100, type = int)

  parser.add_argument("-mp", "--map", help = "key-value file to map tokens", dest = "map_path", required = False, default = None)

  parser.set_defaults(func = run)
  
  args = parser.parse_args()
  args.func(args, parser) 

#This only executes when this file is executed rather than imported
if __name__ == '__main__':
  main(*sys.argv[1:])
