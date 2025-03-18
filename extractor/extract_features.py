
import sys, os, argparse, pdb, numpy as np, tqdm, torch, random
import datasets, models, utils
import umap

from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from topic_modeling import nltk_extract_entity

##https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
##https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
##https://stackoverflow.com/questions/76926025/sentence-embeddings-from-llama-2-huggingface-opensource
##https://github.com/run-llama/llama_index/blob/main/llama_index/embeddings/huggingface.py#L133
##https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
def mean_pooling(features, attention_mask, axis = 1):
  expanded_attention_mask = attention_mask.unsqueeze(-1).expand(features.size()).float()
  numerator = (features * expanded_attention_mask).sum(axis=axis)
  mean = numerator / expanded_attention_mask.sum(axis=axis).clamp(min=1e-9)

  return mean.numpy()

def process_token(token, calc_avg = False):
  token_ids = []
  text_ids  = []
  features  = []
  token_pos = []
  avg_features  = []

  for k in token.keys():
    token_ids.append(k)
    text_ids.append(token[k]["text_ids"])
    features.append(token[k]["features"])
    token_pos.append(token[k]["token_pos"])

    if calc_avg:
      avg_features.append( np.mean(token[k]["features"], axis = 0) )

  return token_ids, token_pos, text_ids, avg_features if calc_avg else features

def merge_token_batch(token_feat, token):
  for k in token.keys():
    if k not in token_feat:
      token_feat[k] = {"text_ids":[], "features": [], "token_pos": []}

    token_feat[k]["text_ids"].extend(token[k]["text_ids"])
    token_feat[k]["features"].extend(token[k]["features"])
    token_feat[k]["token_pos"].extend(token[k]["token_pos"])

def cluster_distance(dist_value, sentence_name):
  #Sturge's rule
  n_clusters = int(1.0 + 3.322 * np.log10(dist_value.shape[0]))
  model      = KMeans(n_clusters = n_clusters, random_state=utils.SEED_VALUE) ##better silhouette score 

  clustering = model.fit(dist_value)
  clusters = np.unique(clustering.labels_)

  order_rows = []
  order_names = []

  for cl in clusters:
    indices = np.where(model.labels_ == cl)[0]
    order_rows.extend(dist_value[indices, :])
    order_names.extend(sentence_name[indices])

  order_rows = np.array(order_rows).T
  order_cols = []

  for cl in clusters:
    indices = np.where(model.labels_ == cl)[0]
    order_cols.extend(order_rows[indices, :])

  return np.array(order_cols).T, np.array(order_names)

## distance:   <dataset>-<number_samples>_<model>_sentence_dist-<dist_name>.npz
def save_distances(args, sentence_feat, token_feat, sentence_name, sentence_label, model_name, data_name, cluster=True):
  print("|- Saving distances (sentence)")
  sh = silhouette_score(sentence_feat, sentence_label)

  for dist_metric in ["cosine", "euclidean"]:
    dist_value = pairwise_distances(sentence_feat, metric = dist_metric)

    if cluster:
      dist_value, sentence_name = cluster_distance(dist_value, sentence_name)

    file_path = "{}-{}_{}_sentence_dist-{}.npz".format(data_name, len(sentence_feat), model_name, dist_metric)
    file_path = os.path.join(args.output_path, file_path)

    np.savez(file_path, text_ids = sentence_name, similarity = dist_value, hd_silhouette_score = sh)    

def cluster_main_token(tkn_ids, tkn_feat, clusters):
  tkn_feat = np.array(tkn_feat)
  tkn_ids  = np.array(tkn_ids)
  rep_samples = []

  for clt in np.unique(clusters):
    tkn_cluster = tkn_feat[clusters == clt]
    centroid  = tkn_cluster.mean(axis = 0)
    aux_feat  = np.append(centroid[None], tkn_cluster, axis = 0)
    #distances from the average centroid
    dist_feat = pairwise_distances(aux_feat, metric = "euclidean")
    #five more important tokens
    indices   = dist_feat[0].argsort()[1:6] - 1
    rep_samples.append( tkn_ids[clusters == clt][indices].tolist() )

  return rep_samples

def tag_indexOf(token, tags):
  index = -1

  for idx, tg in enumerate(tags):
    if token in tg[0]:
      return idx

  return index

def extract_token_info(tkn_ids, tkn_pos, stn_ids, text_data):
  token_tag = []
  token_entity = []
  token_word = []
  sentence_cache = {}
  text_data = text_data.set_index("name")

  for tkn, pos, stn in zip(tkn_ids, tkn_pos, stn_ids):
    tag_aux = []
    entity_aux = []
    word_aux = []

    for txt, p in zip(stn, pos):
      if txt not in sentence_cache:
        sentence_cache[txt] = {"tags": None, "entities": None, txt: None}
        txt_aux = text_data.loc[txt].text
        tags, entities = nltk_extract_entity(txt_aux)
        # tags, entities = spacy_extract_entity(txt_aux)

        sentence_cache[txt]["tags"] = tags
        sentence_cache[txt]["entities"] = entities
        sentence_cache[txt]["txt"] = txt_aux

      original_word = sentence_cache[txt]["txt"][p[1][0]:p[1][1]].strip()
      index = tag_indexOf(original_word, sentence_cache[txt]["tags"])

      if index >= 0:
        tag_aux.append(sentence_cache[txt]["tags"][index])
        entity_aux.append(sentence_cache[txt]["entities"][index])
        word_aux.append(original_word)
        sentence_cache[txt]["tags"][index][0].replace(original_word, "")

    token_tag.append(tag_aux)
    token_entity.append(entity_aux)
    token_word.append(word_aux)

  return token_tag, token_entity, token_word

## token:   <dataset>-<number_samples>_<model>_token_info.npz
def save_token_info(args, token_feat, text_data, model_name, data_name, update = False):
  print("|- Saving info (token)")
  tkn_ids, tkn_pos, stn_ids, _ = process_token(token_feat)
  postag, entity, word = extract_token_info(tkn_ids, tkn_pos, stn_ids, text_data)

  file_path = "{}-{}_{}_token_info.npz".format(data_name, text_data.shape[0], model_name)
  file_path = os.path.join(args.output_path, file_path)

  if update:
    update_file = np.load(file_path, allow_pickle=True)
    update_file = dict(update_file)
    update_file["named_entity"] = np.array(entity, dtype = object)
    update_file["postag"] = np.array(postag, dtype = object)
    update_file["word"] = np.array(word, dtype = object)
    np.savez(file_path, **update_file)
  else:  
    np.savez(file_path, 
            token_ids = tkn_ids, 
            text_ids = np.array(stn_ids, dtype = object), 
            position = np.array(tkn_pos, dtype = object),
            postag = np.array(postag, dtype = object),
            named_entity = np.array(entity, dtype = object),
            word = np.array(word, dtype = object)
           )

## cluster:   <dataset>-<number_samples>_<model>_token_cluster-<name>.npz
def save_clusters(args, token_feat, n_samples, model_name, data_name):
  print("|- Saving clusters (token)")

  tkn_ids, _, stn_ids, tkn_avg_feat = process_token(token_feat, calc_avg=True)

  #Sturge's rule
  n_clusters = int(1.0 + 3.322 * np.log10(len(tkn_avg_feat)))

  aux_dist = pairwise_distances(tkn_avg_feat, metric = "euclidean")
  lower_dist = aux_dist[np.tril_indices(aux_dist.shape[0])]
  lower_dist = np.unique(lower_dist)
  lower_dist.sort()
  epsilon = utils.otsu_threshold(lower_dist)

  for alg in ["dbscan", "hierarchical", "kmeans"]:
    if alg == "dbscan":      
      model = DBSCAN(eps=epsilon)
    elif alg == "hierarchical":      
      model = AgglomerativeClustering(n_clusters=n_clusters)
    elif alg == "kmeans":
      model = KMeans(n_clusters=n_clusters, random_state=utils.SEED_VALUE) 

    clustering = model.fit(tkn_avg_feat)
    rep_cluster_tkn = cluster_main_token(tkn_ids, tkn_avg_feat, clustering.labels_)

    file_path = "{}-{}_{}_token_cluster-{}.npz".format(data_name, n_samples, model_name, alg)
    file_path = os.path.join(args.output_path, file_path)

    np.savez(file_path, 
             token_ids = tkn_ids, 
             text_ids = np.array(stn_ids, dtype = object), 
             clusters = clustering.labels_, 
             clusters_main_token = np.array(rep_cluster_tkn, dtype = object)
             )

## projection: <dataset>-<number_samples>_<model>_sentence_proj-<projection_name>.npz
def save_projection(args, sentence_feat, sentence_name, sentence_label, model_name, data_name):
  print("|- Saving projections (sentence)")

  sentence_feat = np.array(sentence_feat)

  for proj_name in ["PCA", "tSNE", "UMAP"]:
    projection = PCA(n_components = 2, random_state = utils.SEED_VALUE)    

    if proj_name == "tSNE":
      projection  = TSNE(n_components = 2, random_state = utils.SEED_VALUE)
    elif proj_name == "UMAP":
      projection = umap.UMAP(n_components = 2, random_state = utils.SEED_VALUE)

    vis_proj = projection.fit_transform(sentence_feat)
    sh = silhouette_score(vis_proj, sentence_label) 

    file_path = "{}-{}_{}_sentence_proj-{}.npz".format(data_name, len(sentence_feat), model_name, proj_name)
    file_path = os.path.join(args.output_path, file_path)

    np.savez(file_path, text_ids = sentence_name, projection = vis_proj, silhouette_score=sh)

## text: <dataset>-<number_samples>_text.npz
def save_text(args, text_data, data_name):
  print("|- Saving data text")

  nr_samples = text_data.shape[0]
  file_path = os.path.join(args.output_path, f"{data_name}-{nr_samples}_text.npz")

  np.savez(file_path, 
           text_ids = text_data.name.to_numpy(), 
           text = text_data.text.to_numpy(), 
           processed_text = text_data.processed_text.to_numpy(),
           label = text_data.label.to_numpy(),
           topic = text_data.topic.to_numpy() )
  
## features: <dataset>-<number_samples>_<model>_<first_idx>-<last_idx>.npz
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
  pattern_file = f"{dataset}-{nr_samples}_{model}" + "_{" + f":0{pattern_size}d" + "}-{" + f":0{pattern_size}d" + "}.npz"
  sentence_feat = []
  token_feat = {}

  if not os.path.isdir(os.path.join(args.output_path, "sentence_feat")):
    os.makedirs(os.path.join(args.output_path, "sentence_feat"))
  if not os.path.isdir(os.path.join(args.output_path, "token_feat")):
    os.makedirs(os.path.join(args.output_path, "token_feat"))    

  nr_batch = text_data.shape[0] // args.batch_size
  nr_batch = nr_batch if nr_batch * args.batch_size == text_data.shape[0] else nr_batch + 1

  for first_idx in tqdm.tqdm(range(0, text_data.shape[0], args.batch_size), desc = str(dataset) + " " + str(model) + " Batch" , total = nr_batch, unit= "step"):  
    batch = text_data.iloc[first_idx:(first_idx + args.batch_size)]
    tkn_layer_feat, token, attn_mask = model.extract_feature(batch.name.tolist(), batch.text.tolist())

    stn_feat = mean_pooling(tkn_layer_feat[0], attn_mask)
    tkn_ids, tkn_pos, stn_ids, tkn_feat = process_token(token[0])

    sentence_feat.extend(stn_feat)
    merge_token_batch(token_feat, token[0])

    file_path = os.path.join(args.output_path, "sentence_feat", pattern_file.format(first_idx, first_idx + args.batch_size - 1))
    np.savez(file_path, text_ids = batch.name.to_numpy(), sentence_feat = stn_feat )

    file_path = os.path.join(args.output_path, "token_feat", pattern_file.format(first_idx, first_idx + args.batch_size - 1))
    np.savez(file_path, 
             token_ids = tkn_ids,                              #all the read tokens
             text_ids = np.array(stn_ids, dtype = object),     #list of sentences per token
             token_feat = np.array(tkn_feat, dtype = object),  #list of sentence features per token
             token_pos = np.array(tkn_pos, dtype = object)     #list of token positions in a sentence
             )    

  labels = text_data.topic.to_numpy() if text_data.iloc[0].label is None else text_data.label.to_numpy()

  # save_distances(args, sentence_feat, token_feat, text_data.name.to_numpy(), labels, str(model), str(dataset))
  save_clusters(args, token_feat, text_data.shape[0], str(model), str(dataset))
  save_projection(args, sentence_feat, text_data.name.to_numpy(), labels, str(model), str(dataset))
  save_token_info(args, token_feat, text_data, str(model), str(dataset))
  save_text(args, text_data, str(dataset))

  print("|- {} samples - {} tokens".format(text_data.shape[0], len(token_feat)))

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
