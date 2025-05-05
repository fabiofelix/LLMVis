import os, glob, json, numpy as np, time, re
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')

def parse_obj_path(obj_path):
  obj_name = os.path.basename(obj_path)
  split_out = obj_name.split("_")
  dataset = model = obj_type = opt = opt_type = None

  ##text
  if len(split_out) == 2: 
    dataset, obj_type = split_out
    obj_type, _ = obj_type.split(".")
  else:   
    dataset, model, obj_type, opt = split_out

    if "-" in opt:
      opt, opt_type = opt.split("-")
      opt_type, _ = opt_type.split(".")
    ##token info  
    else: 
      opt, _ = opt.split(".")

  ## dataset: <dataset>-<number_samples>
  ## model:   <model>, <model>-b<model_block> 
  ## obj_type: sentence, token, class,       text
  ## opt:      proj,     info,  explanation, None
  ## opt_type: PCA,             LIME
  ##           UMAP,            SHAP
  ##           t-SNE 
  return dataset, model, obj_type, opt, opt_type

@app.route("/")
def index():
  return render_template('index.html')

@app.route("/config")
def load_config():
  objects = glob.glob("data/*npz")
  objects.sort()
  #models: all models listed in the data folder
  config = {"models": []}

  for obj_path in objects:
    dataset, model, obj_type, _, _ = parse_obj_path(obj_path)

    if obj_type not in ["text", "token"]:
      config["models"].append(dataset + "_" + model)

  config["models"] = np.unique(config["models"]).tolist()
  config["models"].sort()

  return config

def process_sentence(config, obj_path, dataset, model, obj_type, opt, opt_type):
  print("|- Loading sentence projection")
  data = np.load(obj_path, allow_pickle=True)
  config["objs"].append(
    {
      "type": "projection",
      "name": opt_type,
      "source": "sentence",
      "ids": data["text_ids"].tolist(),
      "label": None,
      "data": data["projection"].tolist(),
      "silhouette": data["silhouette_score"].item()
    })
  
  return config["objs"][-1]

def process_token(config, obj_path, dataset, model, obj_type, opt, opt_type):
  print("|- Loading token info")
  data = np.load(obj_path, allow_pickle=True)
  config["objs"].append(
    {
      "type": "word",
      "name": opt_type,
      "source": "token",
      "ids": data["token_ids"].tolist(),
      "data": {
        "sentences": data["text_ids"].tolist(),
        "label": data["text_label"].tolist(),
        "position": data["position"].tolist(),
        "postag": data["postag"].tolist(),
        "named_entity": data["named_entity"].tolist()   
      }  
    })
  
  return config["objs"][-1]
      
def process_dataset_text(config, obj_path, dataset, model, obj_type, opt, opt_type):
  print("|- Loading text")
  data = np.load(obj_path, allow_pickle=True)
  config["objs"].append(
    {
      "type": "text",
      "name": opt_type,
      "source": "text",
      "ids": data["text_ids"].tolist(),
      "data": {
        "text": data["text"].tolist(),
        "processed_text": data["processed_text"].tolist(),
        "label": data["topic"].tolist() if data["label"][0] is None else data["label"].tolist()
      }
    })

  return config["objs"][-1]

def process_explanation(config, obj_path, dataset, model, obj_type, opt, opt_type):
  print("|- Loading explanation")
  data = np.load(obj_path, allow_pickle=True)
  config["objs"].append(
    {
      "type": "explanation",
      "name": opt_type,
      "source": "class",
      "ids": data["class_ids"].tolist(),
      "data": {
        "sentences": data["text_ids"].tolist(),
        "label": [],
        "tokens": data["token_ids"].tolist(),
        "explanations": data["explanation"].tolist(),
        "class_report": data["class_report"].tolist(),
        "position": [],
        "postag": [],
        "named_entity": []
      }  
    })

  return config["objs"][-1]

@app.route("/filter", methods = ["POST"])
def filter():
  filter_cfg = request.get_json(force = True)
  #models: selected model
  #projections: possible projections for the selected model
  #distances: possible distances for the selected model
  #objs: first listed projection/distance for the selected model, token clusters, and texts
  config = {"models": [ filter_cfg["config"]["model"] ], "projections": [], "explanations": [], "objs": []}

  filter_dataset, filter_model_block = filter_cfg["config"]["model"].split("_")
  filter_model = re.sub(r"-b[1-9]\d", "", filter_model_block)

  objects = glob.glob("data/*npz")
  objects.sort()

  #Always use the first listed projection
  processed_stn_proj  = False
  #Always use the first listed explanation
  processed_class_exp = False

  text = None
  tkn  = None
  proj = None
  exp  = None

  for obj_path in objects:
    dataset, model, obj_type, opt, opt_type = parse_obj_path(obj_path)

    if dataset == filter_dataset:
      if obj_type == "text" and filter_cfg["type"] in ["model", "projection"]:
        text = process_dataset_text(config, obj_path, dataset, model, obj_type, opt, opt_type)

      elif obj_type == "token" and filter_model in model and filter_cfg["type"] in ["model", "explanation"]:
        tkn = process_token(config, obj_path, dataset, model, obj_type, opt, opt_type)

      elif obj_type == "sentence" and filter_model_block in model:
        config["projections"].append(opt_type)

        if ((filter_cfg["type"] == "model" and not processed_stn_proj) or
            (filter_cfg["type"] == "projection" and filter_cfg["config"]["projection"] == opt_type)):
          processed_stn_proj = True
          proj = process_sentence(config, obj_path, dataset, model, obj_type, opt, opt_type)

      elif obj_type == "class" and filter_model_block in model:
        config["explanations"].append(opt_type)

        if ((filter_cfg["type"] == "model" and not processed_class_exp) or
            (filter_cfg["type"] == "explanation" and filter_cfg["config"]["explanation"] == opt_type)):
          processed_class_exp = True
          exp = process_explanation(config, obj_path, dataset, model, obj_type, opt, opt_type)

  if text is not None and proj is not None:
    print("|-- Copying text labels to projection")

    proj["label"] = text["data"]["label"]
  if tkn is not None and exp is not None:
    print("|-- Copying token info to explanation")

    for tkn_id in exp["data"]["tokens"]:
      idx = tkn["ids"].index(tkn_id)

      exp["data"]["label"].append(tkn["data"]["label"][idx])
      exp["data"]["position"].append(tkn["data"]["position"][idx])
      exp["data"]["postag"].append(tkn["data"]["postag"][idx])
      exp["data"]["named_entity"].append(tkn["data"]["named_entity"][idx])

  print("|- Sending config")

  return config

if __name__ == "main":
  ##Development/debug
  app.run()
  ##Production
  #from waitress import serve
  #serve(app, host="0.0.0.0", port=5000)
