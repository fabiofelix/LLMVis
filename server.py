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
    dataset, model, obj_type, opt, _ = parse_obj_path(obj_path)

    if obj_type != "text" and opt != "info":
      config["models"].append(dataset + "_" + model)

  config["models"] = np.unique(config["models"]).tolist()
  config["models"].sort()

  return config

def process_sentence(config, obj_path, loaded_proj_stn, loaded_dist_stn, dataset, model, obj_type, opt, opt_type):
  if opt == "proj":
    config["projections"].append(opt_type)
    
    if not loaded_proj_stn:
      print("|- Loading sentence projection")
      loaded_proj_stn = True
      data = np.load(obj_path, allow_pickle=True)
      config["objs"].append(
        {
          "type": "projection",
          "name": opt_type,
          "source": "sentence",
          "ids": data["text_ids"].tolist(),
          "label": None,
          "topic": None,
          "data": data["projection"].tolist(),
          "silhouette": data["silhouette_score"].item()
        })
      
  return loaded_proj_stn, loaded_dist_stn       

def set_token_info(config, data, type_):
  for obj in config["objs"]:
    if obj["type"] == type_:
      obj["data"]["position"] = data["position"].tolist()
      obj["data"]["postag"] = data["postag"].tolist()
      obj["data"]["named_entity"] = data["named_entity"].tolist()   
      obj["data"]["word"] = data["word"].tolist()
      break

def process_token(config, obj_path, loaded_info_tkn, dataset, model, obj_type, opt, opt_type):
  if opt == "info":  
    if not loaded_info_tkn:
      print("|- Loading token info")

      loaded_info_tkn = True
      data = np.load(obj_path, allow_pickle=True)

      config["objs"].append(
        {
          "type": "word",
          "name": opt_type,
          "source": "token",
          "ids": data["token_ids"].tolist(),
          "label": None,
          "topic": None,          
          "data": {
            "sentences": data["text_ids"].tolist(),
            "clusters": None,
            "main_token": None,
            "position": None,
            "postag": None,
            "named_entity": None,
            "word": None
          }  
        })  

      set_token_info(config, data, "word")
      set_token_info(config, data, "explanation")

  return loaded_info_tkn
      
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
        "label": data["label"].tolist(),
        "topic": data["topic"].tolist(),
      }
    })  

def process_explanation(config, obj_path, loaded_exp, dataset, model, obj_type, opt, opt_type):
  config["explanations"].append(opt_type)

  if not loaded_exp:
    print("|- Loading explanation")
    loaded_exp = True
    start_time = time.time()
    data = np.load(obj_path, allow_pickle=True)
    # print(time.time() - start_time)
    config["objs"].append(
      {
        "type": "explanation",
        "name": opt_type,
        "source": "class",
        "ids": data["class_ids"].tolist(),
        "label": None,
        "topic": None,          
        "data": {
          "sentences": data["text_ids"].tolist(),
          "tokens": data["token_ids"].tolist(),
          "explanations": data["explanation"].tolist(),
          "clusters": None,
          "main_token": None,
          "position": None,
          "postag": None,
          "named_entity": None,
          "word": None
        }  
      })

  return loaded_exp

@app.route("/filter", methods = ["POST"])
def filter():
  filter_cfg = request.get_json(force = True)
  filter_type = filter_cfg["type"]
  data_source = filter_cfg["source"]
  #models: selected model
  #projections: possible projections for the selected model
  #distances: possible distances for the selected model
  #objs: first listed projection/distance for the selected model, token clusters, and texts
  config = {"models": [ filter_cfg["config"]["model"] ], "projections": [], "distances": [], "clusters": [], "explanations": [], "objs": []}

  objects = glob.glob("data/*npz")
  objects.sort()

  dataset, _ = filter_cfg["config"]["model"].split("_")
  filtered_objects = []

  for obj in objects:
    file_name = os.path.basename(obj)
    config_model_without_block = re.sub(r"-b[1-9]\d", "", filter_cfg["config"]["model"])

    if (filter_cfg["config"]["model"] in file_name or 
        config_model_without_block + "_token_info" in file_name or 
        dataset + "_text" in file_name):
      filtered_objects.append(obj)

  if filter_type == "projection":
    filtered_objects = [ obj for obj in filtered_objects if filter_cfg["config"]["projection"] in os.path.basename(obj) ]
  elif filter_type == "distance":
    filtered_objects = [ obj for obj in filtered_objects if filter_cfg["config"]["distance"] in os.path.basename(obj) ]    
  elif filter_type == "explanation":
    filtered_objects = [ obj for obj in filtered_objects if filter_cfg["config"]["explanation"] in os.path.basename(obj) or "token_info" in os.path.basename(obj) ]        

  loaded_proj_stn = False
  loaded_dist_stn = False
  loaded_info_tkn = False
  loaded_exp = False

  proj = None
  text = None

  for obj_path in filtered_objects:
    dataset, model, obj_type, opt, opt_type = parse_obj_path(obj_path)

    if obj_type == "sentence":
      loaded_proj_stn, loaded_dist_stn = process_sentence(config, obj_path, loaded_proj_stn, loaded_dist_stn, dataset, model, obj_type, opt, opt_type)

      if proj is None and loaded_proj_stn:
        proj = config["objs"][-1]
    elif obj_type == "token":  
      loaded_info_tkn = process_token(config, obj_path, loaded_info_tkn, dataset, model, obj_type, opt, opt_type)
    elif obj_type == "text":  
      process_dataset_text(config, obj_path, dataset, model, obj_type, opt, opt_type)

      if text is None:
        text = config["objs"][-1]
    elif obj_type == "class":
      loaded_exp = process_explanation(config, obj_path, loaded_exp, dataset, model, obj_type, opt, opt_type)                
      

  config["projections"] = np.unique(config["projections"]).tolist()
  config["projections"].sort()

  if text is not None:
    if proj is not None:
      proj["label"] = text["data"]["label"]
      proj["topic"] = text["data"]["topic"]

  print("|- Sending config")

  return config

if __name__ == "main":
  ##Development/debug
  app.run()
  ##Production
  #from waitress import serve
  #serve(app, host="0.0.0.0", port=5000)
