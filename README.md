
# Visual exploration of transformer text embedding spaces using explanation methods

This repo contains a code implementation of the visualization tool proposed by a [paper]() submitted to [??](). It combines projections, word cloud, and explainers to explore LLM embedding spaces.

## **Install**

1. Create a python virtual environment with pip, conda, etc.
2. Install the dependences for extraction and visualization

```
  pip install -r requirements.txt
```

  - If one wants to run only the Visualization tool

```
  pip install -r requirements_vis.txt
```

## **Extract information**

1. Run the following code to extract the features

``` 
  python extract_features.py 
  
  -d  dataset index (required)
      (0) PaperAbstract
      (1) BBCNews
      (2) TinyStories
      (3) Amazon
  -m  model index (required)
      (0) BERT
      (1) DeBERTa 2 (88B)
      (2) Llama 3.1 (8B)
      (3) Gemma 2 (9B)
  -o  path to save the outputs (required)
  -s  path to load the data (if the dataset loads from files, such as PaperAbstract, BBCNews, TinyStories, or Amazon)
  -n  number of samples to load from the dataset (default = 100)
  -b  batch size (default = 100)
``` 

2. Available models

  - [BERT](https://huggingface.co/google-bert/bert-base-uncased) base uncased with 110 milion parameters
  - [DeBERTa](https://huggingface.co/microsoft/deberta-v2-xlarge) version 2 with ~88 bilion parameters
  - [Llama](https://huggingface.co/meta-llama/Llama-3.1-8B) version 3.1 with 8 billion parameters
  - [Gemma](https://huggingface.co/google/gemma-2-9b) version 2 with 9 billion parameters


3. Available datasets

  - [PaperAbstract](https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles) with paper abstracts
  - [TinyStories-V2](https://huggingface.co/datasets/roneneldan/TinyStories) (generated with GPT-4) with small children stories
  - [BBCNews](http://mlg.ucd.ie/datasets/bbc.html)
  - [Amazon](http://snap.stanford.edu/data/web-FineFoods.html) with fine foods reviews

4. Basic configurations

  - All extractions were performed in a NVIDIA A100 GPU, with RAM of 128 GB, and Python 3.12.8
  - We called the code with `-b 100` in all BERT and DeBERTa tests, `-b 50` for Llama , and `-b 25` for Gemma
  - We also used `-n 1000`

> [!WARNING] 
> Even with the code setting `torch.manual_seed` and other seeds, the models' outputs can very between executions. For example, the user can identify suitable changes in the projection as point rotation.


> [!IMPORTANT] 
> Before running, you should set up some paths again.
> For example, in `extractor/models.py`
> 1. `LLAMA_HPC_PATH`
> 2. The method `set_model_path` for DeBERTa, Llama and Gemma models
> 
> In the `extractor/utils.py` file, the `main_cache_path` variable is automatically configured to your `home` folder.


## **Visualization**

1. After extraction, copy all the generated .npz files to the *data folder*

2. Run the following command to run *server.py* code inside the right environment and open the *Running on* link on the browser

```
  flask --app server run --debug
```

4. Basic configurations

  - The whole visualization was tested with Python `3.9.23` or `3.10.6`
  - Different screen resolutions can show different number of tokens in Token-frequency view, for example, 1366x768 or 1920x1160

> [!WARNING] 
> Every time the Word Cloud is updated, the words will be in different positions due to randomness of the used [d3-cloud](https://github.com/jasondavies/d3-cloud) code.


3. Vis tool interaction:
  * Scatter plot: color palette maps classes
    - Select different projections on the top-right drop-down list
    - Click on the plot area and drag to select a region with samples
    - Click on the plot area remove selection
    - Mouse-over on of the on the colored square on the bottom-left side to check class name
    - Click on the colored square on the bottom-left side to (un)select all samples from that class
  * Word cloud: bigger and darker words are more frequent than smaller and lighter ones
    - Mose over one token to show more information
    - Click on one token to (un)select  
    - Use the top-right menu to clear all selections
  * Sankey diagram: left rectangles represent predicted classes and right ones the more important tokens for each predicted class
    - Select different explaners on the top-right drop-down list
    - Click on one class/token to (un)select
    - Mose over one link to show more information
  * Texts: left-most bar color maps classes
    - Click on the title (text id + label) to (un)select the text
    - Use the top-right menu to clear all text selections

## **Basic code structure**

1. Feature extractor:
  - If you want to add a new **model**, you should inheret the class `MyModelFamily` and update the `create_model` function, both in `extractor/models.py`
  - If you want to add a new **dataset**, you should inheret the class `TextDataset` and update the `create_dataset` function, both in `extractor/datasets.py`
  - If you want to add a new **explainer**, you should inheret the class `Explainer` in `extractor/xai.py` and update `save_explanation` function in `extractor/extract_features.py`
  - If you want to add a new **projection**, you should update `save_projection` function in `extractor/extract_features.py`
  - If you want to change the **token info** extraction, you should update `extract_token_info` function in `extractor/extract_features.py`
  - The main funtion is `run` in `extractor/extract_features.py`
  
2. Vis tool:
  - If you want to add a new **visualization**, you should inheret the view manager `VisManager` in `static/js/client.js` and the drawer `MySVG` in `static/js/svg.js`
  - The bind between objects and DOM components is in  `document.addEventListener("DOMContentLoaded", function(){ .. })` of `static/js/client.js`
  - The main HTML is `templates/index.html` and CSS is `static/css/vis.css`
  - The flask server main function is `filter` in `server.py`

## **Contact**
