
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
      (1) TinyStories
      (2) BBCNews
      (3) Amazon
  -m  model index (required)
      (0) BERT
      (1) DeBERTa 2 (88B)
      (2) Llama 3.1 (8B)
      (3) Gemma 2 (9B)
  -o  path to save the outputs (required)
  -s  path to load the data (if the dataset loads from files, such as PaperAbstract, TinyStories, BBCNews, or Amazon)
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

## **Visualization**

1. After extraction, copy all the generated .npz files to the *data folder*

2. Run the following command to run *server.py* code inside the right environment and open the *Running on* link on the browser

```
  flask --app server run --debug
```

3. Vis tool interaction:
  * Scatter plot: color palette maps classes
    - Click on the plot area and drag to select a region with samples
    - Click on the plot area remove selection
    - Click on the class to (un)select all samples from that class
  * Word cloud: bigger and darker words are more frequent than smaller and lighter ones
    - Click on one token to (un)select  
    - Mose over one token to show more information
    - Use the top-right menu to clear selection
  * Sankey diagram: left rectangles represent predicted classes and right ones the more important tokens for each prediction
    - Click on one class/token to (un)select
    - Mose over one link to show more information
  * Texts:
    - Click on the title (text id + label) to (un)select the text
    - Use the top-right menu to clear all text selections
    - Left-most bar color maps classes

## **Basic code structure**

1. Feature extractor:
  - extractor/extract_features.py
  - extractor/models.py: create_model and MyModelFamily
  - extractor/datasets.py: create_dataset and TextDataset
  - extractor/xai.py: Explainer.run
  
2. Vis tool:
  - server.py: load_config, filter
  - templates/index.html
  - static/js/client.js: document.addEventListener("DOMContentLoaded", function(){ .. })
  - static/js/svg.js: MySVG
  - static/css/vis.css

