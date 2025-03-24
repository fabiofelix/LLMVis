
1. Create a python virtual environment with pip, conda, etc.
2. Install the dependences

```
  pip install -r requirements.txt
```

3. Run the following code to extract the features

``` 
  python extract_features.py 
  
  -d  dataset index (required)
      (0) Kaggle
      (1) TinyStories
      (2) BBCNews
  -m  model index (required)
      (0) BERT
      (1) Llama 3.1 (8B)
      (2) Gemma 2 (9B)
  -o  path to save the outputs (required)
  -s  path to load the data (if the dataset loads from files, such as Kaggle, TinyStories, and BBCNews)
  -n  number of samples to load from the dataset (default = 100)
  -b  batch size (default = 100)
``` 

4. Available datasets

  - [Kaggle dataset](https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles?select=test.csv) with paper abstracts
  - [TinyStories-V2](https://huggingface.co/datasets/roneneldan/TinyStories) (generated with GPT-4) with small children stories
  - [BBCNews](http://mlg.ucd.ie/datasets/bbc.html)

5. Copy all the .npz files generated to the *data folder*

6. Run the following command to run *server.py* code inside the right environment and open the *Running on* link on the browser

```
  flask --app server run --debug
```

7. Vis tool interaction:
  * Scatter plot: color palette maps classes
    - Click on the plot area and drag to select a region with samples
    - Click on the plot area remove selection
    - Click on the class to (un)select all samples from that class
  * Word cloud: bigger and darker words are more frequent than smaller and lighter ones
    - Click on one token to (un)select  
    - Mose over one token to show more information
  * Tree map: blue rectangle are clusters, green rectangle are tokens
    - Click on one cluster/token to (un)select
    - Ctrl+click to zoom-in(-out) the cluster
    - Mose over one cluster/token to show more information
    - Light-blue cluster reflect PARTIAL selection of the cluster tokens selected by the other views
    - Very light-blue cluster reflect TOTAL selection of the cluster tokens selected by the other views
    - Very light-green token reflect selection by the other views
  * Texts:
    - Click on the title (text id + label) to (un)select the text
    - Use the top-right menu to clear all text selections
    - Left-most bar color maps classes

8. Code structure
  * Feature extractor:
    - extractor/extract_features.py
    - extractor/models.py: create_model and MyModelFamily
    - extractor/datasets.py: create_dataset and TextDataset
  * Vis tool:
    - server.py: load_config, filter
    - templates/index.html
    - static/js/client.js: document.addEventListener("DOMContentLoaded", function(){ .. })
    - static/js/svg.js: MySVG
    - static/js/canvas.js: MyCanvas
    - static/css/vis.css
