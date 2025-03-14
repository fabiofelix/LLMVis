
1. Create a python virtual environment with pip, conda, etc.
2. Install the dependences

```
  pip install -r requirements.txt
```

3. To generate the features, run the following code

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

4.1 For Kaggle dataset, download data from [link](https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles?select=test.csv)
4.2 For TinyStories, download from [link](https://huggingface.co/datasets/roneneldan/TinyStories)
4.3 For BBCNews, download from [link](http://mlg.ucd.ie/datasets/bbc.html)

5. Copy all the .npz files generated to the *data folder*

6. Run the following command to run *server.py* code and open the *Running on* link on the browser

```
  flask --app server run --debug
```

7. Vis tool interaction:

7.1 Scatter plot
a) Click on the plot area and drag to select a region with samples
b) Click on the plot area remove selection

7.2 Tree map: blue rectangle are clusters, green rectangle are tokens
a) Click on one cluster to select and click again to remove the selection
b) Ctrl + click to look inside the the cluster
c) Light-blue cluster reflect PARTIAL selection of the cluster tokens selected by the other views
d) Very light-blue cluster reflect TOTAL selection of the cluster tokens selected by the other views
e) Very light-green token reflect selection by the other views

7.3 Texts:
a) Click on the title (text id + label) to select the text, click again to remove the text selection
b) Use the top-right menu to clear all text selections

7. Code structure

7.1 Feature extractor:
 - extractor/extract_features.py
 - extractor/models.py: create_model and MyModelFamily
 - extractor/datasets.py: create_dataset and TextDataset

7.2 Vis tool:
  - server.py: load_config, filter
  - templates/index.html
  - static/js/client.js: document.addEventListener("DOMContentLoaded", function(){ .. })
  - static/js/svg.js: MySVG
  - static/js/canvas.js: MyCanvas
  - static/css/vis.css
