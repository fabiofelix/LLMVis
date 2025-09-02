
import os
import torch
import sys
import numba
import numpy as np

SEED_VALUE = 1537

##https://stackoverflow.com/questions/44131691/how-to-clear-cache-or-force-recompilation-in-numba  
##https://numba.pydata.org/numba-doc/0.48.0/developer/caching.html#cache-clearing
##https://numba.pydata.org/numba-doc/0.48.0/reference/envvars.html#envvar-NUMBA_CACHE_DIR
#to save numba cache out the /home folder
main_cache_path = os.path.join("/vast", os.path.basename(os.path.expanduser("~")))
clip_download_root = None
omni_path = os.path.join(os.path.expanduser("~"), ".cache/torch/hub/facebookresearch_omnivore_main")
hugging_path = None

if os.path.isdir(main_cache_path):
  cache_path = os.path.join(main_cache_path, "cache")

  if not os.path.isdir(cache_path):
    os.mkdir(cache_path)

  numba.config.CACHE_DIR = cache_path   #default: ~/.cache
  clip_download_root = os.path.join(cache_path, "clip") #default: ~/.cache/clip
  
  cache_path = os.path.join(cache_path, "torch", "hub")

  if not os.path.isdir(cache_path):
    os.makedirs(cache_path)

  torch.hub.set_dir(cache_path) #default: ~/.cache/torch/hub  
  omni_path = os.path.join(cache_path, "facebookresearch_omnivore_main")

  hugging_path = os.path.join(cache_path, "huggingface", "hub")

#to work with: torch.multiprocessing.set_start_method('spawn')
sys.path.append(omni_path)

def get_filtered_indices(x, lower_threshold = 1):
  x = np.array(x)

  ## discards minimum and maximum values
  min_value = np.max([np.min(x), lower_threshold])
  max_value = np.max(x)
  x_aux     = x[(x > min_value) & (x < max_value)]

  q1 = np.quantile(x_aux, 0.25)
  q3 = np.quantile(x_aux, 0.75)
  iqr = q3 - q1

  filter = (x > np.max((min_value, q1 - 1.5 * iqr))) & (x < np.min((max_value, q3 + 1.5 * iqr)))

  return np.where(filter)[0]
