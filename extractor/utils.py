
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

#to work with: torch.multiprocessing.set_start_method('spawn')
sys.path.append(omni_path)

def remove_otulier(x):
  if type(x) == list:
    x = np.array(x)

  while True:
    q1  = np.percentile(x, 25)
    q3  = np.percentile(x, 75)
    iqr = q3 - q1

    begin_shape = x.shape

    x = x[x >= q1 - iqr, ] #maiores que o limite inferior (remove os menores)
    x = x[x <= q3 + iqr, ] #menores que o limite superior (remove os maiores)

    if begin_shape == x.shape:
      break    

  return x  

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def otsu_threshold(x, bins = 100, remove_out = False, smooth_window = None):    
  #remove outlier
  if remove_out:
    x = remove_otulier(x)

  hist, edge = np.histogram(x, bins = bins)
  mids = [ np.mean([edge[i - 1], edge[i]]) for i in range(1, edge.shape[0])  ]

  if smooth_window is not None:
    hist = moving_average(hist, smooth_window)

  prop     = hist if np.sum(hist) == 1 else hist / np.sum(hist)
  omega    = np.cumsum(prop)
  mu       = np.cumsum(mids * prop)
  muT      = np.sum(mids * prop)    
  max_varB = -1
  maxk     = -1

  for k in range(0, len(prop)):
    varB = (muT * omega[k] - mu[k])**2 / (1 if omega[k] == 0 or omega[k] == 1 else omega[k] * (1 - omega[k]))
 
    if varB > max_varB :
      max_varB = varB
      maxk     = k  

  return mids[maxk]