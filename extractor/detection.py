import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from utils import get_class_args

class OutlierDetector():
  def __init__(self, outlier_size = 0.1, random_state = 42):
    self.outlier_size = outlier_size
    self.model = IsolationForest(n_estimators=100, max_samples="auto", max_features=1.0, contamination="auto", random_state=random_state)

  def fit_predict(self, data):
    self.model.fit(data)

    ## negative values -1.0 and 0.0.
    ## -1.0 indicates severe anomaly.
    scores = self.model.score_samples(data)
    k = max(1, round(len(scores) * 0.1))

    if isinstance(self.outlier_size, float):
      k = max(1, round(len(scores) * self.outlier_size))
    elif isinstance(self.outlier_size, int):
      k = max(1, self.outlier_size)

    top_k = np.argsort(scores)[:k]

    pred = np.ones(len(data), dtype=int)
    pred[top_k] = -1

    return pred

class SubGroupDetector():
  def __init__(self, model_class, criterion = "silhouette", random_state = 42):
    self.model = None
    self.model_class = model_class
    self.criterion = criterion
    self.random_state = random_state

  def find_best_epsilon(self, data, n_neighbors = 3):
    ##n_neighbors can't be less than 3 and greater than the half number of samples
    ## n_neighbors = max(3, n_neighbors)
    ## n_neighbors = min(n_neighbors, data.shape[0] // 2)
    n_neighbors = np.clip(n_neighbors, 3, data.shape[0] // 2)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(data)

    distances, indices = nn.kneighbors(data)
    k_distances = np.sort(distances[:, -1])

    knee = KneeLocator(np.arange(len(k_distances)), k_distances, curve="convex", direction="increasing")

    if knee.knee is None:
      print("[Warning] Unable to find k-distance knee. Returning a 90% percentile")
      eps = np.percentile(k_distances, 90)
      # k = np.argmin(np.abs(np.array(k_distances) - eps))
    else:
      eps = k_distances[knee.knee]
      # k = knee.knee

    return eps, n_neighbors

  def find_best_k(self, model_class, data, criterion, random_state):
    values = []
    k_range = [i for i in range(3, int(np.sqrt(len(data))) + 1)]

    for k in k_range:
      class_args = get_class_args(model_class)

      if "random_state" in class_args:
        model = model_class(n_clusters=k, random_state=random_state)
      else:
        model = model_class(n_clusters=k)

      if criterion == "inertia":
        model.fit(data)
        values.append(model.inertia_)
      elif criterion in ["silhouette", "calinski_harabasz", "davies_bouldin"]:
        pred = model.fit_predict(data)

        if criterion == "silhouette":
          score = silhouette_score(data, pred)
        elif criterion == "calinski_harabasz":
          score = calinski_harabasz_score(data, pred)
        elif criterion == "davies_bouldin":
          score = davies_bouldin_score(data, pred)

        values.append(score)
      else:
        raise ValueError(f"[{criterion}] is not supported")

    if criterion == "inertia":
      knee_locator = KneeLocator(k_range, values, curve="convex", direction="decreasing")

      if knee_locator.knee is None:
        print("[Warning] Unable to find k-distance knee. Returning a 90% percentile")
        target = np.percentile(values, 90)
        k = np.argmin(np.abs(np.array(values) - target))
        return k_range, values, k_range[k]
      else:
        return k_range, values, knee_locator.knee

    return k_range, values, k_range[np.argmax(values)]    

  def fit_predict(self, data):
    class_args = get_class_args(self.model_class)
    self.model = None

    if self.model_class in [KMeans, BisectingKMeans, AgglomerativeClustering]:
      k_range, score_values, n_clusters = self.find_best_k(self.model_class, data, self.criterion, self.random_state)

      if "random_state" in class_args:
        self.model = self.model_class(n_clusters=n_clusters, random_state=self.random_state)
      else:
        self.model = self.model_class(n_clusters=n_clusters)

      # print(f"{self.model_class.__name__} with n_clusters {n_clusters}")    
    elif self.model_class in [DBSCAN, HDBSCAN]:
      eps, min_samples = self.find_best_epsilon(data)

      if "eps" in class_args:
        self.model = self.model_class(min_samples=min_samples, eps=eps, metric="cosine")
      else:
        self.model = self.model_class(min_samples=min_samples, metric="cosine")

      # print(f"{self.model_class.__name__} with eps {eps} and min_samples {min_samples}")    

    self.model.fit(data)
    
    return self.model.labels_