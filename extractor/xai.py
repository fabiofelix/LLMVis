
import numpy as np, pandas as pd, tqdm, pdb
import utils
import shap

from lime import lime_tabular
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Explainer:
  def __init__(self):
    self.scaler = StandardScaler()
    self.classifier = svm.SVC(kernel = "linear", C = 1, verbose = False, max_iter = 1000, probability = True, random_state=utils.SEED_VALUE) 
    self.explainer = None
    self.exp_space = None

  def create_explainer(self, X_train, X_test, feature_names, label_names):
    pass

  def do_run(self):
    pass

  ##Strafified sampling with fixed test_samples_per_class
  def train_test_split(self, data, labels, test_samples_per_class):
    data = np.array(data)
    labels = np.array(labels)
    X_train = []
    y_train = []
    X_test = []
    y_test = []    
    
    for lb in np.unique(labels):
      indices = np.where(labels == lb)[0]
      aux_idx = np.random.choice(range(indices.shape[0]), size=test_samples_per_class, replace=False)
      indices = indices[aux_idx]
      indices.sort()

      X_test.extend(data[indices].tolist())
      y_test.extend(labels[indices].tolist())

      indices = np.array([ i for i in range(data.shape[0]) if i not in indices  ])

      X_train.extend(data[indices].tolist())
      y_train.extend(labels[indices].tolist())

    return X_train, X_test, y_train, y_test

  def run(self, data, labels, feature_names, label_names, test_samples_per_class = 5):
    X_train, X_test, y_train, y_test =  self.train_test_split(data, labels, test_samples_per_class=test_samples_per_class)

    X_train = self.scaler.fit_transform(X_train)
    X_test  = self.scaler.transform(X_test)

    self.classifier.fit(X_train, y_train)

    self.create_explainer(X_train, X_test, feature_names, label_names)

    self.exp_space = {"original_label": [], "predicted_label": [], "predicted_prob": []}

    for feat in feature_names:
      self.exp_space[feat] = []

    return self.do_run(X_test, y_test, feature_names, np.unique(labels), label_names)
  

class MyLime(Explainer):
  def create_explainer(self, X_train, X_test, feature_names, label_names):
    self.explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=feature_names, class_names=label_names,  mode='classification', random_state=utils.SEED_VALUE)
  
  def do_run(self, X_test, y_test, feature_names, label_ids, label_names):
    for sample, label in tqdm.tqdm(zip(X_test, y_test), desc = "|-- LIME explanation" , total = X_test.shape[0], unit= "sample"):
      lime_values = self.explainer.explain_instance(sample, self.classifier.predict_proba, num_features=sample.shape[0], labels=label_ids)
      label_prob  = self.classifier.predict_proba(sample.reshape(1, -1))[0]
      label_pred  = np.argmax(label_prob)

      self.exp_space["original_label"].append(label_names[label])
      self.exp_space["predicted_label"].append(label_names[label_pred])
      self.exp_space["predicted_prob"].append(label_prob[label_pred])

      lime_values = lime_values.as_map()[label_pred]

      for values in lime_values:
        self.exp_space[ feature_names[values[0]] ].append(values[1])

    # pdb.set_trace()
    return pd.DataFrame(self.exp_space)

class MyShap(Explainer):  
  def create_explainer(self, X_train, X_test, feature_names, label_names):
    self.explainer = shap.KernelExplainer(self.classifier.predict, X_test)

  def do_run(self, X_test, y_test, feature_names, label_ids, label_names):    
    label_prob  = self.classifier.predict_proba(X_test)
    label_pred  = np.argmax(label_prob, axis = 1)

    shap_values = self.explainer(X_test, silent=True)

    for label, shap_v, lb_pred, lb_prob in tqdm.tqdm(zip(y_test, shap_values.values, label_pred, label_prob), desc = "|-- SHAP explanation" , total = X_test.shape[0], unit= "sample"):
      self.exp_space["original_label"].append(label_names[label])
      self.exp_space["predicted_label"].append(label_names[lb_pred])
      self.exp_space["predicted_prob"].append(lb_prob[lb_pred])

      for idx, values in enumerate(shap_v):
        self.exp_space[ feature_names[idx] ].append(values)

    # pdb.set_trace()
    return pd.DataFrame(self.exp_space)


# def explain_lime(data, labels, feature_names, label_names, test_size = 10):
#   # pdb.set_trace()
#   # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, stratify = labels, random_state=utils.SEED_VALUE)
#   X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=utils.SEED_VALUE)

#   scaler = StandardScaler()
#   X_train = scaler.fit_transform(X_train)
#   X_test = scaler.transform(X_test)

#   classifier = svm.SVC(kernel = "linear", C = 1, verbose = False, max_iter = 1000, probability = True, random_state=utils.SEED_VALUE)
#   classifier.fit(X_train, y_train)

#   explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=feature_names, class_names=label_names,  mode='classification', random_state=utils.SEED_VALUE)
#   exp_space = {"original_label": [], "predicted_label": [], "predicted_prob": []}

#   for feat in feature_names:
#     exp_space[feat] = []

#   for sample, label in tqdm.tqdm(zip(X_test, y_test), desc = "|-- LIME explanation" , total = X_test.shape[0], unit= "sample"):
#     lime_values = explainer.explain_instance(sample, classifier.predict_proba, num_features=sample.shape[0], labels=np.unique(labels))
#     label_prob  = classifier.predict_proba(sample.reshape(1, -1))[0]
#     label_pred  = np.argmax(label_prob)

#     exp_space["original_label"].append(label_names[label])
#     exp_space["predicted_label"].append(label_names[label_pred])
#     exp_space["predicted_prob"].append(label_prob[label_pred])

#     lime_values = lime_values.as_map()[label_pred]

#     for values in lime_values:
#       exp_space[ feature_names[values[0]] ].append(values[1])

#   # pdb.set_trace()
#   return pd.DataFrame(exp_space)


# def explain_shap(data, labels, feature_names, label_names, test_size = 10):
#   # pdb.set_trace()
#   # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, stratify = labels, random_state=utils.SEED_VALUE)
#   X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=utils.SEED_VALUE)

#   scaler = StandardScaler()
#   X_train = scaler.fit_transform(X_train)
#   X_test = scaler.transform(X_test)

#   classifier = svm.SVC(kernel = "linear", C = 1, verbose = False, max_iter = 1000, probability = True, random_state=utils.SEED_VALUE)
#   classifier.fit(X_train, y_train)
#   label_pred = classifier.predict(X_test)

#   explainer = shap.KernelExplainer(classifier.predict, X_test)
#   shap_values = explainer(X_test)
#   exp_space = {"original_label": [], "predicted_label": []}

#   for feat in feature_names:
#     exp_space[feat] = []

#   for label, shap_v, lb_pred in tqdm.tqdm(zip(y_test, shap_values.values, label_pred), desc = "|-- SHAP explanation" , total = X_test.shape[0], unit= "sample"):
#     exp_space["original_label"].append(label_names[label])
#     exp_space["predicted_label"].append(label_names[lb_pred])

#     for idx, values in enumerate(shap_v):
#       exp_space[ feature_names[idx] ].append(values)

#   # pdb.set_trace()
#   return pd.DataFrame(exp_space)