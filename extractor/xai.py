
import numpy as np, pandas as pd, tqdm, pdb
import utils
import shap
import torch

from lime import lime_tabular
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report

from captum.metrics import infidelity

def model_forward(X_test, classifier):
  label_prob = classifier.predict_proba(X_test.detach().cpu().numpy())
  label_pred = label_prob.argmax(axis = 1)

  return torch.tensor([ prob[idx] for prob, idx in zip(label_prob, label_pred) ], dtype=torch.float32)

def perturb_fn(X_test):
  # Random Gaussian noise perturbation
  noise = 0.01 * torch.randn_like(X_test)
  return noise, X_test - noise  

class Explainer:
  ## As the tokens are represented by their feature-vector norm, a zero value implies in no presence of
  ## a token in a text (sentence). If StandardScaler is used, zero values would be changed, creating a false environment for the explainer
  def __init__(self, scaler_name = "minmax"):
    self.scaler = MinMaxScaler() if scaler_name == "minmax" else StandardScaler()
    self.classifier = svm.SVC(kernel = "linear", C = 1, verbose = False, max_iter = 1000, probability = True, random_state=utils.SEED_VALUE) 
    self.label_encoder = LabelEncoder()
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
      test_indices = []

      if indices.shape[0] > test_samples_per_class:
        test_indices = np.random.choice(indices, size=test_samples_per_class, replace=False)
      #Takes at least one sample per class   
      elif indices.shape[0] > 1:
        test_indices = [indices[0]]

      test_indices.sort()
      train_indices = np.setdiff1d(indices, test_indices)

      X_test.extend(data[test_indices].tolist())
      y_test.extend(labels[test_indices].tolist())

      X_train.extend(data[train_indices].tolist())
      y_train.extend(labels[train_indices].tolist())
    
    print("|-- Train/test split")
    print("|-- {:<10s}{:<10s}{:<10s}{:<10s}".format("class", "#train", "#test", "total"))

    train_labels, train_count = np.unique(y_train, return_counts = True)
    test_labels, test_count = np.unique(y_test, return_counts = True)
    total_train, total_test = 0, 0


    for lb in np.unique(labels):
      total     = np.where(labels == lb)[0]
      train_idx = np.where(train_labels == lb)[0]
      test_idx  = np.where(test_labels == lb)[0]

      count_tr = 0 if train_idx.shape[0] == 0 else train_count[train_idx[0]]  
      count_ts = 0 if test_idx.shape[0] == 0 else test_count[test_idx[0]]

      total_train += count_tr
      total_test += count_ts

      print("|-- {:<10d}{:<10d}{:<10d}{:<10d}".format(lb, count_tr, count_ts, total.shape[0]))

    print("|-- {:<10s}{:<10d}{:<10d}{:<10d}".format("total", total_train, total_test, total_train + total_test))

    return X_train, X_test, y_train, y_test

  def evaluate_classifier(self, X_test, y_test, label_codes, label_names):
    y_pred = self.classifier.predict(X_test)

    return classification_report(y_test, y_pred, zero_division = 0, labels = label_codes, target_names = label_names, output_dict=True)
  
  def evaluate_explainer(self, sample, explanation):
    sample = torch.tensor(sample, dtype=torch.float32).reshape(1, -1)
    explanation = torch.tensor(explanation, dtype=torch.float32).unsqueeze(0)
    infidelity_metric = infidelity(forward_func = model_forward, perturb_func = perturb_fn, inputs = sample, attributions = explanation, n_perturb_samples = 10, additional_forward_args = self.classifier)

    return infidelity_metric.detach().cpu().numpy().item()  

  def run(self, data, labels, feature_names, test_samples_per_class = 5):
    labels = self.label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test =  self.train_test_split(data, labels, test_samples_per_class=test_samples_per_class)
    
    X_train = self.scaler.fit_transform(X_train)
    X_test  = self.scaler.transform(X_test)

    self.classifier.fit(X_train, y_train)
    eval = self.evaluate_classifier(X_test, y_test, self.label_encoder.transform(self.label_encoder.classes_), self.label_encoder.classes_)
    self.create_explainer(X_train, X_test, feature_names, self.label_encoder.classes_)

    self.exp_space = {"original_label": [], "predicted_label": [], "predicted_prob": [], "infidelity": []}

    for feat in feature_names:
      self.exp_space[feat] = []

    return self.do_run(X_test, y_test, feature_names, np.unique(labels), self.label_encoder.classes_), eval

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
      exp_space_array = np.zeros(sample.shape[0])

      for values in lime_values:
        self.exp_space[ feature_names[values[0]] ].append(values[1])
        exp_space_array[values[0]] = values[1]

      infidelity = self.evaluate_explainer(sample, exp_space_array)
      self.exp_space["infidelity"].append(infidelity)        

    # pdb.set_trace()
    return pd.DataFrame(self.exp_space)

class MyShap(Explainer):  
  def create_explainer(self, X_train, X_test, feature_names, label_names):
    self.explainer = shap.KernelExplainer(self.classifier.predict_proba, X_test, link = "logit")

  def do_run(self, X_test, y_test, feature_names, label_ids, label_names):    
    label_prob  = self.classifier.predict_proba(X_test)
    label_pred  = np.argmax(label_prob, axis = 1)

    print("|-- SHAP explanation")
    shap_values = self.explainer(X_test)

    for sample, label, shap_v, lb_pred, lb_prob in zip(X_test, y_test, shap_values.values, label_pred, label_prob):  
      self.exp_space["original_label"].append(label_names[label])
      self.exp_space["predicted_label"].append(label_names[lb_pred])
      self.exp_space["predicted_prob"].append(lb_prob[lb_pred])
      exp_space_array = np.zeros(sample.shape[0])

      for idx, values in enumerate(shap_v[:, lb_pred]):
        self.exp_space[ feature_names[idx] ].append(values)
        exp_space_array[idx] = values
        
      infidelity = self.evaluate_explainer(sample, exp_space_array)
      self.exp_space["infidelity"].append(infidelity)          

    return pd.DataFrame(self.exp_space)
