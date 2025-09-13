import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_model(name, **kwargs):
	"""Devuelve una instancia de modelo según el nombre."""
	if name == 'random_forest':
		return RandomForestClassifier(**kwargs)
	elif name == 'logistic_regression':
		return LogisticRegression(**kwargs)
	elif name == 'svm':
		return SVC(probability=True, **kwargs)
	elif name == 'xgboost':
		return XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
	else:
		raise ValueError(f"Modelo no soportado: {name}")
