import pandas as pd

def load_data(path):
	"""Carga un archivo CSV en un DataFrame."""
	return pd.read_csv(path)

def save_results(results, path):
	"""Guarda resultados en un archivo CSV."""
	pd.DataFrame(results).to_csv(path, index=False)
	
def model_pkl(model, filename):
	"""Guarda un modelo entrenado en un archivo .pkl en la carpeta actual de src."""
	import joblib
	import os
	current_dir = os.path.dirname(__file__)
	full_path = os.path.join(current_dir, filename)
	joblib.dump(model, full_path)
