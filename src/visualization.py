import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_missing(df):
	"""Visualiza valores faltantes en el DataFrame."""
	sns.heatmap(df.isnull(), yticklabels=False, cbar=True)
	plt.title('Valores Faltantes')
	plt.show()

def plot_feature_importances(model, feature_names, top_n=10):
	"""Grafica la importancia de las top_n variables de un modelo."""
	importances = model.feature_importances_
	indices = importances.argsort()[::-1][:top_n]
	plt.figure(figsize=(10,6))
	plt.title(f"Top {top_n} Importancia de Características")
	plt.bar(range(top_n), importances[indices])
	plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
	plt.tight_layout()
	plt.show()
