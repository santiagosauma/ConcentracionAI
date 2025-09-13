import pandas as pd
import numpy as np

def fill_missing_age(df):
	"""Imputa la edad faltante con la mediana."""
	df['Age'] = df['Age'].fillna(df['Age'].median())
	return df

def fill_missing_embarked(df):
	"""Imputa el puerto de embarque faltante con el modo."""
	df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
	return df

def drop_unused(df):
	"""Elimina columnas irrelevantes para el modelado."""
	return df.drop(['Cabin', 'Ticket'], axis=1)

def encode_categoricals(df):
	"""Codifica variables categóricas usando one-hot encoding."""
	df = pd.get_dummies(df, drop_first=True)
	return df

def preprocess(df):
	"""Pipeline de preprocesamiento básico con codificación categórica."""
	df = fill_missing_age(df)
	df = fill_missing_embarked(df)
	df = drop_unused(df)
	df = encode_categoricals(df)
	return df
