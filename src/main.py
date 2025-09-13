from utils import load_data, save_results, model_pkl
from preprocessing import preprocess
from models import get_model
from evaluation import compute_metrics
from visualization import plot_missing, plot_feature_importances
from sklearn.model_selection import train_test_split

# 1. Cargar datos
df = load_data('data/Titanic-Dataset-Canvas.csv')

# 2. Preprocesar
# Preprocesamiento (incluye codificación categórica)
df_clean = preprocess(df)

# 3. Separar variables
X = df_clean.drop('Survived', axis=1)
y = df_clean['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenar modelo
model = get_model('random_forest', n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluar
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
metrics = compute_metrics(y_test, y_pred, y_proba)
print("\nMétricas del modelo:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

model_pkl(model, 'modelo_entrenado.pkl')

# 6. Visualizar
plot_missing(df)
plot_feature_importances(model, X.columns)
