import streamlit as st
import pandas as pd
import pickle
import joblib
import os
import datetime

from modules import (
    render_exploration_page,
    render_prediction_page,
    render_model_analysis_page,
    render_whatif_page
)

st.set_page_config(
    page_title="Dashboard Titanic - Análisis ML",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'loading_messages' not in st.session_state:
    st.session_state.loading_messages = []

def add_loading_message(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    st.session_state.loading_messages.append(full_message)

def show_loading_log():
    if st.session_state.loading_messages:
        with st.expander(f"📋 Log del Sistema ({len(st.session_state.loading_messages)} eventos)", expanded=False):
            st.write("**Historial de carga y operaciones del sistema:**")
            
            for message in reversed(st.session_state.loading_messages[-20:]):
                st.text(message)
            
            if len(st.session_state.loading_messages) > 20:
                st.info(f"Mostrando últimos 20 eventos de {len(st.session_state.loading_messages)} total")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Limpiar Log", key="clear_log"):
                    st.session_state.loading_messages = []
                    st.rerun()
            
            with col2:
                if st.button("📥 Descargar Log", key="download_log"):
                    log_content = "\n".join(st.session_state.loading_messages)
                    st.download_button(
                        label="💾 Descargar",
                        data=log_content,
                        file_name=f"titanic_dashboard_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with col3:
                success_count = len([m for m in st.session_state.loading_messages if "✅" in m])
                warning_count = len([m for m in st.session_state.loading_messages if "⚠️" in m])
                error_count = len([m for m in st.session_state.loading_messages if "❌" in m])
                
                st.write(f"✅ {success_count} | ⚠️ {warning_count} | ❌ {error_count}")
    else:
        with st.expander("📋 Log del Sistema (vacío)", expanded=False):
            st.info("No hay eventos registrados aún")

def load_data():
    add_loading_message("📊 Iniciando carga de datos...")
    
    data_paths = [
        "../data/Titanic_Dataset_Featured.csv",
        "../data/Titanic_Dataset_Imputado.csv", 
        "../data/Titanic-Dataset-Canvas.csv"
    ]
    
    for path in data_paths:
        try:
            if os.path.exists(path):
                add_loading_message(f"📂 Encontrado archivo: {path}")
                df = pd.read_csv(path)
                add_loading_message(f"✅ Datos cargados exitosamente: {len(df)} registros, {len(df.columns)} columnas")
                return df
        except Exception as e:
            add_loading_message(f"❌ Error cargando {path}: {str(e)}")
    
    add_loading_message("❌ No se pudo cargar ningún archivo de datos")
    st.error("❌ No se encontraron archivos de datos válidos")
    return None

def load_models():
    add_loading_message("🤖 Iniciando carga de modelos...")
    
    models = {}
    model_dir = "../models"
    
    if not os.path.exists(model_dir):
        add_loading_message(f"⚠️ Directorio de modelos no encontrado: {model_dir}")
        return models
    
    model_files = {
        "Random Forest": "randomforest_model.pkl",
        "XGBoost": "xgboost_model.pkl", 
        "Logistic Regression": "logisticregression_model.pkl",
        "Neural Network": "NeuralNetwork_Publication_20250831_170008.h5",
        "SVM": "svm_model.pkl"
    }
    
    for name, filename in model_files.items():
        path = os.path.join(model_dir, filename)
        try:
            if os.path.exists(path):
                add_loading_message(f"📂 Cargando modelo: {name}")
                
                if filename.endswith('.h5'):
                    try:
                        import tensorflow as tf
                        model = tf.keras.models.load_model(path)
                        models[name] = model
                        add_loading_message(f"✅ Modelo {name} cargado: {type(model).__name__} (TensorFlow/Keras)")
                    except ImportError:
                        add_loading_message(f"⚠️ TensorFlow no disponible para cargar {name}")
                        continue
                else:
                    model = joblib.load(path)
                    models[name] = model
                    
                    model_type = type(model).__name__
                    model_module = type(model).__module__
                    add_loading_message(f"✅ Modelo {name} cargado: {model_type} (de {model_module})")
                    
                    if hasattr(model, 'n_estimators'):
                        add_loading_message(f"   - n_estimators: {model.n_estimators}")
                    if hasattr(model, 'max_depth'):
                        add_loading_message(f"   - max_depth: {model.max_depth}")
                    if hasattr(model, 'learning_rate'):
                        add_loading_message(f"   - learning_rate: {model.learning_rate}")
                    if hasattr(model, 'C'):
                        add_loading_message(f"   - C (regularización): {model.C}")
                    
                    if name == 'Neural Network':
                        expected_features = 76
                    elif name == 'SVM':
                        expected_features = 5
                    else:
                        expected_features = 89
                    add_loading_message(f"   - Features esperadas: {expected_features}")
                    
                    has_predict_proba = hasattr(model, 'predict_proba')
                    has_predict = hasattr(model, 'predict')
                    add_loading_message(f"   - predict_proba: {'✅' if has_predict_proba else '❌'}")
                    add_loading_message(f"   - predict: {'✅' if has_predict else '❌'}")
                        
                    if 'tensorflow' in str(type(model)).lower() or 'keras' in str(type(model)).lower():
                        try:
                            input_shape = model.input_shape
                            add_loading_message(f"   - input_shape: {input_shape}")
                        except:
                            add_loading_message(f"   - input_shape: No disponible")
                        
            else:
                add_loading_message(f"⚠️ Archivo no encontrado: {path}")
        except Exception as e:
            add_loading_message(f"❌ Error cargando {name}: {str(e)}")
    
    add_loading_message(f"✅ Total de modelos cargados: {len(models)}")
    return models

def main():
    add_loading_message("🚀 Iniciando aplicación..")
    
    st.title("🚢 Dashboard Titanic - Análisis de Supervivencia ML")
    st.markdown("""
    **Dashboard Interactivo de Machine Learning** para el análisis y predicción de supervivencia en el Titanic.
    Explore los datos, genere predicciones y analice modelos de forma interactiva.
    
    **Navegue usando el sidebar** ← para acceder a las diferentes secciones del análisis.
    """)
    
    show_loading_log()
    
    st.markdown("---")
    
    df = load_data()
    if df is None:
        st.stop()
    
    models = load_models()
    
    st.subheader("📊 Información del Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Registros Totales", f"{len(df):,}")
    with col2:
        st.metric("📋 Variables", f"{len(df.columns)}")
    with col3:
        st.metric("🤖 Modelos Cargados", f"{len(models)}")
    
    st.markdown("---")
    
    st.sidebar.markdown("# 🎯 Elegir Sección")
    st.sidebar.markdown("**Seleccione el análisis que desea realizar:**")
    
    page_options = [
        "🔍 Exploración de Datos",
        "🔮 Predicción Interactiva", 
        "📊 Análisis de Modelos",
        "🔄 Análisis What-If"
    ]
    
    selected_page = st.sidebar.selectbox(
        "Navegación:",
        options=page_options,
        index=0,
        help="Navegue entre las diferentes secciones del dashboard"
    )
    
    add_loading_message(f"📄 Navegando a: {selected_page}")
    
    if selected_page == "🔍 Exploración de Datos":
        render_exploration_page(df)
    elif selected_page == "🔮 Predicción Interactiva":
        render_prediction_page(df, models)
    elif selected_page == "📊 Análisis de Modelos":
        render_model_analysis_page(df, models)
    elif selected_page == "🔄 Análisis What-If":
        render_whatif_page(df, models)

if __name__ == "__main__":
    main()