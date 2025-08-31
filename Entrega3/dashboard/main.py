import streamlit as st
import pandas as pd
import pickle
import os
import datetime

# Importar páginas desde el módulo modules
from modules import (
    render_exploration_page,
    render_prediction_page,
    render_model_analysis_page,
    render_whatif_page
)

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Titanic - Análisis ML",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ocultar la navegación automática de Streamlit
st.markdown("""
<style>
    .stAppHeader {display: none;}
    .css-1v0mbdj.etr89bj1 {display: none;}
    .css-1lcbmhc.e1fqkh3o0 {display: none;}
    section[data-testid="stSidebar"] .css-ng1t4o {display: none;}
    section[data-testid="stSidebar"] .css-1d391kg {display: none;}
</style>
""", unsafe_allow_html=True)

# Inicializar session state para logs
if 'loading_messages' not in st.session_state:
    st.session_state.loading_messages = []

def add_loading_message(message):
    """Agregar mensaje al log con timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    st.session_state.loading_messages.append(full_message)
    # NO mostrar mensaje en tiempo real - solo guardar en el log

def show_loading_log():
    """Mostrar dropdown con el log de mensajes de carga"""
    if st.session_state.loading_messages:
        with st.expander(f"📋 Log del Sistema ({len(st.session_state.loading_messages)} eventos)", expanded=False):
            st.write("**Historial de carga y operaciones del sistema:**")
            
            # Mostrar mensajes en orden cronológico inverso (más recientes primero)
            for message in reversed(st.session_state.loading_messages[-20:]):  # Mostrar últimos 20
                st.text(message)
            
            if len(st.session_state.loading_messages) > 20:
                st.info(f"Mostrando últimos 20 eventos de {len(st.session_state.loading_messages)} total")
            
            # Botones de control
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
                # Mostrar estadísticas del log
                success_count = len([m for m in st.session_state.loading_messages if "✅" in m])
                warning_count = len([m for m in st.session_state.loading_messages if "⚠️" in m])
                error_count = len([m for m in st.session_state.loading_messages if "❌" in m])
                
                st.write(f"✅ {success_count} | ⚠️ {warning_count} | ❌ {error_count}")
    else:
        with st.expander("📋 Log del Sistema (vacío)", expanded=False):
            st.info("No hay eventos registrados aún")

def load_data():
    """Cargar y procesar datos del Titanic"""
    add_loading_message("📊 Iniciando carga de datos...")
    
    # Intentar cargar diferentes archivos de datos con las rutas correctas
    data_paths = [
        "../../Entrega2/data/Titanic_Dataset_Featured.csv",
        "../../Entrega2/data/Titanic_Dataset_Imputado.csv", 
        "../../Entrega2/data/Titanic-Dataset-Canvas.csv"
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
    """Cargar modelos entrenados"""
    add_loading_message("🤖 Iniciando carga de modelos...")
    
    models = {}
    model_dir = "../models"  # Esta ruta está correcta desde dashboard/
    
    if not os.path.exists(model_dir):
        add_loading_message(f"⚠️ Directorio de modelos no encontrado: {model_dir}")
        return models
    
    # Lista de modelos a cargar
    model_files = {
        "Random Forest": "randomforest_model.pkl",
        "XGBoost": "xgboost_model.pkl"
    }
    
    for name, filename in model_files.items():
        path = os.path.join(model_dir, filename)
        try:
            if os.path.exists(path):
                add_loading_message(f"📂 Cargando modelo: {name}")
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
                add_loading_message(f"✅ Modelo {name} cargado exitosamente")
            else:
                add_loading_message(f"⚠️ Archivo no encontrado: {path}")
        except Exception as e:
            add_loading_message(f"❌ Error cargando {name}: {str(e)}")
    
    add_loading_message(f"✅ Total de modelos cargados: {len(models)}")
    return models

def main():
    """Función principal del dashboard"""
    add_loading_message("🚀 Iniciando aplicación..")
    
    # Título principal
    st.title("🚢 Dashboard Titanic - Análisis de Supervivencia ML")
    st.markdown("""
    **Dashboard Interactivo de Machine Learning** para el análisis y predicción de supervivencia en el Titanic.
    Explore los datos, genere predicciones y analice modelos de forma interactiva.
    
    **Navegue usando el sidebar** ← para acceder a las diferentes secciones del análisis.
    """)
    
    # Mostrar log de carga
    show_loading_log()
    
    st.markdown("---")
    
    # Cargar datos
    df = load_data()
    if df is None:
        st.stop()
    
    # Cargar modelos
    models = load_models()
    
    # === NAVEGACIÓN ===
    # Selector de página en sidebar (simple y limpio)
    page_options = [
        "🔍 Exploración de Datos",
        "🔮 Predicción Interactiva", 
        "📊 Análisis de Modelos",
        "🔄 Análisis What-If"
    ]
    
    selected_page = st.sidebar.selectbox(
        "Seleccione una sección:",
        options=page_options,
        index=0,  # Por defecto: Exploración de Datos
        help="Navegue entre las diferentes secciones del dashboard"
    )
    
    # Información adicional en sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Información del Dataset")
    st.sidebar.info(f"📈 **{len(df)}** registros totales")
    st.sidebar.info(f"📋 **{len(df.columns)}** variables")
    st.sidebar.info(f"🤖 **{len(models)}** modelos cargados")
    
    # Renderizar página seleccionada
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