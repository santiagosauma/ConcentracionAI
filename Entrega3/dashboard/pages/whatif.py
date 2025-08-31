import streamlit as st
import pandas as pd
import numpy as np

def render_whatif_page(df, models):
    """Página de análisis What-If"""
    st.header("🔄 Análisis What-If")
    
    st.markdown("""
    **🔬 Análisis Contrafactual**: Esta sección permite explorar cómo cambios en las características 
    de un pasajero afectarían su probabilidad de supervivencia.
    """)
    
    st.info("🚧 Esta sección está en desarrollo. Próximamente incluirá:")
    
    with st.expander("📋 Funcionalidades Planificadas", expanded=True):
        st.markdown("""
        **🎛️ Herramientas Interactivas:**
        - Sliders para modificar características
        - Visualización en tiempo real de cambios
        - Comparación antes/después
        - Análisis de sensibilidad
        
        **📊 Visualizaciones:**
        - Gráficos de tornado (sensitivity analysis)
        - Heatmaps de probabilidad
        - Contour plots multidimensionales
        - Árboles de decisión explicativos
        
        **🔍 Análisis Avanzado:**
        - Generación de contrafactuales
        - Distancia mínima para cambio de predicción
        - Plausibilidad de escenarios alternativos
        - Recomendaciones de intervención
        """)
    
    # Placeholder para demostración básica
    st.subheader("🎯 Demo Básico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Escenario Base:**")
        st.write("👤 Hombre, 30 años, Tercera Clase")
        st.write("💰 Tarifa: £7.25")
        st.write("👨‍👩‍👧‍👦 Solo (sin familia)")
        
        if models:
            # Ejemplo básico con el primer modelo disponible
            model_name = list(models.keys())[0]
            model = models[model_name]
            
            try:
                # Características base: [Pclass=3, Sex=male, Age=30, SibSp=0, Parch=0, Fare=7.25, C=0, Q=0, S=1]
                base_features = np.array([[3, 0, 30, 0, 0, 7.25, 0, 0, 1]])
                base_prob = model.predict_proba(base_features)[0][1]
                
                st.metric("🎯 Probabilidad Base", f"{base_prob:.1%}")
            except:
                st.write("⚠️ Error calculando probabilidad base")
    
    with col2:
        st.markdown("**Escenario Alternativo:**")
        st.write("👤 Mujer, 25 años, Primera Clase")
        st.write("💰 Tarifa: £100")
        st.write("👨‍👩‍👧‍👦 Con 1 hijo")
        
        if models:
            try:
                # Características alternativas: [Pclass=1, Sex=female, Age=25, SibSp=0, Parch=1, Fare=100, C=1, Q=0, S=0]
                alt_features = np.array([[1, 1, 25, 0, 1, 100, 1, 0, 0]])
                alt_prob = model.predict_proba(alt_features)[0][1]
                
                st.metric("🎯 Probabilidad Alternativa", f"{alt_prob:.1%}")
                
                # Mostrar diferencia
                diff = alt_prob - base_prob
                st.metric("📈 Cambio", f"{diff:+.1%}")
                
            except:
                st.write("⚠️ Error calculando probabilidad alternativa")
