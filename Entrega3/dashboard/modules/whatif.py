import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def create_feature_vector_whatif(pclass, sex, age, sibsp, parch, fare, embarked, model_name):
    """
    Crear vector de características para análisis What-If
    """
    if model_name == 'Neural Network':
        from .prediction import create_feature_vector_neural_network
        return create_feature_vector_neural_network(pclass, sex, age, sibsp, parch, fare, embarked)
    else:
        from .prediction import create_feature_vector_simple
        return create_feature_vector_simple(pclass, sex, age, sibsp, parch, fare, embarked)

def get_prediction(model, X_transformed, model_name):
    """
    Obtener predicción del modelo
    """
    try:
        if 'tensorflow' in str(type(model)).lower() or 'keras' in str(type(model)).lower():
            return float(model.predict(X_transformed, verbose=0)[0][0])
        elif hasattr(model, 'predict_proba'):
            return float(model.predict_proba(X_transformed)[0][1])
        else:
            return float(model.predict(X_transformed)[0])
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return 0.0

def render_whatif_page(df, models):
    """Página de análisis What-If - Análisis Contrafactual"""
    st.header("🔄 Análisis What-If")
    
    st.markdown("""
    **🎯 Herramienta de Análisis Contrafactual**: Explore cómo cambios específicos en las características 
    de un pasajero afectan su probabilidad de supervivencia usando sliders interactivos.
    """)
    
    if not models:
        st.error("❌ No hay modelos disponibles para el análisis What-If")
        return
    
    # Selección de modelo
    st.subheader("🤖 Configuración del Análisis")
    
    selected_model_name = st.selectbox(
        "Seleccione el modelo para el análisis:",
        options=list(models.keys()),
        help="Elija el modelo que desea usar para el análisis contrafactual"
    )
    
    st.markdown("---")
    
    # Configuración del pasajero base
    st.subheader("👤 Configuración del Pasajero Base")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_pclass = st.selectbox("🎫 Clase:", [1, 2, 3], index=1)
        base_sex = st.selectbox("👤 Género:", ['male', 'female'], index=1)
        base_age = st.number_input("📅 Edad:", 0, 100, 30)
    
    with col2:
        base_sibsp = st.number_input("👨‍👩‍👧‍👦 Hermanos/Cónyuges:", 0, 8, 1)
        base_parch = st.number_input("👶 Padres/Hijos:", 0, 6, 0)
        base_embarked = st.selectbox("🚢 Puerto de Embarque:", ['C', 'Q', 'S'], index=2)
    
    with col3:
        base_fare = st.number_input("💰 Tarifa (£):", 0.0, 500.0, 32.0)
    
    # Calcular predicción base
    try:
        selected_model = models[selected_model_name]
        X_base = create_feature_vector_whatif(base_pclass, base_sex, base_age, 
                                            base_sibsp, base_parch, base_fare, 
                                            base_embarked, selected_model_name)
        base_prob = get_prediction(selected_model, X_base, selected_model_name)
        
        st.success(f"📊 **Probabilidad Base de Supervivencia:** {base_prob:.1%}")
        
    except Exception as e:
        st.error(f"Error calculando probabilidad base: {str(e)}")
        return
    
    st.markdown("---")
    
    # Análisis What-If con sliders
    st.subheader("🔄 Análisis Contrafactual con Sliders")
    st.markdown("**Modifique las características usando los sliders y observe cómo cambia la probabilidad:**")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 🎯 Modificar Características")
        
        new_pclass = st.slider("🎫 Clase:", 1, 3, base_pclass, 
                              help="1=Primera, 2=Segunda, 3=Tercera")
        new_sex = st.selectbox("👤 Género:", ['male', 'female'], 
                             index=0 if base_sex == 'male' else 1, key="new_sex")
        new_age = st.slider("📅 Edad:", 0, 100, base_age)
        new_sibsp = st.slider("👨‍👩‍👧‍👦 Hermanos/Cónyuges:", 0, 8, base_sibsp)
        new_parch = st.slider("👶 Padres/Hijos:", 0, 6, base_parch)
        new_embarked = st.selectbox("🚢 Puerto:", ['C', 'Q', 'S'], 
                                  index=['C', 'Q', 'S'].index(base_embarked), key="new_embarked")
        new_fare = st.slider("💰 Tarifa (£):", 0.0, 500.0, base_fare)
    
    with col_right:
        st.markdown("### 📊 Cambio en Probabilidad")
        
        # Calcular nueva predicción
        try:
            X_new = create_feature_vector_whatif(new_pclass, new_sex, new_age,
                                               new_sibsp, new_parch, new_fare,
                                               new_embarked, selected_model_name)
            new_prob = get_prediction(selected_model, X_new, selected_model_name)
            prob_change = new_prob - base_prob
            
            # Mostrar métricas de cambio
            col_metric1, col_metric2 = st.columns(2)
            
            with col_metric1:
                st.metric("Nueva Probabilidad", f"{new_prob:.1%}")
            
            with col_metric2:
                st.metric("Cambio", f"{prob_change:+.1%}", 
                         delta_color="normal" if prob_change >= 0 else "inverse")
            
            # Visualización del cambio con gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = new_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilidad de Supervivencia"},
                delta = {'reference': base_prob * 100, 'suffix': '%'},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': base_prob * 100
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error en análisis What-If: {str(e)}")
            new_prob = base_prob
            prob_change = 0
    
    # Visualización de comparación
    st.markdown("---")
    st.subheader("📊 Visualización del Cambio en Probabilidad")
    
    fig_bar = go.Figure()
    
    fig_bar.add_trace(go.Bar(
        x=['Escenario Base', 'Escenario Modificado'],
        y=[base_prob, new_prob],
        marker_color=['lightblue', 'lightcoral' if new_prob < base_prob else 'lightgreen'],
        text=[f'{base_prob:.1%}', f'{new_prob:.1%}'],
        textposition='auto',
        name="Probabilidad"
    ))
    
    fig_bar.update_layout(
        title=f"Comparación de Probabilidades - {selected_model_name}",
        yaxis_title="Probabilidad de Supervivencia",
        showlegend=False,
        height=400
    )
    
    fig_bar.update_yaxes(range=[0, 1], tickformat='.0%')
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Análisis de factores modificados
    if abs(prob_change) > 0.001:
        st.markdown("### 🔍 Análisis de Factores Modificados")
        
        changes = []
        if new_pclass != base_pclass:
            direction = "⬆️" if new_pclass < base_pclass else "⬇️"
            changes.append(f"{direction} **Clase:** {base_pclass} → {new_pclass}")
        if new_sex != base_sex:
            changes.append(f"🔄 **Género:** {base_sex} → {new_sex}")
        if new_age != base_age:
            direction = "⬆️" if new_age > base_age else "⬇️"
            changes.append(f"{direction} **Edad:** {base_age} → {new_age}")
        if new_sibsp != base_sibsp:
            direction = "⬆️" if new_sibsp > base_sibsp else "⬇️"
            changes.append(f"{direction} **Hermanos/Cónyuges:** {base_sibsp} → {new_sibsp}")
        if new_parch != base_parch:
            direction = "⬆️" if new_parch > base_parch else "⬇️"
            changes.append(f"{direction} **Padres/Hijos:** {base_parch} → {new_parch}")
        if new_embarked != base_embarked:
            changes.append(f"🔄 **Puerto:** {base_embarked} → {new_embarked}")
        if abs(new_fare - base_fare) > 0.1:
            direction = "⬆️" if new_fare > base_fare else "⬇️"
            changes.append(f"{direction} **Tarifa:** £{base_fare:.1f} → £{new_fare:.1f}")
        
        if changes:
            for change in changes:
                st.write(f"• {change}")
        
        # Interpretación del impacto
        if abs(prob_change) > 0.15:
            impact = "🔴 **Impacto muy significativo**" if prob_change < 0 else "🟢 **Mejora muy significativa**"
        elif abs(prob_change) > 0.05:
            impact = "🟠 **Impacto moderado**" if prob_change < 0 else "🟡 **Mejora moderada**"
        else:
            impact = "🔹 **Impacto menor**"
        
        st.info(f"{impact}: {prob_change:+.1%} en probabilidad de supervivencia")