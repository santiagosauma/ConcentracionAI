import streamlit as st
import pandas as pd
import pickle
import numpy as np

@st.dialog("🚢 Información Histórica de Tarifas del Titanic (1912)")
def show_fare_info_modal():
    """Modal con información histórica de tarifas"""
    
    # Crear tabs para diferentes secciones
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Por Clase", "🎫 Tipos de Cabina", "📊 Estadísticas", "🏆 Casos Famosos"])
    
    with tab1:
        st.markdown("#### 💰 Tarifas por Clase Social")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🥇 Primera Clase**")
            st.write("• Rango: £30 - £900")
            st.write("• Promedio: £84")
            st.write("• Cabinas de lujo")
            st.write("• Servicio completo")
        
        with col2:
            st.markdown("**🥈 Segunda Clase**")
            st.write("• Rango: £10 - £30")
            st.write("• Promedio: £21")
            st.write("• Cabinas cómodas")
            st.write("• Buen servicio")
        
        with col3:
            st.markdown("**🥉 Tercera Clase**")
            st.write("• Rango: £3 - £15")
            st.write("• Promedio: £14")
            st.write("• Compartimientos")
            st.write("• Servicio básico")
    
    with tab2:
        st.markdown("#### 🎫 Tipos Especiales de Cabina")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏰 Suites de Lujo**")
            st.write("• Parlour Suites: £500-£2,560")
            st.write("• Promenade Suites: £300-£700")
            st.write("• Incluían salón privado")
            st.write("• Balcón o terraza")
        
        with col2:
            st.markdown("**🛏️ Cabinas Especiales**")
            st.write("• Single berth: +50% del precio")
            st.write("• Cabinas con ventana: +25%")
            st.write("• Cerca del comedor: +15%")
            st.write("• Cabinas interiores: -20%")
    
    with tab3:
        st.markdown("#### 📊 Estadísticas Interesantes")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💸 Datos de Precio**")
            st.write("• Tarifa más cara: £2,560 (Suite)")
            st.write("• Tarifa más barata: £0 (Empleados)")
            st.write("• Tarifa promedio general: £32.20")
            st.write("• 15% viajaron gratis (tripulación)")
        
        with col2:
            st.markdown("**🔢 Distribución**")
            st.write("• 37% pagaron menos de £10")
            st.write("• 45% pagaron £10-£50")
            st.write("• 15% pagaron £50-£200")
            st.write("• 3% pagaron más de £200")
    
    with tab4:
        st.markdown("#### 🏆 Casos Famosos")
        
        st.markdown("**💎 Pasajeros VIP**")
        st.write("• **Col. Archibald Gracie**: £2,560 (Suite más cara)")
        st.write("• **Benjamin Guggenheim**: £2,000+ (Magnate minero)")
        st.write("• **Isidor Straus**: £1,200+ (Dueño de Macy's)")
        st.write("• **John Jacob Astor**: £500+ (El más rico a bordo)")
        
        st.markdown("**👨‍👩‍👧‍👦 Familias Completas**")
        st.write("• **Familia Carter**: £1,200 total (5 personas)")
        st.write("• **Familia Sage**: £70 total (11 personas)")
        st.write("• **Huérfanos Navratil**: £60 (historia famosa)")

def render_prediction_page(df, models):
    """Página de predicción interactiva"""
    st.header("🔮 Predicción de Supervivencia")
    
    st.markdown("""
    **🎯 Predicción Interactiva**: Ingrese las características de un pasajero para predecir su probabilidad 
    de supervivencia usando múltiples modelos de Machine Learning entrenados.
    """)
    
    # Formulario de predicción
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Información Personal")
            
            pclass = st.selectbox(
                "🎫 Clase del Pasajero:",
                options=[1, 2, 3],
                help="1 = Primera Clase, 2 = Segunda Clase, 3 = Tercera Clase"
            )
            
            sex = st.selectbox(
                "👤 Género:",
                options=['male', 'female'],
                help="Género del pasajero"
            )
            
            age = st.number_input(
                "📅 Edad:",
                min_value=0,
                max_value=100,
                value=30,
                help="Edad en años del pasajero"
            )
        
        with col2:
            st.subheader("👨‍👩‍👧‍👦 Información Familiar")
            
            sibsp = st.number_input(
                "👫 Hermanos/Cónyuge a bordo:",
                min_value=0,
                max_value=10,
                value=0,
                help="Número de hermanos/hermanas o cónyuge a bordo"
            )
            
            parch = st.number_input(
                "👶 Padres/Hijos a bordo:",
                min_value=0,
                max_value=10,
                value=0,
                help="Número de padres/hijos a bordo"
            )
            
            # Información sobre tarifas con botón de ayuda
            col_fare, col_info = st.columns([3, 1])
            with col_fare:
                fare = st.text_input(
                    "💰 Tarifa Pagada (£):",
                    value="32.0",
                    help="Tarifa pagada por el boleto en libras esterlinas"
                )
            with col_info:
                st.write("")  # Espaciado
                if st.button("💰 Ver Info Histórica de Tarifas", help="Información sobre tarifas del Titanic"):
                    show_fare_info_modal()
        
        with col3:
            st.subheader("🚢 Información del Viaje")
            
            embarked = st.selectbox(
                "🏃‍♂️ Puerto de Embarque:",
                options=['C', 'Q', 'S'],
                index=2,  # Southampton por defecto
                help="C = Cherbourg, Q = Queenstown, S = Southampton"
            )
        
        # Botón de predicción
        submit = st.form_submit_button(
            "🔮 Obtener Predicción",
            type="primary",
            help="Haga clic para obtener la predicción de todos los modelos disponibles"
        )

    if submit:
        try:
            # Validar y convertir tarifa
            try:
                fare_value = float(fare)
                if fare_value < 0:
                    st.error("❌ La tarifa no puede ser negativa")
                    return
            except ValueError:
                st.error("❌ Por favor ingrese un valor numérico válido para la tarifa")
                return
            
            # Crear array de características
            features = np.array([[pclass, 1 if sex == 'female' else 0, age, sibsp, parch, fare_value, 
                                1 if embarked == 'C' else 0, 1 if embarked == 'Q' else 0, 1 if embarked == 'S' else 0]])
            
            # Realizar predicciones con todos los modelos
            predictions = {}
            for model_name, model in models.items():
                try:
                    prob = model.predict_proba(features)[0][1]
                    predictions[model_name] = prob
                except Exception as e:
                    st.error(f"Error con modelo {model_name}: {str(e)}")
            
            if predictions:
                # Mostrar resultados
                st.success("✅ Predicción completada exitosamente")
                
                # Crear visualización de resultados
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🎯 Resultados por Modelo")
                    
                    for model_name, prob in predictions.items():
                        confidence = "Alta" if abs(prob - 0.5) > 0.3 else "Media" if abs(prob - 0.5) > 0.15 else "Baja"
                        
                        st.metric(
                            f"🤖 {model_name}",
                            f"{prob:.1%}",
                            delta=f"Confianza: {confidence}"
                        )
                        
                        # Barra de progreso visual
                        st.progress(prob, text=f"Probabilidad de supervivencia: {prob:.1%}")
                        st.markdown("---")
                
                with col2:
                    # Resumen de consenso
                    avg_prob = np.mean(list(predictions.values()))
                    max_prob = max(predictions.values())
                    min_prob = min(predictions.values())
                    
                    st.subheader("📊 Consenso de Modelos")
                    st.metric("📈 Promedio", f"{avg_prob:.1%}")
                    st.metric("⬆️ Máximo", f"{max_prob:.1%}")
                    st.metric("⬇️ Mínimo", f"{min_prob:.1%}")
                    
                    # Interpretación
                    if avg_prob > 0.7:
                        st.success("💚 **Alta probabilidad de supervivencia**")
                    elif avg_prob > 0.4:
                        st.warning("🟡 **Probabilidad moderada de supervivencia**")
                    else:
                        st.error("🔴 **Baja probabilidad de supervivencia**")
        
        except Exception as e:
            st.error(f"❌ Error en la predicción: {str(e)}")
            st.error("Por favor verifique que todos los campos estén completados correctamente.")
