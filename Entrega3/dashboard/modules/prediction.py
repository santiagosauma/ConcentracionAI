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

def create_feature_vector_simple(pclass, sex, age, sibsp, parch, fare, embarked):
    """
    Crear un vector de características usando solo las características básicas
    y aplicando One-Hot Encoding manual para que sea compatible con los modelos
    """
    import numpy as np
    
    # Características numéricas básicas
    features = [
        pclass,           # Pclass
        age,              # Age  
        sibsp,            # SibSp
        parch,            # Parch
        fare,             # Fare
    ]
    
    # Feature engineering básico
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_per_person = fare / family_size if family_size > 0 else fare
    
    features.extend([
        family_size,      # FamilySize
        is_alone,         # IsAlone
        fare_per_person,  # FarePerPerson
        1 if age < 18 else 0,  # IsMinor
        0,                # Has_Cabin (asumimos no)
    ])
    
    # One-Hot Encoding para Sex
    features.extend([
        1 if sex == 'male' else 0,    # Sex_male
        1 if sex == 'female' else 0,  # Sex_female
    ])
    
    # One-Hot Encoding para Embarked
    features.extend([
        1 if embarked == 'C' else 0,  # Embarked_C
        1 if embarked == 'Q' else 0,  # Embarked_Q  
        1 if embarked == 'S' else 0,  # Embarked_S
    ])
    
    # One-Hot Encoding para Title
    if sex == 'male':
        title_master = 1 if age < 18 else 0
        title_mr = 1 if age >= 18 else 0
        title_miss = 0
        title_mrs = 0
    else:
        title_master = 0
        title_mr = 0
        title_miss = 1 if age < 25 else 0  # Aproximación
        title_mrs = 1 if age >= 25 else 0  # Aproximación
    
    features.extend([
        title_master,     # Title_Master
        title_miss,       # Title_Miss  
        title_mr,         # Title_Mr
        title_mrs,        # Title_Mrs
    ])
    
    # Age Groups
    age_group_child = 1 if age < 18 else 0
    age_group_young = 1 if 18 <= age < 35 else 0
    age_group_adult = 1 if 35 <= age < 60 else 0
    age_group_senior = 1 if age >= 60 else 0
    
    features.extend([
        age_group_child,   # AgeGroup_Child
        age_group_young,   # AgeGroup_YoungAdult
        age_group_adult,   # AgeGroup_Adult
        age_group_senior,  # AgeGroup_Senior
    ])
    
    # Family Size Categories
    family_solo = 1 if family_size == 1 else 0
    family_small = 1 if 2 <= family_size <= 3 else 0
    family_large = 1 if family_size > 3 else 0
    
    features.extend([
        family_solo,       # FamilySize_Solo
        family_small,      # FamilySize_Small
        family_large,      # FamilySize_Large
    ])
    
    # Rellenar con ceros hasta llegar a 89 características
    while len(features) < 89:
        features.append(0)
    
    # Asegurarse de que tenemos exactamente 89 características
    features = features[:89]
    
    return np.array(features).reshape(1, -1)

def render_prediction_page(df, models):
    """Página de predicción interactiva"""
    st.header("🔮 Predicción de Supervivencia")
    
    st.markdown("""
    **🎯 Predicción Interactiva**: Ingrese las características de un pasajero para predecir su probabilidad 
    de supervivencia usando múltiples modelos de Machine Learning entrenados.
    """)
    
    # Botón de información histórica FUERA del formulario
    col_info1, col_info2, col_info3 = st.columns([1, 2, 1])
    with col_info2:
        if st.button("💰 Ver Información Histórica de Tarifas del Titanic", 
                    help="Información sobre tarifas del Titanic", 
                    use_container_width=True):
            show_fare_info_modal()
    
    st.markdown("---")
    
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
            
            # Campo de tarifa simplificado
            fare = st.text_input(
                "💰 Tarifa Pagada (£):",
                value="32.0",
                help="Tarifa pagada por el boleto en libras esterlinas (ver información histórica arriba)"
            )
        
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
            
            # Crear vector de características compatible con los modelos
            try:
                X_transformed = create_feature_vector_simple(pclass, sex, age, sibsp, parch, fare_value, embarked)
                
                # Realizar predicciones con todos los modelos
                predictions = {}
                model_info = {}
                for model_name, model in models.items():
                    try:
                        # Información de depuración del modelo
                        model_type = type(model).__name__
                        model_info[model_name] = {
                            'type': model_type,
                            'module': type(model).__module__
                        }
                        
                        # Verificar si el modelo tiene predict_proba
                        if hasattr(model, 'predict_proba'):
                            prob = float(model.predict_proba(X_transformed)[0][1])  # Convertir a float de Python
                        else:
                            # Si no tiene predict_proba, usar predict y convertir
                            pred = model.predict(X_transformed)[0]
                            prob = float(pred)  # Convertir a float de Python
                        
                        predictions[model_name] = prob
                    except Exception as e:
                        st.error(f"Error con modelo {model_name}: {str(e)}")
                        st.error(f"Forma de datos enviada: {X_transformed.shape}")
                
                # Mostrar información de depuración
                with st.expander("🔍 Información de Depuración de Modelos", expanded=False):
                    st.write("**Modelos cargados:**")
                    for name, info in model_info.items():
                        st.write(f"- **{name}**: {info['type']} (de {info['module']})")
                    
                    st.write("**Características enviadas:**")
                    st.write(f"- Forma del vector: {X_transformed.shape}")
                    st.write(f"- Primeros 10 valores: {X_transformed[0][:10].tolist()}")
                    
                    st.write("**Predicciones obtenidas:**")
                    for name, prob in predictions.items():
                        st.write(f"- **{name}**: {prob:.4f} ({prob:.1%})")
                    
                    # Prueba adicional: verificar con casos extremos
                    st.write("**🧪 Prueba de Diferenciación de Modelos:**")
                    st.write("Probando con casos extremos para verificar diferencias...")
                    
                    # Caso 1: Mujer, primera clase, joven
                    test1 = create_feature_vector_simple(1, 'female', 25, 0, 0, 100, 'C')
                    # Caso 2: Hombre, tercera clase, mayor  
                    test2 = create_feature_vector_simple(3, 'male', 60, 0, 0, 7, 'S')
                    
                    for i, (test_case, description) in enumerate([(test1, "Mujer 1ra clase"), (test2, "Hombre 3ra clase")], 1):
                        st.write(f"**Caso {i} ({description}):**")
                        for model_name, model in models.items():
                            try:
                                if hasattr(model, 'predict_proba'):
                                    test_prob = float(model.predict_proba(test_case)[0][1])
                                    st.write(f"  - {model_name}: {test_prob:.4f}")
                            except:
                                st.write(f"  - {model_name}: Error en predicción de prueba")
                        
            except Exception as e:
                st.error(f"❌ Error creando vector de características: {str(e)}")
                return
            
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
                    prob_values = list(predictions.values())
                    avg_prob = float(np.mean(prob_values))
                    max_prob = float(max(prob_values))
                    min_prob = float(min(prob_values))
                    
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
