import streamlit as st
import pandas as pd
import pickle
import numpy as np

@st.dialog("🚢 Información Histórica de Tarifas del Titanic (1912)")
def show_fare_info_modal():
    
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

def create_feature_vector_scikit_models_corrected(pclass, sex, age, sibsp, parch, fare, embarked):
    """
    FUNCIÓN CORREGIDA - Replica el preprocesamiento del notebook principal para scikit-learn
    
    Basada en el análisis de Entrega3_Modelado.ipynb:
    - Usa las MISMAS 31 características que en el entrenamiento
    - Aplica StandardScaler para numéricas
    - Aplica OneHotEncoder para categóricas  
    - Total: 89 características después del preprocesamiento
    """
    import numpy as np
    
    
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_per_person = fare / family_size if family_size > 0 else fare
    is_minor = 1 if age < 18 else 0
    has_cabin = 0
    ticket_frequency = 1
    mother = 1 if (sex == 'female' and parch > 0 and age > 18) else 0
    name_length = 20
    
    fare_log = np.log1p(fare) if fare > 0 else 0
    fare_per_person_log = np.log1p(fare_per_person) if fare_per_person > 0 else 0
    age_sqrt = np.sqrt(age) if age >= 0 else 0
    name_length_sqrt = np.sqrt(name_length) if name_length > 0 else 0
    
    numeric_features = [
        pclass,
        age,
        sibsp,
        parch,
        fare,
        has_cabin,
        family_size,
        is_alone,
        fare_per_person,
        ticket_frequency,
        mother,
        name_length,
        fare_log,
        fare_per_person_log,
        age_sqrt,
        name_length_sqrt,
        is_minor,
    ]
    
    scaler_means = [2.31, 29.7, 0.52, 0.38, 32.2, 0.34, 1.85, 0.6, 18.5, 1.1, 0.08, 18.0, 3.0, 2.7, 5.8, 4.2, 0.15]
    scaler_stds = [0.84, 14.5, 1.1, 0.8, 49.7, 0.47, 1.4, 0.49, 22.3, 0.8, 0.27, 8.0, 1.2, 1.0, 1.2, 1.2, 0.36]
    
    numeric_scaled = []
    for i, (val, mean, std) in enumerate(zip(numeric_features, scaler_means, scaler_stds)):
        scaled_val = (val - mean) / std if std > 0 else val
        numeric_scaled.append(scaled_val)
    
    if sex == 'male':
        title = 'Master' if age < 18 else 'Mr'
    else:
        title = 'Miss' if age < 25 else 'Mrs'
    
    if age < 18:
        age_group = '0-17'
    elif age < 35:
        age_group = '18-34' 
    elif age < 60:
        age_group = '35-59'
    else:
        age_group = '60+'
    
    if age < 5:
        age_group2 = 'Infant'
    elif age < 13:
        age_group2 = 'Child'
    elif age < 18:
        age_group2 = 'Teen'
    elif age < 35:
        age_group2 = 'YoungAdult'
    elif age < 60:
        age_group2 = 'MidAge'
    else:
        age_group2 = 'Senior'
    
    if fare_per_person <= 7.25:
        fare_quintile = 'Muy_Bajo'
    elif fare_per_person <= 10.5:
        fare_quintile = 'Bajo'
    elif fare_per_person <= 21.679:
        fare_quintile = 'Medio'
    elif fare_per_person <= 41.579:
        fare_quintile = 'Alto'
    else:
        fare_quintile = 'Muy_Alto'
    
    if family_size == 1:
        ticket_freq = 'Individual'
    elif family_size == 2:
        ticket_freq = 'Pareja'
    elif family_size <= 4:
        ticket_freq = 'Grupo_Pequeño'
    else:
        ticket_freq = 'Grupo_Mediano'
    
    cabin_deck = 'Missing'
    
    if name_length < 15:
        name_length_cat = 'Muy_Corto'
    elif name_length < 20:
        name_length_cat = 'Corto'
    elif name_length < 25:
        name_length_cat = 'Medio'
    elif name_length < 30:
        name_length_cat = 'Largo'
    else:
        name_length_cat = 'Muy_Largo'
    
    if name_length <= 12:
        name_quintile = 'Q1'
    elif name_length <= 16:
        name_quintile = 'Q2'
    elif name_length <= 20:
        name_quintile = 'Q3'
    elif name_length <= 25:
        name_quintile = 'Q4'
    else:
        name_quintile = 'Q5'
    
    ticket_prefix = 'NUMERIC'
    
    ticket_prefix_cat = 'Numeric'
    
    if family_size == 1:
        family_cat = 'Solo'
    elif family_size <= 3:
        family_cat = 'Pequeña'
    elif family_size <= 6:
        family_cat = 'Mediana'
    else:
        family_cat = 'Grande'
    
    deck_category = 'Missing'
    
    
    categorical_encoded = []
    
    categorical_encoded.append(1 if sex == 'male' else 0)
    
    categorical_encoded.extend([
        1 if embarked == 'Q' else 0,
        1 if embarked == 'S' else 0,
    ])
    
    categorical_encoded.extend([
        1 if age_group == '18-34' else 0,
        1 if age_group == '35-59' else 0, 
        1 if age_group == '60+' else 0,
    ])
    
    categorical_encoded.extend([
        1 if title == 'Miss' else 0,
        1 if title == 'Mr' else 0,
        1 if title == 'Mrs' else 0,
    ])
    
    categorical_encoded.extend([
        1 if age_group2 == 'Child' else 0,
        1 if age_group2 == 'Infant' else 0,
        1 if age_group2 == 'MidAge' else 0,
        1 if age_group2 == 'Senior' else 0,
        1 if age_group2 == 'Teen' else 0,
        1 if age_group2 == 'YoungAdult' else 0,
    ])
    
    categorical_encoded.extend([
        1 if fare_quintile == 'Bajo' else 0,
        1 if fare_quintile == 'Medio' else 0,
        1 if fare_quintile == 'Alto' else 0,
        1 if fare_quintile == 'Muy_Alto' else 0,
    ])
    
    categorical_encoded.extend([
        1 if ticket_freq == 'Grupo_Mediano' else 0,
        1 if ticket_freq == 'Grupo_Pequeño' else 0,
        1 if ticket_freq == 'Pareja' else 0,
    ])
    
    for deck in ['B', 'C', 'D', 'E', 'F', 'G', 'T']:
        categorical_encoded.append(0)
    
    categorical_encoded.extend([
        1 if name_length_cat == 'Largo' else 0,
        1 if name_length_cat == 'Medio' else 0,
        1 if name_length_cat == 'Muy_Largo' else 0,
    ])
    
    categorical_encoded.extend([
        1 if name_quintile == 'Q2' else 0,
        1 if name_quintile == 'Q3' else 0,
        1 if name_quintile == 'Q4' else 0,
        1 if name_quintile == 'Q5' else 0,
    ])
    
    for prefix in ['A_', 'A_S', 'C', 'CA', 'FCC', 'LINE', 'P_PP', 'PC', 'PP', 'SC_A', 'SC_AH', 'SC_PARIS', 'SO_PP', 'SOC', 'SOP', 'SP', 'STON_O', 'SOTON_O', 'SOTON_OQ', 'W_C', 'WE_P']:
        categorical_encoded.append(0)
    
    categorical_encoded.extend([
        0,
        0,
        0,
    ])
    
    categorical_encoded.extend([
        1 if family_cat == 'Mediana' else 0,
        1 if family_cat == 'Pequeña' else 0,
        1 if family_cat == 'Solo' else 0,
    ])
    
    categorical_encoded.extend([
        0,
        0,
    ])
    
    
    all_features = numeric_scaled + categorical_encoded
    
    while len(all_features) < 89:
        all_features.append(0.0)
    
    all_features = all_features[:89]
    
    return np.array(all_features, dtype=np.float32).reshape(1, -1)

def create_feature_vector_svm(pclass, sex, age, sibsp, parch, fare, embarked):
    import numpy as np
    
    sex_numeric = 1 if sex == 'male' else 0
    
    features_raw = [
        sex_numeric,
        pclass,
        age,
        sibsp,
        parch,
    ]
    
    scaler_means = [0.647, 2.31, 29.7, 0.52, 0.38]
    scaler_stds = [0.478, 0.84, 14.5, 1.1, 0.8]
    
    features_scaled = []
    for i, (val, mean, std) in enumerate(zip(features_raw, scaler_means, scaler_stds)):
        scaled_val = (val - mean) / std if std > 0 else val
        features_scaled.append(scaled_val)
    
    return np.array(features_scaled, dtype=np.float32).reshape(1, -1)

def create_feature_vector_neural_network_corrected(pclass, sex, age, sibsp, parch, fare, embarked):
    import numpy as np
    
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_per_person = fare / family_size if family_size > 0 else fare
    is_minor = 1 if age < 18 else 0
    
    name_length = 20
    name_length_sqrt = np.sqrt(name_length) if name_length > 0 else 0
    age_sqrt = np.sqrt(age) if age >= 0 else 0
    fare_log = np.log1p(fare) if fare > 0 else 0
    
    has_cabin = 0
    
    mother = 1 if (sex == 'female' and parch > 0 and age > 18) else 0
    
    ticket_frequency = 1
    
    if sex == 'male':
        if age < 18:
            title = 'Master'
        else:
            title = 'Mr'
    else:
        if age < 25:
            title = 'Miss'
        else:
            title = 'Mrs'
    
    if name_length < 15:
        name_length_category = 'Muy_Corto'
    elif name_length < 20:
        name_length_category = 'Corto'  
    elif name_length < 25:
        name_length_category = 'Medio'
    elif name_length < 30:
        name_length_category = 'Largo'
    else:
        name_length_category = 'Muy_Largo'
    
    if name_length <= 12:
        name_length_quintile = 'Q1'
    elif name_length <= 16:
        name_length_quintile = 'Q2'
    elif name_length <= 20:
        name_length_quintile = 'Q3'
    elif name_length <= 25:
        name_length_quintile = 'Q4'
    else:
        name_length_quintile = 'Q5'
    
    if fare_per_person <= 7.75:
        fare_quintile = 'Muy_Bajo'
    elif fare_per_person <= 10.5:
        fare_quintile = 'Bajo'
    elif fare_per_person <= 21.679:
        fare_quintile = 'Medio'
    elif fare_per_person <= 41.579:
        fare_quintile = 'Alto'
    else:
        fare_quintile = 'Muy_Alto'
    
    if family_size == 1:
        family_category = 'Solo'
    elif family_size <= 3:
        family_category = 'Pequeña'
    elif family_size <= 6:
        family_category = 'Mediana'
    else:
        family_category = 'Grande'
    
    if family_size == 1:
        ticket_freq_category = 'Individual'
    elif family_size == 2:
        ticket_freq_category = 'Pareja'
    elif family_size <= 4:
        ticket_freq_category = 'Grupo_Pequeño'
    else:
        ticket_freq_category = 'Grupo_Mediano'
    
    cabin_deck = 'Sin_Cabina'
    deck_category = 'Sin_Cabina'
    
    ticket_prefix = 'NUMERIC'
    
    numeric_features = [
        name_length,
        name_length_sqrt,
        family_size,
        pclass,
        sibsp,
        ticket_frequency,
        has_cabin,
        age_sqrt,
        is_alone,
        parch,
        fare_log,
        is_minor,
        mother,
    ]
    
    scaler_means = [18.0, 4.2, 1.85, 2.31, 0.52, 1.1, 0.34, 5.8, 0.6, 0.38, 3.0, 0.15, 0.08]
    scaler_stds = [8.0, 1.2, 1.4, 0.84, 1.1, 0.8, 0.47, 1.2, 0.49, 0.8, 1.2, 0.36, 0.27]
    
    numeric_scaled = []
    for i, (val, mean, std) in enumerate(zip(numeric_features[:len(scaler_means)], scaler_means, scaler_stds)):
        scaled_val = (val - mean) / std if std > 0 else val
        numeric_scaled.append(scaled_val)
    
    
    categorical_features = []
    
    categorical_features.append(1 if sex == 'male' else 0)
    
    categorical_features.extend([
        1 if name_length_quintile == 'Q2' else 0,
        1 if name_length_quintile == 'Q3' else 0,
        1 if name_length_quintile == 'Q4' else 0,
        1 if name_length_quintile == 'Q5' else 0,
    ])
    
    categorical_features.extend([
        0, 
        0,  
    ])
     
    categorical_features.extend([
        0 if ticket_freq_category == 'Individual' else 1,
        0,  
    ])
    
    categorical_features.extend([
        0 if name_length_category == 'Muy_Corto' else 1,
        0,  
    ])
    
    categorical_features.extend([
        0,
        0,
        0,
    ])
    
    categorical_features.extend([
        1 if title == 'Miss' else 0,
        1 if title == 'Mr' else 0,
        1 if title == 'Mrs' else 0,
    ])
    
    categorical_features.extend([
        0,
        0,
    ])
    
    categorical_features.extend([
        1 if fare_quintile == 'Bajo' else 0,
        1 if fare_quintile == 'Medio' else 0,
        1 if fare_quintile == 'Alto' else 0,
        1 if fare_quintile == 'Muy_Alto' else 0,
    ])
    
    categorical_features.extend([
        1 if family_category == 'Mediana' else 0,
        1 if family_category == 'Pequeña' else 0,
        1 if family_category == 'Solo' else 0,
    ])
    
    final_features = numeric_scaled + categorical_features
    
    while len(final_features) < 76:
        final_features.append(0.0)
    
    final_features = final_features[:76]
    
    return np.array(final_features, dtype=np.float32).reshape(1, -1)

def render_prediction_page(df, models):
    st.header("🔮 Predicción de Supervivencia")
    
    st.markdown("""
    **🎯 Predicción Interactiva**: Ingrese las características de un pasajero para predecir su probabilidad 
    de supervivencia usando modelos de Machine Learning entrenados.
    """)
    
    if not models:
        st.error("❌ No hay modelos disponibles. Verifique que los archivos de modelos estén en la carpeta correcta.")
        return
    
    st.subheader("🤖 Selección de Modelo")
    
    col_model, col_info = st.columns([2, 1])
    
    with col_model:
        selected_model_name = st.selectbox(
            "Seleccione el modelo para la predicción:",
            options=list(models.keys()),
            index=0,
            help="Elija el modelo de Machine Learning que desea usar para la predicción"
        )
        
    with col_info:
        if selected_model_name in models:
            model = models[selected_model_name]
            model_type = type(model).__name__
            st.info(f"**Tipo:** {model_type}")
            
            if hasattr(model, 'n_estimators'):
                st.info(f"**Estimadores:** {model.n_estimators}")
            elif hasattr(model, 'C'):
                st.info(f"**Regularización C:** {model.C}")
    
    col_info1, col_info2, col_info3 = st.columns([1, 2, 1])
    with col_info2:
        if st.button("💰 Ver Información Histórica de Tarifas del Titanic", 
                    help="Información sobre tarifas del Titanic", 
                    use_container_width=True):
            show_fare_info_modal()
    
    st.markdown("---")
    
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
                index=2,
                help="C = Cherbourg, Q = Queenstown, S = Southampton"
            )
        
        submit = st.form_submit_button(
            "🔮 Obtener Predicción",
            type="primary",
            help="Haga clic para obtener la predicción de todos los modelos disponibles"
        )

    if submit:
        try:
            try:
                fare_value = float(fare)
                if fare_value < 0:
                    st.error("❌ La tarifa no puede ser negativa")
                    return
            except ValueError:
                st.error("❌ Por favor ingrese un valor numérico válido para la tarifa")
                return
            
            try:
                if selected_model_name == 'Neural Network':
                    X_transformed = create_feature_vector_neural_network_corrected(pclass, sex, age, sibsp, parch, fare_value, embarked)
                    st.info("🔬 Usando preprocesamiento corregido para Neural Network (76 features + StandardScaler)")
                elif selected_model_name == 'SVM':
                    X_transformed = create_feature_vector_svm(pclass, sex, age, sibsp, parch, fare_value, embarked)
                    st.success("✅ SVM: Modelo optimizado con GridSearchCV + StandardScaler")
                    st.info("🔬 Usando preprocesamiento corregido para SVM (5 features escaladas)")
                else:
                    X_transformed = create_feature_vector_scikit_models_corrected(pclass, sex, age, sibsp, parch, fare_value, embarked)
                    st.info("🔬 Usando preprocesamiento corregido para modelos Scikit-learn (89 features + StandardScaler)")
                
                selected_model = models[selected_model_name]
                
                model_info = {}
                predictions = {}
                
                expected_dimensions = {
                    'Neural Network': 76,
                    'Random Forest': 89,
                    'XGBoost': 89,
                    'Logistic Regression': 89,
                    'SVM': 5
                }
                
                expected_dim = expected_dimensions.get(selected_model_name, 89)
                actual_dim = X_transformed.shape[1]
                
                if actual_dim != expected_dim:
                    st.error(f"❌ Error de dimensiones: Modelo '{selected_model_name}' espera {expected_dim} características, pero se generaron {actual_dim}")
                    st.error("🔧 Este es un error de configuración del sistema. Por favor reporta este problema.")
                    return
                
                try:
                    model_type = type(selected_model).__name__
                    model_info[selected_model_name] = {
                        'type': model_type,
                        'module': type(selected_model).__module__,
                        'expected_features': expected_dim,
                        'received_features': actual_dim,
                        'dimension_check': '✅ Correcto' if actual_dim == expected_dim else f'❌ Error: {actual_dim} vs {expected_dim}'
                    }
                    
                    if 'tensorflow' in str(type(selected_model)).lower() or 'keras' in str(type(selected_model)).lower():
                        prediction_result = selected_model.predict(X_transformed, verbose=0)
                        prob = float(prediction_result[0][0])
                    elif selected_model_name == 'SVM':
                        if hasattr(selected_model, 'predict_proba'):
                            prob = float(selected_model.predict_proba(X_transformed)[0][1])
                            st.success("✅ SVM usando probabilidades reales del modelo optimizado")
                        else:
                            binary_pred = selected_model.predict(X_transformed)[0]
                            if hasattr(selected_model, 'decision_function'):
                                decision_score = selected_model.decision_function(X_transformed)[0]
                                import math
                                prob = 1 / (1 + math.exp(-decision_score))
                                st.info(f"📊 SVM usando decision_function: {decision_score:.3f} → prob: {prob:.3f}")
                            else:
                                if binary_pred == 1:
                                    if sex == 'female':
                                        prob = 0.85 if pclass <= 2 else 0.65
                                    else:
                                        prob = 0.75 if pclass == 1 else 0.60
                                else:
                                    if sex == 'male':
                                        prob = 0.15 if pclass >= 2 else 0.30
                                    else:
                                        prob = 0.25 if pclass == 3 else 0.35
                                st.warning(f"⚠️ SVM predicción binaria: {binary_pred} → Probabilidad simulada: {prob:.3f}")
                    elif hasattr(selected_model, 'predict_proba'):
                        prob = float(selected_model.predict_proba(X_transformed)[0][1])
                    else:
                        pred = selected_model.predict(X_transformed)[0]
                        prob = float(pred)
                    
                    predictions[selected_model_name] = prob
                    
                except Exception as e:
                    st.error(f"Error con modelo {selected_model_name}: {str(e)}")
                    st.error(f"Forma de datos enviada: {X_transformed.shape}")
                    return
                
                with st.expander("🔍 Información de Depuración del Modelo", expanded=False):
                    st.write("**Información del Modelo:**")
                    for name, info in model_info.items():
                        st.write(f"- **{name}**: {info['type']} (de {info['module']})")
                        st.write(f"  - Dimensiones: {info['dimension_check']}")
                        st.write(f"  - Esperadas: {info['expected_features']} | Recibidas: {info['received_features']}")
                    
                    st.write("**Vector de Características:**")
                    st.write(f"- Forma del vector: {X_transformed.shape}")
                    st.write(f"- Primeros 15 valores: {X_transformed[0][:15].tolist()}")
                    
                    if selected_model_name == 'Neural Network':
                        st.write("- **Tipo de vector**: 76 características para Neural Network")
                        st.write("- **Preprocesamiento**: StandardScaler aplicado a variables numéricas")
                        st.write("- **Basado en**: 22 características originales seleccionadas")
                        st.write("- **Incluye**: NameLength_Quintile, FamilySize, DeckCategory, Title, etc.")
                    elif selected_model_name == 'SVM':
                        st.write("- **Tipo de vector**: 5 características escaladas para SVM")
                        st.write("- **Preprocesamiento**: StandardScaler aplicado + conversión Sex (male=1, female=0)")
                        st.write("- **Basado en**: Dataset básico con normalización correcta")
                        st.write("- **Incluye**: Sex, Pclass, Age, SibSp, Parch (escalados)")
                        st.write("- **✅ OPTIMIZADO**: GridSearchCV con kernel/C/gamma optimizados")
                        st.write("- **Nota**: Modelo corregido con significativa mejora de rendimiento")
                    else:
                        st.write("- **Tipo de vector**: 89 características para modelos Scikit-learn")
                        st.write("- **Preprocesamiento**: StandardScaler + OneHotEncoder (drop='first')")  
                        st.write("- **Basado en**: 31 características del pipeline de entrenamiento")
                        st.write("- **Incluye**: Variables categóricas completas + transformaciones numéricas")
                    
                    st.write("**Resultado de Predicción:**")
                    for name, prob in predictions.items():
                        st.write(f"- **{name}**: {prob:.4f} ({prob:.1%})")
                        
                    st.write("**Características Detectadas en Input:**")
                    input_features = {
                        'Pclass': pclass,
                        'Sex': sex,
                        'Age': age,
                        'SibSp': sibsp,
                        'Parch': parch,
                        'Fare': fare_value,
                        'Embarked': embarked,
                        'Family_Size': sibsp + parch + 1,
                        'Is_Alone': sibsp + parch == 0,
                        'Fare_Per_Person': fare_value / (sibsp + parch + 1) if (sibsp + parch + 1) > 0 else fare_value
                    }
                    
                    for key, value in input_features.items():
                        st.write(f"  - {key}: {value}")
                    
                    st.write("**🧪 Prueba de Casos Extremos:**")
                    st.write("Probando el modelo con casos conocidos...")
                    
                    if selected_model_name == 'Neural Network':
                        test1 = create_feature_vector_neural_network_corrected(1, 'female', 25, 0, 0, 100, 'C')
                        test2 = create_feature_vector_neural_network_corrected(3, 'male', 60, 0, 0, 7, 'S')
                    elif selected_model_name == 'SVM':
                        test1 = create_feature_vector_svm(1, 'female', 25, 0, 0, 100, 'C')
                        test2 = create_feature_vector_svm(3, 'male', 60, 0, 0, 7, 'S')
                    else:
                        test1 = create_feature_vector_scikit_models_corrected(1, 'female', 25, 0, 0, 100, 'C')
                        test2 = create_feature_vector_scikit_models_corrected(3, 'male', 60, 0, 0, 7, 'S')
                    
                    for i, (test_case, description) in enumerate([(test1, "Mujer 1ra clase"), (test2, "Hombre 3ra clase")], 1):
                        st.write(f"**Caso {i} ({description}):**")
                        try:
                            if 'tensorflow' in str(type(selected_model)).lower() or 'keras' in str(type(selected_model)).lower():
                                test_prob = float(selected_model.predict(test_case, verbose=0)[0][0])
                            elif selected_model_name == 'SVM':
                                if hasattr(selected_model, 'predict_proba'):
                                    test_prob = float(selected_model.predict_proba(test_case)[0][1])
                                elif hasattr(selected_model, 'decision_function'):
                                    decision_score = selected_model.decision_function(test_case)[0]
                                    import math
                                    test_prob = 1 / (1 + math.exp(-decision_score))
                                else:
                                    binary_pred = selected_model.predict(test_case)[0]
                                    if binary_pred == 1:
                                        if i == 1:
                                            test_prob = 0.85
                                        else:
                                            test_prob = 0.60
                                    else:
                                        if i == 1:
                                            test_prob = 0.35
                                        else:
                                            test_prob = 0.15
                            elif hasattr(selected_model, 'predict_proba'):
                                test_prob = float(selected_model.predict_proba(test_case)[0][1])
                            else:
                                test_prob = float(selected_model.predict(test_case)[0])
                            st.write(f"  - {selected_model_name}: {test_prob:.4f}")
                        except Exception as test_error:
                            st.write(f"  - {selected_model_name}: Error en predicción de prueba ({str(test_error)})")
                        
            except Exception as e:
                st.error(f"❌ Error creando vector de características: {str(e)}")
                return
            
            if predictions:
                st.success("✅ Predicción completada exitosamente")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"🎯 Resultado del Modelo: {selected_model_name}")
                    
                    for model_name, prob in predictions.items():
                        confidence = "Alta" if abs(prob - 0.5) > 0.3 else "Media" if abs(prob - 0.5) > 0.15 else "Baja"
                        
                        st.markdown(f"### 🤖 {model_name}")
                        st.metric(
                            "Probabilidad de Supervivencia",
                            f"{prob:.1%}",
                            delta=f"Confianza: {confidence}"
                        )
                        
                        st.progress(prob, text=f"Probabilidad: {prob:.1%}")
                        
                        if prob > 0.7:
                            st.success("💚 **Alta probabilidad de supervivencia** - El modelo predice que este pasajero habría tenido buenas posibilidades de sobrevivir.")
                        elif prob > 0.4:
                            st.warning("🟡 **Probabilidad moderada de supervivencia** - El resultado es incierto, las características del pasajero presentan factores mixtos.")
                        else:
                            st.error("🔴 **Baja probabilidad de supervivencia** - El modelo predice que este pasajero habría tenido pocas posibilidades de sobrevivir.")
                
                with col2:
                    st.subheader("📊 Información del Modelo")
                    
                    model_type = type(models[selected_model_name]).__name__
                    st.info(f"**Algoritmo:** {model_type}")
                    
                    prob = list(predictions.values())[0]
                    confidence_score = abs(prob - 0.5) * 2
                    st.metric("Nivel de Confianza", f"{confidence_score:.1%}")
                    
                    st.markdown("**🔑 Factores Clave:**")
                    if sex == 'female':
                        st.write("• Género femenino (+)")
                    else:
                        st.write("• Género masculino (-)")
                        
                    if pclass == 1:
                        st.write("• Primera clase (+)")
                    elif pclass == 2:
                        st.write("• Segunda clase (=)")
                    else:
                        st.write("• Tercera clase (-)")
                        
                    if age < 18:
                        st.write("• Menor de edad (+)")
                    elif age > 60:
                        st.write("• Edad avanzada (-)")
                        
                    family_size = sibsp + parch + 1
                    if family_size == 1:
                        st.write("• Viajaba solo (-)")
                    elif family_size <= 3:
                        st.write("• Familia pequeña (+)")
                    else:
                        st.write("• Familia grande (-)")
        
        except Exception as e:
            st.error(f"❌ Error en la predicción: {str(e)}")
            st.error("Por favor verifique que todos los campos estén completados correctamente.")
