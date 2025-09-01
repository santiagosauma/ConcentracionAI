import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

def render_exploration_page(df):
    st.header("🔍 Exploración de Datos Avanzada")
    
    # Información contextual mejorada
    st.markdown("""
    **📊 Análisis Exploratorio Interactivo**: Esta sección permite analizar los datos históricos del Titanic 
    con herramientas avanzadas de visualización y estadística. Use los filtros para explorar patrones específicos 
    y descubrir insights automáticos basados en los datos.
    """)
    
    st.sidebar.header("🎛️ Filtros de Exploración Avanzados")
    
    with st.sidebar.expander("👥 Filtros Demográficos", expanded=True):
        selected_sex = st.multiselect(
            "Género:",
            options=df['Sex'].unique(),
            default=df['Sex'].unique(),
            help="Seleccione uno o más géneros"
        )
        
        selected_class = st.multiselect(
            "Clase del Pasajero:",
            options=sorted(df['Pclass'].unique()),
            default=sorted(df['Pclass'].unique()),
            help="1 = Primera, 2 = Segunda, 3 = Tercera Clase"
        )
        
        age_range = st.slider(
            "Rango de Edad:",
            min_value=int(df['Age'].min()),
            max_value=int(df['Age'].max()),
            value=(int(df['Age'].min()), int(df['Age'].max())),
            help="Seleccione el rango de edades"
        )
    
    with st.sidebar.expander("🚢 Filtros de Viaje", expanded=False):
        embarked_options = df['Embarked'].dropna().unique()
        selected_embarked = st.multiselect(
            "Puerto de Embarque:",
            options=embarked_options,
            default=embarked_options,
            help="C = Cherbourg, Q = Queenstown, S = Southampton"
        )
        
        fare_min, fare_max = df['Fare'].min(), df['Fare'].max()
        selected_fare_range = st.slider(
            "Rango de Tarifa (£):",
            min_value=float(fare_min),
            max_value=float(fare_max),
            value=(float(fare_min), float(fare_max)),
            help="Seleccione el rango de tarifas pagadas"
        )
        
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        family_size_range = st.slider(
            "Tamaño de Familia:",
            min_value=int(df['FamilySize'].min()),
            max_value=int(df['FamilySize'].max()),
            value=(int(df['FamilySize'].min()), int(df['FamilySize'].max())),
            help="Incluye al pasajero + cónyuge/hermanos + padres/hijos"
        )
    
    filtered_df = df[
        (df['Sex'].isin(selected_sex)) &
        (df['Pclass'].isin(selected_class)) &
        (df['Age'].between(age_range[0], age_range[1])) &
        (df['Embarked'].isin(selected_embarked)) &
        (df['Fare'].between(selected_fare_range[0], selected_fare_range[1])) &
        (df['FamilySize'].between(family_size_range[0], family_size_range[1]))
    ]
    
    st.subheader("📊 Métricas del Segmento Filtrado")
    
    # Calcular métricas clave
    total_passengers = len(filtered_df)
    survival_rate = filtered_df['Survived'].mean() if len(filtered_df) > 0 else 0
    avg_age = filtered_df['Age'].mean() if len(filtered_df) > 0 else 0
    avg_fare = filtered_df['Fare'].mean() if len(filtered_df) > 0 else 0
    
    # Métricas generales para comparación
    total_general = len(df)
    survival_rate_general = df['Survived'].mean()
    
    # Dashboard de métricas con indicadores
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_passengers = total_passengers - total_general
        st.metric(
            "👥 Pasajeros",
            f"{total_passengers:,}",
            delta=f"{delta_passengers:+,} vs total"
        )
    
    with col2:
        delta_survival = (survival_rate - survival_rate_general) * 100
        st.metric(
            "💚 Tasa Supervivencia",
            f"{survival_rate:.1%}",
            delta=f"{delta_survival:+.1f}% vs promedio",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "📅 Edad Promedio",
            f"{avg_age:.1f} años" if not pd.isna(avg_age) else "N/A",
        )
    
    with col4:
        st.metric(
            "💰 Tarifa Promedio",
            f"£{avg_fare:.1f}" if not pd.isna(avg_fare) else "N/A",
        )
    
    # Mensaje de filtrado
    if len(filtered_df) != len(df):
        st.info(f"📊 Mostrando **{len(filtered_df):,}** de **{len(df):,}** pasajeros ({len(filtered_df)/len(df):.1%} del total)")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados. Ajuste los criterios.")
        return
    
    st.subheader("📈 Análisis Visual Interactivo")
    
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "📊 Distribuciones", 
        "🔥 Correlaciones", 
        "📈 Comparaciones", 
        "🎯 Análisis Avanzado"
    ])
    
    with viz_tab1:
        st.markdown("#### 📊 Distribuciones por Supervivencia")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_age = px.histogram(
                filtered_df, 
                x='Age', 
                color='Survived', 
                nbins=20,
                title="Distribución de Edades por Supervivencia",
                labels={'Survived': 'Sobrevivió', 'Age': 'Edad'},
                color_discrete_map={0: '#ff7f7f', 1: '#7fdf7f'}
            )
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            fig_fare = px.histogram(
                filtered_df[filtered_df['Fare'] < filtered_df['Fare'].quantile(0.95)], 
                x='Fare', 
                color='Survived',
                nbins=20,
                title="Distribución de Tarifas por Supervivencia",
                labels={'Survived': 'Sobrevivió', 'Fare': 'Tarifa (£)'},
                color_discrete_map={0: '#ff7f7f', 1: '#7fdf7f'}
            )
            fig_fare.update_layout(height=400)
            st.plotly_chart(fig_fare, use_container_width=True)
    
    with viz_tab2:
        st.markdown("#### 🔥 Análisis de Correlaciones")
        
        survival_by_class_sex = filtered_df.groupby(['Pclass', 'Sex'])['Survived'].agg(['mean', 'count']).reset_index()
        survival_pivot = survival_by_class_sex.pivot(index='Pclass', columns='Sex', values='mean')
        
        fig_heatmap = px.imshow(
            survival_pivot,
            title="Tasa de Supervivencia por Clase y Género",
            labels=dict(x="Género", y="Clase", color="Tasa Supervivencia"),
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Matriz de correlación para variables numéricas
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Matriz de Correlación - Variables Numéricas",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with viz_tab3:
        st.markdown("#### 📈 Comparaciones por Grupos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_box_age = px.box(
                filtered_df,
                x='Pclass',
                y='Age',
                color='Survived',
                title="Distribución de Edades por Clase y Supervivencia",
                labels={'Pclass': 'Clase', 'Age': 'Edad', 'Survived': 'Sobrevivió'}
            )
            fig_box_age.update_layout(height=400)
            st.plotly_chart(fig_box_age, use_container_width=True)
        
        with col2:
            fig_violin = px.violin(
                filtered_df[filtered_df['Fare'] < filtered_df['Fare'].quantile(0.95)],
                x='Pclass',
                y='Fare',
                color='Survived',
                title="Distribución de Tarifas por Clase y Supervivencia",
                labels={'Pclass': 'Clase', 'Fare': 'Tarifa (£)', 'Survived': 'Sobrevivió'}
            )
            fig_violin.update_layout(height=400)
            st.plotly_chart(fig_violin, use_container_width=True)
    
    with viz_tab4:
        st.markdown("#### 🎯 Análisis Estadístico Avanzado")
        
        # Análisis de outliers
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(
                filtered_df,
                x='Age',
                y='Fare',
                color='Survived',
                size='FamilySize',
                symbol='Pclass',
                title="Análisis Multidimensional: Edad vs Tarifa",
                labels={
                    'Age': 'Edad', 
                    'Fare': 'Tarifa (£)', 
                    'Survived': 'Sobrevivió',
                    'FamilySize': 'Tamaño Familia'
                },
                hover_data=['Sex', 'Embarked']
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Análisis de supervivencia por tamaño de familia
            family_survival = filtered_df.groupby('FamilySize').agg({
                'Survived': ['mean', 'count'],
                'Fare': 'mean'
            }).round(3)
            
            family_survival.columns = ['Tasa_Supervivencia', 'Cantidad_Pasajeros', 'Tarifa_Promedio']
            family_survival = family_survival.reset_index()
            
            fig_family = px.scatter(
                family_survival,
                x='FamilySize',
                y='Tasa_Supervivencia',
                size='Cantidad_Pasajeros',
                color='Tarifa_Promedio',
                title="Supervivencia por Tamaño de Familia",
                labels={
                    'FamilySize': 'Tamaño de Familia',
                    'Tasa_Supervivencia': 'Tasa de Supervivencia',
                    'Cantidad_Pasajeros': 'Cantidad',
                    'Tarifa_Promedio': 'Tarifa Promedio'
                }
            )
            fig_family.update_layout(height=400)
            st.plotly_chart(fig_family, use_container_width=True)
    
    st.subheader("💡 Insights Automáticos del Segmento")
    
    insights = []
    
    if len(selected_sex) > 1:
        survival_by_sex = filtered_df.groupby('Sex')['Survived'].mean()
        best_sex = survival_by_sex.idxmax()
        worst_sex = survival_by_sex.idxmin()
        diff = (survival_by_sex[best_sex] - survival_by_sex[worst_sex]) * 100
        insights.append(f"👥 **Género**: {best_sex} tiene {diff:.1f}% mayor tasa de supervivencia que {worst_sex}")
    
    if len(selected_class) > 1:
        survival_by_class = filtered_df.groupby('Pclass')['Survived'].mean()
        best_class = survival_by_class.idxmax()
        worst_class = survival_by_class.idxmin()
        diff = (survival_by_class[best_class] - survival_by_class[worst_class]) * 100
        insights.append(f"🎫 **Clase**: Clase {best_class} tiene {diff:.1f}% mayor supervivencia que Clase {worst_class}")
    
    if survival_rate > 0:
        survived_age = filtered_df[filtered_df['Survived'] == 1]['Age'].mean()
        not_survived_age = filtered_df[filtered_df['Survived'] == 0]['Age'].mean()
        if not pd.isna(survived_age) and not pd.isna(not_survived_age):
            age_diff = abs(survived_age - not_survived_age)
            if age_diff > 3:
                younger_group = "supervivientes" if survived_age < not_survived_age else "no supervivientes"
                insights.append(f"📅 **Edad**: Los {younger_group} son en promedio {age_diff:.1f} años más jóvenes")
    
    if len(filtered_df) > 10:
        high_fare_survival = filtered_df[filtered_df['Fare'] > filtered_df['Fare'].median()]['Survived'].mean()
        low_fare_survival = filtered_df[filtered_df['Fare'] <= filtered_df['Fare'].median()]['Survived'].mean()
        fare_diff = (high_fare_survival - low_fare_survival) * 100
        if abs(fare_diff) > 10:
            better_group = "altas" if high_fare_survival > low_fare_survival else "bajas"
            insights.append(f"💰 **Tarifa**: Pasajeros con tarifas {better_group} tienen {abs(fare_diff):.1f}% mejor supervivencia")
    
    if insights:
        for insight in insights:
            st.success(insight)
    else:
        st.info("💡 Seleccione diferentes filtros para generar insights comparativos")
    
    with st.expander("📊 Estadísticas Detalladas del Segmento", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Estadísticas Descriptivas**")
            desc_stats = filtered_df[['Age', 'Fare', 'FamilySize']].describe().round(2)
            st.dataframe(desc_stats)
        
        with col2:
            st.markdown("**🎯 Análisis por Categorías**")
            
            survival_summary = []
            
            for col in ['Sex', 'Pclass', 'Embarked']:
                if col in filtered_df.columns:
                    survival_by_cat = filtered_df.groupby(col)['Survived'].agg(['count', 'mean']).round(3)
                    survival_by_cat.columns = ['Cantidad', 'Tasa_Supervivencia']
                    survival_by_cat['Categoria'] = col
                    survival_by_cat = survival_by_cat.reset_index()
                    survival_summary.append(survival_by_cat)
            
            if survival_summary:
                combined_stats = pd.concat(survival_summary, ignore_index=True)
                st.dataframe(combined_stats)

def calculate_statistical_significance(df, group_col, target_col):
    groups = df[group_col].unique()
    if len(groups) == 2:
        group1 = df[df[group_col] == groups[0]][target_col]
        group2 = df[df[group_col] == groups[1]][target_col]
        
        statistic, p_value = stats.ttest_ind(group1, group2)
        return p_value < 0.05, p_value
    return False, None
