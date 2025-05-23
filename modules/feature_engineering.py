import streamlit as st
import pandas as pd
import numpy as np
import featuretools as ft
from autofeat import AutoFeatRegressor, AutoFeatClassifier

from sklearn.preprocessing import LabelEncoder
from utilities import encode_categorical_variables

def feature_engineering_section():
    """Section de feature engineering."""
    st.header("2. Feature Engineering")
    
    if not st.session_state.data_loaded:
        st.error("Veuillez d'abord charger des données.")
        return
    
    st.write("""
    **Instructions :**
    1. Choisissez une méthode de feature engineering (manuelle ou automatique).
    2. Configurez les paramètres pour cette méthode.
    3. Appliquez les transformations pour générer de nouvelles caractéristiques.
    """)
    
    main_df = st.session_state.dataframes[st.session_state.main_table]
    target = st.session_state.target_column
    target_type = st.session_state.target_type
    
     
    method = st.radio("Choisissez la méthode de feature engineering", 
                      ["Manuel", "AutoFeat", "Featuretools"])
    
    if method == "Manuel":
        manual_feature_engineering(main_df, target)
    
    elif method == "AutoFeat":
        autofeat_feature_engineering(main_df, target, target_type)
    
    elif method == "Featuretools":
        featuretools_feature_engineering(target)
    
     
    if st.session_state.feature_engineering_done:
        st.subheader("Aperçu des données après feature engineering")
        st.dataframe(st.session_state.X.head())
        
        # 
        st.subheader("Statistiques des caractéristiques")
        st.dataframe(st.session_state.X.describe())
        
        if st.button("Continuer vers la modélisation"):
            st.session_state.active_tab = "Modélisation"
            st.rerun()

def manual_feature_engineering(df, target):
    """Feature engineering manuel avec transformations personnalisées."""
    st.subheader("Feature Engineering Manuel")
    
     
    possible_features = [col for col in df.columns if col != target]
    features = st.multiselect("Sélectionnez les caractéristiques à utiliser", 
                             possible_features, 
                             default=possible_features)
    
    if not features:
        st.warning("Veuillez sélectionner au moins une caractéristique.")
        return
    
    # Identifier les colonnes catégorielles
    cat_candidates = df[features].select_dtypes(include=["object", "category", "string"]).columns.tolist()
    categorical_cols = st.multiselect("Colonnes catégorielles à encoder", cat_candidates, default=cat_candidates)
    
    # Options de transformation
    st.subheader("Transformations")
    st.write("Sélectionnez les transformations à appliquer pour chaque colonne numérique:")
    
    transform_options = {
        "Racine carrée": np.sqrt,
        "Logarithme": np.log1p,
        "Inverse": lambda x: 1/(x+0.001),  # Ajout d'une petite valeur pour éviter la division par zéro
        "Carré": lambda x: x**2,
        "Normalisation": lambda x: (x - x.mean()) / x.std()
    }
    
    # Sélection des transformations par colonne
    transform_cols = {}
    numeric_cols = df[features].select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    if numeric_cols:
        cols_per_row = 3
        rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
        
        for i in range(rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < len(numeric_cols):
                    col_name = numeric_cols[idx]
                    with cols[j]:
                        transform_cols[col_name] = st.multiselect(
                            f"{col_name}",
                            list(transform_options.keys()),
                            key=f"transform_{col_name}"
                        )
    
    # Interactions entre caractéristiques
    st.subheader("Interactions entre caractéristiques")
    create_interactions = st.checkbox("Créer des interactions entre caractéristiques numériques")
    selected_interactions = []
    
    if create_interactions and len(numeric_cols) >= 2:
        num_interactions = st.slider("Nombre d'interactions à créer", 
                                    min_value=1, 
                                    max_value=min(10, len(numeric_cols) * (len(numeric_cols) - 1) // 2),
                                    value=3)
        
        for i in range(num_interactions):
            col1, col2, op = st.columns(3)
            with col1:
                first_col = st.selectbox(f"Première colonne #{i+1}", numeric_cols, key=f"first_col_{i}")
            with col2:
                second_col = st.selectbox(f"Deuxième colonne #{i+1}", 
                                         [c for c in numeric_cols if c != first_col], 
                                         key=f"second_col_{i}")
            with op:
                operation = st.selectbox(f"Opération #{i+1}", 
                                        ["Multiplication", "Division", "Addition", "Soustraction"],
                                        key=f"op_{i}")
            
            selected_interactions.append((first_col, second_col, operation))
    
    # Exécuter le feature engineering
    if st.button("Appliquer le feature engineering manuel"):
        with st.spinner("Application des transformations..."):
            # Copie du dataframe
            df_transformed = df.copy()
            
            # Encoder les variables catégorielles
            if categorical_cols:
                df_transformed, encoders = encode_categorical_variables(df_transformed, categorical_cols)
                st.session_state.encoders = encoders
            
            # Appliquer les transformations sélectionnées
            for col, transforms in transform_cols.items():
                for transform in transforms:
                    new_col_name = f"{col}_{transform}"
                    try:
                        df_transformed[new_col_name] = transform_options[transform](df_transformed[col])
                    except Exception as e:
                        st.warning(f"Erreur lors de la transformation {transform} pour {col}: {e}")
            
            # Créer les interactions
            for first_col, second_col, operation in selected_interactions:
                new_col_name = f"{first_col}_{operation}_{second_col}"
                try:
                    if operation == "Multiplication":
                        df_transformed[new_col_name] = df_transformed[first_col] * df_transformed[second_col]
                    elif operation == "Division":
                        df_transformed[new_col_name] = df_transformed[first_col] / (df_transformed[second_col] + 0.001)
                    elif operation == "Addition":
                        df_transformed[new_col_name] = df_transformed[first_col] + df_transformed[second_col]
                    elif operation == "Soustraction":
                        df_transformed[new_col_name] = df_transformed[first_col] - df_transformed[second_col]
                except Exception as e:
                    st.warning(f"Erreur lors de la création de l'interaction {operation} entre {first_col} et {second_col}: {e}")
            
            # Stocker les résultats
            features_for_model = [col for col in df_transformed.columns if col != target]
            st.session_state.X = df_transformed[features_for_model]
            st.session_state.y = df_transformed[target]
            st.session_state.feature_engineering_done = True
            
            st.success("Feature engineering manuel terminé avec succès!")

def autofeat_feature_engineering(df, target, target_type):
    """Feature engineering automatique avec AutoFeat."""
    st.subheader("Feature Engineering Automatique avec AutoFeat")
    
    # Options AutoFeat
    featsel_runs = st.slider("Nombre d'exécutions pour la sélection de caractéristiques", 
                           min_value=1, max_value=10, value=5)
   
    
    # Exécuter AutoFeat
    if st.button("Appliquer AutoFeat"):
        with st.spinner("Feature engineering avec AutoFeat en cours..."):
            try:
                # Copie du dataframe
                df_copy = df.copy()
                
                # Encoder les colonnes catégorielles
                cat_cols = df_copy.select_dtypes(include=["object", "category", "string"]).columns.tolist()
                if cat_cols:
                    for col in cat_cols:
                        if col != target or (col == target and target_type == "Classification"):
                            le = LabelEncoder()
                            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                
                # Utiliser AutoFeat selon le type de problème
                if target_type == "Régression":
                    autofeat_model = AutoFeatRegressor(featsel_runs=featsel_runs, 
                                                    verbose=True)
                else:
                    autofeat_model = AutoFeatClassifier(featsel_runs=featsel_runs,
                                                     verbose=True)
                
                # Transformer les données
                X_input = df_copy.drop(columns=[target])
                y_input = df_copy[target]
                
                X_transformed = autofeat_model.fit_transform(X_input, y_input)
                
                # Stocker les résultats
                st.session_state.X = pd.DataFrame(X_transformed, 
                                              columns=X_transformed.columns)
                st.session_state.y = y_input
                st.session_state.feature_engineering_done = True
                
                # Afficher les caractéristiques importantes
                if hasattr(autofeat_model, 'get_support'):
                    feat_importance = autofeat_model.get_support()
                    st.write("Importance des caractéristiques (Top 10):")
                    importance_df = pd.DataFrame({
                        'Feature': X_transformed.columns,
                        'Importance': feat_importance
                    }).sort_values('Importance', ascending=False).head(10)
                    st.dataframe(importance_df)
                
                st.success("Feature engineering avec AutoFeat terminé avec succès!")
                
            except Exception as e:
                st.error(f"Erreur lors de l'exécution d'AutoFeat: {e}")

def featuretools_feature_engineering(target):
    """Feature engineering automatique avec Featuretools."""
    st.subheader("Feature Engineering Automatique avec Featuretools")
    
    if len(st.session_state.dataframes) < 1:
        st.error("Veuillez charger au moins un fichier CSV.")
        return
    
    
    dataframes = {}
    entity_set_name ="entityset"
    
     
    for file_name, df in st.session_state.dataframes.items():
        st.subheader(f"Configuration pour {file_name}")
        
        entity_name = st.text_input(f"Nom de l'entité pour {file_name}", 
                                  value=file_name.split('.')[0],
                                  key=f"entity_{file_name}")
        
        index_col = st.selectbox(f"Colonne d'index pour {entity_name}", 
                               df.columns,
                               key=f"index_{file_name}")
        if not df[index_col].is_unique:
            st.warning(f"La colonne '{index_col}' n'est pas unique. Une colonne 'auto_index' sera ajoutée.")
            df['auto_index'] = range(1, len(df) + 1)
            index_col = 'auto_index'
        
        time_col = st.selectbox(f"Colonne temporelle (optionnel)", 
                              ["Aucune"] + list(df.columns),
                              key=f"time_{file_name}")
        
        
        if time_col != "Aucune":
            dataframes[entity_name] = (df, index_col, time_col)
        else:
            dataframes[entity_name] = (df, index_col)
    
    # Définir les relations si plusieurs dataframes
    relationships = []
    if len(dataframes) > 1:
        st.subheader("Définition des relations")
        
        num_rels = st.number_input("Nombre de relations à définir", 
                                 min_value=0, 
                                 max_value=10, 
                                 value=min(1, len(dataframes)-1))
        
        df_names = list(dataframes.keys())
        for i in range(int(num_rels)):
            st.markdown(f"**Relation {i+1}**")
            
            parent_df = st.selectbox(f"Entité parent", df_names, key=f"parent_df_{i}")
            parent_df_data = st.session_state.dataframes[
                [k for k in st.session_state.dataframes.keys() 
                 if k.split('.')[0] == parent_df][0]
            ]
            parent_col = st.selectbox(f"Colonne parent", 
                                    parent_df_data.columns, 
                                    key=f"parent_col_{i}")
            
            child_df = st.selectbox(f"Entité enfant", 
                                  [df for df in df_names if df != parent_df], 
                                  key=f"child_df_{i}")
            child_df_data = st.session_state.dataframes[
                [k for k in st.session_state.dataframes.keys() 
                 if k.split('.')[0] == child_df][0]
            ]
            child_col = st.selectbox(f"Colonne enfant", 
                                   child_df_data.columns, 
                                   key=f"child_col_{i}")
            
            relationships.append((parent_df, parent_col, child_df, child_col))
    
    # Paramètres Deep Feature Synthesis
    st.subheader("Paramètres Deep Feature Synthesis")
    
    max_depth = st.slider("Profondeur maximale", min_value=1, max_value=5, value=2)
    
    
    target_entity = st.selectbox("Entité cible", list(dataframes.keys()), 
                               index=0)
    
    # Exécuter Featuretools
    if st.button("Appliquer Featuretools"):
        with st.spinner("Feature engineering avec Featuretools en cours..."):
            try:
                 
                es = ft.EntitySet(id=entity_set_name)
                
                 
                for entity_name, (df, index, *time_index) in dataframes.items():
                    if time_index:
                        es.add_dataframe(
                            dataframe_name=entity_name,
                            dataframe=df,
                            index=index,
                            time_index=time_index[0]
                        )
                    else:
                        es.add_dataframe(
                            dataframe_name=entity_name,
                            dataframe=df,
                            index=index
                        )
                
                
                
                
                feature_matrix, feature_defs = ft.dfs(
                    dataframes=dataframes,
                    target_dataframe_name=target_entity,
                    relationships=relationships,
                )
                
                
                
                # Stocker les résultats
                if target in feature_matrix.columns:
                    st.session_state.X = feature_matrix.drop(columns=[target])
                    st.session_state.y = feature_matrix[target]
                else:
                    main_df = st.session_state.dataframes[st.session_state.main_table]
                    index_col = [col for entity, (df, col, *_) in dataframes.items() 
                                if entity == target_entity][0]
                    
                    # Joindre la colonne cible
                    merged_df = pd.merge(
                        feature_matrix,
                        main_df[[index_col, target]],
                        on=index_col
                    )
                    
                    st.session_state.X = merged_df.drop(columns=[target, index_col])
                    st.session_state.y = merged_df[target]
                
                st.session_state.feature_engineering_done = True
                
                 
                st.write(f"Nombre de caractéristiques générées: {len(feature_defs)}")
                st.write("Aperçu des caractéristiques générées:")
                st.dataframe(feature_matrix.head())
                
                st.success("Feature engineering avec Featuretools terminé avec succès!")
                
            except Exception as e:
                st.error(f"Erreur lors de l'exécution de Featuretools: {e}")
                st.error(str(e))
