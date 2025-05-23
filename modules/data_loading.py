import streamlit as st
import pandas as pd
import numpy as np
import io
from utilities import load_uploaded_csv_files, handle_missing_values

def load_data_section():
    """Section de chargement et préparation des données."""
    st.header("1. Chargement et préparation des données")
    
    st.write("""
    **Instructions :**
    1. Téléchargez d'abord la table principale contenant la colonne cible.
    2. Téléchargez ensuite les autres tables si nécessaire.
    3. Examinez les données et traitez les valeurs manquantes.
    """)
    
    # File upload
    uploaded_files = st.file_uploader("Téléchargez vos fichiers CSV", type=["csv"], accept_multiple_files=True)
    
    if not uploaded_files:
        st.info("Veuillez télécharger au moins un fichier CSV.")
        st.session_state.data_loaded = False
        return
    
    # Progress indicator
    st.session_state.dataframes = {}
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        df = load_uploaded_csv_files(file)
        progress_val = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress_val)
    
    progress_bar.empty()
    
    # Affichage des données et traitement
    st.subheader("Aperçu des données")
    
    for file in uploaded_files:
        df = st.session_state.dataframes[file.name]
        
        with st.expander(f"Fichier: {file.name}", expanded=(file == uploaded_files[0])):
            st.dataframe(df.head())
            
            st.write("Informations:")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
            st.write("Statistiques descriptives:")
            st.dataframe(df.describe())
            
            # Traitement des valeurs manquantes
            missing_values = df.isnull().sum()
            if missing_values.any():
                st.write("Valeurs manquantes:")
                st.dataframe(missing_values[missing_values > 0])
                
                missing_value_option = st.selectbox(
                    "Méthode de traitement des valeurs manquantes",
                    ["Supprimer les lignes", "Supprimer les colonnes", "Imputation (moyenne)", "Imputation (médiane)", "Imputation (mode)"],
                    key=f"missing_value_option_{file.name}"
                )
                
                if st.button(f"Appliquer le traitement pour {file.name}"):
                    with st.spinner("Traitement des valeurs manquantes..."):
                        df_cleaned = handle_missing_values(df, missing_value_option)
                        st.session_state.dataframes[file.name] = df_cleaned
                        st.success(f"Valeurs manquantes traitées pour {file.name}")
                        st.experimental_rerun()
    
    # Sélection de la table principale et colonne cible
    if len(uploaded_files) > 0:
        st.subheader("Configuration de l'analyse")
        col1, col2 = st.columns(2)
        
        with col1:
            main_table = st.selectbox("Sélectionnez la table principale", 
                                      [file.name for file in uploaded_files],
                                      index=0)
        
        main_df = st.session_state.dataframes[main_table]
        
        with col2:
            target_column = st.selectbox("Sélectionnez la colonne cible", main_df.columns)
        
        # Déterminer automatiquement le type de la variable cible
        target_type = "Classification" if main_df[target_column].nunique() < 20 else "Régression"
        st.radio("Type de problème", ["Classification", "Régression"], index=0 if target_type == "Classification" else 1, key="problem_type")
        
        # Enregistrer les informations de configuration
        if st.button("Confirmer et continuer"):
            st.session_state.main_table = main_table
            st.session_state.target_column = target_column
            st.session_state.target_type = st.session_state.problem_type
            st.session_state.data_loaded = True
            st.session_state.active_tab = "Feature Engineering"
            st.success("Données chargées avec succès! Passez à l'étape de Feature Engineering.")
            st.rerun()
