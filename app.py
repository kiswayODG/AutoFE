import streamlit as st
import pandas as pd
import numpy as np
from modules.data_loading import load_data_section
from modules.feature_engineering import feature_engineering_section
from modules.modeling import modeling_section
from modules.results_analysis import results_analysis_section
import traceback


st.set_page_config(
    page_title="AutoML App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

 
st.title("Application de Feature Engineering automatisée")

 
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'feature_engineering_done' not in st.session_state:
    st.session_state.feature_engineering_done = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'result' not in st.session_state:
    st.session_state.result = False
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Chargement des données"

 
with st.sidebar:
    st.title("Navigation")
    
     
    tabs = ["Chargement des données", "Feature Engineering", "Modélisation", "Analyse des résultats"]
    
    if st.button("Chargement des données"):
        st.session_state.active_tab = "Chargement des données"
    
    if st.button("Feature Engineering", disabled=not st.session_state.data_loaded):
        st.session_state.active_tab = "Feature Engineering"
    
    if st.button("Modélisation", disabled=not st.session_state.feature_engineering_done):
        st.session_state.active_tab = "Modélisation"
    
    if st.button("Analyse des résultats", disabled=not st.session_state.model_trained):
        st.session_state.active_tab = "Analyse des résultats"
    
     
    st.divider()
    st.subheader("État de l'application")
    
    status_color = {True: "✅", False: "❌"} 
    
    st.markdown(f"{status_color[st.session_state.data_loaded]} Données chargées")
    st.markdown(f"{status_color[st.session_state.feature_engineering_done]} Feature Engineering terminé")
    st.markdown(f"{status_color[st.session_state.model_trained]} Modèle entraîné") 

 
try:
    if st.session_state.active_tab == "Chargement des données":
        load_data_section()
    
    elif st.session_state.active_tab == "Feature Engineering":
        feature_engineering_section()
    
    elif st.session_state.active_tab == "Modélisation":
        modeling_section()
    
    elif st.session_state.active_tab == "Analyse des résultats":
        results_analysis_section()
        print("Analyse des résultats")

except Exception as e:
    st.error(f"Une erreur s'est produite: {e}")
    st.code(traceback.format_exc())
