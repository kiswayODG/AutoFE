import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, explained_variance_score,
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import time
import pickle
import io
import base64
from utilities import (
    plot_confusion_matrix, plot_feature_importance, 
    plot_regression_results, 
    
)

print("Loading results_analysis.py...")

def results_analysis_section():
    """Section d'analyse des résultats du modèle."""
    st.header("4. Analyse des résultats")
    
    
    if not st.session_state.model_trained:
        st.error("Veuillez d'abord entraîner un modèle.")
        return
    
    st.write("""
    **Instructions :**
    1. Explorez les différentes métriques et visualisations pour évaluer les performances du modèle.
    2. Analysez les caractéristiques importantes du modèle.
    3. Exportez le modèle ou les résultats pour une utilisation ultérieure.
    """)
    
    
    model = st.session_state.model
    model_name = st.session_state.model_name
    model_results = st.session_state.model_results
    target_type = st.session_state.target_type
    
    X_test = model_results['X_test']
    y_test = model_results['y_test']
    y_pred = model_results['y_pred']
    training_time = model_results['training_time']
    
     
    st.subheader("Résumé des performances")
    st.write(f"**Modèle:** {model_name}")
    st.write(f"**Temps d'entraînement:** {training_time:.2f} secondes")
    
    
    if target_type == "Classification":
        classification_analysis(model, X_test, y_test, y_pred)
    else:   
        regression_analysis(y_test, y_pred)
    
     
    if 'feature_importance' in st.session_state:
        feature_analysis()
    
    
    export_model(model)
    
    st.session_state.model_trained = True

def classification_analysis(model, X_test, y_test, y_pred):
    """Analyse détaillée pour les modèles de classification."""
    st.subheader("Métriques de classification")
    
     
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Précision", f"{accuracy:.4f}")
    
     
    st.subheader("Rapport de classification")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0))
    
    # Matrice de confusion
    st.subheader("Matrice de confusion")
    cm = confusion_matrix(y_test, y_pred)
    class_names = sorted(y_test.unique())
    fig = plot_confusion_matrix(cm, class_names)
    st.pyplot(fig)
    
    
def regression_analysis(y_test, y_pred):
    """Analyse détaillée pour les modèles de régression."""
    st.subheader("Métriques de régression")
    
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MSE", f"{mse:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
    with col2:
        st.metric("MAE", f"{mae:.4f}")
        st.metric("R²", f"{r2:.4f}")
    with col3:
        st.metric("Variance expliquée", f"{evs:.4f}")
    
     
    st.subheader("Prédictions vs Valeurs réelles")
    fig = plot_regression_results(y_test, y_pred)
    st.pyplot(fig)
    
     
    st.subheader("Distribution des erreurs")
    residuals = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
     
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title("Distribution des résidus")
    ax1.set_xlabel("Erreur")
    
     
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title("Résidus vs Prédictions")
    ax2.set_xlabel("Prédictions")
    ax2.set_ylabel("Résidus")
    
    plt.tight_layout()
    st.pyplot(fig)

def feature_analysis():
    """Analyse de l'importance des caractéristiques."""
    st.subheader("Importance des caractéristiques")
    
    feature_importance = st.session_state.feature_importance
    
     
    importance_df = pd.DataFrame({
        'Feature': feature_importance['features'],
        'Importance': feature_importance['importance']
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    
    st.dataframe(importance_df.style.bar(subset=['Importance'], color='#5fba7d'))
    
    
     
    

def export_model(model):
    """Exporter le modèle entraîné."""
    st.subheader("Exporter le modèle")
    
    if st.button("Sauvegarder le modèle (PKL)"):
        try:
            
            model_pkl = pickle.dumps(model)
            
            
            b64 = base64.b64encode(model_pkl).decode()
            href = f'<a href="data:file/pickle;base64,{b64}" download="trained_model.pkl">Télécharger le modèle (PKL)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            st.success("Modèle prêt à télécharger!")
        except Exception as e:
            st.error(f"Erreur lors de la sérialisation du modèle: {e}")
    
    '''
    # Exportation des prédictions
    if 'model_results' in st.session_state:
        if st.button("Exporter les prédictions (CSV)"):
            try:
                
                results = pd.DataFrame({
                    'y_test': st.session_state.model_results['y_test'],
                    'y_pred': st.session_state.model_results['y_pred']
                })
                
                 
                csv = results.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Télécharger les prédictions (CSV)</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.success("Prédictions prêtes à télécharger!")
            except Exception as e:
                st.error(f"Erreur lors de l'exportation des prédictions: {e}")
'''