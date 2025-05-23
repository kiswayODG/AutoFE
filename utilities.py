import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import io
import matplotlib.pyplot as plt
import seaborn as sns

def load_uploaded_csv_files(file):
    """Charge un fichier CSV téléchargé."""
    try:
        # Vérifier si le fichier a déjà été lu
        if hasattr(file, 'name') and file.name in st.session_state.dataframes:
            return st.session_state.dataframes[file.name].copy()
        
        # Lire le fichier
        file_bytes = file.read()
        file.seek(0)  # Réinitialiser le pointeur de fichier
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
        except UnicodeDecodeError:
            # Essayer une autre encodage si utf-8 échoue
            file.seek(0)
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='latin1')
        
        # Stocker le dataframe dans la session
        if hasattr(file, 'name'):
            st.session_state.dataframes[file.name] = df.copy()
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {e}")
        return None

def handle_missing_values(df, method='Imputation (moyenne)'):
    """Traite les valeurs manquantes selon la méthode spécifiée."""
    if method == "Supprimer les lignes":
        return df.dropna()
    elif method == "Supprimer les colonnes":
        return df.dropna(axis=1)
    elif method == "Imputation (moyenne)":
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='mean')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
        
        return df
    elif method == "Imputation (médiane)":
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
        
        return df
    elif method == "Imputation (mode)":
        imputer = SimpleImputer(strategy='most_frequent')
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df

def encode_categorical_variables(df, categorical_cols):
    """Encode les variables catégorielles avec LabelEncoder."""
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders

def plot_confusion_matrix(cm, class_names):
    """Affiche une matrice de confusion avec seaborn."""
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Prédiction')
    plt.ylabel('Valeur réelle')
    plt.title('Matrice de confusion')
    return plt

def plot_feature_importance(importance, feature_names, title='Importance des caractéristiques'):
    """Affiche l'importance des caractéristiques."""
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.bar(range(len(indices)), importance[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    return plt

def plot_regression_results(y_true, y_pred):
    """Affiche les résultats d'un modèle de régression."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.title('Valeurs réelles vs. Prédictions')
    return plt

def plot_precision_recall_curve(precision, recall, avg_precision):
    """Affiche la courbe précision-rappel."""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Précision-Rappel (AP = {avg_precision:.2f})')
    plt.xlabel('Rappel')
    plt.ylabel('Précision')
    plt.title('Courbe Précision-Rappel')
    plt.legend(loc="lower left")
    return plt


