from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor


import matplotlib.pyplot as plt
import seaborn as sns
import time

def modeling_section():
    """Section de modélisation et entraînement."""
    st.header("3. Modélisation")
    
    if not st.session_state.feature_engineering_done:
        st.error("Veuillez d'abord compléter le feature engineering.")
        return
    
    st.write("""
    **Instructions :**
    1. Sélectionnez le modèle à utiliser selon le type de problème.
    2. Configurez les paramètres du modèle.
    3. Entraînez le modèle sur vos données.
    """)
    
    X = st.session_state.X
    columns = X.columns

    
    for col in X.columns:
        if X[col].dtype.name == 'category' or X[col].dtype == object:
            X[col], _ = pd.factorize(X[col])

   
    y = st.session_state.y
    target_type = st.session_state.target_type
    
     

    model_name = choose_model(target_type)
    

    model, params = configure_model(model_name, target_type)
    

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Taille de l'ensemble de test (%)", 
                            min_value=10, max_value=50, value=20, step=5) / 100
    
    with col2:
        random_state = st.number_input("Graine aléatoire (random_state)", 
                                     min_value=0, max_value=1000, value=42)
    
    normalize_data = st.checkbox("Normaliser les données", value=True)
    
    

    if st.button("Entraîner le modèle"):
        with st.spinner(f"Entraînement du modèle {model_name} en cours..."):
            try:
                print((X))
                if normalize_data:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X_data = pd.DataFrame(X_scaled, columns=columns)
                    
                else:
                    X_data = X
                

                X_train, X_test, y_train, y_test = train_test_split(
                    X_data, y, test_size=test_size, random_state=random_state
                )
                

                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                

                y_pred = model.predict(X_test)
                

                cv_scores = cross_val_score(model, X_data, y)
                

                train_sizes, train_scores, test_scores = learning_curve(
                    model, X_data, y, n_jobs=-1, 
                    train_sizes=np.linspace(0.1, 1.0, 10)
                )
                

                st.session_state.model = model
                st.session_state.model_name = model_name
                st.session_state.model_results = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'training_time': training_time,
                    'cv_scores': cv_scores,
                    'learning_curve': {
                        'train_sizes': train_sizes,
                        'train_scores': train_scores,
                        'test_scores': test_scores
                    }
                }
                

                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                    st.session_state.feature_importance = {
                        'importance': feature_importance,
                        'features': columns
                    }
                elif hasattr(model, 'coef_'):

                    feature_importance = np.abs(model.coef_)
                    if feature_importance.ndim > 1:  # For multi-class problems
                        feature_importance = np.mean(feature_importance, axis=0)
                    st.session_state.feature_importance = {
                        'importance': feature_importance,
                        'features': columns
                    }
                

                if target_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    st.success(f"Modèle entraîné avec succès! Précision: {accuracy:.4f}")
                    

                    cm = confusion_matrix(y_test, y_pred)
                    st.subheader("Matrice de confusion")
                    fig, ax = plt.subplots(figsize=(10, 7))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    plt.xlabel('Prédiction')
                    plt.ylabel('Valeur réelle')
                    st.pyplot(fig)
                    
                else:   
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.success(f"Modèle entraîné avec succès! MSE: {mse:.4f}, R²: {r2:.4f}")
                    

                    st.subheader("Prédictions vs. Valeurs Réelles")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                    ax.set_xlabel('Valeurs Réelles')
                    ax.set_ylabel('Prédictions')
                    st.pyplot(fig)
                
                

                st.session_state.model_trained = True
               
                
            except Exception as e:
                st.error(f"Erreur pendant l'entraînement du modèle: {e}")
    if st.session_state.model_trained:            
        if st.button("Continuer vers l'analyse des résultats"):
            st.session_state.active_tab = "Analyse des résultats"
            print("on est rentré dans l'analyse des résultats")
            st.rerun()

def choose_model(target_type):
    """Sélection du modèle selon le type de problème."""
    if target_type == "Classification":
        return st.selectbox(
            "Choisissez le modèle de classification",
            [
                "Random Forest Classifier",
                "XGBoost Classifier",
                "Logistic Regression",
                "Support Vector Machine (SVM)",
                "K-Nearest Neighbors (KNN)",
                "Naive Bayes",
                "Decision Tree Classifier",
                "Gradient Boosting Classifier",
                "AdaBoost Classifier",
                "Neural Networks (MLPClassifier)",
                "CatBoost Classifier"
            ]
        )
    else:   
        return st.selectbox(
            "Choisissez le modèle de régression",
            [
                "Random Forest Regressor",
                "Linear Regression",
                "XGBoost Regressor",
                "Support Vector Regressor (SVR)",
                "K-Nearest Neighbors Regressor (KNN)",
                "Decision Tree Regressor",
                "Gradient Boosting Regressor",
                "AdaBoost Regressor",
                "Ridge Regression",
                "Lasso Regression",
                "Elastic Net Regression",
                "CatBoost Regressor",
            ]
        )

def configure_model(model_name, target_type):
    """Configure le modèle sélectionné avec les paramètres appropriés."""
    st.subheader("Configuration du modèle")
    

    params = {}
    model = None
    

    if model_name == "Random Forest Classifier":
        n_estimators = st.slider("Nombre d'arbres", 50, 500, 100)
        max_depth = st.slider("Profondeur maximale", 2, 30, 10)
        min_samples_split = st.slider("Échantillons minimums pour division", 2, 20, 2)
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': 42
        }
        model = RandomForestClassifier(**params)
    
    elif model_name == "Random Forest Regressor":
        n_estimators = st.slider("Nombre d'arbres", 50, 500, 100)
        max_depth = st.slider("Profondeur maximale", 2, 30, 10)
        min_samples_split = st.slider("Échantillons minimums pour division", 2, 20, 2)
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': 42
        }
        model = RandomForestRegressor(**params)
    
    elif model_name == "XGBoost Classifier":
        n_estimators = st.slider("Nombre d'arbres", 50, 500, 100)
        learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, step=0.01)
        max_depth = st.slider("Profondeur maximale", 2, 15, 6)
        
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'random_state': 42
        }
        model = XGBClassifier(**params)
    
    elif model_name == "XGBoost Regressor":
        n_estimators = st.slider("Nombre d'arbres", 50, 500, 100)
        learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, step=0.01)
        max_depth = st.slider("Profondeur maximale", 2, 15, 6)
        
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'random_state': 42
        }
        model = XGBRegressor(**params)
    
    elif model_name == "Logistic Regression":
        C = st.slider("Inverse de la force de régularisation", 0.01, 10.0, 1.0)
         
        params = {
            'C': C, 
            'max_iter': 1000,
            'random_state': 42
        }
        model = LogisticRegression(**params)
    
    elif model_name == "Linear Regression":
        fit_intercept = st.checkbox("Ajuster l'ordonnée à l'origine", value=True)
        
        params = {
            'fit_intercept': fit_intercept,
            'n_jobs': -1
        }
        model = LinearRegression(**params)
    
    elif model_name == "Support Vector Machine (SVM)":
        C = st.slider("Paramètre de régularisation C", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Noyau", ["linear", "poly", "rbf", "sigmoid"])
        
        params = {
            'C': C,
            'kernel': kernel,
            'random_state': 42
        }
        model = SVC(**params, probability=True)
    
    elif model_name == "Support Vector Regressor (SVR)":
        C = st.slider("Paramètre de régularisation C", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Noyau", ["linear", "poly", "rbf", "sigmoid"])
        
        params = {
            'C': C,
            'kernel': kernel
        }
        model = SVR(**params)
    
    elif model_name == "K-Nearest Neighbors (KNN)":
        n_neighbors = st.slider("Nombre de voisins", 1, 20, 5)
        weights = st.selectbox("Méthode de pondération", ["uniform", "distance"])
        
        params = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'n_jobs': -1
        }
        model = KNeighborsClassifier(**params)
    
    elif model_name == "K-Nearest Neighbors Regressor (KNN)":
        n_neighbors = st.slider("Nombre de voisins", 1, 20, 5)
        weights = st.selectbox("Méthode de pondération", ["uniform", "distance"])
        
        params = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'n_jobs': -1
        }
        model = KNeighborsRegressor(**params)
    
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    
    elif model_name == "Decision Tree Classifier":
        max_depth = st.slider("Profondeur maximale", 2, 30, 10)
        min_samples_split = st.slider("Échantillons minimums pour division", 2, 20, 2)
        
        params = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': 42
        }
        model = DecisionTreeClassifier(**params)
    
    elif model_name == "Decision Tree Regressor":
        max_depth = st.slider("Profondeur maximale", 2, 30, 10)
        min_samples_split = st.slider("Échantillons minimums pour division", 2, 20, 2)
        
        params = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': 42
        }
        model = DecisionTreeRegressor(**params)
    
    elif model_name == "Gradient Boosting Classifier":
        n_estimators = st.slider("Nombre d'arbres", 50, 500, 100)
        learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, step=0.01)
        max_depth = st.slider("Profondeur maximale", 2, 15, 3)
        
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'random_state': 42
        }
        model = GradientBoostingClassifier(**params)
    
    elif model_name == "Gradient Boosting Regressor":
        n_estimators = st.slider("Nombre d'arbres", 50, 500, 100)
        learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, step=0.01)
        max_depth = st.slider("Profondeur maximale", 2, 15, 3)
        
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'random_state': 42
        }
        model = GradientBoostingRegressor(**params)
    
    elif model_name == "AdaBoost Classifier":
        n_estimators = st.slider("Nombre d'estimateurs", 50, 500, 100)
        learning_rate = st.slider("Taux d'apprentissage", 0.01, 2.0, 1.0, step=0.01)
        
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'random_state': 42
        }
        model = AdaBoostClassifier(**params)
    
    elif model_name == "AdaBoost Regressor":
        n_estimators = st.slider("Nombre d'estimateurs", 50, 500, 100)
        learning_rate = st.slider("Taux d'apprentissage", 0.01, 2.0, 1.0, step=0.01)
        
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'random_state': 42
        }
        model = AdaBoostRegressor(**params)
    
   
    
    elif model_name == "Ridge Regression":
        alpha = st.slider("Paramètre de régularisation alpha", 0.01, 10.0, 1.0)
        
        params = {
            'alpha': alpha,
            'random_state': 42
        }
        model = Ridge(**params)
    
    elif model_name == "Lasso Regression":
        alpha = st.slider("Paramètre de régularisation alpha", 0.01, 10.0, 1.0)
        
        params = {
            'alpha': alpha,
            'random_state': 42
        }
        model = Lasso(**params)
    
    elif model_name == "Elastic Net Regression":
        alpha = st.slider("Paramètre de régularisation alpha", 0.01, 10.0, 1.0)
        l1_ratio = st.slider("Ratio L1", 0.0, 1.0, 0.5, step=0.01)
        
        params = {
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'random_state': 42
        }
        model = ElasticNet(**params)
    

    
    elif model_name == "CatBoost Classifier":
        n_estimators = st.slider("Nombre d'itérations", 50, 500, 100)
        learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, step=0.01)
        depth = st.slider("Profondeur", 2, 15, 6)
        
        params = {
            'iterations': n_estimators,
            'learning_rate': learning_rate,
            'depth': depth,
            'random_state': 42
        }
        model = CatBoostClassifier(**params, verbose=0)
    
    elif model_name == "CatBoost Regressor":
        n_estimators = st.slider("Nombre d'itérations", 50, 500, 100)
        learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, step=0.01)
        depth = st.slider("Profondeur", 2, 15, 6)
        
        params = {
            'iterations': n_estimators,
            'learning_rate': learning_rate,
            'depth': depth,
            'random_state': 42
        }
        model = CatBoostRegressor(**params, verbose=0)
    
    return model, params
