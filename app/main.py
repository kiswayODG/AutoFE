import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report
import featuretools as ft
import autofeat
import io
from utilities import load_uploaded_csv_files
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="AutoFE App",page_icon="📊", layout="centered")




st.title("Application d'Automated Feature Engineering")

st.write("""
    **Instructions :**
    1. Téléchargez d'abord la table principale contenant la colonne cible.
    2. Téléchargez ensuite les autres tables si nécessaire.
""")

uploaded_files = st.file_uploader("Téléchargez vos fichiers CSV", type=["csv"], accept_multiple_files=True)
target_type = " "
method = " "

col_specs = [1.5, 1.5, 1.5]  # Spécifications des colonnes : largeur relative
columns = st.columns(col_specs)
if uploaded_files:
    with columns[0]:
        method = st.selectbox("Choisissez la méthode d'AutoFE", ["AutoFeat", "Featuretools", "Manuel"])
else:
    with columns[0]:
        if st.button("Choisir méthode d'AutoFE"):
            st.warning("Veuillez d'abord télécharger un fichier CSV.")




if uploaded_files:
   
    df = load_uploaded_csv_files(uploaded_files[0])
    with columns[1]:
        target = st.selectbox("Choisissez la colonne cible", df.columns)
    with columns[2]:
        target_type = st.selectbox("Type de la variable cible", ["Classification", "Régression"])

    
    st.write("Aperçu des données :")
    for file in uploaded_files:
        try:
            df = load_uploaded_csv_files(file)

            st.write(f"Fichier : {file.name}")
            with st.container():
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier {file.name} : {e}")
     
        missing_values = df.isnull().sum()
        if missing_values.any():
            st.write(f"Valeurs manquantes dans {file.name} :")
            st.write(missing_values[missing_values > 0])
            
            
            st.write("Choisissez une méthode pour traiter les valeurs manquantes :")
            missing_value_option = st.selectbox(
                "Méthode de traitement des valeurs manquantes",
                ["Supprimer les lignes", "Supprimer les colonnes", "Imputation (moyenne)", "Imputation (médiane)", "Imputation (mode)"],
                key=f"missing_value_option_{file.name}"
            )

            if missing_value_option == "Supprimer les lignes":
                df = df.dropna()
            elif missing_value_option == "Supprimer les colonnes":
                df = df.dropna(axis=1)
            elif missing_value_option == "Imputation (moyenne)":
                imputer = SimpleImputer(strategy='mean')
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            elif missing_value_option == "Imputation (médiane)":
                imputer = SimpleImputer(strategy='median')
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            elif missing_value_option == "Imputation (mode)":
                imputer = SimpleImputer(strategy='most_frequent')
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


if target_type == "Classification":
    with columns[0]:
        model = st.selectbox(
        "Choisissez le modèle de Machine Learning",
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
            "LightGBM Classifier",
            "CatBoost Classifier"
        ]
    )
elif target_type == "Régression":
    with columns[0]:
        model = st.selectbox(
        "Choisissez le modèle de Machine Learning",
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
            "LightGBM Regressor",
            "CatBoost Regressor",
            "Neural Networks (MLPRegressor)"
        ]
    )
    
if method == "Manuel":
    st.write("Sélectionnez les colonnes pour le feature engineering manuel :")
    df = load_uploaded_csv_files(uploaded_files[0])

    col1, col2, col3 = st.columns(3)

    with col1:
        target = st.selectbox(" Colonne cible", df.columns)

    with col2:
        possible_features = [col for col in df.columns if col != target]
        features = st.multiselect(" Colonnes de caractéristiques", possible_features)

    with col3:
        if not features:
            features = possible_features
            #st.info(" Toutes les colonnes (sauf la cible) sont utilisées.")
        cat_candidates = df[features].select_dtypes(include=["object", "category", "string"]).columns.tolist()
        categorical_cols = st.multiselect("Colonnes catégoriques", cat_candidates)

    st.write("Sélectionnez les colonnes pour appliquer des transformations :")
    transform_options = {
        "Racine carrée": np.sqrt,
        "Logarithme": np.log1p,
        "Inverse": np.reciprocal
    }

    transform_cols = {}
    cols_per_row = 3
    rows = (len(features) + cols_per_row - 1) // cols_per_row

    for i in range(rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < len(features):
                col_name = features[idx]
                with cols[j]:
                    transform_cols[col_name] = st.multiselect(
                        f"{col_name}",
                        list(transform_options.keys()),
                        key=f"transform_{col_name}"
                )
    
if method == "Featuretools" and len(uploaded_files) > 1:
    st.write("Spécifiez les colonnes de référence pour relier les tables :")
    reference_columns = {}
    for file in uploaded_files[1:]: 
        df = pd.read_csv(file)
        col = st.selectbox(f"Colonne de référence pour {file.name}", df.columns, key=f"ref_{file.name}")
        reference_columns[file.name] = col

if st.button("Exécuter"):
    if uploaded_files:
        with st.spinner("Traitement en cours... Veuillez patienter."):
            dfs = [load_uploaded_csv_files(file) for file in uploaded_files]

            if method == "AutoFeat":
                features = autofeat.AutoFeatRegressor()
                X = features.fit_transform(dfs[0].drop(columns=[target]), dfs[0][target])
                st.write(X)
                
                
            elif method == "Featuretools":
                es = ft.EntitySet(id="id")
                
                # Ajouter la table principale
                es.entity_from_dataframe(entity_id="table_0", dataframe=dfs[0], index="index_col")

                for i, df in enumerate(dfs[1:], start=1):
                    table_name = f"table_{i}"
                    es.entity_from_dataframe(entity_id=table_name, dataframe=df, index="index_col")
                    ref_col = reference_columns[uploaded_files[i].name]
                    es.relationships.append(ft.Relationship(es[f"table_{i-1}"]['index_col'], es[table_name][ref_col]))

                feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name="table_0")
                X = feature_matrix.drop(columns=[target])
                
                
            elif method == "Manuel":
                df = dfs[0]
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    
                for col, transforms in transform_cols.items():
                    for transform in transforms:
                        new_col_name = f"{col}_{transform}"
                        df[new_col_name] = transform_options[transform](df[col])
                        if new_col_name not in features:
                            features.append(new_col_name)
                X = df[features]

            y = dfs[0][target]

            if target_type == "Classification":
                if model == "Random Forest Classifier":
                    clf = RandomForestClassifier()
                elif model == "XGBoost Classifier":
                    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                elif model == "Logistic Regression":
                    clf = LogisticRegression()
                elif model == "Support Vector Machine (SVM)":
                    clf = SVC()
                elif model == "K-Nearest Neighbors (KNN)":
                    clf = KNeighborsClassifier()
                elif model == "Naive Bayes":
                    clf = GaussianNB()
                elif model == "Decision Tree Classifier":
                    clf = DecisionTreeClassifier()
                elif model == "Gradient Boosting Classifier":
                    clf = GradientBoostingClassifier()
                elif model == "AdaBoost Classifier":
                    clf = AdaBoostClassifier()
                elif model == "Neural Networks (MLPClassifier)":
                    clf = MLPClassifier()
                elif model == "LightGBM Classifier":
                    clf = LGBMClassifier()
                elif model == "CatBoost Classifier":
                    clf = CatBoostClassifier(verbose=0)
            elif target_type == "Régression":
                if model == "Random Forest Regressor":
                    clf = RandomForestRegressor()
                elif model == "Linear Regression":
                    clf = LinearRegression()
                elif model == "XGBoost Regressor":
                    clf = XGBRegressor()
                elif model == "Support Vector Regressor (SVR)":
                    clf = SVR()
                elif model == "K-Nearest Neighbors Regressor (KNN)":
                    clf = KNeighborsRegressor()
                elif model == "Decision Tree Regressor":
                    clf = DecisionTreeRegressor()
                elif model == "Gradient Boosting Regressor":
                    clf = GradientBoostingRegressor()
                elif model == "AdaBoost Regressor":
                    clf = AdaBoostRegressor()
                elif model == "Ridge Regression":
                    clf = Ridge()
                elif model == "Lasso Regression":
                    clf = Lasso()
                elif model == "Elastic Net Regression":
                    clf = ElasticNet()
                elif model == "LightGBM Regressor":
                    clf = LGBMRegressor()
                elif model == "CatBoost Regressor":
                    clf = CatBoostRegressor(verbose=0)
                elif model == "Neural Networks (MLPRegressor)":
                    clf = MLPRegressor()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)

            if target_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Précision du modèle : {accuracy:.2f}")

                st.write("Matrice de confusion :")
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(10, 7))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Prédit')
                plt.ylabel('Réel')
                st.pyplot(plt)

                st.write("Rapport de classification :")
                st.text(classification_report(y_test, y_pred))

            elif target_type == "Régression":
                mse = mean_squared_error(y_test, y_pred)
                
                st.write("Racinne -- Erreur quadratique moyenne (MSE) :")
                st.write(f"{np.sqrt(mse):.2f}")
                        
                st.write("Précision :")
                r2 = r2_score(y_test, y_pred)
                st.write(f"{r2:.2f}")
            

                st.write("Valeurs réelles vs prédites :")
                plt.figure(figsize=(10, 7))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.xlabel('Réel')
                plt.ylabel('Prédit')
                plt.title('Réel vs Prédit')
                st.pyplot(plt)
                
            st.write("Téléchargez le dataset complété :")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                X.to_excel(writer, sheet_name='Dataset Complété', index=False)
            output.seek(0)
            st.download_button(label="Télécharger le dataset complété", data=output, file_name="dataset_complet.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("Veuillez charger un fichier CSV et choisir une méthode d'AutoFE.")

