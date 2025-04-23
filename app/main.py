import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import featuretools as ft
import autofeat

st.set_page_config(page_title="AutoFE App", layout="wide")

st.title("Application d'Automated Feature Engineering")

uploaded_files = st.file_uploader("Téléchargez vos fichiers CSV", type=["csv"], accept_multiple_files=True)

method = st.selectbox("Choisissez la méthode d'AutoFE", ["AutoFeat", "Featuretools", "Manuel"])
target_type = " "

if uploaded_files:
    df = pd.read_csv(uploaded_files[0])
    target = st.selectbox("Choisissez la colonne cible", df.columns)
    target_type = st.selectbox("Type de la variable cible", ["Classification", "Régression"])


if target_type == "Classification":
    model = st.selectbox("Choisissez le modèle de Machine Learning", ["Random Forest", "XGBoost", "Logistic Regression"])
elif target_type == "Régression":
    model = st.selectbox("Choisissez le modèle de Machine Learning", ["Random Forest Regressor", "Linear Regression", "XGBoost Regressor"])


if method == "Featuretools" and len(uploaded_files) > 1:
    st.write("Spécifiez les colonnes de référence pour relier les tables :")
    reference_columns = {}
    for file in uploaded_files:
        df = pd.read_csv(file)
        col = st.selectbox(f"Colonne de référence pour {file.name}", df.columns)
        reference_columns[file.name] = col

if st.button("Exécuter"):
    dfs = [pd.read_csv(file) for file in uploaded_files]

    if method == "AutoFeat":
        features = autofeat.AutoFeatRegressor()
        X = features.fit_transform(dfs[0].drop(columns=[target]), df[target])
    elif method == "Featuretools":
        es = ft.EntitySet(id="id")
        for i, df in enumerate(dfs):
            es.entity_from_dataframe(entity_id=f"table_{i}", dataframe=df, index="index_col")
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name="table_0")
        X = feature_matrix.drop(columns=[target])
    elif method == "Manuel":
        X = dfs[0].drop(columns=[target])

    y = df[target]
    if model == "Random Forest":
        clf = RandomForestClassifier()
    elif model == "XGBoost":
        clf = XGBClassifier()
    clf.fit(X, y)

    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    st.write(f"Précision du modèle : {accuracy:.2f}")


'''
par la suite vous pouvez ajouter des fonctionnalités supplémentaires telles que la visualisation des résultats, l'optimisation des hyperparamètres, etc.

lorsque un utilisateur upload son jeu de données il doit pouvoir avoir un aperçu de ses données, et choisir la colonne cible, le type de la variable cible (classification ou régression), et le modèle de machine learning qu'il souhaite utiliser.

Pour le manuel feature engineering, on va ajouter une section où l'utilisateur peut sélectionner les colonnes qu'il souhaite utiliser comme caractéristiques et la colonne cible.
aussi lui permttre de choisir des colonnes à catégoriser,
et d'autres options de prétraitement des données de feature enginering comme faire mettre sous racine, faire le log, .....


revoir la partie de featuretools pour qu'il puisse relier les tables entre elles, et faire le feature engineering sur l'ensemble des tables.

revoir aussi le design de l'interface pour qu'il soit plus convivial et intuitif.


'''